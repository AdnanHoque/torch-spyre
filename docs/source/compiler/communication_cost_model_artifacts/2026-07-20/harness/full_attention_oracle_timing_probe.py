#!/usr/bin/env python3
"""Warm HBM/LX/oracle timing for full stable-softmax attention."""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from core.op_parser import resolve_custom_op
from core.profiler import create_profiler
from ops.builtin_ops import (
    create_tensors,
    get_operation_target,
    get_tensor_preparer,
    prepare_module_target,
)


OP = "experimental.flash_attn_stable_softmax"
B, H, LQ, LK, D = 1, 4, 512, 4096, 128
GROUP_SIZE = 8
SHAPES = [(B, H, LQ, D), (B, H, LK, D), (B, H, LK, D)]
RUNS = int(os.environ.get("FULL_ATTN_RUNS", "30"))
SEED = int(os.environ.get("FULL_ATTN_SEED", "0"))
MODE = os.environ["FULL_ATTN_MODE"]
if MODE not in {"hbm", "lx", "oracle"}:
    raise ValueError(f"unsupported FULL_ATTN_MODE={MODE!r}")


NORMAL_INVENTORY = [
    "mul",
    "mul",
    "ReStickifyOpHBM",
    "shuffle",
    "batchmatmul",
    "max",
    "sub",
    "exp",
    "sum",
    "reciprocal",
    "mul",
    "batchmatmul",
]
ORACLE_INVENTORY = [op for op in NORMAL_INVENTORY if op != "shuffle"]
HBM_INVENTORY = ORACLE_INVENTORY
PREFIX_INVENTORY = NORMAL_INVENTORY[: NORMAL_INVENTORY.index("shuffle") + 1]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def allocations(spec: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for wrapper in spec.get("dscs_", []):
        dsc = next(iter(wrapper.values()))
        for node in dsc.get("scheduleTree_", []):
            if node.get("nodeType_") != "allocate":
                continue
            result.append(
                {
                    key: copy.deepcopy(node.get(key))
                    for key in (
                        "name_",
                        "ldsIdx_",
                        "component_",
                        "layoutDimOrder_",
                        "maxDimSizes_",
                        "startAddressCoreCorelet_",
                        "coordinates_",
                    )
                }
            )
    return result


def artifact_report(cache: Path) -> dict[str, Any]:
    roots: list[dict[str, Any]] = []
    for path in sorted(cache.rglob("sdsc_*.json")):
        root_name, spec = next(iter(json.loads(path.read_text()).items()))
        canonical = copy.deepcopy(spec)
        canonical.pop("debug_handle_", None)
        roots.append(
            {
                "root_name": root_name,
                "op": root_name.split("_", 1)[1],
                "canonical_sha256": json_hash(canonical),
                "allocations": allocations(spec),
            }
        )
    roots.sort(key=lambda root: int(root["root_name"].split("_", 1)[0]))
    bundles = sorted(cache.rglob("bundle.mlir"))
    if len(bundles) != 1:
        raise RuntimeError(f"expected one bundle in {cache}, found {bundles}")
    bundle = bundles[0]
    bundle_text = bundle.read_text()
    return {
        "roots": roots,
        "op_inventory": [root["op"] for root in roots],
        "bundle": str(bundle.relative_to(cache)),
        "bundle_token": bundle.parent.name,
        "bundle_sha256": sha256_bytes(bundle.read_bytes()),
        "bundle_input_arg_extract_count": bundle_text.count("input_arg_extract"),
        "bundle_s2_synthetic_name_mentions": bundle_text.count(
            "__spyre_lx_relayout_destination__"
        ),
    }


def one_root(report: dict[str, Any], op: str, occurrence: int = 0) -> dict[str, Any]:
    matches = [root for root in report["roots"] if root["op"] == op]
    if occurrence >= len(matches):
        raise RuntimeError(
            f"missing {op} occurrence {occurrence}; inventory={report['op_inventory']}"
        )
    return matches[occurrence]


def allocation_for_lds(root: dict[str, Any], lds_idx: int) -> dict[str, Any]:
    matches = [a for a in root["allocations"] if a["ldsIdx_"] == lds_idx]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one LDS {lds_idx} allocation in {root['root_name']}: {matches}"
        )
    return matches[0]


def physical_allocation(allocation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(allocation[key])
        for key in (
            "component_",
            "layoutDimOrder_",
            "maxDimSizes_",
            "startAddressCoreCorelet_",
        )
    }


def restickify_components(report: dict[str, Any]) -> tuple[str, str]:
    root = one_root(report, "ReStickifyOpHBM")
    source = allocation_for_lds(root, 0)["component_"]
    destination = allocation_for_lds(root, 1)["component_"]
    return str(source), str(destination)


def main() -> None:
    run_dir = Path(os.environ["FULL_ATTN_RUN_DIR"])
    trace_dir = run_dir / "trace"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    import torch_spyre
    from torch_spyre._inductor import config as spyre_config
    from torch_spyre._inductor.propagate_hints import (
        _reset_counter as reset_hint_counter,
    )
    from torch_spyre._inductor.propagate_named_dims import reset as reset_named_dims

    torch_spyre._autoload()
    torch._dynamo.reset()
    torch.random.default_generator.manual_seed(SEED)
    custom_module, resolved_op = resolve_custom_op(OP, None)
    cpu_tensors = create_tensors(torch, SHAPES, resolved_op, "tsp", custom_module)
    query_cpu, key_cpu, value_cpu = cpu_tensors
    query_cpu = torch.where(query_cpu >= 0.5, 2.0, -2.0).to(torch.float16)
    key_cpu = torch.where(key_cpu >= 0.5, 2.0, -2.0).to(torch.float16)
    cpu_tensors = (query_cpu.contiguous(), key_cpu.contiguous(), value_cpu.contiguous())

    reference = get_operation_target(resolved_op, torch, "cpu", SHAPES, custom_module)
    with torch.no_grad():
        expected = reference(*cpu_tensors).cpu().clone()

    device = torch.device("spyre")
    run_tensors = tuple(t.to(device) for t in cpu_tensors)
    tensor_preparer = get_tensor_preparer(
        torch, SHAPES, resolved_op, "tsp", custom_module
    )

    normal_target = get_operation_target(
        resolved_op, torch, "tsp", SHAPES, custom_module
    )
    prefix_target = get_operation_target(
        resolved_op, torch, "tsp", SHAPES, custom_module
    )
    oracle_target = get_operation_target(
        resolved_op, torch, "tsp", SHAPES, custom_module
    )

    class NormalGraph(torch.nn.Module):
        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            return normal_target(q, k, v)

    class PrefixGraph(torch.nn.Module):
        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            return prefix_target(q, k, v)

    class OracleGraph(torch.nn.Module):
        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            return oracle_target(q, k, v)

    configs = {
        "timed": {
            "test_preseeded_lx_relayout": False,
            "test_lx_relayout_preseed_only": False,
        },
        "prefix": {
            "test_preseeded_lx_relayout": False,
            "test_lx_relayout_preseed_only": True,
        },
        "oracle": {
            "test_preseeded_lx_relayout": True,
            "test_lx_relayout_preseed_only": False,
        },
    }
    if MODE == "oracle":
        caches = {
            "prefix": run_dir / "prefix_cache",
            "oracle": run_dir / "oracle_cache",
        }
    else:
        caches = {"timed": run_dir / "cache"}
    for cache in caches.values():
        cache.mkdir(parents=True, exist_ok=True)

    def prepare() -> tuple[torch.Tensor, ...]:
        reset_named_dims()
        reset_hint_counter()
        return tensor_preparer(run_tensors)

    def compile_first(
        name: str, graph: torch.nn.Module, *, copy_output: bool
    ) -> tuple[Any, torch.Tensor | None]:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(caches[name])
        prepared = prepare()
        try:
            with spyre_config.patch(configs[name]):
                compiled = torch.compile(
                    prepare_module_target(graph, cpu_tensors, device=device),
                    fullgraph=True,
                )
            with torch.no_grad(), spyre_config.patch(configs[name]):
                result = compiled(*prepared)
                torch.spyre.synchronize()
                copied = result.cpu().clone() if copy_output else None
            return compiled, copied
        finally:
            reset_named_dims()
            reset_hint_counter()

    def cached_call(name: str, compiled: Any) -> Any:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(caches[name])
        with torch.no_grad(), spyre_config.patch(configs[name]):
            return compiled(*run_tensors)

    prefix_compiled = None
    if MODE == "oracle":
        prefix_compiled, _ = compile_first(
            "prefix", PrefixGraph(), copy_output=False
        )
        timed_compiled, _ = compile_first(
            "oracle", OracleGraph(), copy_output=True
        )
        cached_call("prefix", prefix_compiled)
        torch.spyre.synchronize()
        warm = cached_call("oracle", timed_compiled)
        torch.spyre.synchronize()
        warm = warm.cpu().clone()
    else:
        timed_compiled, _ = compile_first(
            "timed", NormalGraph(), copy_output=True
        )
        warm = cached_call("timed", timed_compiled)
        torch.spyre.synchronize()
        warm = warm.cpu().clone()

    profiler = create_profiler(
        torch, str(trace_dir), profile_memory=True, with_stack=False
    )
    oracle_walls_ms: list[float] = []
    prefix_walls_ms: list[float] = []
    result = None
    profiler.start()
    if MODE == "oracle":
        assert prefix_compiled is not None
        for _ in range(RUNS):
            prefix_started = time.perf_counter()
            cached_call("prefix", prefix_compiled)
            torch.spyre.synchronize()
            prefix_walls_ms.append((time.perf_counter() - prefix_started) * 1000.0)

            oracle_started = time.perf_counter()
            result = cached_call("oracle", timed_compiled)
            torch.spyre.synchronize()
            oracle_walls_ms.append((time.perf_counter() - oracle_started) * 1000.0)
            profiler.step()
    else:
        for _ in range(RUNS):
            started = time.perf_counter()
            result = cached_call("timed", timed_compiled)
            torch.spyre.synchronize()
            oracle_walls_ms.append((time.perf_counter() - started) * 1000.0)
            profiler.step()
    profiler.stop()

    assert result is not None
    actual = result.cpu().clone()
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    close = torch.isclose(actual, expected, rtol=1e-2, atol=1e-2)
    correctness_gate = bool(
        close.all()
        and torch.isfinite(actual).all()
        and torch.isfinite(expected).all()
    )
    traces = sorted(glob.glob(str(trace_dir / "*.pt.trace.json")))

    if MODE == "oracle":
        prefix_art = artifact_report(caches["prefix"])
        oracle_art = artifact_report(caches["oracle"])
        prefix_s2 = allocation_for_lds(one_root(prefix_art, "shuffle"), 1)
        oracle_s2 = allocation_for_lds(
            one_root(oracle_art, "batchmatmul", 0), 1
        )
        materialization_gates = {
            "prefix_inventory_exact": prefix_art["op_inventory"]
            == PREFIX_INVENTORY,
            "prefix_ends_at_shuffle": prefix_art["op_inventory"][-1:]
            == ["shuffle"],
            "prefix_has_no_consumer": "batchmatmul"
            not in prefix_art["op_inventory"],
            "oracle_inventory_exact": oracle_art["op_inventory"]
            == ORACLE_INVENTORY,
            "prefix_restickify_lx_to_lx": restickify_components(prefix_art)
            == ("lx", "lx"),
            "prefix_s2_physical_equals_oracle_consumer": physical_allocation(
                prefix_s2
            )
            == physical_allocation(oracle_s2),
            "oracle_s2_is_lx": oracle_s2["component_"] == "lx",
            "oracle_no_hidden_s2_bundle_input": oracle_art[
                "bundle_s2_synthetic_name_mentions"
            ]
            == 0,
        }
        materialization_gate = all(materialization_gates.values())
        artifact_data: dict[str, Any] = {
            "prefix_artifacts": prefix_art,
            "oracle_artifacts": oracle_art,
            "prefix_s2_allocation": prefix_s2,
            "oracle_consumer_s2_allocation": oracle_s2,
        }
        prefix_token = prefix_art["bundle_token"]
        oracle_token = oracle_art["bundle_token"]
        materialization_class = "oracle_prefix_then_no_shuffle"
    else:
        timed_art = artifact_report(caches["timed"])
        expected_inventory = HBM_INVENTORY if MODE == "hbm" else NORMAL_INVENTORY
        expected_restickify = ("lx", "hbm") if MODE == "hbm" else ("lx", "lx")
        materialization_gates = {
            "inventory_exact": timed_art["op_inventory"] == expected_inventory,
            "restickify_path_exact": restickify_components(timed_art)
            == expected_restickify,
            "shuffle_count_exact": timed_art["op_inventory"].count("shuffle")
            == (1 if MODE == "lx" else 0),
        }
        materialization_gate = all(materialization_gates.values())
        artifact_data = {"timed_artifacts": timed_art}
        prefix_token = None
        oracle_token = timed_art["bundle_token"]
        materialization_class = (
            "hbm_no_shuffle_lx_to_hbm"
            if MODE == "hbm"
            else "lx_one_shuffle_lx_to_lx"
        )

    summary = {
        "mode": MODE,
        "seed": SEED,
        "runs": RUNS,
        "op": OP,
        "shapes": SHAPES,
        "group_size": GROUP_SIZE,
        "source_bytes_per_core": H * LK * D * 2 // (H * GROUP_SIZE),
        "trace": traces[-1] if traces else None,
        "expected_trace_kernel_events": RUNS * (2 if MODE == "oracle" else 1),
        "measured_kernel_events": RUNS,
        "setup_kernel_events": RUNS if MODE == "oracle" else 0,
        "prefix_bundle_token": prefix_token,
        "oracle_bundle_token": oracle_token,
        "oracle_only_wall_ms_mean": statistics.fmean(oracle_walls_ms),
        "oracle_only_wall_ms_median": statistics.median(oracle_walls_ms),
        "prefix_wall_ms_mean": (
            statistics.fmean(prefix_walls_ms) if prefix_walls_ms else None
        ),
        "prefix_wall_ms_median": (
            statistics.median(prefix_walls_ms) if prefix_walls_ms else None
        ),
        "warm_correct": bool(
            torch.isclose(warm, expected, rtol=1e-2, atol=1e-2).all()
        ),
        "allclose_rtol_1e2_atol_1e2": bool(close.all()),
        "bitwise_equal_to_cpu": bool(torch.equal(actual, expected)),
        "actual_finite": bool(torch.isfinite(actual).all()),
        "expected_finite": bool(torch.isfinite(expected).all()),
        "mismatch_count": int((~close).sum().item()),
        "element_count": close.numel(),
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "correctness_gate": correctness_gate,
        "materialization_gates": materialization_gates,
        "materialization_gate": materialization_gate,
        "materialization_class": materialization_class,
        "prefix_setup_excluded_by_strict_trace_classification": MODE == "oracle",
        **artifact_data,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "mode",
                    "seed",
                    "runs",
                    "trace",
                    "warm_correct",
                    "correctness_gate",
                    "materialization_gates",
                    "materialization_gate",
                    "materialization_class",
                    "prefix_bundle_token",
                    "oracle_bundle_token",
                )
            },
            sort_keys=True,
        )
    )
    if not (summary["warm_correct"] and correctness_gate and materialization_gate):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
