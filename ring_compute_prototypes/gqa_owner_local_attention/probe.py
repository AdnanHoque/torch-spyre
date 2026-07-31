#!/usr/bin/env python3
"""Compile, validate, and time owner-local multi-block GQA decode attention."""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from model import DecodeGQAConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--q-route", choices=("broadcast", "hbm"), default="broadcast"
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--query-tokens", type=int, default=1)
    parser.add_argument("--physical-query-rows", type=int, default=64)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--debug-plans", action="store_true")
    return parser.parse_args()


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
                "canonical_sha256": hashlib.sha256(
                    json.dumps(
                        canonical, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "num_cores": spec.get("numCoresUsed_"),
                "physical_core_ids": [
                    int(core) for core in spec.get("coreIdToDsc_", {})
                ],
                "work_slices": copy.deepcopy(spec.get("numWkSlicesPerDim_")),
                "core_id_to_work_slice": copy.deepcopy(
                    spec.get("coreIdToWkSlice_")
                ),
                "allocations": allocations(spec),
                "logical_shapes": [
                    copy.deepcopy(next(iter(wrapper.values())).get("N_"))
                    for wrapper in spec.get("dscs_", [])
                    if next(iter(wrapper.values())).get("N_") is not None
                ],
            }
        )
    roots.sort(key=lambda root: int(root["root_name"].split("_", 1)[0]))
    bundles = sorted(cache.rglob("bundle.mlir"))
    if len(bundles) != 1:
        raise RuntimeError(f"expected one bundle in {cache}, found {bundles}")
    bundle = bundles[0]
    return {
        "roots": roots,
        "op_inventory": [root["op"] for root in roots],
        "bundle": str(bundle),
        "bundle_token": bundle.parent.name,
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
    }


def allocation_components(root: dict[str, Any]) -> list[str]:
    by_lds = {
        row["ldsIdx_"]: str(row["component_"])
        for row in root["allocations"]
        if isinstance(row["ldsIdx_"], int)
    }
    return [by_lds[index] for index in sorted(by_lds)]


def logical_element_count(root: dict[str, Any]) -> int:
    shapes = root.get("logical_shapes") or []
    if len(shapes) != 1:
        raise RuntimeError(
            f"expected one logical shape in {root['root_name']}: {shapes}"
        )
    extents = [
        value
        for name, value in shapes[0].items()
        if name in {"x_", "mb_", "out_", "in_"}
    ]
    if not extents or not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in extents
    ):
        raise RuntimeError(
            f"non-concrete logical shape in {root['root_name']}: {shapes[0]}"
        )
    return math.prod(extents)


def contiguous_kv_owner_grid(root: dict[str, Any]) -> bool:
    """Recognize core=4*kv_head+owner independent of SDSC dim labels."""

    work_slices = root.get("work_slices") or {}
    mapping = root.get("core_id_to_work_slice") or {}
    if root.get("num_cores") != 32 or len(mapping) != 32:
        return False
    kv_dims = [name for name, count in work_slices.items() if count == 8]
    owner_dims = [name for name, count in work_slices.items() if count == 4]
    for kv_dim in kv_dims:
        for owner_dim in owner_dims:
            if all(
                mapping.get(str(core), {}).get(kv_dim) == core // 4
                and mapping.get(str(core), {}).get(owner_dim) == core % 4
                for core in range(32)
            ):
                return True
    return False


def one_root(artifacts: dict[str, Any], op: str) -> dict[str, Any]:
    matches = [root for root in artifacts["roots"] if root["op"] == op]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {op}, got {len(matches)}")
    return matches[0]


def roots_containing(
    artifacts: dict[str, Any], fragment: str
) -> list[dict[str, Any]]:
    return [root for root in artifacts["roots"] if fragment in root["op"]]


def make_inputs(
    config: DecodeGQAConfig, seed: int, physical_query_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    valid_q = torch.randn(
        (
            config.batch,
            config.kv_heads,
            config.query_groups,
            config.query_tokens,
            config.head_dim,
        ),
        generator=generator,
        dtype=torch.float32,
    )
    if physical_query_rows % config.query_groups:
        raise ValueError("physical query rows must divide evenly across query groups")
    rows_per_group = physical_query_rows // config.query_groups
    if config.query_tokens > rows_per_group:
        raise ValueError("query tokens do not fit in one physical query-group lane")
    q = torch.zeros(
        (
            config.batch,
            config.kv_heads,
            physical_query_rows,
            config.head_dim,
        ),
        dtype=torch.float32,
    )
    k = torch.randn(
        (
            config.batch,
            config.kv_heads,
            config.context,
            config.head_dim,
        ),
        generator=generator,
        dtype=torch.float32,
    )
    kv_head = torch.arange(config.kv_heads, dtype=torch.float32).view(
        1, config.kv_heads, 1, 1, 1
    )
    owner = torch.arange(config.context, dtype=torch.float32).div(
        config.context // config.key_owners, rounding_mode="floor"
    ).view(
        1, 1, config.context, 1
    )
    local_block = torch.arange(config.context, dtype=torch.float32).remainder(
        config.context // config.key_owners
    ).div(
        config.block_size, rounding_mode="floor"
    ).view(
        1, 1, config.context, 1
    )
    query_group = torch.arange(config.query_groups, dtype=torch.float32).view(
        1, 1, config.query_groups, 1, 1
    )
    valid_q = 0.05 * valid_q + 0.003 * kv_head + 0.005 * query_group
    for group in range(config.query_groups):
        start = group * rows_per_group
        q[:, :, start : start + config.query_tokens, :] = valid_q[:, :, group]
    k = 0.05 * k + 0.004 * owner + 0.002 * local_block
    return (
        q.to(torch.float16).contiguous(),
        k.to(torch.float16).transpose(-1, -2).contiguous(),
    )


def computation(
    q: torch.Tensor,
    key_t: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    # Separate nontrivial producer roots make Q's source grid and K/V's owner
    # grid explicit.  The two QK scale factors multiply to 1/sqrt(D).
    q_lx = q * scale
    k_lx = key_t * scale
    scores = torch.matmul(q_lx, k_lx)
    # Reduce the unsplit query-row axis.  The N/context axis remains split over
    # stationary K owners, so this is an owner-local checksum rather than a
    # cross-owner softmax reduction.
    return torch.sum(scores, dim=-2)


def copy_to_cpu(
    result: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(result, tuple):
        return tuple(value.cpu().clone() for value in result)
    return result.cpu().clone()


def compare_tensor(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    close = torch.isclose(actual, expected, rtol=2e-2, atol=2e-2)
    per_kv_head = []
    if actual.ndim >= 2 and actual.shape[1] == 8:
        for kv_head in range(actual.shape[1]):
            head_difference = difference[:, kv_head]
            head_close = close[:, kv_head]
            per_kv_head.append(
                {
                    "kv_head": kv_head,
                    "mismatch_count": int((~head_close).sum()),
                    "element_count": head_close.numel(),
                    "max_abs_error": float(head_difference.max()),
                    "mean_abs_error": float(head_difference.mean()),
                }
            )
    return {
        "allclose_rtol_2e2_atol_2e2": bool(close.all()),
        "mismatch_count": int((~close).sum()),
        "element_count": close.numel(),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "actual_finite": bool(torch.isfinite(actual).all()),
        "expected_finite": bool(torch.isfinite(expected).all()),
        "per_kv_head": per_kv_head,
    }


def compare_tree(
    actual: torch.Tensor | tuple[torch.Tensor, ...],
    expected: torch.Tensor | tuple[torch.Tensor, ...],
) -> list[dict[str, Any]]:
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    if len(actual_values) != len(expected_values):
        raise RuntimeError("actual and expected output arity differ")
    return [
        compare_tensor(actual_value, expected_value)
        for actual_value, expected_value in zip(actual_values, expected_values)
    ]


def trace_report(
    trace_path: Path, *, bundle_token: str, expected_runs: int
) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text())
    events = [
        event
        for event in trace.get("traceEvents", [])
        if event.get("cat") == "kernel"
    ]
    matched = [
        float(event["dur"])
        for event in events
        if bundle_token in str(event.get("name"))
    ]
    return {
        "gate": len(events) == expected_runs
        and len(matched) == expected_runs
        and all(duration > 0 for duration in matched),
        "kernel_event_count": len(events),
        "matched_event_count": len(matched),
        "kernel_names": [str(event.get("name")) for event in events],
        "device_us": (
            {
                "median": statistics.median(matched),
                "mean": statistics.fmean(matched),
                "min": min(matched),
                "max": max(matched),
                "samples": matched,
            }
            if matched
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    config = DecodeGQAConfig(
        query_tokens=args.query_tokens,
        context=args.context,
    )
    if args.physical_query_rows < config.query_groups * config.query_tokens:
        raise ValueError("physical query rows cannot be smaller than logical Q")
    if args.run_dir.exists():
        raise FileExistsError(f"run directory already exists: {args.run_dir}")
    cache = args.run_dir / "cache"
    trace_dir = args.run_dir / "trace"
    cache.mkdir(parents=True)
    trace_dir.mkdir()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    os.environ.setdefault("SPYRE_LX_PLANNER_RELAYOUT", "1")
    os.environ.setdefault("DXP_LX_FRAC_AVAIL", "0.2")

    import torch_spyre
    try:
        from core.profiler import create_profiler
    except ModuleNotFoundError:
        from torch.profiler import ProfilerActivity, profile

        def create_profiler(
            torch_module: Any,
            trace_directory: str,
            *,
            profile_memory: bool,
            with_stack: bool,
        ) -> Any:
            return profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                profile_memory=profile_memory,
                with_stack=with_stack,
                on_trace_ready=torch_module.profiler.tensorboard_trace_handler(
                    trace_directory
                ),
            )
    from torch_spyre._inductor import config as spyre_config
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._inductor.propagate_hints import _reset_counter
    from torch_spyre._inductor.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
        reset as reset_named_dims,
    )

    if args.debug_plans:
        import torch_spyre._inductor.lx_relayout as relayout_module
        import torch_spyre._inductor.scratchpad.allocator as allocator_module

        original_ratio = relayout_module._destination_size_ratio
        original_collect = allocator_module.collect_lx_relayout_plans
        original_view = relayout_module._per_core_view_on_buf

        def debug_ratio(*ratio_args: Any, **ratio_kwargs: Any) -> int | None:
            value = original_ratio(*ratio_args, **ratio_kwargs)
            print(
                "LX_RELAYOUT_GEOMETRY "
                + json.dumps(
                    {
                        "producer_map": ratio_args[0],
                        "producer_splits": ratio_args[1],
                        "consumer_map": ratio_args[2],
                        "consumer_splits": ratio_args[3],
                        "destination_size_ratio": value,
                    },
                    sort_keys=True,
                )
            )
            return value

        def debug_collect(*collect_args: Any, **collect_kwargs: Any) -> Any:
            plans = original_collect(*collect_args, **collect_kwargs)
            print(
                "LX_RELAYOUT_PLANS "
                + json.dumps(
                    [
                        {
                            "source": plan.source_name,
                            "consumer": plan.consumer_name,
                            "destination_size_ratio": plan.destination_size_ratio,
                        }
                        for plan in plans
                    ],
                    sort_keys=True,
                )
            )
            return plans

        def debug_view(*view_args: Any, **view_kwargs: Any) -> Any:
            value = original_view(*view_args, **view_kwargs)
            print(
                "LX_RELAYOUT_VIEW "
                + json.dumps(
                    {
                        "op": view_args[0].get_name(),
                        "buffer": view_args[2],
                        "view": repr(value[0]),
                        "has_partial_reduction": value[1],
                        "representable": value[2],
                    },
                    sort_keys=True,
                )
            )
            return value

        relayout_module._destination_size_ratio = debug_ratio
        relayout_module._per_core_view_on_buf = debug_view
        allocator_module.collect_lx_relayout_plans = debug_collect

    torch_spyre._autoload()
    torch._dynamo.reset()
    scale = 1.0 / math.sqrt(math.sqrt(config.head_dim))
    work_div = {
        "num_kv_heads": config.kv_heads,
        "max_seqlen_kv": config.key_owners,
    }
    q_work_div = {"num_kv_heads": config.kv_heads}
    q_source_core_ids = tuple(
        config.key_owners * kv_head for kv_head in range(config.kv_heads)
    )
    cpu_inputs = make_inputs(config, args.seed, args.physical_query_rows)
    with torch.no_grad():
        expected = copy_to_cpu(
            computation(*cpu_inputs, scale=scale)
        )
    device = torch.device("spyre")
    device_inputs = tuple(value.to(device) for value in cpu_inputs)

    class Graph(torch.nn.Module):
        def forward(
            self,
            q: torch.Tensor,
            key_t: torch.Tensor,
        ) -> torch.Tensor:
            with spyre_hint(
                work_div=q_work_div,
                physical_core_ids=list(q_source_core_ids),
            ):
                q_lx = q * scale
            with spyre_hint(physical_core_order="work_div_inner_first"):
                with spyre_hint(work_div=work_div):
                    k_lx = key_t * scale
                    scores = torch.matmul(q_lx, k_lx)
                    return torch.sum(scores, dim=-2)

    def prepare() -> tuple[torch.Tensor, torch.Tensor]:
        reset_named_dims()
        _reset_counter()
        for name, size in (
            ("batch_size", config.batch),
            ("num_kv_heads", config.kv_heads),
            ("query_row", args.physical_query_rows),
            ("max_seqlen_kv", config.context),
            ("head_dim", config.head_dim),
        ):
            declare_tensor_dim(name, size)
        q, key_t = device_inputs
        name_tensor_dims(
            q,
            [
                "batch_size",
                "num_kv_heads",
                "query_row",
                "head_dim",
            ],
        )
        name_tensor_dims(
            key_t,
            [
                "batch_size",
                "num_kv_heads",
                "head_dim",
                "max_seqlen_kv",
            ],
        )
        return q, key_t

    prepared = prepare()
    patch = {
        "lx_planning": True,
        "lx_planner_relayout": args.q_route == "broadcast",
    }
    try:
        with spyre_config.patch(patch):
            compiled: Callable[..., Any] = torch.compile(
                Graph().to(device), fullgraph=True
            )
        with torch.no_grad(), spyre_config.patch(patch):
            compile_result = compiled(*prepared)
            torch.spyre.synchronize()
            compile_result = copy_to_cpu(compile_result)
    finally:
        reset_named_dims()
        _reset_counter()

    for _ in range(args.warmups):
        with torch.no_grad(), spyre_config.patch(patch):
            compiled(*device_inputs)
        torch.spyre.synchronize()

    profiler = create_profiler(
        torch, str(trace_dir), profile_memory=True, with_stack=False
    )
    walls_ms: list[float] = []
    measured = None
    profiler.start()
    for _ in range(args.runs):
        started = time.perf_counter()
        with torch.no_grad(), spyre_config.patch(patch):
            measured = compiled(*device_inputs)
        torch.spyre.synchronize()
        walls_ms.append((time.perf_counter() - started) * 1000.0)
        profiler.step()
    profiler.stop()
    if measured is None:
        raise RuntimeError("no measured output")
    measured = copy_to_cpu(measured)

    artifacts = artifact_report(cache)
    shuffles = roots_containing(artifacts, "shuffle")
    bmms = roots_containing(artifacts, "batchmatmul")
    expected_bmms = 1
    q_logical_elements = (
        config.batch
        * config.kv_heads
        * args.physical_query_rows
        * config.head_dim
    )
    q_shuffle_elements = (
        logical_element_count(shuffles[0]) if len(shuffles) == 1 else None
    )
    qk_bmm = bmms[0] if bmms else None
    muls = [root for root in artifacts["roots"] if root["op"] == "mul"]
    q_producer = muls[0] if muls else None
    shuffle_index = (
        artifacts["roots"].index(shuffles[0]) if len(shuffles) == 1 else -1
    )
    qk_index = artifacts["roots"].index(qk_bmm) if qk_bmm else -1
    common_structural_gates = {
        "expected_bmm_count": len(bmms) == expected_bmms,
        "q_producer_uses_sparse_cohort_roots": q_producer is not None
        and q_producer["physical_core_ids"] == list(q_source_core_ids),
        "qk_uses_contiguous_kv_owner_grid": qk_bmm is not None
        and contiguous_kv_owner_grid(qk_bmm),
        "multiple_k_blocks_per_owner": config.blocks_per_owner > 1,
    }
    if args.q_route == "broadcast":
        route_structural_gates = {
            "one_q_shuffle": len(shuffles) == 1,
            "q_shuffle_is_lx_to_lx": len(shuffles) == 1
            and allocation_components(shuffles[0]) == ["lx", "lx"],
            "q_shuffle_descriptor_is_exact_q": q_shuffle_elements
            == q_logical_elements,
            "all_bmm_operands_and_outputs_are_lx": len(bmms) == expected_bmms
            and all(
                allocation_components(root) == ["lx", "lx", "lx"]
                for root in bmms
            ),
            "q_shuffle_precedes_qk": 0 <= shuffle_index < qk_index,
            "no_restickify_between_q_shuffle_and_qk": 0
            <= shuffle_index
            < qk_index
            and all(
                "restickify" not in root["op"]
                for root in artifacts["roots"][shuffle_index + 1 : qk_index]
            ),
            "only_q_is_shuffled": len(shuffles) == 1,
        }
    else:
        route_structural_gates = {
            "no_shuffle": not shuffles,
            "q_reads_hbm_while_k_and_output_use_lx": len(bmms) == 1
            and allocation_components(bmms[0]) == ["hbm", "lx", "lx"],
        }
    structural_gates = {
        **common_structural_gates,
        **route_structural_gates,
    }
    structural_gate = all(structural_gates.values())
    comparisons = {
        "compile": compare_tree(compile_result, expected),
        "measured": compare_tree(measured, expected),
    }
    correctness_gate = all(
        row["allclose_rtol_2e2_atol_2e2"]
        and row["actual_finite"]
        and row["expected_finite"]
        for rows in comparisons.values()
        for row in rows
    )
    traces = sorted(glob.glob(str(trace_dir / "*.pt.trace.json")))
    if len(traces) != 1:
        raise RuntimeError(f"expected one trace, found {traces}")
    trace = trace_report(
        Path(traces[0]),
        bundle_token=artifacts["bundle_token"],
        expected_runs=args.runs,
    )

    report = {
        "schema_version": 1,
        "scope": "fused_multi_block_decode_qk_with_gqa_owner_local_kv",
        "q_route": args.q_route,
        "config": config.report(),
        "physical_query_rows": args.physical_query_rows,
        "physical_q_bytes": q_logical_elements * config.dtype_bytes,
        "q_padding_factor": args.physical_query_rows
        / (config.query_groups * config.query_tokens),
        "work_div": work_div,
        "q_work_div": q_work_div,
        "q_source_core_ids": list(q_source_core_ids),
        "physical_core_order": "work_div_inner_first",
        "dataflow": {
            "q": "four logical GQA rows in physical M64, then P4 LX broadcast",
            "k_v": "unique [kv_head, context-owner] shards remain owner-local",
            "qk": "one BMM root consumes all local K64 blocks",
            "state": "owner-local checksum; softmax-state merge not included",
        },
        "correctness_gate": correctness_gate,
        "comparisons": comparisons,
        "structural_gate": structural_gate,
        "structural_gates": structural_gates,
        "artifacts": artifacts,
        "trace_path": traces[0],
        "trace": trace,
        "wall_ms": {
            "median": statistics.median(walls_ms),
            "mean": statistics.fmean(walls_ms),
            "samples": walls_ms,
        },
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "q_route": args.q_route,
                "correctness_gate": correctness_gate,
                "structural_gate": structural_gate,
                "trace_gate": trace["gate"],
                "device_median_us": (
                    trace["device_us"]["median"] if trace["device_us"] else None
                ),
                "op_inventory": artifacts["op_inventory"],
                "run_dir": str(args.run_dir),
            },
            sort_keys=True,
        )
    )
    if not (correctness_gate and structural_gate and trace["gate"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
