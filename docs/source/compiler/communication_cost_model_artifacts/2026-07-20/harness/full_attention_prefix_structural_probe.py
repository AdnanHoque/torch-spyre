#!/usr/bin/env python3
"""Validate the allocation-identical full-attention S2 prefix initializer."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from core.op_parser import resolve_custom_op
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
SEED = int(os.environ.get("FULL_PREFIX_SEED", "17"))
ORDER = os.environ.get("FULL_PREFIX_COMPILE_ORDER", "normal_prefix_oracle")
VALID_ORDERS = {
    "normal_prefix_oracle": ("normal", "prefix", "oracle"),
    "oracle_prefix_normal": ("oracle", "prefix", "normal"),
}
if ORDER not in VALID_ORDERS:
    raise ValueError(f"unsupported FULL_PREFIX_COMPILE_ORDER={ORDER!r}")


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
                "num_cores": spec.get("numCoresUsed_"),
                "work_slices": spec.get("numWkSlicesPerDim_"),
                "core_id_to_work_slice": spec.get("coreIdToWkSlice_"),
                "allocations": allocations(spec),
            }
        )
    roots.sort(key=lambda root: int(root["root_name"].split("_", 1)[0]))
    bundles = sorted(cache.rglob("bundle.mlir"))
    if len(bundles) != 1:
        raise RuntimeError(f"expected one bundle in {cache}, found {bundles}")
    bundle_text = bundles[0].read_text()
    return {
        "roots": roots,
        "op_inventory": [root["op"] for root in roots],
        "bundle": str(bundles[0].relative_to(cache)),
        "bundle_sha256": sha256_bytes(bundles[0].read_bytes()),
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
    """Fields that identify the same LX storage and physical tensor layout.

    The SHUFFLE output and BMM input intentionally carry different logical
    coordinate views (the transfer maps cores/corelets into the consumer view),
    while sharing these exact physical allocation fields.
    """
    return {
        key: copy.deepcopy(allocation[key])
        for key in (
            "component_",
            "layoutDimOrder_",
            "maxDimSizes_",
            "startAddressCoreCorelet_",
        )
    }


def comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    close = torch.isclose(actual, expected, rtol=1e-2, atol=1e-2)
    return {
        "allclose_rtol_1e2_atol_1e2": bool(close.all()),
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "exact_mismatch_count": int((actual != expected).sum().item()),
        "actual_finite": bool(torch.isfinite(actual).all()),
        "expected_finite": bool(torch.isfinite(expected).all()),
        "mismatch_count": int((~close).sum().item()),
        "element_count": close.numel(),
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
    }


def main() -> None:
    run_dir = Path(os.environ["FULL_PREFIX_RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)

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
    # High-contrast deterministic logits make a wrong S2 preseed visibly wrong,
    # rather than merely bit-different but still within the production 1e-2
    # tolerance used by near-uniform random attention inputs.
    query_cpu = torch.where(query_cpu >= 0.5, 2.0, -2.0).to(torch.float16)
    key_cpu = torch.where(key_cpu >= 0.5, 2.0, -2.0).to(torch.float16)
    cpu_tensors = (query_cpu.contiguous(), key_cpu.contiguous(), value_cpu.contiguous())
    poison_key_cpu = torch.zeros_like(key_cpu)

    reference = get_operation_target(resolved_op, torch, "cpu", SHAPES, custom_module)
    with torch.no_grad():
        expected = reference(query_cpu, key_cpu, value_cpu).cpu().clone()

    device = torch.device("spyre")
    query = query_cpu.to(device)
    key = key_cpu.to(device)
    value = value_cpu.to(device)
    poison_key = poison_key_cpu.to(device)
    real_inputs = (query, key, value)
    poison_inputs = (query, poison_key, value)
    tensor_preparer = get_tensor_preparer(
        torch, SHAPES, resolved_op, "tsp", custom_module
    )

    # Distinct forward code objects are required here. Dynamo otherwise shares
    # the first compiled closure across modes, bypassing the mode-specific cache
    # and codegen patch even though torch.compile returned three wrappers.
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

    graph_factories = {
        "normal": NormalGraph,
        "prefix": PrefixGraph,
        "oracle": OracleGraph,
    }

    configs = {
        "normal": {
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
    caches = {name: run_dir / f"{name}_cache" for name in configs}
    for cache in caches.values():
        cache.mkdir(parents=True, exist_ok=True)

    compiled: dict[str, Any] = {}
    compile_first: dict[str, torch.Tensor | None] = {}

    def prepare(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        reset_named_dims()
        reset_hint_counter()
        return tensor_preparer(inputs)

    def invoke(
        name: str,
        fn: Any,
        inputs: tuple[torch.Tensor, ...],
        *,
        copy_output: bool,
    ) -> torch.Tensor | None:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(caches[name])
        prepared = prepare(inputs)
        try:
            with torch.no_grad(), spyre_config.patch(configs[name]):
                result = fn(*prepared)
                torch.spyre.synchronize()
                if not copy_output:
                    return None
                return result.cpu().clone()
        finally:
            reset_named_dims()
            reset_hint_counter()

    def compile_first_run(name: str) -> None:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(caches[name])
        target = graph_factories[name]()
        target = prepare_module_target(target, cpu_tensors, device=device)
        with spyre_config.patch(configs[name]):
            fn = torch.compile(target, fullgraph=True)
        compiled[name] = fn
        compile_first[name] = invoke(
            name, fn, real_inputs, copy_output=name != "prefix"
        )

    for name in VALID_ORDERS[ORDER]:
        compile_first_run(name)

    # A zero-key preseed makes the first score BMM use uniform logits. The
    # oracle's own real-key producer still executes, so a mismatch proves that
    # the omitted transfer is not being bypassed through HBM or its S1 source.
    invoke("prefix", compiled["prefix"], poison_inputs, copy_output=False)
    negative = invoke("oracle", compiled["oracle"], real_inputs, copy_output=True)
    assert negative is not None

    invoke("prefix", compiled["prefix"], real_inputs, copy_output=False)
    positive = invoke("oracle", compiled["oracle"], real_inputs, copy_output=True)
    assert positive is not None

    # The full oracle's max root reuses the S2 address. A second call without a
    # new prefix must therefore fail, proving timing has to preseed every event.
    no_reseed = invoke("oracle", compiled["oracle"], real_inputs, copy_output=True)
    assert no_reseed is not None

    invoke("prefix", compiled["prefix"], real_inputs, copy_output=False)
    restored = invoke("oracle", compiled["oracle"], real_inputs, copy_output=True)
    assert restored is not None
    normal = invoke("normal", compiled["normal"], real_inputs, copy_output=True)
    assert normal is not None

    outputs = {
        "normal_vs_cpu": comparison(normal, expected),
        "negative_wrong_preseed_vs_cpu": comparison(negative, expected),
        "positive_correct_preseed_vs_cpu": comparison(positive, expected),
        "repeat_without_reseed_vs_cpu": comparison(no_reseed, expected),
        "restored_after_reseed_vs_cpu": comparison(restored, expected),
        "negative_vs_positive": comparison(negative, positive),
        "repeat_without_reseed_vs_positive": comparison(no_reseed, positive),
        "positive_vs_normal": comparison(positive, normal),
        "restored_vs_normal": comparison(restored, normal),
    }

    artifacts = {name: artifact_report(cache) for name, cache in caches.items()}
    normal_art = artifacts["normal"]
    prefix_art = artifacts["prefix"]
    oracle_art = artifacts["oracle"]
    normal_shuffle = one_root(normal_art, "shuffle")
    prefix_shuffle = one_root(prefix_art, "shuffle")
    normal_bmm = one_root(normal_art, "batchmatmul", 0)
    oracle_bmm = one_root(oracle_art, "batchmatmul", 0)
    normal_s2 = allocation_for_lds(normal_shuffle, 1)
    prefix_s2 = allocation_for_lds(prefix_shuffle, 1)
    normal_consumer_s2 = allocation_for_lds(normal_bmm, 1)
    oracle_consumer_s2 = allocation_for_lds(oracle_bmm, 1)
    normal_source = allocation_for_lds(normal_shuffle, 0)
    prefix_source = allocation_for_lds(prefix_shuffle, 0)

    normal_without_shuffle = [
        op for op in normal_art["op_inventory"] if op != "shuffle"
    ]
    normal_root_hashes_without_shuffle = [
        root["canonical_sha256"]
        for root in normal_art["roots"]
        if root["op"] != "shuffle"
    ]
    gates = {
        "normal_correct": outputs["normal_vs_cpu"][
            "allclose_rtol_1e2_atol_1e2"
        ],
        "wrong_preseed_mismatches": not outputs["negative_vs_positive"][
            "bitwise_equal"
        ]
        and outputs["negative_vs_positive"]["exact_mismatch_count"] > 0,
        "correct_preseed_matches": outputs[
            "positive_correct_preseed_vs_cpu"
        ]["allclose_rtol_1e2_atol_1e2"]
        and outputs["positive_vs_normal"]["bitwise_equal"],
        "repeat_without_reseed_mismatches": not outputs[
            "repeat_without_reseed_vs_positive"
        ]["bitwise_equal"]
        and outputs["repeat_without_reseed_vs_positive"][
            "exact_mismatch_count"
        ]
        > 0,
        "restore_matches": outputs["restored_after_reseed_vs_cpu"][
            "allclose_rtol_1e2_atol_1e2"
        ]
        and outputs["restored_vs_normal"]["bitwise_equal"],
        "all_expected_finite": all(
            row["expected_finite"] for row in outputs.values()
        ),
        "normal_has_one_shuffle": normal_art["op_inventory"].count("shuffle")
        == 1,
        "prefix_is_exact_normal_through_shuffle": prefix_art["op_inventory"]
        == normal_art["op_inventory"][
            : normal_art["op_inventory"].index("shuffle") + 1
        ],
        "prefix_ends_at_shuffle": prefix_art["op_inventory"][-1:] == ["shuffle"],
        "prefix_has_no_consumer_or_later_root": "batchmatmul"
        not in prefix_art["op_inventory"],
        "oracle_is_normal_minus_shuffle": oracle_art["op_inventory"]
        == normal_without_shuffle,
        "prefix_root_specs_equal_normal_prefix": [
            root["canonical_sha256"] for root in prefix_art["roots"]
        ]
        == [
            root["canonical_sha256"]
            for root in normal_art["roots"][: len(prefix_art["roots"])]
        ],
        "oracle_root_specs_equal_normal_minus_shuffle": [
            root["canonical_sha256"] for root in oracle_art["roots"]
        ]
        == normal_root_hashes_without_shuffle,
        "prefix_source_allocation_equal_normal": prefix_source == normal_source,
        "prefix_s2_allocation_equal_normal": prefix_s2 == normal_s2,
        "normal_shuffle_s2_physical_equals_consumer_s2": physical_allocation(
            normal_s2
        )
        == physical_allocation(normal_consumer_s2),
        "prefix_s2_physical_equals_oracle_consumer_s2": physical_allocation(
            prefix_s2
        )
        == physical_allocation(oracle_consumer_s2),
        "oracle_consumer_allocation_equals_normal_consumer": oracle_consumer_s2
        == normal_consumer_s2,
        "s2_is_lx": normal_s2["component_"]
        == prefix_s2["component_"]
        == oracle_consumer_s2["component_"]
        == "lx",
        "shuffle_source_is_lx": normal_source["component_"]
        == prefix_source["component_"]
        == "lx",
        "oracle_has_no_hidden_s2_bundle_input": oracle_art[
            "bundle_s2_synthetic_name_mentions"
        ]
        == 0
        and oracle_art["bundle_input_arg_extract_count"]
        == normal_art["bundle_input_arg_extract_count"],
        "prefix_has_fewer_bundle_inputs": prefix_art[
            "bundle_input_arg_extract_count"
        ]
        < normal_art["bundle_input_arg_extract_count"],
    }
    report = {
        "scope": {
            "op": OP,
            "B": B,
            "H": H,
            "Lq": LQ,
            "Lk": LK,
            "D": D,
            "group_size": GROUP_SIZE,
            "source_bytes_per_core": H * LK * D * 2 // (H * GROUP_SIZE),
            "compile_order": ORDER,
        },
        "outputs": outputs,
        "compile_first_output_copied": {
            name: value is not None for name, value in compile_first.items()
        },
        "artifacts": artifacts,
        "normal_source_allocation": normal_source,
        "prefix_source_allocation": prefix_source,
        "normal_s2_allocation": normal_s2,
        "prefix_s2_allocation": prefix_s2,
        "normal_consumer_s2_allocation": normal_consumer_s2,
        "oracle_consumer_s2_allocation": oracle_consumer_s2,
        "gates": gates,
        "all_gates": all(gates.values()),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "inventories": {
                    name: value["op_inventory"] for name, value in artifacts.items()
                },
                "outputs": outputs,
                "gates": gates,
                "all_gates": report["all_gates"],
            },
            sort_keys=True,
        )
    )
    if not report["all_gates"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
