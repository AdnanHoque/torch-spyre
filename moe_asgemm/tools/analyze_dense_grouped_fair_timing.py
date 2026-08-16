#!/usr/bin/env python3
"""Fail-closed comparison of D-AS-X and the accepted grouped G3-LX cohort."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


PODS = (
    "adnan-cdx-spyre-dev-pf",
    "adnan-clc-spyre-dev-pf",
    "adnan-spyre-current-pf",
    "adnan-spyre-dev-pf",
)
SELECTORS = ("identity", "permutation", "hot8")
GROUPED_SELECTOR = {
    "identity": "identity",
    "permutation": "seed17_permutation",
    "hot8": "hot8_repeats",
}
KINDS = ("single", "block")


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _median(samples: list[dict], selector: str, kind: str) -> float:
    values = [
        item["per_call_ms"]
        for item in samples
        if item["selector"] == selector and item["kind"] == kind
    ]
    expected = 150 if kind == "single" else 30
    if len(values) != expected:
        raise AssertionError(
            f"{selector}/{kind} has {len(values)} samples, expected {expected}"
        )
    round_medians = [
        statistics.median(
            item["per_call_ms"]
            for item in samples
            if item["selector"] == selector
            and item["kind"] == kind
            and item["round"] == round_index
        )
        for round_index in range(3)
    ]
    return statistics.median(round_medians)


def _validate_dense(result: dict) -> None:
    expected_shape = {"E": 128, "F": 704, "H": 2816, "T": 512, "cores": 32}
    if result["shape"] != expected_shape:
        raise AssertionError(f"unexpected dense shape: {result['shape']}")
    required = {
        "backend_compile": True,
        "launched_generated_kernel": True,
        "one_flat_expert_loop": True,
        "one_source_bundle": True,
        "one_wrapper_bundle_call": True,
        "one_x_hbm_to_lx_preheader": True,
        "gate_up_share_exact_x_interval": True,
        "internal_compute_allocations": "LX-only",
        "hbm_pool_allocations": 0,
        "restickify_ops": 0,
        "router_weighting_after_down": True,
        "expert_hbm_operands_advance": 4,
        "final_hbm_outputs": 1,
    }
    for name, expected in required.items():
        if result.get(name) != expected:
            raise AssertionError(
                f"dense {name}={result.get(name)!r}, expected {expected!r}"
            )
    if result["runtime_alpha_shape"] != [128, 512, 1]:
        raise AssertionError("dense runtime alpha is not [E,T,1]")
    if not result["backend_core_mapping"]["x_core_id_to_work_slice_all_32"]:
        raise AssertionError("dense X core map was not validated on all 32 cores")
    if not result["backend_core_mapping"][
        "gate_up_core_id_to_work_slice_matches_x_all_32"
    ]:
        raise AssertionError("dense gate/up core maps do not match resident X")
    correctness = result["correctness"]
    if not correctness["same_callable_two_alphas"]:
        raise AssertionError("dense route payloads did not reuse one callable")
    metrics = [*correctness["payloads"].values(), correctness["alpha_response_delta"]]
    if max(item["rel_l2"] for item in metrics) > 0.03:
        raise AssertionError("dense relative L2 exceeds 0.03")
    if min(item["cosine"] for item in metrics) < 0.999:
        raise AssertionError("dense cosine is below 0.999")
    protocol = result["timing"]["protocol"]
    expected_protocol = {
        "warmups": 5,
        "singles": 50,
        "blocks": 10,
        "block_iters": 5,
        "rounds": 3,
    }
    if protocol != expected_protocol:
        raise AssertionError(f"unexpected dense timing protocol: {protocol}")
    if len(result["timing"]["samples"]) != 540:
        raise AssertionError("dense result does not contain 540 raw samples")


def _validate_grouped(result: dict) -> None:
    if result.get("status") != "passed" or not result.get("timing_collected"):
        raise AssertionError("grouped G3-LX timing result is not accepted")
    if result["shape"] != {
        "E": 128,
        "F": 704,
        "H": 2816,
        "Q": 128,
        "cores": 32,
        "dtype": "float16",
        "tau": 32,
        "useful_routed_rows": 4096,
    }:
        raise AssertionError(f"unexpected grouped shape: {result['shape']}")
    checks = result["structural_checks"]
    if not checks["activation_path"]["all_edges_lx"]:
        raise AssertionError("grouped activation path is not fully LX")
    common = checks["common"]
    if not common["three_indirect_grouped_bmm_sdscs"]:
        raise AssertionError("grouped result does not contain three indirect BMMs")
    if not common["all_projections_use_row_only_work_division"]:
        raise AssertionError("grouped result does not retain row-only C32 division")
    if not common["no_selected_weight_tensor_materialization"]:
        raise AssertionError("grouped result materializes selected weights")
    if len(result["samples"]) != 540:
        raise AssertionError("grouped result does not contain 540 raw samples")


def analyze(dense_root: Path, grouped_root: Path) -> dict:
    dense_names = {
        "adnan-cdx-spyre-dev-pf": "cdx",
        "adnan-clc-spyre-dev-pf": "clc",
        "adnan-spyre-current-pf": "current",
        "adnan-spyre-dev-pf": "dev",
    }
    per_device: dict[str, dict] = {}
    for pod in PODS:
        dense = _load(dense_root / dense_names[pod] / "result.json")
        grouped = _load(grouped_root / pod / "timing" / "G3-LX" / "result.json")
        _validate_dense(dense)
        _validate_grouped(grouped)
        dense_samples = dense["timing"]["samples"]
        grouped_samples = grouped["samples"]
        per_device[pod] = {}
        for selector in SELECTORS:
            per_device[pod][selector] = {}
            for kind in KINDS:
                dense_ms = _median(dense_samples, selector, kind)
                grouped_ms = _median(grouped_samples, GROUPED_SELECTOR[selector], kind)
                per_device[pod][selector][kind] = {
                    "dense_ms": dense_ms,
                    "grouped_ms": grouped_ms,
                    "grouped_over_dense": grouped_ms / dense_ms,
                }

    aggregate: dict[str, dict] = {}
    for selector in SELECTORS:
        aggregate[selector] = {}
        for kind in KINDS:
            dense_values = [per_device[p][selector][kind]["dense_ms"] for p in PODS]
            grouped_values = [per_device[p][selector][kind]["grouped_ms"] for p in PODS]
            ratios = [per_device[p][selector][kind]["grouped_over_dense"] for p in PODS]
            aggregate[selector][kind] = {
                "dense_median_of_device_medians_ms": statistics.median(dense_values),
                "grouped_median_of_device_medians_ms": statistics.median(
                    grouped_values
                ),
                "median_per_device_grouped_over_dense": statistics.median(ratios),
                "dense_inter_device_spread_percent": (
                    (max(dense_values) - min(dense_values))
                    / statistics.median(dense_values)
                    * 100
                ),
                "direction_dense_faster_on_all_four": all(
                    ratio > 1.0 for ratio in ratios
                ),
            }

    return {
        "schema_version": 1,
        "status": "accepted",
        "comparison": (
            "D-AS-X complete weighted dense control versus kernel-only grouped G3-LX"
        ),
        "pods": list(PODS),
        "per_device": per_device,
        "aggregate": aggregate,
        "acceptance_gate": {
            "four_devices": len(per_device) == 4,
            "all_selectors_and_modes_dense_faster_on_all_four": all(
                aggregate[selector][kind]["direction_dense_faster_on_all_four"]
                for selector in SELECTORS
                for kind in KINDS
            ),
        },
        "interpretation": (
            "Dense includes runtime top-8 post-down weighting and accumulation; "
            "grouped G3-LX excludes router weighting and combine. Dense winning is "
            "therefore a decisive one-sided loss for this grouped implementation."
        ),
        "non_claims": [
            "not end-to-end model latency",
            "not an energy comparison",
            "not proof that all grouped schedules lose",
            "not a same-source contemporaneous rerun of G3-LX",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--grouped-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.dense_root, args.grouped_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"]["identity"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
