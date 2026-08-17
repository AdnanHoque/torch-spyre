#!/usr/bin/env python3
"""Analyze the retained D-AS-X expert and component timing controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _median_full(result: dict) -> float:
    samples = result["timing"]["samples"]
    values = [
        item["per_call_ms"]
        for item in samples
        if item["selector"] == "identity" and item["kind"] == "block"
    ]
    assert len(values) == 30
    return statistics.median(values)


def _median_control(result: dict) -> float:
    samples = result["samples"]
    assert len(samples) in {60, 90}
    assert all(item["calls"] == 5 for item in samples)
    return statistics.median(item["per_call_ms"] for item in samples)


def _linear_fit(points: list[tuple[int, float]]) -> dict:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum(
        (x - x_mean) ** 2 for x in xs
    )
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * x for x in xs]
    residual = sum((y - estimate) ** 2 for y, estimate in zip(ys, predicted))
    total = sum((y - y_mean) ** 2 for y in ys)
    return {
        "intercept_ms": intercept,
        "slope_ms_per_expert": slope,
        "r_squared": 1.0 - residual / total,
        "points": [{"experts": x, "median_ms": y} for x, y in points],
    }


def _validate_full(result: dict, experts: int) -> None:
    assert result["shape"]["E"] == experts
    assert result["hbm_pool_allocations"] == 0
    assert result["restickify_ops"] == 0
    assert result["expert_hbm_operands_advance"] == 4
    assert result["internal_compute_allocations"] == "LX-only"
    assert result["one_flat_expert_loop"]
    assert result["one_x_hbm_to_lx_preheader"]
    assert result["final_hbm_outputs"] == 1
    samples = result["timing"]["samples"]
    assert len(samples) == 540
    assert sum(item["calls"] for item in samples) == 900


def _validate_component(result: dict, mode: str, experts: int) -> None:
    assert result["mode"] == mode
    assert result["shape"]["E"] == experts
    structure = result["structure"]
    assert structure["bmm_count"] == 3
    assert structure["hbm_pool_allocations"] == 0
    assert structure["restickify_ops"] == 0
    assert structure["all_internal_compute_lx"]
    assert structure["streamed_weight_arg_indices"] == [2, 3, 4]
    assert result["correctness"]["rel_l2"] <= 0.03
    assert result["correctness"]["cosine"] >= 0.999


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    artifacts = repo / "moe_asgemm" / "artifacts"
    decomposition = artifacts / "decomposition"

    expert_points = []
    for experts in (2, 8, 32, 64):
        result = _load(decomposition / "expert_sweep" / f"e{experts}" / "result.json")
        _validate_full(result, experts)
        expert_points.append((experts, _median_full(result)))
    full_128 = _load(artifacts / "clean_reproduction" / "cdx" / "compile_result.json")
    _validate_full(full_128, 128)
    expert_points.append((128, _median_full(full_128)))
    expert_fit = _linear_fit(expert_points)

    control_fits = {}
    for mode in ("full", "no_gelu", "hidden_add", "route_add"):
        points = []
        for experts in (2, 32):
            result = _load(
                decomposition
                / "component_controls"
                / f"{mode}_e{experts}"
                / "result.json"
            )
            _validate_component(result, mode, experts)
            points.append((experts, _median_control(result)))
        control_fits[mode] = _linear_fit(points)

    full_slope = control_fits["full"]["slope_ms_per_expert"]
    pointwise_deltas = {
        "gelu_us_per_expert": 1000
        * (full_slope - control_fits["no_gelu"]["slope_ms_per_expert"]),
        "hidden_mul_minus_add_us_per_expert": 1000
        * (full_slope - control_fits["hidden_add"]["slope_ms_per_expert"]),
        "route_mul_minus_add_us_per_expert": 1000
        * (full_slope - control_fits["route_add"]["slope_ms_per_expert"]),
    }

    leaf = {}
    for kind in ("gate", "down"):
        result = _load(decomposition / "matmul_leaf" / kind / "result.json")
        assert result["structure"]["one_bmm"]
        assert result["structure"]["work_division"] == "M32,N1,K1"
        assert result["correctness"]["rel_l2"] <= 0.03
        assert result["correctness"]["cosine"] >= 0.999
        leaf[kind] = {
            "median_ms": _median_control(result),
            "extra_hbm_bytes_vs_loop": result["structure"]["logical_bytes"][
                "extra_vs_dasx_loop"
            ],
        }

    fixed_ms = expert_fit["intercept_ms"]
    peak_bytes_per_ms = 150_000_000
    for values in leaf.values():
        values["extra_hbm_time_at_150GBps_ms"] = (
            values["extra_hbm_bytes_vs_loop"] / peak_bytes_per_ms
        )
        values["fixed_and_peak_extra_adjusted_proxy_ms"] = (
            values["median_ms"] - fixed_ms - values["extra_hbm_time_at_150GBps_ms"]
        )

    matmul_proxy_ms = (
        2 * leaf["gate"]["fixed_and_peak_extra_adjusted_proxy_ms"]
        + leaf["down"]["fixed_and_peak_extra_adjusted_proxy_ms"]
    )
    measured_slope = expert_fit["slope_ms_per_expert"]
    residual_proxy_ms = max(0.0, measured_slope - matmul_proxy_ms)
    removable_proxy_ms = fixed_ms + 128 * residual_proxy_ms
    full_block_ms = _median_full(full_128)

    component_probe = repo / "experiments" / "dasx_component_sweep_probe.py"
    leaf_probe = repo / "experiments" / "dasx_matmul_leaf_probe.py"
    output = {
        "status": "accepted",
        "measurement_scope": "one AIU; amortized synchronized block calls",
        "expert_scaling": expert_fit,
        "component_controls": control_fits,
        "pointwise_substitution_deltas": pointwise_deltas,
        "standalone_matmul_proxies": leaf,
        "native_ddl_prediction": {
            "clean_full_block_ms": full_block_ms,
            "three_matmul_adjusted_proxy_ms_per_expert": matmul_proxy_ms,
            "non_matmul_and_sequence_residual_proxy_ms_per_expert": residual_proxy_ms,
            "fixed_plus_residual_proxy_ms_at_e128": removable_proxy_ms,
            "proxy_fraction_of_full_percent": 100 * removable_proxy_ms / full_block_ms,
            "preregistered_expected_gain_percent_at_most": 5.0,
            "preregistered_falsification_gain_percent": 10.0,
        },
        "important_limits": [
            "AIUPTI exposes the program as one device job, not per-SDSC timestamps",
            "standalone matmul leaves include extra HBM traffic and a separate job launch",
            "the 150 GB/s adjustment is a calibrated proxy, not bus telemetry",
            "pointwise controls measure substitution deltas, not absolute pointwise latency",
        ],
        "instrument_sha256": {
            "component": hashlib.sha256(component_probe.read_bytes()).hexdigest(),
            "matmul_leaf": hashlib.sha256(leaf_probe.read_bytes()).hexdigest(),
        },
    }
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
