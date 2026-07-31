#!/usr/bin/env python3
"""Summarize the matched SenDNN versus Design A down-projection study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
from typing import Any


EXPECTED_SHAPE = {"M": 512, "K": 12800, "N": 4096}
EXPECTED_DESIGN_A_SPLIT = {"M": 4, "N": 8, "K": 1}
FLOPS = 2 * EXPECTED_SHAPE["M"] * EXPECTED_SHAPE["K"] * EXPECTED_SHAPE["N"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict[str, float | int]:
    require(len(values) >= 2, "need at least two timing events")
    return {
        "count": len(values),
        "mean_us": statistics.mean(values),
        "median_us": statistics.median(values),
        "minimum_us": min(values),
        "maximum_us": max(values),
        "stdev_us": statistics.stdev(values),
        "first_ten_median_us": statistics.median(values[:10]),
        "last_ten_median_us": statistics.median(values[-10:]),
    }


def summarize_sendnn(run_root: Path, relative: str) -> tuple[dict[str, Any], list[float]]:
    profile = run_root / relative / "profile"
    result_path = profile / "result.json"
    trace_path = profile / "kineto_trace.json"
    result = load_json(result_path)
    require(result["mode"] == "fp16", f"not FP16: {result_path}")
    require(result["logical_shape"] == EXPECTED_SHAPE, f"wrong shape: {result_path}")
    require(result["correctness"]["passed"], f"correctness failed: {result_path}")
    events = result["trace_summary"]["kernel_events"]
    require(len(events) == result["repetitions"] == 60, f"event count: {result_path}")
    require({event["name"] for event in events} == {"fp16_bmm"}, str(result_path))
    values = [float(event["duration_us"]) for event in events]
    require(all(value > 0 for value in values), f"nonpositive event: {result_path}")
    return (
        {
            "profile": str(profile),
            "warmups": result["warmups"],
            "repetitions": result["repetitions"],
            "device_event": {"category": "kernel", "name": "fp16_bmm"},
            "timing": distribution(values),
            "correctness": result["correctness"],
            "software": result["software"],
            "environment": result["environment"],
            "hashes": {
                "result_sha256": sha256(result_path),
                "trace_sha256": sha256(trace_path),
                "benchmark_sha256": result["benchmark_script"]["sha256"],
                "wrapper_sha256": result["wrapper"]["sha256"],
            },
            "excluded_device_categories": ["gpu_memcpy", "gpu_memset"],
        },
        values,
    )


def summarize_design_a(
    run_root: Path, relative: str
) -> tuple[dict[str, Any], list[float]]:
    directory = run_root / relative
    summary_path = directory / "summary.json"
    summary = load_json(summary_path)
    trace_paths = list((directory / "trace").glob("*.json"))
    require(len(trace_paths) == 1, f"expected one trace below {directory}")
    require(summary["status"] == "pass", f"failed Design A run: {directory}")
    require(summary["correctness_gate"], f"correctness failed: {directory}")
    require(summary["structural_gate"], f"structure failed: {directory}")
    timing_gate = summary["timing"]["trace"]["gate"]
    require(
        all(timing_gate.values()) if isinstance(timing_gate, dict) else timing_gate,
        str(directory),
    )
    shape = summary["shape"]
    require(
        {"M": shape["logical_m"], "K": shape["k"], "N": shape["n"]}
        == EXPECTED_SHAPE,
        f"wrong shape: {directory}",
    )
    require(summary["candidate_work_division"] == EXPECTED_DESIGN_A_SPLIT, str(directory))
    require(
        all(
            bundle["roots"] == ["batchmatmul"]
            for bundle in summary["artifacts"]["bundles"]
        ),
        f"non-BMM root: {directory}",
    )
    paid = summary["paid_boundary"]
    require(
        paid["inputs_ready_on_device"]
        and paid["per_arm_native_activation_layout"]
        and paid["per_arm_native_weight_layout"]
        and paid["bmm_native_output_layout"]
        and not paid["candidate_padding_inside_event"]
        and not paid["candidate_restickify_inside_event"]
        and not paid["candidate_output_slice_inside_event"],
        f"not the compute-only boundary: {directory}",
    )
    values = [
        float(value)
        for value in summary["timing"]["trace"]["candidate"]["durations_us"]
    ]
    require(len(values) == 60 and all(value > 0 for value in values), str(directory))
    return (
        {
            "directory": str(directory),
            "warmups": summary["warmups"],
            "blocks": summary["blocks"],
            "candidate_work_division": summary["candidate_work_division"],
            "device_event": {"category": "kernel", "roots": ["batchmatmul"]},
            "timing": distribution(values),
            "pt_service": summary["pt_service"]["arms"]["candidate"],
            "paid_boundary": paid,
            "correctness": summary["correctness"],
            "provenance": summary["provenance"],
            "hashes": {
                "summary_sha256": sha256(summary_path),
                "trace_sha256": sha256(trace_paths[0]),
            },
        },
        values,
    )


def compare(sendnn: list[float], design_a: list[float]) -> dict[str, Any]:
    sendnn_median = statistics.median(sendnn)
    design_a_median = statistics.median(design_a)
    return {
        "winner": "sendnn",
        "sendnn_median_us": sendnn_median,
        "design_a_median_us": design_a_median,
        "sendnn_speedup_over_design_a": design_a_median / sendnn_median,
        "sendnn_latency_reduction_percent_vs_design_a": 100.0
        * (design_a_median - sendnn_median)
        / design_a_median,
        "design_a_latency_increase_percent_vs_sendnn": 100.0
        * (design_a_median - sendnn_median)
        / sendnn_median,
        "sendnn_effective_tflops": FLOPS / (sendnn_median * 1e-6) / 1e12,
        "design_a_effective_tflops": FLOPS / (design_a_median * 1e-6) / 1e12,
        "event_ranges_disjoint": max(sendnn) < min(design_a),
    }


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    require(root.is_dir(), f"run root does not exist: {root}")

    sendnn_runs: dict[str, Any] = {}
    sendnn_values: dict[str, list[float]] = {}
    for label, relative in (
        ("pre", "sendnn_pre_v3"),
        ("post", "sendnn_post_v1"),
        ("final", "sendnn_final_v1"),
    ):
        sendnn_runs[label], sendnn_values[label] = summarize_sendnn(root, relative)

    design_a_runs: dict[str, Any] = {}
    design_a_values: dict[str, list[float]] = {}
    for label, relative in (
        ("warm10", "design_a_mid_v2"),
        ("warm30", "design_a_mid_v3_warm30"),
    ):
        design_a_runs[label], design_a_values[label] = summarize_design_a(
            root, relative
        )

    primary_sendnn = sendnn_values["pre"] + sendnn_values["post"]
    sensitivity_sendnn = sendnn_values["post"] + sendnn_values["final"]
    pooled_sendnn = sum(sendnn_values.values(), [])
    pooled_design_a = sum(design_a_values.values(), [])
    report = {
        "schema": "sendnn_vs_design_a_m512_down_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "device": platform.node(),
        "shape": EXPECTED_SHAPE,
        "dtype": "FP16",
        "flops": FLOPS,
        "method": {
            "launch_order": "SenDNN, Design A warm10, SenDNN, Design A warm30, SenDNN",
            "performance_scope": "Kineto cat==kernel complete-event duration",
            "sen_dnn_boundary": (
                "static weight preparation excluded; gpu_memcpy/gpu_memset and host wall "
                "excluded; one fp16_bmm kernel event per Predict"
            ),
            "design_a_boundary": (
                "activation and weight preplaced in native device layouts; one BMM root; "
                "native device output; conversions and host transfer excluded"
            ),
            "not_matched": [
                "not one-process ABBA across frameworks",
                "SenDNN uses torch 2.10 while Design A uses torch 2.11",
                "operand values differ, although dense FP16 scheduling is data-independent",
            ],
            "not_claimed": [
                "host or Granite end-to-end speedup",
                "hardware stall or ring-utilization counters",
                "identical generated program or input layout",
            ],
        },
        "runs": {"sendnn": sendnn_runs, "design_a": design_a_runs},
        "comparisons": {
            "primary_equal_warmup_bracket": compare(
                primary_sendnn, design_a_values["warm10"]
            ),
            "longer_design_a_warmup_sensitivity": compare(
                sensitivity_sendnn, design_a_values["warm30"]
            ),
            "all_events_pooled": compare(pooled_sendnn, pooled_design_a),
        },
        "all_gates": True,
        "decision": (
            "Design A beats the Torch-Spyre incumbent for this shape but does not beat "
            "the pinned SenDNN FP16 matmul. SenDNN remains the performance target."
        ),
    }
    output = args.output or root / "comparison.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
