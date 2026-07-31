#!/usr/bin/env python3
"""Measure Granite FP8 per-row activation-scale derivation on Spyre."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        choices=("fp32-input", "fp16-reduction"),
        required=True,
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def floor_scale(scale: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(torch.float32).eps
    return torch.relu(scale - eps) + eps


def fp32_input_scale(activation: torch.Tensor) -> torch.Tensor:
    activation_fp32 = activation.to(torch.float32)
    max_abs = torch.amax(torch.abs(activation_fp32), dim=1, keepdim=True)
    return floor_scale(max_abs / float(torch.finfo(torch.float8_e4m3fn).max))


def fp16_reduction_scale(activation: torch.Tensor) -> torch.Tensor:
    # The input is already FP16, so abs/max selects an exactly representable
    # FP16 value.  Convert only the M row maxima to FP32 before division.
    max_abs = torch.amax(torch.abs(activation), dim=1, keepdim=True)
    return floor_scale(
        max_abs.to(torch.float32)
        / float(torch.finfo(torch.float8_e4m3fn).max)
    )


def synchronize() -> None:
    spyre_module = getattr(torch, "spyre", None)
    if spyre_module is not None and hasattr(spyre_module, "synchronize"):
        spyre_module.synchronize()


def summarize_trace(trace_path: Path) -> dict[str, object]:
    with trace_path.open() as handle:
        events = json.load(handle).get("traceEvents", [])

    kernel_events = []
    kernel_by_name = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    for event in events:
        duration_us = float(event.get("dur", 0.0) or 0.0)
        if event.get("cat") != "kernel" or duration_us <= 0:
            continue
        name = str(event.get("name", "<unknown>"))
        kernel_events.append({"name": name, "duration_us": duration_us})
        kernel_by_name[name]["count"] += 1
        kernel_by_name[name]["duration_us"] += duration_us
    return {
        "kernel_events": kernel_events,
        "kernel_by_name": dict(sorted(kernel_by_name.items())),
    }


def main() -> None:
    args = parse_args()
    if min(args.m, args.k, args.reps) < 1 or args.warmups < 0:
        raise ValueError("M, K, and reps must be positive; warmups must be non-negative")

    import torch_spyre

    if hasattr(torch_spyre, "_autoload"):
        torch_spyre._autoload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "kineto_trace.json"
    result_path = args.output_dir / "result.json"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260731)
    activation_host = (
        torch.randint(
            -64,
            65,
            (args.m, args.k),
            dtype=torch.int16,
            generator=generator,
        ).to(torch.float32)
        / 8.0
    ).to(torch.float16)
    activation = activation_host.to("spyre")
    fn = (
        fp32_input_scale
        if args.implementation == "fp32-input"
        else fp16_reduction_scale
    )
    compiled = torch.compile(
        fn,
        backend="inductor",
        dynamic=False,
        fullgraph=True,
        options={"epilogue_fusion": False},
    )

    compile_start_ns = time.perf_counter_ns()
    actual_device = compiled(activation)
    synchronize()
    compile_and_first_run_ms = (time.perf_counter_ns() - compile_start_ns) / 1.0e6

    actual = actual_device.cpu()
    reference = fp32_input_scale(activation_host)
    error = (actual - reference).abs()
    passed = bool(
        torch.isfinite(actual).all()
        and torch.allclose(actual, reference, rtol=1.0e-6, atol=1.0e-8)
    )
    if not passed:
        raise RuntimeError(
            "scale correctness failed: "
            f"max_abs_error={error.max().item()}"
        )

    for _ in range(args.warmups):
        actual_device = compiled(activation)
        synchronize()

    from torch.profiler import ProfilerActivity, profile

    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=False,
    )
    profiler.start()
    wall_start_ns = time.perf_counter_ns()
    for _ in range(args.reps):
        actual_device = compiled(activation)
        synchronize()
        profiler.step()
    wall_total_us = (time.perf_counter_ns() - wall_start_ns) / 1000.0
    profiler.stop()
    profiler.export_chrome_trace(str(trace_path))

    trace = summarize_trace(trace_path)
    durations = [event["duration_us"] for event in trace["kernel_events"]]
    if not durations:
        raise RuntimeError("Kineto trace contains no cat='kernel' events")
    kernel_total_us = sum(durations)
    kernel_mean_us = kernel_total_us / args.reps
    result = {
        "schema_version": 1,
        "component": "dynamic_fp8_per_row_activation_scale",
        "implementation": args.implementation,
        "logical_shape": {"M": args.m, "K": args.k},
        "correctness": {
            "passed": passed,
            "max_abs_error": error.max().item(),
            "mean_abs_error": error.mean().item(),
        },
        "compile_and_first_run_ms": compile_and_first_run_ms,
        "warmups": args.warmups,
        "repetitions": args.reps,
        "kernel_mean_us_per_iteration": kernel_mean_us,
        "kernel_event_duration_median_us": statistics.median(durations),
        "wall_mean_us": wall_total_us / args.reps,
        "trace_summary": trace,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_spyre_module": torch_spyre.__file__,
        },
        "environment": {
            key: os.environ.get(key)
            for key in ("TORCHINDUCTOR_CACHE_DIR", "DXP_LX_FRAC_AVAIL")
        },
    }
    if not math.isfinite(kernel_mean_us):
        raise RuntimeError("non-finite kernel time")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
