#!/usr/bin/env python3
"""Kineto wrapper for the preserved direct SenDNN FP16/FP8 control."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch_sendnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fp16", "fp8"), required=True)
    parser.add_argument("--benchmark-script", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_control(path: Path):
    spec = importlib.util.spec_from_file_location("direct_sendnn_control", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import control script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def register_profiler_backend() -> None:
    torch_sendnn.sendnn_backend.is_available = lambda: False
    try:
        torch.utils.rename_privateuse1_backend("aiu")
    except RuntimeError as error:
        if "already been set" not in str(error):
            raise
    torch._register_device_module("aiu", torch_sendnn.sendnn_backend)
    torch.utils.generate_methods_for_privateuse1_backend()


def summarize_trace(trace_path: Path) -> dict:
    with trace_path.open() as handle:
        events = json.load(handle).get("traceEvents", [])

    category_summary: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "duration_us": 0.0}
    )
    positive_events = []
    for event in events:
        duration = float(event.get("dur", 0.0) or 0.0)
        if duration <= 0:
            continue
        category = str(event.get("cat", "<none>"))
        category_summary[category]["count"] += 1
        category_summary[category]["duration_us"] += duration
        positive_events.append(
            {
                "name": str(event.get("name", "<unknown>")),
                "category": category,
                "duration_us": duration,
            }
        )

    kernel_events = [
        event for event in positive_events if event["category"] == "kernel"
    ]
    privateuse_events = [
        event
        for event in positive_events
        if event["category"]
        in {
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
            "privateuse1",
            "PrivateUse1",
        }
    ]
    return {
        "category_summary": dict(sorted(category_summary.items())),
        "kernel_events": kernel_events,
        "privateuse_or_device_events": privateuse_events,
        "positive_event_count": len(positive_events),
    }


def main() -> None:
    args = parse_args()
    if args.warmups < 1 or args.repetitions < 2:
        raise ValueError("need at least one warmup and two measured repetitions")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = args.run_dir / "kineto_trace.json"
    result_path = args.run_dir / "result.json"

    register_profiler_backend()
    control = load_control(args.benchmark_script)

    activation, weight = control.make_data(args.mode)
    loader, contracts, compile_statuses = control.build_and_compile(args.mode)
    lifecycle_statuses = control.prepare_and_initialize(loader, args.mode, weight)
    output, sendnn_outputs, inputs = control.make_execution_io(loader, activation)

    execute_status = control.execute_once(loader, sendnn_outputs, inputs)
    reference, reference_ms = control.cpu_reference(args.mode, activation, weight)
    correctness = control.correctness_metrics(args.mode, output, reference)
    correctness["execute_status"] = control.status_text(execute_status)
    correctness["cpu_reference_wall_ms"] = reference_ms
    if not correctness["passed"]:
        raise RuntimeError(f"correctness failed: {correctness}")

    for _ in range(args.warmups):
        control.execute_once(loader, sendnn_outputs, inputs)

    from torch.profiler import ProfilerActivity, profile

    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=False,
    )
    profiler.start()
    wall_start_ns = time.perf_counter_ns()
    for _ in range(args.repetitions):
        control.execute_once(loader, sendnn_outputs, inputs)
        profiler.step()
    wall_end_ns = time.perf_counter_ns()
    profiler.stop()
    profiler.export_chrome_trace(str(trace_path))

    trace_summary = summarize_trace(trace_path)
    kernel_events = trace_summary["kernel_events"]
    kernel_total_us = sum(event["duration_us"] for event in kernel_events)
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "logical_shape": {"M": control.M, "K": control.K, "N": control.N},
        "graph_contract": (
            "FP16 PrimaryInput -> Identity(output FP8) -> BatchScaledMatmul"
            if args.mode == "fp8"
            else "FP16 PrimaryInput -> BatchMatMul"
        ),
        "correctness": correctness,
        "compile_statuses": compile_statuses,
        "lifecycle_statuses": lifecycle_statuses,
        "supernode_contracts": contracts,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "wall_total_us": (wall_end_ns - wall_start_ns) / 1000.0,
        "wall_mean_us": (wall_end_ns - wall_start_ns)
        / 1000.0
        / args.repetitions,
        "kernel_event_count": len(kernel_events),
        "kernel_total_us": kernel_total_us,
        "kernel_mean_us_per_predict": (
            kernel_total_us / args.repetitions if kernel_events else None
        ),
        "trace_summary": trace_summary,
        "trace_path": str(trace_path),
        "software": control.software_versions(),
        "torch_sendnn_module": torch_sendnn.__file__,
        "benchmark_script": {
            "path": str(args.benchmark_script),
            "sha256": sha256(args.benchmark_script),
        },
        "wrapper": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "environment": {
            key: os.getenv(key)
            for key in (
                "DT_OPT",
                "DTCOMPILER_EXPORT_DIR",
                "DEEPRT_EXPORT_DIR",
                "DTCOMPILER_KEEP_EXPORT",
                "TORCH_DEVICE_BACKEND_AUTOLOAD",
                "DXP_LX_FRAC_AVAIL",
                "LX_PLANNING",
            )
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
