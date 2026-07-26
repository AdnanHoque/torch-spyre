#!/usr/bin/env python3
"""Analyze the exact SenDNN one-layer B1/S512 prefill contract.

The SenDNN profiler emits one fused device-program event per measured prefill.
The matching DtoH event spans almost the same interval, so adding both event
durations would double-count device time.  This analyzer records that overlap
explicitly and derives the compiler-cycle utilization from the PREFILL compile
section rather than whichever ``Total`` row happens to appear first.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any


FIRST_TOKEN_RE = re.compile(r"First-token latency: ([0-9.]+) ms")
PROFILE_VALUE_RE = re.compile(
    r"^\[SPYRE-PERF-SUITE\] (CPU TIME|Spyre TIME|Kernel TIME) "
    r"\(in ms\):\s+([0-9.]+)$",
    re.MULTILINE,
)
COMPILE_RE = re.compile(r"\[DEM\] Compiling \((\d+)/(\d+)\): \{")
TOTAL_RE = re.compile(r"^Total\s+(\d+)\s*$")


def read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return handle.read()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "population_stdev_ms": statistics.pstdev(values),
    }


def interval_overlap_us(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start = float(left["ts"])
    right_start = float(right["ts"])
    left_end = left_start + float(left["dur"])
    right_end = right_start + float(right["dur"])
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def analyze_trace(path: Path, iterations: int) -> dict[str, Any]:
    trace = read_json(path)
    events = trace["traceEvents"]
    kernels = sorted(
        [event for event in events if event.get("cat") == "kernel" and "dur" in event],
        key=lambda event: event["ts"],
    )
    h2d = sorted(
        [
            event
            for event in events
            if event.get("cat") == "gpu_memcpy"
            and "HtoD"
            in (event.get("args", {}).get("call", "") + event.get("name", ""))
            and "dur" in event
        ],
        key=lambda event: event["ts"],
    )
    dtoh = sorted(
        [
            event
            for event in events
            if event.get("cat") == "gpu_memcpy"
            and "DtoH"
            in (event.get("args", {}).get("call", "") + event.get("name", ""))
            and "dur" in event
        ],
        key=lambda event: event["ts"],
    )
    memsets = [
        event
        for event in events
        if event.get("cat") == "gpu_memset" and "dur" in event
    ]

    if len(kernels) != iterations:
        raise ValueError(f"expected {iterations} kernel events, found {len(kernels)}")
    if len(dtoh) != iterations:
        raise ValueError(f"expected {iterations} DtoH events, found {len(dtoh)}")
    kernel_names = sorted({str(event["name"]) for event in kernels})
    if kernel_names != ["embedding"]:
        raise ValueError(
            f"expected the fused SenDNN program name, found {kernel_names}"
        )

    kernel_ms = [float(event["dur"]) / 1000.0 for event in kernels]
    dtoh_ms = [float(event["dur"]) / 1000.0 for event in dtoh]
    overlaps_ms = [
        interval_overlap_us(kernel, copy) / 1000.0
        for kernel, copy in zip(kernels, dtoh)
    ]
    overlap_percent = [
        min(100.0, 100.0 * overlap / kernel_duration)
        for overlap, kernel_duration in zip(overlaps_ms, kernel_ms)
    ]

    return {
        "path": str(path),
        "sha256": sha256(path),
        "measured_iterations": iterations,
        "kernel_names": kernel_names,
        "kernel": stats(kernel_ms),
        "h2d": {
            "event_count": len(h2d),
            "events_per_iteration": len(h2d) / iterations,
            "mean_total_per_iteration_ms": sum(float(event["dur"]) for event in h2d)
            / iterations
            / 1000.0,
        },
        "dtoh": {
            **stats(dtoh_ms),
            "events_per_iteration": len(dtoh) / iterations,
        },
        "memset": {
            "event_count": len(memsets),
            "events_per_iteration": len(memsets) / iterations,
            "mean_total_per_iteration_ms": sum(
                float(event["dur"]) for event in memsets
            )
            / iterations
            / 1000.0,
        },
        "kernel_dtoh_overlap": {
            "mean_overlap_ms": statistics.mean(overlaps_ms),
            "mean_kernel_covered_percent": statistics.mean(overlap_percent),
            "interpretation": (
                "Kernel and DtoH intervals overlap; do not add their durations. "
                "Use the kernel event as device-program time."
            ),
        },
    }


def analyze_run_log(path: Path, iterations: int) -> dict[str, Any]:
    text = read_text(path)
    first_token_ms = [float(value) for value in FIRST_TOKEN_RE.findall(text)]
    if len(first_token_ms) != iterations:
        raise ValueError(
            f"expected {iterations} first-token measurements, "
            f"found {len(first_token_ms)}"
        )
    profiler = {
        label.lower().replace(" ", "_") + "_ms": float(value)
        for label, value in PROFILE_VALUE_RE.findall(text)
    }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "first_token_latency": stats(first_token_ms),
        "profiler_reported": profiler,
    }


def analyze_compiler_log(
    path: Path, kernel_mean_ms: float, core_frequency_mhz: float
) -> dict[str, Any]:
    lines = read_text(path).splitlines()
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    collecting_order = False

    for line in lines:
        compile_match = COMPILE_RE.search(line)
        if compile_match:
            current = {
                "index": int(compile_match.group(1)),
                "compile_count": int(compile_match.group(2)),
                "kind": None,
                "total_cycles": None,
                "execution_order": [],
            }
            sections.append(current)
            collecting_order = False
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped in {"PREFILL", "DECODING"} and current["kind"] is None:
            current["kind"] = stripped.lower()
        if stripped == "== Final Execution Order ==":
            collecting_order = True
            continue
        if collecting_order:
            if not stripped:
                collecting_order = False
            else:
                current["execution_order"].append(stripped)
            continue
        total_match = TOTAL_RE.match(line)
        if total_match and current["total_cycles"] is None:
            current["total_cycles"] = int(total_match.group(1))

    if not sections:
        raise ValueError("compiler log contains no compile sections")
    prefill = [section for section in sections if section["kind"] == "prefill"]
    if len(prefill) != 1 or prefill[0]["total_cycles"] is None:
        raise ValueError(f"expected one PREFILL section with cycles, found {prefill}")

    required_markers = (
        "embedding",
        "mean-LayerNormNorm",
        "bmm-BMM_1",
        "_safe_softmax-Exp",
        "bmm_1-BMM_1",
        "silu",
        "mm_7-BMM_1",
        "div",
    )
    order = prefill[0]["execution_order"]
    missing = [
        marker for marker in required_markers if not any(marker in op for op in order)
    ]
    if missing:
        raise ValueError(f"PREFILL execution order is missing {missing}")

    cycles = int(prefill[0]["total_cycles"])
    ideal_duration_ms = cycles / core_frequency_mhz / 1000.0
    return {
        "path": str(path),
        "sha256": sha256(path),
        "compile_sections": sections,
        "prefill": {
            "total_cycles": cycles,
            "core_frequency_mhz": core_frequency_mhz,
            "ideal_duration_ms": ideal_duration_ms,
            "trace_kernel_mean_ms": kernel_mean_ms,
            "compiler_cycle_utilization_percent": 100.0
            * ideal_duration_ms
            / kernel_mean_ms,
            "required_execution_markers": list(required_markers),
            "execution_order_count": len(order),
            "interpretation": (
                "Derived from compiler cycles divided by trace kernel time; "
                "this is not a physical utilization counter."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--compiler-log", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--core-frequency-mhz", type=float, default=1100.0)
    parser.add_argument("--torch-metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    trace = analyze_trace(args.trace, args.iterations)
    run_log = analyze_run_log(args.run_log, args.iterations)
    compiler = analyze_compiler_log(
        args.compiler_log,
        float(trace["kernel"]["mean_ms"]),
        args.core_frequency_mhz,
    )
    result = {
        "contract": {
            "backend": "sendnn",
            "model": "ibm-granite/granite-3.3-8b-instruct",
            "decoder_layers": 1,
            "batch_size": 1,
            "prompt_length": 512,
            "max_new_tokens_contract": 4,
            "mode": "prefill_only",
            "measured_iterations": args.iterations,
        },
        "trace": trace,
        "run_log": run_log,
        "compiler": compiler,
    }
    if args.torch_metrics:
        torch_metrics = read_json(args.torch_metrics)
        torch_prefill = torch_metrics["reproduction"]["phase_metrics"]["prefill"]
        sendnn_mean_ms = float(trace["kernel"]["mean_ms"])
        torch_phase_mean_ms = float(torch_prefill["phase_total"]["mean_ms"])
        result["torch_spyre_comparison"] = {
            "torch_metrics_path": str(args.torch_metrics),
            "scope": (
                "The SenDNN fused program contains embedding, one decoder block, "
                "final norm, and output head, so compare it with the Torch-Spyre "
                "one-layer phase_total rather than block_total."
            ),
            "sendnn_fused_phase_mean_ms": sendnn_mean_ms,
            "torch_spyre_one_layer_phase_mean_ms": torch_phase_mean_ms,
            "torch_spyre_decoder_block_mean_ms": float(
                torch_prefill["block_total"]["mean_ms"]
            ),
            "sendnn_minus_torch_spyre_phase_ms": sendnn_mean_ms
            - torch_phase_mean_ms,
            "sendnn_minus_torch_spyre_phase_percent": 100.0
            * (sendnn_mean_ms / torch_phase_mean_ms - 1.0),
            "torch_spyre_over_sendnn_ratio": torch_phase_mean_ms / sendnn_mean_ms,
        }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
