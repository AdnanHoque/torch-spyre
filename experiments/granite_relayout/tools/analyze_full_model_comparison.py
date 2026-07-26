#!/usr/bin/env python3
"""Compare matched full-model SenDNN and Torch-Spyre Granite traces."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from pathlib import Path
from typing import Any

from analyze_granite_trace import analyze as analyze_torch_spyre


PHASES = ("prefill", "decode_first", "decode_steady_1", "decode_steady_2")
TIMING_RE = re.compile(
    r"Per-token timing information:\s*"
    r"([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+) ms"
)
TOTAL_CYCLES_RE = re.compile(r"^Total\s+(\d+)\s*$", re.MULTILINE)


def read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


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


def parse_profiled_wall(path: Path, iterations: int) -> dict[str, Any]:
    matches = [
        [float(value) for value in match]
        for match in TIMING_RE.findall(path.read_text(encoding="utf-8"))
    ]
    if len(matches) < iterations:
        raise ValueError(
            f"{path} has {len(matches)} per-token rows; expected at least {iterations}"
        )
    # Torch-Spyre logs one unprofiled materialization row before the measured
    # rows. SenDNN's warmup row is redirected to the compiler log. Taking the
    # final rows is therefore correct for both launchers.
    measured = matches[-iterations:]
    return {
        "rows_ms": measured,
        "prefill": stats([row[0] for row in measured]),
        "decode": stats([value for row in measured for value in row[1:]]),
        "interpretation": (
            "Profiled Python wall time; retained as diagnostics and excluded "
            "from the device-program comparison."
        ),
    }


def analyze_sendnn(
    trace_path: Path,
    run_log_path: Path,
    compiler_log_path: Path,
    iterations: int,
    core_frequency_mhz: float,
) -> dict[str, Any]:
    trace = read_json(trace_path)
    events = trace["traceEvents"]
    kernels = sorted(
        (
            event
            for event in events
            if event.get("cat") == "kernel" and "dur" in event
        ),
        key=lambda event: event["ts"],
    )
    expected_kernel_count = iterations * len(PHASES)
    if len(kernels) != expected_kernel_count:
        raise ValueError(
            f"expected {expected_kernel_count} SenDNN kernels, found {len(kernels)}"
        )
    kernel_names = sorted({event["name"] for event in kernels})
    if kernel_names != ["embedding"]:
        raise ValueError(f"unexpected fused SenDNN kernel names: {kernel_names}")

    phase_values = {phase: [] for phase in PHASES}
    for run_index in range(iterations):
        row = kernels[run_index * len(PHASES) : (run_index + 1) * len(PHASES)]
        for phase, event in zip(PHASES, row):
            phase_values[phase].append(event["dur"] / 1000.0)
    decode_values = [
        value
        for phase in PHASES[1:]
        for value in phase_values[phase]
    ]

    dtoh = sorted(
        (
            event
            for event in events
            if event.get("cat") == "gpu_memcpy"
            and "dur" in event
            and (
                event.get("args", {}).get("call") == "DtoH"
                or "DtoH" in event.get("name", "")
            )
        ),
        key=lambda event: event["ts"],
    )
    if len(dtoh) != len(kernels):
        raise ValueError(
            f"expected one overlapping DtoH event per kernel, found {len(dtoh)}"
        )
    overlap_percent = []
    for kernel, copy in zip(kernels, dtoh):
        start = max(kernel["ts"], copy["ts"])
        end = min(kernel["ts"] + kernel["dur"], copy["ts"] + copy["dur"])
        overlap_percent.append(100.0 * max(0.0, end - start) / kernel["dur"])

    compiler_log = compiler_log_path.read_text(encoding="utf-8")
    softmax_indices = sorted(
        {int(value) for value in re.findall(r"_safe_softmax_(\d+)", compiler_log)}
    )
    silu_indices = sorted(
        {int(value) for value in re.findall(r"silu_(\d+)", compiler_log)}
    )
    expected_indexed_layers = list(range(1, 40))
    if (
        "_safe_softmax-" not in compiler_log
        or "silu-" not in compiler_log
        or softmax_indices != expected_indexed_layers
        or silu_indices != expected_indexed_layers
    ):
        raise ValueError(
            "SenDNN compiler log does not contain the expected 40 decoder "
            "layer markers"
        )
    totals = [int(value) for value in TOTAL_CYCLES_RE.findall(compiler_log)]
    if len(totals) != 2 or totals[1] <= totals[0]:
        raise ValueError(f"unexpected SenDNN compiler cycle totals: {totals}")
    decode_cycles, prefill_cycles = totals
    prefill_mean = statistics.mean(phase_values["prefill"])
    decode_mean = statistics.mean(decode_values)
    prefill_ideal_ms = prefill_cycles / (core_frequency_mhz * 1000.0)
    decode_ideal_ms = decode_cycles / (core_frequency_mhz * 1000.0)

    return {
        "path": str(trace_path),
        "kernel_events": len(kernels),
        "generation_runs": iterations,
        "fused_kernel_name": kernel_names[0],
        "phase_metrics": {
            phase: stats(values) for phase, values in phase_values.items()
        },
        "decode_average": stats(decode_values),
        "dtoh_overlap": {
            "events": len(dtoh),
            "mean_kernel_interval_covered_percent": statistics.mean(
                overlap_percent
            ),
            "interpretation": (
                "DtoH spans the fused device program and must not be added to "
                "the kernel duration."
            ),
        },
        "compiler_cycle_proxy": {
            "core_frequency_mhz": core_frequency_mhz,
            "prefill_cycles": prefill_cycles,
            "prefill_ideal_ms": prefill_ideal_ms,
            "prefill_utilization_percent": 100.0 * prefill_ideal_ms / prefill_mean,
            "decode_cycles": decode_cycles,
            "decode_ideal_ms": decode_ideal_ms,
            "decode_utilization_percent": 100.0 * decode_ideal_ms / decode_mean,
            "interpretation": (
                "Compiler-cycle proxy, not a physical utilization counter."
            ),
        },
        "compiler_structure": {
            "decoder_layers": 40,
            "base_layer_markers": ["_safe_softmax", "silu"],
            "indexed_layer_range": [1, 39],
            "interpretation": (
                "The base marker plus contiguous suffixes 1 through 39 prove "
                "that both attention softmax and SwiGLU occur for 40 layers."
            ),
        },
        "profiled_wall": parse_profiled_wall(run_log_path, iterations),
    }


def compare(sendnn_ms: float, torch_spyre_ms: float) -> dict[str, float]:
    return {
        "sendnn_device_ms": sendnn_ms,
        "torch_spyre_device_ms": torch_spyre_ms,
        "sendnn_minus_torch_spyre_ms": sendnn_ms - torch_spyre_ms,
        "sendnn_lower_percent": 100.0 * (1.0 - sendnn_ms / torch_spyre_ms),
        "torch_spyre_over_sendnn_ratio": torch_spyre_ms / sendnn_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sendnn-trace", type=Path, required=True)
    parser.add_argument("--sendnn-run-log", type=Path, required=True)
    parser.add_argument("--sendnn-compiler-log", type=Path, required=True)
    parser.add_argument("--torch-spyre-trace", type=Path, required=True)
    parser.add_argument("--torch-spyre-run-log", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--core-frequency-mhz", type=float, default=1100.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sendnn = analyze_sendnn(
        args.sendnn_trace,
        args.sendnn_run_log,
        args.sendnn_compiler_log,
        args.iterations,
        args.core_frequency_mhz,
    )
    torch_spyre = analyze_torch_spyre(args.torch_spyre_trace)
    if torch_spyre["generation_runs"] != args.iterations:
        raise ValueError(
            "Torch-Spyre generation count mismatch: "
            f"expected {args.iterations}, got {torch_spyre['generation_runs']}"
        )
    for phase in PHASES:
        if torch_spyre["phase_metrics"][phase]["layer_count"] != 40:
            raise ValueError(f"Torch-Spyre {phase} trace is not a 40-layer run")

    torch_decode_ms = statistics.mean(
        torch_spyre["phase_metrics"][phase]["phase_total"]["mean_ms"]
        for phase in PHASES[1:]
    )
    torch_spyre["decode_average"] = {
        "mean_ms": torch_decode_ms,
        "definition": "Mean of first-decode and two steady-decode phase means.",
    }
    torch_spyre["profiled_wall"] = parse_profiled_wall(
        args.torch_spyre_run_log, args.iterations
    )

    result = {
        "contract": {
            "model": "ibm-granite/granite-3.3-8b-instruct",
            "decoder_layers": 40,
            "batch_size": 1,
            "fixed_prompt_tokens": 512,
            "generated_token_phases": 4,
            "iterations": args.iterations,
            "dtype": "fp16",
            "weights": "unfused",
            "attention": "SDPA",
            "comparison_scope": (
                "Trace-derived device-program time: fused SenDNN kernel versus "
                "the sum of Torch-Spyre phase kernels."
            ),
        },
        "sendnn": sendnn,
        "torch_spyre": torch_spyre,
        "comparison": {
            "prefill": compare(
                sendnn["phase_metrics"]["prefill"]["mean_ms"],
                torch_spyre["phase_metrics"]["prefill"]["phase_total"][
                    "mean_ms"
                ],
            ),
            "decode_average": compare(
                sendnn["decode_average"]["mean_ms"], torch_decode_ms
            ),
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
