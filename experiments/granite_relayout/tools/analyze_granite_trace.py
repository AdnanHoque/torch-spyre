#!/usr/bin/env python3
"""Analyze the exact four-token Granite trace contract used in this repository.

The contract is one prefill followed by three decode calls.  Kernel phases end
at the sampling ``div`` kernel, which lets the same analyzer handle Antoni's
40-layer trace and the reproduced one-layer, 20-iteration trace.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path
from typing import Any


PHASES = (
    ("prefill", 5),
    ("decode_first", 7),
    ("decode_steady_1", 6),
    ("decode_steady_2", 6),
)

# The current fused Torch-Spyre checkout emits one giant prefill kernel per
# decoder block, three kernels for the first decode block (cache setup), and
# one kernel for each steady decode block.  Keep this opt-in so the historical
# five/seven/six/six artifact contract remains strict by default.
FUSED_PHASES = (
    ("prefill", 1),
    ("decode_first", 3),
    ("decode_steady_1", 1),
    ("decode_steady_2", 1),
)

PREFILL_COMPONENTS = (
    "projection_norm",
    "attention_qk",
    "attention_av",
    "swiglu",
    "residual",
)


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


def is_input_kernel(name: str) -> bool:
    return "index_select_0" in name or "sdsc_fused_mul_0_" in name


def split_phases(
    kernels: list[dict[str, Any]], phase_specs: tuple[tuple[str, int], ...]
) -> list[list[dict[str, Any]]]:
    phases: list[list[dict[str, Any]]] = []
    start = 0
    for index, kernel in enumerate(kernels):
        if "sdsc_fused_div_0_" not in kernel["name"]:
            continue
        phases.append(kernels[start : index + 1])
        start = index + 1
    if start != len(kernels):
        raise ValueError(f"{len(kernels) - start} kernels remain after the last phase")
    if len(phases) % len(phase_specs):
        raise ValueError(
            f"expected a multiple of {len(phase_specs)} phases, found {len(phases)}"
        )
    return phases


def analyze(
    path: Path,
    phase_specs: tuple[tuple[str, int], ...] = PHASES,
    *,
    allow_fused_block_layout: bool = False,
    selected_kinds: set[str] | None = None,
) -> dict[str, Any]:
    trace = read_json(path)
    kernels = [
        {
            "name": event["name"],
            "duration_ms": event["dur"] / 1000.0,
            "timestamp_us": event["ts"],
        }
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel" and "dur" in event
    ]
    phases = split_phases(kernels, phase_specs)

    selected_kinds = selected_kinds or {name for name, _ in phase_specs}
    by_kind: dict[str, dict[str, list[float] | int]] = {
        name: {
            "layers": [],
            "block_total": [],
            "one_off": [],
            "phase_total": [],
            "layer_count": 0,
        }
        for name, _ in phase_specs
        if name in selected_kinds
    }
    prefill_components = {name: [] for name in PREFILL_COMPONENTS}

    for phase_index, phase in enumerate(phases):
        kind, kernels_per_layer = phase_specs[phase_index % len(phase_specs)]
        if kind not in selected_kinds:
            continue
        phase_total = sum(kernel["duration_ms"] for kernel in phase)

        body_start = 0
        while body_start < len(phase) and is_input_kernel(phase[body_start]["name"]):
            body_start += 1

        if len(phase) - body_start < 3:
            raise ValueError(f"{kind} phase is missing output-head kernels")
        tail = phase[-3:]
        expected_tail = ("add_mean_mul_rsqrt", "bmm_transpose", "div_0")
        for kernel, marker in zip(tail, expected_tail):
            if marker not in kernel["name"]:
                raise ValueError(
                    f"{kind} phase tail expected {marker}, got {kernel['name']}"
                )

        block = phase[body_start:-3]
        if len(block) % kernels_per_layer:
            if not allow_fused_block_layout or not block:
                raise ValueError(
                    f"{kind} has {len(block)} block kernels; expected a multiple of "
                    f"{kernels_per_layer}"
                )
            # A nonstandard fused kernel count is accepted only in the
            # explicit analysis mode; it is never a silent relaxation of the
            # historical production artifact contract.
            kernels_per_layer = len(block)
            layer_count = 1
        else:
            layer_count = len(block) // kernels_per_layer
        recorded_layer_count = int(by_kind[kind]["layer_count"])
        if recorded_layer_count not in (0, layer_count):
            raise ValueError(
                f"{kind} layer count changed from {recorded_layer_count} to {layer_count}"
            )
        by_kind[kind]["layer_count"] = layer_count

        layer_totals = []
        for offset in range(0, len(block), kernels_per_layer):
            layer = block[offset : offset + kernels_per_layer]
            layer_totals.append(sum(kernel["duration_ms"] for kernel in layer))
            if kind == "prefill" and len(layer) == len(PREFILL_COMPONENTS):
                for label, kernel in zip(PREFILL_COMPONENTS, layer):
                    prefill_components[label].append(kernel["duration_ms"])
            elif kind == "prefill" and allow_fused_block_layout:
                prefill_components.setdefault("fused_block", []).append(
                    sum(kernel["duration_ms"] for kernel in layer)
                )

        block_total = sum(layer_totals)
        values = by_kind[kind]
        values["layers"].extend(layer_totals)  # type: ignore[union-attr]
        values["block_total"].append(block_total)  # type: ignore[union-attr]
        values["one_off"].append(phase_total - block_total)  # type: ignore[union-attr]
        values["phase_total"].append(phase_total)  # type: ignore[union-attr]

    phase_metrics: dict[str, Any] = {}
    for kind, _ in phase_specs:
        if kind not in selected_kinds:
            continue
        values = by_kind[kind]
        phase_metrics[kind] = {
            "layer_count": values["layer_count"],
            "layer": stats(values["layers"]),  # type: ignore[arg-type]
            "block_total": stats(values["block_total"]),  # type: ignore[arg-type]
            "one_off": stats(values["one_off"]),  # type: ignore[arg-type]
            "phase_total": stats(values["phase_total"]),  # type: ignore[arg-type]
        }

    return {
        # Preserve the caller-provided path so checked-in metrics are identical
        # in every clone. Absolute provenance paths live in environment.json.
        "path": str(path),
        "kernel_events": len(kernels),
        "generation_runs": len(phases) // len(phase_specs),
        "phase_metrics": phase_metrics,
        "prefill_components": {
            name: stats(values)
            for name, values in prefill_components.items()
            if values
        },
    }


def comparison(
    reference: dict[str, Any],
    reproduction: dict[str, Any],
    phase_specs: tuple[tuple[str, int], ...] = PHASES,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for kind, _ in phase_specs:
        ref = reference["phase_metrics"][kind]
        repro = reproduction["phase_metrics"][kind]
        projected_block = repro["layer"]["mean_ms"] * ref["layer_count"]
        projected_total = projected_block + repro["one_off"]["mean_ms"]
        actual_total = ref["phase_total"]["mean_ms"]
        rows[kind] = {
            "reference_layers": ref["layer_count"],
            "reference_layer_mean_ms": ref["layer"]["mean_ms"],
            "reference_phase_total_ms": actual_total,
            "reproduction_layers": repro["layer_count"],
            "reproduction_layer_mean_ms": repro["layer"]["mean_ms"],
            "reproduction_one_off_mean_ms": repro["one_off"]["mean_ms"],
            "projected_block_ms": projected_block,
            "projected_phase_total_ms": projected_total,
            "projected_minus_reference_ms": projected_total - actual_total,
            "projected_minus_reference_percent": 100.0
            * (projected_total / actual_total - 1.0),
        }

    decode_kinds = [name for name, _ in phase_specs if name.startswith("decode")]
    result = {
        "phases": rows,
    }
    if decode_kinds:
        reference_decode = statistics.mean(
            rows[name]["reference_phase_total_ms"] for name in decode_kinds
        )
        projected_decode = statistics.mean(
            rows[name]["projected_phase_total_ms"] for name in decode_kinds
        )
        result["decode_average"] = {
            "reference_device_ms": reference_decode,
            "projected_device_ms": projected_decode,
            "projected_minus_reference_ms": projected_decode - reference_decode,
            "projected_minus_reference_percent": 100.0
            * (projected_decode / reference_decode - 1.0),
            "reference_gap_to_reported_177_ms": 177.0 - reference_decode,
            "projected_gap_to_reported_177_ms": 177.0 - projected_decode,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tokens-per-generation",
        type=int,
        choices=range(1, len(PHASES) + 1),
        default=len(PHASES),
        help="number of traced model phases in each measured generation",
    )
    parser.add_argument(
        "--fused-block-layout",
        action="store_true",
        help="use current giant-kernel counts (prefill=1, first decode=3, steady=1)",
    )
    parser.add_argument(
        "--only-phase",
        choices=[name for name, _ in PHASES],
        help="validate and compare only this phase while retaining full-cycle parsing",
    )
    args = parser.parse_args()

    phase_specs = (
        FUSED_PHASES if args.fused_block_layout else PHASES
    )[: args.tokens_per_generation]
    selected_kinds = {args.only_phase} if args.only_phase else None
    comparison_phase_specs = tuple(
        spec for spec in phase_specs if selected_kinds is None or spec[0] in selected_kinds
    )
    reference = analyze(
        args.reference,
        phase_specs,
        allow_fused_block_layout=args.fused_block_layout,
        selected_kinds=selected_kinds,
    )
    reproduction = analyze(
        args.reproduction,
        phase_specs,
        allow_fused_block_layout=args.fused_block_layout,
        selected_kinds=selected_kinds,
    )
    result = {
        "reference": reference,
        "reproduction": reproduction,
        "comparison": comparison(reference, reproduction, comparison_phase_specs),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
