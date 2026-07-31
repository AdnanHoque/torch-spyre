#!/usr/bin/env python3
"""Standalone torch-spyre Q/O FP8 matmul proof-of-concept benchmark.

By default, the FP8 variants exercise torch-spyre's device-native path. Odd-M
cases use the existing channel-packed activation operation; the experimental
DD2 PoC uses the minibatch-packed operation for even M:

    FP16 activation -> qfp8ch/qfp8mb --+
                                        +-> batchmatmulfp8[mb] -> FP16
    FP16 weight     -> qfp8wt ----------+

With ``--prepack-weight``, weight quantization instead runs once in a separate
compiled graph before correctness, warmups, and profiling:

    FP16 weight -> scale/clamp/qfp8wt -> persistent QFP8WT tensor

The timed graph then consumes that QFP8WT tensor directly.  This models a
static model weight while retaining dynamic activation quantization.

The raw variants stop at the FP16 matmul output.  The scaled variants
additionally apply the output scales explicitly:

    raw FP16 -> FP32 -> per-row scale -> per-column scale -> FP16

The current FP8 enablement branch accepts the scale arguments required by
``aten._scaled_mm`` but does not apply them in the lowering.  Unit FP16 schema
scales make that omission neutral; the explicit pointwise operations model the
two scale-application stages without relying on incomplete lowering semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch


FP8_VARIANTS = {
    "fp8_baseline",
    "fp8_optimized",
    "fp8_raw_baseline",
    "fp8_raw_optimized",
}
OPTIMIZED_VARIANTS = {"fp8_optimized", "fp8_raw_optimized"}
SCALED_VARIANTS = {"fp8_baseline", "fp8_optimized"}
VARIANTS = ("fp16", *sorted(FP8_VARIANTS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument(
        "--m-split",
        type=int,
        help="override the optimized variant's M work-division split",
    )
    parser.add_argument(
        "--n-split",
        type=int,
        help="override the optimized variant's N work-division split",
    )
    parser.add_argument(
        "--prepack-weight",
        action="store_true",
        help=(
            "quantize the FP16 weight once in a separate compiled graph and "
            "exclude that work from the timed graph"
        ),
    )
    parser.add_argument(
        "--activation-packing",
        choices=("auto", "channel", "minibatch"),
        default="auto",
        help=(
            "select the FP8 activation stick layout; 'channel' is a "
            "diagnostic control for the existing QFP8CH path and 'minibatch' "
            "forces the experimental DD2 QFP8MB path"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(args.m, args.k, args.n) < 1:
        shape = (args.m, args.k, args.n)
        raise ValueError(f"matrix dimensions must be positive: {shape}")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.reps < 1:
        raise ValueError("--reps must be positive")
    if (args.m_split is None) != (args.n_split is None):
        raise ValueError("--m-split and --n-split must be provided together")
    if args.m_split is not None and min(args.m_split, args.n_split) < 1:
        raise ValueError("work-division splits must be positive")
    if args.prepack_weight and args.variant == "fp16":
        raise ValueError("--prepack-weight is only valid for FP8 variants")
    if args.variant == "fp16" and args.activation_packing != "auto":
        raise ValueError("--activation-packing is only valid for FP8 variants")
    if args.activation_packing == "minibatch" and (
        args.m % 2 != 0 or args.k % 64 != 0
    ):
        raise ValueError(
            "minibatch activation packing requires M % 2 == 0 and K % 64 == 0"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def spyre_layout_metadata(tensor: torch.Tensor) -> dict[str, object]:
    """Return JSON-safe physical-layout metadata without relying on STL repr."""

    layout = tensor.device_tensor_layout()
    if layout is None:
        raise RuntimeError("Spyre tensor has no device layout")
    element_arrangement = layout.element_arrangement
    device_dtype = layout.device_dtype
    return {
        "device_size": [int(size) for size in layout.device_size],
        "stride_map": [int(stride) for stride in layout.stride_map],
        "device_dtype": getattr(device_dtype, "name", str(device_dtype)),
        "element_arrangement": getattr(
            element_arrangement,
            "name",
            str(element_arrangement),
        ),
    }


def make_host_data(
    m: int, k: int, n: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create deterministic, FP8-representable operands and unit scales."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260729)

    # Multiples of 1/8 in [-1, 1] are exactly representable near this range in
    # E4M3.  qfp8wt owns the physical FP8 weight arrangement, so its FP16 input
    # follows the ordinary contiguous logical [K,N] contract used by the
    # torch-spyre FP8 tests.
    activation_fp16 = (
        torch.randint(
            -8,
            9,
            (m, k),
            dtype=torch.int16,
            generator=generator,
        ).to(torch.float32)
        / 8.0
    ).to(torch.float16)
    weight_fp16 = (
        torch.randint(
            -8,
            9,
            (k, n),
            dtype=torch.int16,
            generator=generator,
        ).to(torch.float32)
        / 8.0
    ).to(torch.float16)

    quant_scale_a = torch.ones((), dtype=torch.float16)
    quant_scale_b = torch.ones((), dtype=torch.float16)
    output_scale_a = torch.ones((m, 1), dtype=torch.float32)
    output_scale_b = torch.ones((1, n), dtype=torch.float32)
    return (
        activation_fp16,
        weight_fp16,
        quant_scale_a,
        quant_scale_b,
        output_scale_a,
        output_scale_b,
    )


def optimized_work_division(
    m: int,
    n: int,
    m_split_override: int | None = None,
    n_split_override: int | None = None,
) -> dict[str, int]:
    """Return the experimental 32-core MBxOUT split.

    Large-M cases use the SenDNN-inspired MB8 x OUT4 split.  DD2's qfp8mb
    layout packs pairs of adjacent M rows, so every M split must leave an even
    number of rows on each core.  The smaller power-of-two cases therefore use
    M=2 -> 1x32, M=4 -> 2x16, and M=8 -> 4x8.  OUT is divided in device sticks
    (64 FP16 elements per stick).
    """

    if m_split_override is not None and n_split_override is not None:
        m_split = m_split_override
        n_split = n_split_override
    else:
        m_split = 1 if m == 1 else min(m // 2, 8)
        n_split = 32 // m_split
    if m % m_split != 0 or 32 % m_split != 0:
        raise ValueError(
            f"optimized variant requires min(M,8) to divide both M and 32; "
            f"got M={m}, split={m_split}"
        )
    if m_split * n_split > 32:
        raise ValueError(
            f"optimized split uses {m_split * n_split} cores; maximum is 32"
        )
    if m > 1 and (m // m_split) % 2 != 0:
        raise ValueError(
            "DD2 qfp8mb packs two M rows per physical stick; optimized "
            f"M split={m_split} leaves odd per-core M={m // m_split}"
        )
    n_sticks = (n + 63) // 64
    if n_sticks % n_split != 0:
        raise ValueError(
            f"optimized OUT split={n_split} does not divide N={n} "
            f"({n_sticks} FP16 sticks)"
        )
    return {"M": m_split, "N": n_split}


def annotate_named_dimensions(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output_scale_a: torch.Tensor,
    output_scale_b: torch.Tensor,
    m: int,
    k: int,
    n: int,
) -> None:
    from torch_spyre._inductor.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
        reset,
    )

    reset()
    declare_tensor_dim("M", m)
    declare_tensor_dim("K", k)
    declare_tensor_dim("N", n)
    declare_tensor_dim("ONE", 1)
    name_tensor_dims(activation, ["M", "K"])
    name_tensor_dims(weight, ["K", "N"])
    name_tensor_dims(output_scale_a, ["M", "ONE"])
    name_tensor_dims(output_scale_b, ["ONE", "N"])


def raw_scaled_mm(
    activation_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    quant_scale_a: torch.Tensor,
    quant_scale_b: torch.Tensor,
) -> torch.Tensor:
    # Torch 2.11 requires both scale tensors in the public schema.  The current
    # FP8 enablement lowering ignores them.  They are unit scalars here, so this
    # is neutral; explicit output scaling below is the only non-unit scale path.
    return torch.ops.aten._scaled_mm.default(
        activation_fp8,
        weight_fp8,
        quant_scale_a,
        quant_scale_b,
        None,
        None,
        torch.float16,
        False,
    )


def quantize_activation_fp8(
    activation: torch.Tensor,
    quant_scale_a: torch.Tensor,
    activation_packing: str,
) -> torch.Tensor:
    """Quantize activation while allowing an explicit physical-layout control."""

    if activation_packing == "auto":
        return torch.ops.spyre.quantize_fp8_with_scale(
            activation,
            quant_scale_a,
        )

    inv_scale = torch.reciprocal(quant_scale_a)
    scaled = activation * inv_scale
    clamped = torch.ops.spyre.clamp(scaled, -448.0, 448.0)
    if activation_packing == "channel":
        return torch.ops.spyre.qfp8ch(clamped)
    if activation_packing == "minibatch":
        return torch.ops.spyre.qfp8mb(clamped)
    raise AssertionError(f"unknown activation packing: {activation_packing}")


def make_benchmark_fn(
    variant: str,
    work_division: dict[str, int] | None,
    prepack_weight: bool,
    activation_packing: str,
) -> Callable[..., torch.Tensor]:
    if variant == "fp16":

        def fp16_fn(
            activation: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            return torch.matmul(activation, weight)

        return fp16_fn

    optimized = variant in OPTIMIZED_VARIANTS
    scaled = variant in SCALED_VARIANTS

    if optimized:
        if work_division is None:
            raise AssertionError("optimized variant requires a work-division hint")
        from torch_spyre._inductor import spyre_hint

        def fp8_optimized_fn(
            activation: torch.Tensor,
            weight: torch.Tensor,
            quant_scale_a: torch.Tensor,
            quant_scale_b: torch.Tensor,
            output_scale_a: torch.Tensor,
            output_scale_b: torch.Tensor,
        ) -> torch.Tensor:
            activation_fp8 = quantize_activation_fp8(
                activation,
                quant_scale_a,
                activation_packing,
            )
            weight_fp8 = (
                weight
                if prepack_weight
                else torch.ops.spyre.quantize_weight_fp8_with_scale(
                    weight, quant_scale_b
                )
            )
            with spyre_hint(work_div=work_division):
                result = raw_scaled_mm(
                    activation_fp8,
                    weight_fp8,
                    quant_scale_a,
                    quant_scale_b,
                )
            if not scaled:
                return result
            with spyre_hint(work_div=work_division):
                result = result.to(torch.float32)
            with spyre_hint(work_div=work_division):
                result = result * output_scale_a
            with spyre_hint(work_div=work_division):
                result = result * output_scale_b
            with spyre_hint(work_div=work_division):
                result = result.to(torch.float16)
            return result

        return fp8_optimized_fn

    def fp8_baseline_fn(
        activation: torch.Tensor,
        weight: torch.Tensor,
        quant_scale_a: torch.Tensor,
        quant_scale_b: torch.Tensor,
        output_scale_a: torch.Tensor,
        output_scale_b: torch.Tensor,
    ) -> torch.Tensor:
        activation_fp8 = quantize_activation_fp8(
            activation,
            quant_scale_a,
            activation_packing,
        )
        weight_fp8 = (
            weight
            if prepack_weight
            else torch.ops.spyre.quantize_weight_fp8_with_scale(
                weight, quant_scale_b
            )
        )
        result = raw_scaled_mm(
            activation_fp8,
            weight_fp8,
            quant_scale_a,
            quant_scale_b,
        )
        if not scaled:
            return result
        result = result.to(torch.float32)
        result = result * output_scale_a
        result = result * output_scale_b
        return result.to(torch.float16)

    return fp8_baseline_fn


def synchronize() -> None:
    spyre_module = getattr(torch, "spyre", None)
    if spyre_module is not None and hasattr(spyre_module, "synchronize"):
        spyre_module.synchronize()


def cpu_reference(
    variant: str,
    activation_fp16: torch.Tensor,
    weight_fp16: torch.Tensor,
    quant_scale_a: torch.Tensor,
    quant_scale_b: torch.Tensor,
    output_scale_a: torch.Tensor,
    output_scale_b: torch.Tensor,
) -> torch.Tensor:
    if variant in FP8_VARIANTS:
        activation = (
            (activation_fp16 / quant_scale_a)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(torch.float32)
        )
        weight = (
            (weight_fp16 / quant_scale_b)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(torch.float32)
        )
    else:
        activation = activation_fp16.to(torch.float32)
        weight = weight_fp16.to(torch.float32)

    result = torch.matmul(activation, weight).to(torch.float16)
    if variant in SCALED_VARIANTS:
        result = result.to(torch.float32)
        result = result * output_scale_a
        result = result * output_scale_b
        result = result.to(torch.float16)
    return result


def correctness_metrics(
    variant: str, actual: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | bool]:
    actual_fp32 = actual.to(torch.float32)
    reference_fp32 = reference.to(torch.float32)
    error = (actual_fp32 - reference_fp32).abs()
    reference_norm = torch.linalg.vector_norm(reference_fp32)
    relative_l2 = (
        torch.linalg.vector_norm(actual_fp32 - reference_fp32)
        / reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
    ).item()

    if variant in FP8_VARIANTS:
        # The PT reduction order is not the same as the CPU reference.  Use
        # an elementwise guard large enough for cancellation-heavy outputs,
        # together with the substantially tighter aggregate relative-L2 gate.
        rtol, atol, relative_l2_limit = 0.10, 2.0, 0.10
    else:
        rtol, atol, relative_l2_limit = 0.02, 1.0, 0.01
    allclose = torch.allclose(actual_fp32, reference_fp32, rtol=rtol, atol=atol)
    finite = bool(torch.isfinite(actual_fp32).all())
    passed = bool(
        allclose
        and finite
        and math.isfinite(relative_l2)
        and relative_l2 <= relative_l2_limit
    )
    return {
        "passed": passed,
        "all_finite": finite,
        "allclose": bool(allclose),
        "rtol": rtol,
        "atol": atol,
        "relative_l2_limit": relative_l2_limit,
        "relative_l2_error": relative_l2,
        "max_abs_error": error.max().item(),
        "mean_abs_error": error.mean().item(),
        "rmse": torch.sqrt(torch.mean(error.square())).item(),
    }


def summarize_trace(trace_path: Path) -> dict:
    with trace_path.open() as handle:
        events = json.load(handle).get("traceEvents", [])

    category_summary: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "duration_us": 0.0}
    )
    kernel_events: list[dict[str, float | str]] = []
    kernel_by_name: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "duration_us": 0.0}
    )
    for event in events:
        duration_us = float(event.get("dur", 0.0) or 0.0)
        if duration_us <= 0:
            continue
        category = str(event.get("cat", "<none>"))
        name = str(event.get("name", "<unknown>"))
        category_summary[category]["count"] += 1
        category_summary[category]["duration_us"] += duration_us
        if category == "kernel":
            kernel_events.append(
                {
                    "name": name,
                    "category": category,
                    "duration_us": duration_us,
                }
            )
            kernel_by_name[name]["count"] += 1
            kernel_by_name[name]["duration_us"] += duration_us

    return {
        "category_summary": dict(sorted(category_summary.items())),
        "kernel_events": kernel_events,
        "kernel_by_name": dict(sorted(kernel_by_name.items())),
    }


def effective_matmul_tflops(m: int, k: int, n: int, mean_us: float) -> float:
    return (2.0 * m * k * n) / (mean_us * 1.0e6)


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch_spyre

    if hasattr(torch_spyre, "_autoload"):
        torch_spyre._autoload()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "kineto_trace.json"
    result_path = output_dir / "result.json"

    (
        activation_fp16,
        weight_fp16,
        quant_scale_a_host,
        quant_scale_b_host,
        output_scale_a_host,
        output_scale_b_host,
    ) = make_host_data(args.m, args.k, args.n)

    activation = activation_fp16.to("spyre")
    weight = weight_fp16.to("spyre")
    quant_scale_a = quant_scale_a_host.to("spyre")
    quant_scale_b = quant_scale_b_host.to("spyre")
    output_scale_a = output_scale_a_host.to("spyre")
    output_scale_b = output_scale_b_host.to("spyre")
    compile_options = {"epilogue_fusion": False}

    timed_weight = weight
    prepack_metadata: dict[str, object] = {
        "enabled": False,
        "separate_compiled_graph": False,
        "excluded_from_profile": True,
    }
    if args.prepack_weight:
        if quant_scale_b_host.shape != torch.Size([]):
            raise AssertionError("weight prepack requires a scalar quantization scale")
        if quant_scale_b_host.item() != 1.0:
            raise AssertionError("weight prepack requires a unit quantization scale")

        def weight_prepack_fn(
            weight_input: torch.Tensor,
            unit_scale: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.spyre.quantize_weight_fp8_with_scale(
                weight_input,
                unit_scale,
            )

        compiled_weight_prepack = torch.compile(
            weight_prepack_fn,
            backend="inductor",
            dynamic=False,
            fullgraph=True,
            options=compile_options,
        )
        prepack_start_ns = time.perf_counter_ns()
        timed_weight = compiled_weight_prepack(weight, quant_scale_b)
        synchronize()
        prepack_compile_and_first_run_ms = (
            time.perf_counter_ns() - prepack_start_ns
        ) / 1.0e6

        if timed_weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"weight prepack returned {timed_weight.dtype}, expected E4M3FN"
            )
        if tuple(timed_weight.shape) != (args.k, args.n):
            raise RuntimeError(
                "weight prepack returned shape "
                f"{tuple(timed_weight.shape)}, expected {(args.k, args.n)}"
            )

        from torch_spyre._C import ElementArrangement

        prepacked_layout = timed_weight.device_tensor_layout()
        if prepacked_layout is None:
            raise RuntimeError("prepacked weight has no SpyreTensorLayout")
        if prepacked_layout.element_arrangement != ElementArrangement.QFP8WT:
            raise RuntimeError(
                "prepacked weight did not retain QFP8WT arrangement: "
                f"{prepacked_layout.element_arrangement}"
            )
        prepack_metadata = {
            "enabled": True,
            "separate_compiled_graph": True,
            "excluded_from_profile": True,
            "operation": "spyre.quantize_weight_fp8_with_scale",
            "scale": {
                "shape": [],
                "dtype": str(quant_scale_b_host.dtype),
                "value": float(quant_scale_b_host.item()),
            },
            "input": {
                "shape": [args.k, args.n],
                "dtype": str(weight.dtype),
            },
            "output": {
                "shape": [args.k, args.n],
                "dtype": str(timed_weight.dtype),
                "layout": spyre_layout_metadata(timed_weight),
            },
            "compile": {
                "backend": "inductor",
                "dynamic": False,
                "fullgraph": True,
                "options": compile_options,
                "compile_and_first_run_ms": prepack_compile_and_first_run_ms,
            },
        }

    work_division = (
        optimized_work_division(
            args.m,
            args.n,
            args.m_split,
            args.n_split,
        )
        if args.variant in OPTIMIZED_VARIANTS
        else None
    )
    if args.variant in OPTIMIZED_VARIANTS:
        annotate_named_dimensions(
            activation,
            timed_weight,
            output_scale_a,
            output_scale_b,
            args.m,
            args.k,
            args.n,
        )

    fn = make_benchmark_fn(
        args.variant,
        work_division,
        args.prepack_weight,
        args.activation_packing,
    )
    compiled = torch.compile(
        fn,
        # Torch-Spyre registers its device lowering with Inductor.  Some
        # development trees also register a "spyre" alias, but the pinned
        # Kineto environment used for this study does not.
        backend="inductor",
        dynamic=False,
        fullgraph=True,
        options=compile_options,
    )
    device_args = (
        (activation, weight)
        if args.variant == "fp16"
        else (
            activation,
            timed_weight,
            quant_scale_a,
            quant_scale_b,
            output_scale_a,
            output_scale_b,
        )
    )

    compile_start_ns = time.perf_counter_ns()
    actual_device = compiled(*device_args)
    synchronize()
    compile_and_first_run_ms = (time.perf_counter_ns() - compile_start_ns) / 1.0e6

    # Correctness is a hard precondition for collecting performance data.
    actual = actual_device.cpu()
    reference_start_ns = time.perf_counter_ns()
    reference = cpu_reference(
        args.variant,
        activation_fp16,
        weight_fp16,
        quant_scale_a_host,
        quant_scale_b_host,
        output_scale_a_host,
        output_scale_b_host,
    )
    cpu_reference_ms = (time.perf_counter_ns() - reference_start_ns) / 1.0e6
    correctness = correctness_metrics(args.variant, actual, reference)
    correctness["cpu_reference_ms"] = cpu_reference_ms
    if not correctness["passed"]:
        raise RuntimeError(f"correctness failed before profiling: {correctness}")

    for _ in range(args.warmups):
        actual_device = compiled(*device_args)
        synchronize()

    from torch.profiler import ProfilerActivity, profile

    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=False,
    )
    profiler.start()
    wall_start_ns = time.perf_counter_ns()
    for _ in range(args.reps):
        actual_device = compiled(*device_args)
        synchronize()
        profiler.step()
    wall_total_us = (time.perf_counter_ns() - wall_start_ns) / 1000.0
    profiler.stop()
    profiler.export_chrome_trace(str(trace_path))

    trace_summary = summarize_trace(trace_path)
    kernel_events = trace_summary["kernel_events"]
    if not kernel_events:
        raise RuntimeError(
            "Kineto trace contains no positive-duration events with cat='kernel'"
        )
    kernel_durations_us = [float(event["duration_us"]) for event in kernel_events]
    kernel_total_us = sum(kernel_durations_us)
    kernel_mean_us = kernel_total_us / args.reps

    script_path = Path(__file__).resolve()
    result = {
        "schema_version": 1,
        "variant": args.variant,
        "logical_shape": {"M": args.m, "K": args.k, "N": args.n},
        "graph_contract": {
            "host_prequantized_fp8": False,
            "device_activation_quantization": args.variant in FP8_VARIANTS,
            "activation_packing": (
                args.activation_packing
                if args.variant in FP8_VARIANTS
                else "not_applicable"
            ),
            "device_weight_quantization": (
                args.variant in FP8_VARIANTS and not args.prepack_weight
            ),
            "device_weight_quantization_in_timed_graph": (
                args.variant in FP8_VARIANTS and not args.prepack_weight
            ),
            "prepacked_weight_input": args.prepack_weight,
            "quantization_scale_derivation": "excluded; unit FP16 scalars supplied",
            "raw_scaled_mm_output_dtype": "torch.float16",
            "output_scale_application": (
                "FP16 -> FP32 -> row FP32 mul -> column FP32 mul -> FP16"
                if args.variant in SCALED_VARIANTS
                else "none"
            ),
            "scaled_mm_schema_scales": (
                "unit FP16 scalars; ignored by current PR #2286 lowering"
            ),
            "use_fast_accum": False,
        },
        "weight_prepack": prepack_metadata,
        "work_division_hint": work_division,
        "compile": {
            "backend": "inductor",
            "dynamic": False,
            "fullgraph": True,
            "options": compile_options,
            "compile_and_first_run_ms": compile_and_first_run_ms,
        },
        "correctness": correctness,
        "warmups": args.warmups,
        "repetitions": args.reps,
        "wall_total_us": wall_total_us,
        "wall_mean_us": wall_total_us / args.reps,
        "kernel_event_count": len(kernel_events),
        "kernel_total_us": kernel_total_us,
        "kernel_mean_us_per_iteration": kernel_mean_us,
        "kernel_event_duration_median_us": statistics.median(kernel_durations_us),
        "effective_matmul_tflops": effective_matmul_tflops(
            args.m, args.k, args.n, kernel_mean_us
        ),
        "effective_tflops_numerator": "2*M*K*N",
        "trace_summary": trace_summary,
        "trace_path": str(trace_path),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_spyre": package_version("torch-spyre"),
            "torch_spyre_module": torch_spyre.__file__,
        },
        "script": {"path": str(script_path), "sha256": sha256(script_path)},
        "environment": {
            key: os.environ.get(key)
            for key in (
                "TORCHINDUCTOR_CACHE_DIR",
                "DXP_LX_FRAC_AVAIL",
                "SENCORES",
                "SENCORELETS",
                "DT_OPT",
            )
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
