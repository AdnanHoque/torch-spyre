#!/usr/bin/env python3
"""Measure Granite K and V projections with one reusable FP8 activation.

The FP8 graph deliberately spells the dynamic scale/normalize/pack chain twice,
as two independent linear modules do.  Torch-Spyre's focused reuse pass must
canonicalize those chains to one producer before lowering.  Static weight
packing is compiled and executed before profiling.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from bench_qo_fp8_poc import (
    correctness_metrics,
    cpu_reference,
    dynamic_fp8_row_scale,
    make_host_data,
    quantize_activation_fp8,
    raw_scaled_mm,
    scaled_mm,
    sha256,
    summarize_trace,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("fp16", "fp8_shared"), required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--weight-data-divisor", type=int, default=8)
    parser.add_argument(
        "--compact-fp16-scales",
        action="store_true",
        help=(
            "keep dynamic row scales and static column scales in the FP16 "
            "format consumed by the two backend scale programs"
        ),
    )
    parser.add_argument(
        "--explicit-activation-clamp",
        action="store_true",
        help=(
            "retain the explicit E4M3 saturation clamp in the compact FP16 "
            "scale path instead of relying on the conversion operation"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def annotate_named_dimensions(
    activation: torch.Tensor,
    weight_k: torch.Tensor,
    weight_v: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
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
    name_tensor_dims(weight_k, ["K", "N"])
    name_tensor_dims(weight_v, ["K", "N"])
    name_tensor_dims(scale_a, ["M", "ONE"])
    name_tensor_dims(scale_b, ["ONE", "N"])


def generated_op_counts(cache_dir: Path) -> dict[str, int]:
    counts = {
        "quantscalepertokenfp8": 0,
        "qfp8mb": 0,
        "batchmatmulfp8mb": 0,
        "batchnormfwd": 0,
    }
    for source in cache_dir.rglob("*.py"):
        text = source.read_text(errors="replace")
        for op in counts:
            counts[op] += text.count(f"op='{op}'")
    return counts


def main() -> None:
    args = parse_args()
    if min(args.m, args.k, args.n, args.reps) < 1 or args.warmups < 0:
        raise ValueError("M/K/N/reps must be positive and warmups non-negative")
    if args.compact_fp16_scales and args.variant != "fp8_shared":
        raise ValueError("--compact-fp16-scales requires --variant fp8_shared")
    if args.explicit_activation_clamp and not args.compact_fp16_scales:
        raise ValueError("--explicit-activation-clamp requires --compact-fp16-scales")
    if args.m % 2 or args.k % 64 or args.n % 64:
        raise ValueError(
            "the DD2 K/V pair probe requires even M and K/N multiples of 64"
        )

    import torch_spyre

    if hasattr(torch_spyre, "_autoload"):
        torch_spyre._autoload()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "kineto_trace.json"

    (
        activation_host,
        weight_k_host,
        quant_scale_a_host,
        quant_scale_b_host,
        output_scale_a_host,
        output_scale_b_host,
    ) = make_host_data(args.m, args.k, args.n, args.weight_data_divisor)
    # Keep V deterministic and representable while ensuring it is not the same
    # tensor value as K.
    weight_v_host = torch.roll(weight_k_host, shifts=1, dims=1)

    activation = activation_host.to("spyre")
    weight_k = weight_k_host.to("spyre")
    weight_v = weight_v_host.to("spyre")
    quant_scale_b = quant_scale_b_host.to("spyre")
    output_scale_a = output_scale_a_host.to("spyre")
    output_scale_b = (
        output_scale_b_host.to(torch.float16).to("spyre")
        if args.compact_fp16_scales
        else output_scale_b_host.to("spyre")
    )
    compile_options = {"epilogue_fusion": False}

    if args.variant == "fp16":

        def graph_fn(a, wk, wv):
            return torch.matmul(a, wk), torch.matmul(a, wv)

        device_args = (activation, weight_k, weight_v)
        reference_variant = "fp16"
    else:

        def prepack_weight(weight, unit_scale):
            return torch.ops.spyre.quantize_weight_fp8_with_scale(weight, unit_scale)

        compiled_prepack = torch.compile(
            prepack_weight,
            backend="inductor",
            dynamic=False,
            fullgraph=True,
            options=compile_options,
        )
        packed_weight_k = compiled_prepack(weight_k, quant_scale_b)
        packed_weight_v = compiled_prepack(weight_v, quant_scale_b)
        synchronize()

        # Keep the two source expressions independent here.  The compiler pass,
        # rather than benchmark source sharing, owns their canonicalization.
        def graph_fn(a, wk, wv, column_scale):
            scale_k = dynamic_fp8_row_scale(
                a,
                fused=True,
                keep_backend_fp16=args.compact_fp16_scales,
            )
            activation_k = quantize_activation_fp8(
                a,
                scale_k,
                "minibatch",
                direct_divide=args.compact_fp16_scales,
                skip_clamp=(
                    args.compact_fp16_scales and not args.explicit_activation_clamp
                ),
            )
            if args.compact_fp16_scales:
                zero = torch.zeros((), dtype=torch.float16, device=a.device)
                result_k = raw_scaled_mm(activation_k, wk, True)
                result_k = torch.ops.spyre.apply_fp8_scale(result_k, scale_k, zero)
                result_k = torch.ops.spyre.apply_fp8_scale(result_k, column_scale, zero)
            else:
                result_k = scaled_mm(activation_k, wk, scale_k, column_scale, True)

            scale_v = dynamic_fp8_row_scale(
                a,
                fused=True,
                keep_backend_fp16=args.compact_fp16_scales,
            )
            activation_v = quantize_activation_fp8(
                a,
                scale_v,
                "minibatch",
                direct_divide=args.compact_fp16_scales,
                skip_clamp=(
                    args.compact_fp16_scales and not args.explicit_activation_clamp
                ),
            )
            if args.compact_fp16_scales:
                zero = torch.zeros((), dtype=torch.float16, device=a.device)
                result_v = raw_scaled_mm(activation_v, wv, True)
                result_v = torch.ops.spyre.apply_fp8_scale(result_v, scale_v, zero)
                result_v = torch.ops.spyre.apply_fp8_scale(result_v, column_scale, zero)
            else:
                result_v = scaled_mm(activation_v, wv, scale_v, column_scale, True)
            return result_k, result_v

        annotate_named_dimensions(
            activation,
            packed_weight_k,
            packed_weight_v,
            output_scale_a,
            output_scale_b,
            args.m,
            args.k,
            args.n,
        )
        device_args = (
            activation,
            packed_weight_k,
            packed_weight_v,
            output_scale_b,
        )
        reference_variant = "fp8_optimized"

    compiled = torch.compile(
        graph_fn,
        backend="inductor",
        dynamic=False,
        fullgraph=True,
        options=compile_options,
    )
    compile_start_ns = time.perf_counter_ns()
    actual_device = compiled(*device_args)
    synchronize()
    compile_and_first_run_ms = (time.perf_counter_ns() - compile_start_ns) / 1.0e6

    actual_k, actual_v = (tensor.cpu() for tensor in actual_device)
    reference_start_ns = time.perf_counter_ns()
    reference_k = cpu_reference(
        reference_variant,
        activation_host,
        weight_k_host,
        quant_scale_a_host,
        quant_scale_b_host,
        output_scale_a_host,
        output_scale_b_host,
        args.variant == "fp8_shared",
        args.variant == "fp8_shared",
    )
    reference_v = cpu_reference(
        reference_variant,
        activation_host,
        weight_v_host,
        quant_scale_a_host,
        quant_scale_b_host,
        output_scale_a_host,
        output_scale_b_host,
        args.variant == "fp8_shared",
        args.variant == "fp8_shared",
    )
    reference_ms = (time.perf_counter_ns() - reference_start_ns) / 1.0e6
    correctness = {
        "K": correctness_metrics(reference_variant, actual_k, reference_k),
        "V": correctness_metrics(reference_variant, actual_v, reference_v),
    }
    if not all(metrics["passed"] for metrics in correctness.values()):
        raise RuntimeError(f"correctness failed before profiling: {correctness}")

    for _ in range(args.warmups):
        actual_device = compiled(*device_args)
        synchronize()

    from torch.profiler import ProfilerActivity, profile

    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=False,
        acc_events=True,
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
    kernel_durations = [
        float(event["duration_us"]) for event in trace_summary["kernel_events"]
    ]
    if not kernel_durations:
        raise RuntimeError("Kineto trace contains no kernel events")
    kernel_mean_us = sum(kernel_durations) / args.reps
    cache_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    op_counts = generated_op_counts(cache_dir)
    if args.variant == "fp8_shared" and (
        op_counts["quantscalepertokenfp8"] != 1 or op_counts["qfp8mb"] != 1
    ):
        raise RuntimeError(f"activation reuse did not materialize: {op_counts}")

    script_path = Path(__file__).resolve()
    result = {
        "schema_version": 1,
        "variant": args.variant,
        "logical_shape": {"projections": 2, "M": args.m, "K": args.k, "N": args.n},
        "graph_contract": {
            "projection_family": "Granite K+V",
            "dynamic_activation_quantization": args.variant == "fp8_shared",
            "static_weight_prepack_excluded": args.variant == "fp8_shared",
            "activation_reuse_source": "compiler canonicalization of two independent chains",
            "output_scale_passes_per_projection": 2
            if args.variant == "fp8_shared"
            else 0,
            "compact_fp16_scales_poc": args.compact_fp16_scales,
            "direct_activation_divide_poc": args.compact_fp16_scales,
            "skip_activation_clamp_poc": (
                args.compact_fp16_scales and not args.explicit_activation_clamp
            ),
        },
        "generated_op_counts": op_counts,
        "correctness": correctness,
        "cpu_reference_ms": reference_ms,
        "compile_and_first_run_ms": compile_and_first_run_ms,
        "warmups": args.warmups,
        "repetitions": args.reps,
        "kernel_event_count": len(kernel_durations),
        "kernel_total_us": sum(kernel_durations),
        "kernel_mean_us_per_iteration": kernel_mean_us,
        "kernel_event_duration_median_us": statistics.median(kernel_durations),
        "wall_mean_us": wall_total_us / args.reps,
        "effective_pair_tflops": (4.0 * args.m * args.k * args.n)
        / (kernel_mean_us * 1.0e6),
        "effective_tflops_numerator": "2 projections * 2*M*K*N",
        "trace_summary": trace_summary,
        "trace_path": str(trace_path),
        "script": {"path": str(script_path), "sha256": sha256(script_path)},
        "environment": {
            key: os.environ.get(key)
            for key in (
                "TORCHINDUCTOR_CACHE_DIR",
                "DXP_LX_FRAC_AVAIL",
                "SENCORES",
                "SENCORELETS",
                "SPYRE_LX_PLANNER_RELAYOUT",
                "TORCH_SPYRE_FP8_LX_POC_M_SPLIT",
                "TORCH_SPYRE_FP8_LX_POC_N_SPLIT",
                "TORCH_SPYRE_FP8_LX_POC_RELEASE_QFP8MB_INPUT",
                "TORCH_SPYRE_LX_RELAYOUT_MIN_SOURCE_BYTES",
                "TORCH_SPYRE_LX_RELAYOUT_MAX_SOURCE_BYTES",
                "SENARCH",
                "SENTARGET",
            )
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
