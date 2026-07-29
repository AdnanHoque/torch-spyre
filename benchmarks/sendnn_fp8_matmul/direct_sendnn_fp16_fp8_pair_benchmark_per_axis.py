#!/usr/bin/env python3
"""Paired direct-SenDNN FP16/FP8 BatchMatMul correctness and wall benchmark.

The FP8 graph deliberately uses an FP16 host activation followed by an
on-device Identity whose declared output is FP8.  DeepTools lowers that
producer plus BatchScaledMatmul to Qfp8 -> BatchMatMulV2 -> two scale-recovery
stages.  This is not a host-prequantized-FP8 benchmark.

The scaled-matmul inputs use Granite-compatible scale shapes: one FP32 scale
per activation row and one FP32 scale per output channel.  Their values are
one so that correctness can be checked against an FP8-cast CPU reference.
Scale derivation is upstream of BatchScaledMatmul and is not part of this
standalone graph.

Timing is host wall time around synchronous sendnn.Predict for execution
supernode 2.  Compilation, static-model preparation, device initialization,
input construction, and correctness/reference calculation are excluded.
Activation HostPrep and FP16 result transfer back to the host are included.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import sendnn
import torch


M = int(os.getenv("FP8_BENCH_M", "512"))
K = int(os.getenv("FP8_BENCH_K", "4096"))
N = int(os.getenv("FP8_BENCH_N", "1024"))
if min(M, K, N) < 1:
    raise ValueError(f"matrix dimensions must be positive: M={M}, K={K}, N={N}")
ACTIVATION_SHAPE = [1, 1, M, K]
WEIGHT_SHAPE = [1, 1, K, N]
OUTPUT_SHAPE = [1, 1, M, N]
ACTIVATION_SCALE_SHAPE = [1, 1, M, 1]
WEIGHT_SCALE_SHAPE = [1, 1, 1, N]
DATA_FORMULA = {
    "activation": "(((row * 13 + k * 7) % 37) - 18) / 19, rounded to FP16",
    "weight": "(((k * 11 + col * 5) % 41) - 20) / 21, rounded to FP16",
}


def tensor_info(datatype, shape):
    return sendnn.TensorInfo(
        datatype, sendnn.TensorShape(shape), sendnn.TensorLayout.NCHW
    )


def torch_dtype(datatype):
    mapping = {
        sendnn.sen_datatype_enum.boolean: torch.bool,
        sendnn.sen_datatype_enum.float8: torch.float8_e4m3fn,
        sendnn.sen_datatype_enum.float16: torch.float16,
        sendnn.sen_datatype_enum.float32: torch.float32,
        sendnn.sen_datatype_enum.float64: torch.float64,
        sendnn.sen_datatype_enum.int8: torch.int8,
        sendnn.sen_datatype_enum.int16: torch.int16,
        sendnn.sen_datatype_enum.int32: torch.int32,
        sendnn.sen_datatype_enum.int64: torch.int64,
        sendnn.sen_datatype_enum.uint8: torch.uint8,
        sendnn.sen_datatype_enum.uint16: torch.uint16,
        sendnn.sen_datatype_enum.uint32: torch.uint32,
        sendnn.sen_datatype_enum.uint64: torch.uint64,
    }
    return mapping[datatype]


def status_text(status):
    return str(status)


def require_ok(status, label):
    if not status.IsOk():
        raise RuntimeError(f"{label}: {status}")


def allocate_outputs(graph_loader, supernode):
    torch_outputs = []
    sendnn_outputs = []
    for info in graph_loader.GetOutputs(supernode):
        output = torch.empty(info.Shape().DimsInt(), dtype=torch_dtype(info.DataType()))
        torch_outputs.append(output)
        sendnn_outputs.append(sendnn.AsTensor(info, output.data_ptr()))
    return torch_outputs, sendnn_outputs


def as_sendnn_inputs(graph_loader, supernode, values_by_name):
    inputs = []
    for index, info in enumerate(graph_loader.GetInputs(supernode)):
        name = graph_loader.GetInputName(supernode, index)
        value = values_by_name[name]
        inputs.append(sendnn.AsConstTensor(info, value.data_ptr()))
    return inputs


def make_data(mode):
    row = torch.arange(M, dtype=torch.int32).reshape(M, 1)
    reduction = torch.arange(K, dtype=torch.int32).reshape(1, K)
    activation = (
        (torch.remainder(row * 13 + reduction * 7, 37).float() - 18.0) / 19.0
    ).to(torch.float16)
    activation = activation.reshape(ACTIVATION_SHAPE).contiguous()
    del row, reduction

    reduction = torch.arange(K, dtype=torch.int32).reshape(K, 1)
    column = torch.arange(N, dtype=torch.int32).reshape(1, N)
    weight_fp16 = (
        (torch.remainder(reduction * 11 + column * 5, 41).float() - 20.0)
        / 21.0
    ).to(torch.float16)
    weight_fp16 = weight_fp16.reshape(WEIGHT_SHAPE).contiguous()
    del reduction, column

    weight = (
        weight_fp16.to(torch.float8_e4m3fn).contiguous()
        if mode == "fp8"
        else weight_fp16
    )
    return activation, weight


def tensor_digest(tensor):
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_summary(tensor):
    values = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": values.min().item(),
        "max": values.max().item(),
        "mean": values.mean().item(),
        "std": values.std().item(),
        "sha256": tensor_digest(tensor),
        "first_values": values.flatten()[:8].tolist(),
    }


def build_and_compile(mode):
    fp8 = sendnn.sen_datatype_enum.float8
    fp32 = sendnn.sen_datatype_enum.float32
    fp16 = sendnn.sen_datatype_enum.float16

    builder = sendnn.GraphBuilder()
    activation_node = builder.PrimaryInput(
        "activation", tensor_info(fp16, ACTIVATION_SHAPE)
    )
    weight_datatype = fp8 if mode == "fp8" else fp16
    weight_node = builder.ModelInput(
        "weight", tensor_info(weight_datatype, WEIGHT_SHAPE)
    )

    if mode == "fp8":
        fp8_activation_node = builder.Identity(
            "activation_to_fp8",
            tensor_info(fp8, ACTIVATION_SHAPE),
            activation_node,
        )
        activation_scale_node = builder.ModelInput(
            "activation_scale", tensor_info(fp32, ACTIVATION_SCALE_SHAPE)
        )
        weight_scale_node = builder.ModelInput(
            "weight_scale", tensor_info(fp32, WEIGHT_SCALE_SHAPE)
        )
        output_node = builder.BatchScaledMatmul(
            "fp8_scaled_bmm",
            tensor_info(fp16, OUTPUT_SHAPE),
            fp8_activation_node,
            weight_node,
            activation_scale_node,
            weight_scale_node,
            True,
        )
    else:
        output_node = builder.BatchMatMul(
            "fp16_bmm",
            tensor_info(fp16, OUTPUT_SHAPE),
            activation_node,
            weight_node,
            False,
            False,
        )
    builder.PrimaryOutput("output", output_node)

    graph = sendnn.Graph()
    finalize_status = builder.Finalize(graph)
    require_ok(finalize_status, "Finalize")

    loader = sendnn.GraphLoader("SEN:0")
    load_status = loader.LoadGraph(graph, False)
    require_ok(load_status, "LoadGraph")
    compile_status = loader.CompileGraph()
    require_ok(compile_status, "CompileGraph")
    parse_status = loader.ParseGraph()
    require_ok(parse_status, "ParseGraph")

    contracts = []
    for supernode in (0, 1, 2):
        contracts.append(
            {
                "supernode": supernode,
                "inputs": [
                    loader.GetInputName(supernode, i)
                    for i in range(len(loader.GetInputs(supernode)))
                ],
                "outputs": len(loader.GetOutputs(supernode)),
            }
        )
    expected = [
        {
            "supernode": 0,
            "inputs": (
                ["weight", "activation_scale", "weight_scale"]
                if mode == "fp8"
                else ["weight"]
            ),
            "outputs": 1,
        },
        {"supernode": 1, "inputs": ["PrepareModel"], "outputs": 0},
        {"supernode": 2, "inputs": ["activation"], "outputs": 1},
    ]
    if contracts != expected:
        raise RuntimeError(f"unexpected supernode contracts: {contracts}")

    statuses = {
        "finalize": status_text(finalize_status),
        "load": status_text(load_status),
        "compile": status_text(compile_status),
        "parse": status_text(parse_status),
    }
    return loader, contracts, statuses


def prepare_and_initialize(loader, mode, weight):
    static_values = {"weight": weight}
    if mode == "fp8":
        static_values["activation_scale"] = torch.ones(
            ACTIVATION_SCALE_SHAPE, dtype=torch.float32
        )
        static_values["weight_scale"] = torch.ones(
            WEIGHT_SCALE_SHAPE, dtype=torch.float32
        )

    prepare_outputs, prepare_sendnn_outputs = allocate_outputs(loader, 0)
    prepare_inputs = as_sendnn_inputs(loader, 0, static_values)
    prepare_status = sendnn.Predict(
        loader, prepare_sendnn_outputs, prepare_inputs, 0
    )
    require_ok(prepare_status, "PrepareModel Predict")

    if len(prepare_outputs) != 1:
        raise RuntimeError(f"expected one PrepareModel token: {len(prepare_outputs)}")
    init_outputs, init_sendnn_outputs = allocate_outputs(loader, 1)
    init_inputs = as_sendnn_inputs(
        loader, 1, {"PrepareModel": prepare_outputs[0]}
    )
    init_status = sendnn.Predict(loader, init_sendnn_outputs, init_inputs, 1)
    require_ok(init_status, "DeviceInit Predict")
    if init_outputs:
        raise RuntimeError(f"expected zero DeviceInit outputs: {len(init_outputs)}")

    return {
        "prepare": status_text(prepare_status),
        "device_init": status_text(init_status),
    }


def make_execution_io(loader, activation):
    outputs, sendnn_outputs = allocate_outputs(loader, 2)
    if len(outputs) != 1:
        raise RuntimeError(f"expected one execution output: {len(outputs)}")
    inputs = as_sendnn_inputs(loader, 2, {"activation": activation})
    return outputs[0], sendnn_outputs, inputs


def execute_once(loader, sendnn_outputs, inputs):
    status = sendnn.Predict(loader, sendnn_outputs, inputs, 2)
    require_ok(status, "Execute Predict")
    return status


def cpu_reference(mode, activation, weight):
    activation_ref = activation[0, 0]
    if mode == "fp8":
        activation_ref = activation_ref.to(torch.float8_e4m3fn)
    start_ns = time.perf_counter_ns()
    reference = torch.matmul(activation_ref.float(), weight[0, 0].float())
    reference = reference.to(torch.float16).reshape(OUTPUT_SHAPE)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    return reference, elapsed_ms


def correctness_metrics(mode, actual, reference):
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    absolute_error = (actual_fp32 - reference_fp32).abs()
    reference_norm = torch.linalg.vector_norm(reference_fp32)
    error_norm = torch.linalg.vector_norm(actual_fp32 - reference_fp32)
    relative_l2 = (
        error_norm / reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
    ).item()
    if mode == "fp8":
        rtol, atol, relative_l2_limit = 0.08, 0.5, 0.08
    else:
        # The K=12800 Granite down projection passes the elementwise
        # tolerance but accumulates about 0.04 relative L2 error. Keep the
        # elementwise gate unchanged and allow that expected aggregate error.
        rtol, atol, relative_l2_limit = 0.02, 0.25, 0.06
    allclose = torch.allclose(actual_fp32, reference_fp32, rtol=rtol, atol=atol)
    passed = bool(allclose and math.isfinite(relative_l2) and relative_l2 <= relative_l2_limit)
    return {
        "passed": passed,
        "rtol": rtol,
        "atol": atol,
        "relative_l2_limit": relative_l2_limit,
        "allclose": bool(allclose),
        "max_abs_error": absolute_error.max().item(),
        "mean_abs_error": absolute_error.mean().item(),
        "rmse": torch.sqrt(torch.mean(absolute_error.square())).item(),
        "relative_l2_error": relative_l2,
        "actual_min": actual_fp32.min().item(),
        "actual_max": actual_fp32.max().item(),
        "reference_min": reference_fp32.min().item(),
        "reference_max": reference_fp32.max().item(),
    }


def percentile(sorted_values, quantile):
    if not sorted_values:
        raise ValueError("empty values")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def timing_metrics(loader, sendnn_outputs, inputs, warmups, iterations):
    for _ in range(warmups):
        execute_once(loader, sendnn_outputs, inputs)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    durations_ms = []
    try:
        for _ in range(iterations):
            start_ns = time.perf_counter_ns()
            status = sendnn.Predict(loader, sendnn_outputs, inputs, 2)
            end_ns = time.perf_counter_ns()
            require_ok(status, "timed Execute Predict")
            durations_ms.append((end_ns - start_ns) / 1_000_000.0)
    finally:
        if gc_was_enabled:
            gc.enable()

    ordered = sorted(durations_ms)
    mean_ms = statistics.fmean(durations_ms)
    stdev_ms = statistics.stdev(durations_ms) if len(durations_ms) > 1 else 0.0
    return {
        "warmups": warmups,
        "iterations": iterations,
        "durations_ms": durations_ms,
        "min_ms": ordered[0],
        "p10_ms": percentile(ordered, 0.10),
        "median_ms": statistics.median(ordered),
        "mean_ms": mean_ms,
        "p90_ms": percentile(ordered, 0.90),
        "max_ms": ordered[-1],
        "stdev_ms": stdev_ms,
        "coefficient_of_variation": stdev_ms / mean_ms if mean_ms else None,
        "clock": "time.perf_counter_ns",
        "timed_call": "synchronous sendnn.Predict(loader, outputs, inputs, supernode=2)",
        "includes": [
            "Python-to-SenDNN call overhead",
            "FP16 activation HostPrep and host-to-device path",
            "device execution",
            "FP16 output device-to-host path and HostPrep",
            "completion wait",
        ],
        "excludes": [
            "graph construction and compilation",
            "static weight/scales PrepareModel",
            "DeviceInit",
            "input generation",
            "CPU reference and correctness metrics",
        ],
    }


def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def software_versions():
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_sendnn_distribution": distribution_version("torch_sendnn"),
        "sendnn_distribution": distribution_version("sendnn"),
        "sendnn_module": sendnn.__file__,
        "platform": platform.platform(),
    }


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fp16", "fp8"), required=True)
    parser.add_argument(
        "--phase", choices=("correctness", "timing"), required=True
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.phase == "timing" and (args.warmups < 1 or args.iterations < 2):
        raise ValueError("timing requires at least 1 warmup and 2 iterations")

    print(
        "configuration",
        f"mode={args.mode}",
        f"phase={args.phase}",
        f"M={M}",
        f"K={K}",
        f"N={N}",
        flush=True,
    )
    activation, weight = make_data(args.mode)
    print("data_ready", flush=True)
    loader, contracts, compile_statuses = build_and_compile(args.mode)
    print("compile_statuses", compile_statuses, flush=True)
    lifecycle_statuses = prepare_and_initialize(loader, args.mode, weight)
    print("lifecycle_statuses", lifecycle_statuses, flush=True)
    output, sendnn_outputs, inputs = make_execution_io(loader, activation)

    result = {
        "schema_version": 1,
        "mode": args.mode,
        "phase": args.phase,
        "logical_shape": {"M": M, "K": K, "N": N},
        "physical_shapes_nchw": {
            "activation": ACTIVATION_SHAPE,
            "weight": WEIGHT_SHAPE,
            "output": OUTPUT_SHAPE,
            "activation_scale": (
                ACTIVATION_SCALE_SHAPE if args.mode == "fp8" else None
            ),
            "weight_scale": WEIGHT_SCALE_SHAPE if args.mode == "fp8" else None,
        },
        "graph_contract": (
            "FP16 PrimaryInput -> Identity(output FP8) -> BatchScaledMatmul"
            if args.mode == "fp8"
            else "FP16 PrimaryInput -> BatchMatMul"
        ),
        "host_prequantized_fp8": False,
        "use_fast_accum": True if args.mode == "fp8" else None,
        "scales": (
            {
                "activation": {
                    "shape": ACTIVATION_SCALE_SHAPE,
                    "value": 1.0,
                },
                "weight": {
                    "shape": WEIGHT_SCALE_SHAPE,
                    "value": 1.0,
                },
                "derivation_in_graph": False,
            }
            if args.mode == "fp8"
            else None
        ),
        "data_formula": DATA_FORMULA,
        "input_summary": tensor_summary(activation),
        "weight_summary": tensor_summary(weight),
        "software_versions": software_versions(),
        "compile_statuses": compile_statuses,
        "lifecycle_statuses": lifecycle_statuses,
        "supernode_contracts": contracts,
        "environment": {
            key: os.getenv(key)
            for key in (
                "DT_OPT",
                "DTCOMPILER_EXPORT_DIR",
                "DEE_DUMP_GRAPHS",
                "DXP_LX_FRAC_AVAIL",
                "LX_PLANNING",
                "TORCH_DEVICE_BACKEND_AUTOLOAD",
            )
        },
    }

    if args.phase == "correctness":
        execute_status = execute_once(loader, sendnn_outputs, inputs)
        reference, reference_ms = cpu_reference(args.mode, activation, weight)
        metrics = correctness_metrics(args.mode, output, reference)
        metrics["execute_status"] = status_text(execute_status)
        metrics["cpu_reference_wall_ms"] = reference_ms
        result["correctness"] = metrics
        print("correctness", json.dumps(metrics, sort_keys=True), flush=True)
        write_result(args.result_json, result)
        if not metrics["passed"]:
            raise SystemExit(2)
    else:
        result["timing"] = timing_metrics(
            loader,
            sendnn_outputs,
            inputs,
            args.warmups,
            args.iterations,
        )
        result["post_timing_output"] = {
            "min": output.float().min().item(),
            "max": output.float().max().item(),
            "mean": output.float().mean().item(),
        }
        print("timing", json.dumps(result["timing"], sort_keys=True), flush=True)
        write_result(args.result_json, result)

    print("result_json", args.result_json, flush=True)


if __name__ == "__main__":
    main()
