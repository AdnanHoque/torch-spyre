#!/usr/bin/env python3
"""Compare two persistent Gemma4 expert schedules on one real layer.

Layer 0's expert weights are read directly from the checkpoint.  A saved real
layer input and route payload is repeated from 64 to 512 rows, then paired with
those weights.  We run:

* common-row: every operation divides the token rows over all 32 cores;
* optimized: gate/up split the reduction, pointwise uses all token rows, and
  down/output split output features.

Each invocation runs one schedule in its own process and cache.  Its result is
checked against the same sparse FP32 CPU calculation before device timing.
The two saved outputs are compared after both invocations finish.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", "/home/adnan/hub")
os.environ.setdefault("DXP_LX_FRAC_AVAIL", "0.2")
os.environ.setdefault("TORCH_SPYRE_NATIVE_PACKER", "0")

import torch
from safetensors import safe_open

_schema_shim = torch.library.Library("spyre", "FRAGMENT")
if not hasattr(torch.ops.spyre, "all_gather_async"):
    _schema_shim.define(
        'all_gather_async(Tensor input, SymInt group_size=1, '
        'str group_name="default") -> Tensor'
    )

from torch._inductor.utils import run_and_get_code
import torch_spyre
from torch_spyre._inductor import config, spyre_hint
from torch_spyre._inductor.wsr.propagate_named_dims import (
    declare_tensor_dim,
    name_tensor_dims,
    reset,
)


torch_spyre._autoload()
import torch_spyre._C as extension

if not hasattr(extension, "SymbolicArg"):
    extension.SymbolicArg = object

MODEL = os.environ.get("MODEL", "google/gemma-4-26B-A4B-it")
SNAPSHOT = Path(
    os.environ.get(
        "MODEL_SNAPSHOT",
        "/home/adnan/hub/models--google--gemma-4-26B-A4B-it/"
        "snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "REAL_SCHEDULE_OUTPUT",
        "/tmp/gemma4-real-layer-schedule-comparison",
    )
)
REPEATS = int(os.environ.get("REAL_SCHEDULE_REPEATS", "7"))
SCHEDULE = os.environ.get("REAL_SCHEDULE", "optimized")
CAPTURE = Path(
    os.environ.get(
        "REAL_LAYER_CAPTURE",
        "/tmp/layer0-persistent-chunked-20260818-08.pt",
    )
)


def common_row(x, gate_backing, up_backing, down_backing, route_backing):
    """Use the same row ownership for the whole expert body."""
    gate = gate_backing.permute(1, 0, 2)
    up = up_backing.permute(1, 0, 2)
    down = down_backing.permute(1, 0, 2)
    route = route_backing.permute(1, 0, 2)
    experts = gate.shape[0]
    name_tensor_dims(x, ["T", "H"])
    name_tensor_dims(gate, ["E", "H", "F"])
    name_tensor_dims(up, ["E", "H", "F"])
    name_tensor_dims(down, ["E", "F", "H"])
    name_tensor_dims(route, ["E", "T", "ONE"])
    with spyre_hint(num_tiles_per_dim={"E": experts}):
        with spyre_hint(work_div={"T": 32}):
            gate_out = torch.matmul(x.unsqueeze(0), gate)
            up_out = torch.matmul(x.unsqueeze(0), up)
            hidden = torch.nn.functional.gelu(
                gate_out, approximate="tanh"
            ) * up_out
            down_out = torch.matmul(hidden, down)
            return (down_out * route).sum(dim=0)


def optimized(x, gate_backing, up_backing, down_backing, route_backing):
    """Use each matmul's preferred split and LX relayout between them."""
    gate = gate_backing.permute(1, 0, 2)
    up = up_backing.permute(1, 0, 2)
    down = down_backing.permute(1, 0, 2)
    route = route_backing.permute(1, 0, 2)
    experts = gate.shape[0]
    name_tensor_dims(x, ["T", "H"])
    name_tensor_dims(gate, ["E", "H", "F"])
    name_tensor_dims(up, ["E", "H", "F"])
    name_tensor_dims(down, ["E", "F", "H"])
    name_tensor_dims(route, ["E", "T", "ONE"])
    with spyre_hint(num_tiles_per_dim={"E": experts}):
        with spyre_hint(work_div={"T": 8, "H": 4}):
            gate_out = torch.matmul(x.unsqueeze(0), gate)
            up_out = torch.matmul(x.unsqueeze(0), up)
        with spyre_hint(work_div={"T": 32}):
            hidden = torch.nn.functional.gelu(
                gate_out, approximate="tanh"
            ) * up_out
        with spyre_hint(work_div={"T": 8, "H": 4}):
            down_out = torch.matmul(hidden, down)
            return (down_out * route).sum(dim=0)


def sparse_cpu_reference(x, route, gate_up, down):
    """Calculate only the four selected experts per token in FP32."""
    x = x.float()
    route = route.float()
    result = torch.zeros_like(x)
    intermediate = gate_up.shape[1] // 2
    for expert in range(route.shape[1]):
        rows = torch.nonzero(route[:, expert], as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        selected = x.index_select(0, rows)
        gate_value = torch.nn.functional.linear(
            selected, gate_up[expert, :intermediate].float()
        )
        up_value = torch.nn.functional.linear(
            selected, gate_up[expert, intermediate:].float()
        )
        hidden = torch.nn.functional.gelu(
            gate_value, approximate="tanh"
        ) * up_value
        contribution = torch.nn.functional.linear(
            hidden, down[expert].float()
        )
        contribution *= route[rows, expert, None]
        result.index_add_(0, rows, contribution)
    return result


def metrics(actual, expected):
    actual = actual.float()
    expected = expected.float()
    delta = actual - expected
    expected_norm = torch.linalg.vector_norm(expected)
    return {
        "rel_l2": float(torch.linalg.vector_norm(delta) / expected_norm),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0
            )
        ),
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
    }


def time_device(fn, args, repeats):
    samples = []
    for _ in range(2):
        fn(*args)
        torch.spyre.synchronize()
    for _ in range(repeats):
        started = time.perf_counter()
        fn(*args)
        torch.spyre.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def source_summary(source):
    return {
        "loop_specs": source.count("LoopSpec("),
        "batchmatmul_ops": source.count("op='batchmatmul'"),
        "hbm_pool_allocations": source.count("'hbm_pool'"),
        "hbm_restickifies": source.count("ReStickifyOpHBM"),
        "lx_relayout_markers": source.count("lx_relayout"),
        "completed_reduction_markers": source.count("completed_reduction"),
    }


def pack_weight(weight):
    """Match the adapter's physical [K,E,N] expert-weight packing."""
    return weight.permute(1, 0, 2).contiguous().to("spyre")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading layer-0 expert weights from the real checkpoint", flush=True)
    load_started = time.perf_counter()
    shard = SNAPSHOT / "model-00001-of-00002.safetensors"
    gate_up_key = "model.language_model.layers.0.experts.gate_up_proj"
    down_key = "model.language_model.layers.0.experts.down_proj"
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        # The checkpoint stores BF16, while the HF Spyre path requests FP16.
        gate_up_cpu = checkpoint.get_tensor(gate_up_key).to(torch.float16)
        down_cpu = checkpoint.get_tensor(down_key).to(torch.float16)
    load_s = time.perf_counter() - load_started
    print(f"layer-0 weights ready in {load_s:.1f}s", flush=True)

    captured = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    x_base = captured["expert_input"]
    route_base = captured["route"]
    if x_base.shape[0] != 64 or route_base.shape[:2] != (64, 128):
        raise RuntimeError(
            f"unexpected captured shapes x={x_base.shape}, route={route_base.shape}"
        )
    # The saved values are exact inputs from a real Gemma4 layer.  Repeat the
    # 64 captured rows eight times to exercise the production T=512 geometry.
    x_cpu = x_base.repeat(8, 1).contiguous()
    route_cpu = route_base.repeat(8, 1).contiguous()
    x_device = x_cpu.to("spyre")
    route = route_cpu.unsqueeze(-1).contiguous().to("spyre")
    intermediate = gate_up_cpu.shape[1] // 2
    gate_cpu = gate_up_cpu[:, :intermediate].transpose(1, 2).contiguous()
    up_cpu = gate_up_cpu[:, intermediate:].transpose(1, 2).contiguous()
    down_logical_cpu = down_cpu.transpose(1, 2).contiguous()
    gate_backing = pack_weight(gate_cpu)
    up_backing = pack_weight(up_cpu)
    down_backing = pack_weight(down_logical_cpu)

    tokens, hidden = x_device.shape
    experts, _, intermediate = gate_cpu.shape
    reset()
    for name, extent in (
        ("E", experts),
        ("T", tokens),
        ("H", hidden),
        ("F", intermediate),
        ("ONE", 1),
    ):
        declare_tensor_dim(name, extent)

    schedules = {"optimized": optimized, "common_row": common_row}
    if SCHEDULE not in schedules:
        raise ValueError(f"REAL_SCHEDULE must be one of {tuple(schedules)}")
    compiled = torch.compile(schedules[SCHEDULE], dynamic=False)
    settings = {
        "sencores": 32,
        "lx_planning": True,
        "allow_all_ops_in_lx_planning": True,
        "lx_planner_relayout": True,
        "native_layout_packer": False,
    }
    args = (x_device, gate_backing, up_backing, down_backing, route)
    with config.patch(settings):
        actual_value, generated_code = run_and_get_code(compiled, *args)
        torch.spyre.synchronize()

        actual_cpu = actual_value.cpu().float()
        print("starting sparse FP32 CPU reference", flush=True)
        reference = sparse_cpu_reference(
            x_cpu,
            route_cpu,
            gate_up_cpu,
            down_cpu,
        )
        comparison = metrics(actual_cpu, reference)
        correctness_pass = comparison["rel_l2"] < 0.03
        timing = None
        if correctness_pass:
            print("correctness passed; starting layer timing", flush=True)
            timing = time_device(compiled, args, REPEATS)
        else:
            print("correctness failed; timing skipped", flush=True)

    generated_source = "\n".join(generated_code)
    (OUTPUT_DIR / "generated.py").write_text(generated_source)
    torch.save(
        {"actual": actual_cpu, "reference": reference},
        OUTPUT_DIR / "numeric.pt",
    )
    result = {
        "model": MODEL,
        "schedule": SCHEDULE,
        "layer": 0,
        "input": {
            "capture": str(CAPTURE),
            "checkpoint_shard": str(shard),
            "captured_rows": int(x_base.shape[0]),
            "row_repetitions": 8,
        },
        "shape": {
            "E": experts,
            "T": tokens,
            "H": hidden,
            "F": intermediate,
        },
        "load_s": load_s,
        "route_nonzero": int(torch.count_nonzero(route_cpu)),
        "correctness_pass": correctness_pass,
        "comparison": comparison,
        "source": source_summary(generated_source),
        "timing": timing,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("REAL_LAYER_RESULT " + json.dumps(result), flush=True)
    print(f"saved {OUTPUT_DIR / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
