#!/usr/bin/env python3
"""Measure stage slopes inside the activation-stationary expert loop.

AIUPTI exposes the compiled program as one device job, so it cannot timestamp
the twelve internal SDSCs independently.  This probe decomposes the same
compiler mechanism with nested stage controls and an expert-count sweep.

Every mode keeps one input in LX, streams one or two expert-weight operands,
uses one flat expert loop, accumulates in LX, and drains one final HBM output.
No tensor payload is retained; the result contains source, structure,
correctness metrics, and amortized block timings only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import time

import torch
from torch._inductor.utils import run_and_get_code

import torch_spyre
from torch_spyre._inductor import config, spyre_hint
import torch_spyre._inductor.wsr.propagate_named_dims as pnd

import dasx_shared_lhs_c32_schedule_probe as base


T = 512
H = 2816
F = 704
CORES = 32
E = 2


def _full_variant(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    alpha: torch.Tensor,
    *,
    gelu: bool,
    hidden_mul: bool,
    route_mul: bool,
) -> torch.Tensor:
    with spyre_hint(num_tiles_per_dim={"E": E}):
        gate = torch.ops.spyre.activation_stationary_shared_lhs_mm_prepacked(x, gate_w)
        up = torch.ops.spyre.activation_stationary_shared_lhs_mm_prepacked(x, up_w)
        activated = torch.nn.functional.gelu(gate, approximate="tanh") if gelu else gate
        hidden = activated * up if hidden_mul else activated + up
        down = torch.ops.spyre.activation_stationary_expert_mm_prepacked(hidden, down_w)
        routed = down * alpha if route_mul else down + alpha
        return routed.sum(dim=0)


def full(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return _full_variant(
        x, gate_w, up_w, down_w, alpha, gelu=True, hidden_mul=True, route_mul=True
    )


def no_gelu(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return _full_variant(
        x, gate_w, up_w, down_w, alpha, gelu=False, hidden_mul=True, route_mul=True
    )


def hidden_add(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return _full_variant(
        x, gate_w, up_w, down_w, alpha, gelu=True, hidden_mul=False, route_mul=True
    )


def route_add(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return _full_variant(
        x, gate_w, up_w, down_w, alpha, gelu=True, hidden_mul=True, route_mul=False
    )


MODE_SPECS = {
    "full": {"fn": full, "bmms": 3, "alpha": True, "out": H},
    "no_gelu": {"fn": no_gelu, "bmms": 3, "alpha": True, "out": H},
    "hidden_add": {"fn": hidden_add, "bmms": 3, "alpha": True, "out": H},
    "route_add": {"fn": route_add, "bmms": 3, "alpha": True, "out": H},
}


def _ops_from_source(source: str) -> tuple[list[ast.Call], list[ast.Call], ast.Call]:
    tree = ast.parse(source)
    sdsc_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sdsc"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "async_compile"
    ]
    assert len(sdsc_calls) == 1, f"expected one source bundle, got {len(sdsc_calls)}"
    specs = sdsc_calls[0].args[1]
    assert isinstance(specs, ast.List)
    top_specs = specs.elts
    loops = [
        node
        for node in top_specs
        if isinstance(node, ast.Call) and base._call_name(node) == "LoopSpec"
    ]
    assert len(loops) == 1, f"expected one expert loop, got {len(loops)}"
    loop = loops[0]
    body = base._keyword(loop, "body")
    assert isinstance(body, ast.List)
    top_ops = [
        node
        for node in top_specs
        if isinstance(node, ast.Call) and base._call_name(node) == "OpSpec"
    ]
    loop_ops = [
        node
        for node in body.elts
        if isinstance(node, ast.Call) and base._call_name(node) == "OpSpec"
    ]
    return top_ops, loop_ops, loop


def _analyze(source: str, mode: str) -> dict:
    top_ops, loop_ops, loop = _ops_from_source(source)
    assert int(base._sympify_call(base._keyword(loop, "count"))) == E
    all_ops = [*top_ops, *loop_ops]

    for op in all_ops:
        name = base._op_name(op).lower()
        chain = base._debug_chain(op)
        assert "restickify" not in name
        assert not any("restickify" in item.lower() for item in chain)
        assert all(
            "hbm_pool" not in tensor["allocation"] for tensor in base._tensor_args(op)
        )

    x_copies = [
        op
        for op in top_ops
        if any(
            item.startswith("coarse_tile_read_copy_0_arg0")
            for item in base._debug_chain(op)
        )
    ]
    assert len(x_copies) == 1
    x_args = base._tensor_args(x_copies[0])
    assert len(x_args) == 2
    assert x_args[0]["allocation"].keys() == {"hbm"}
    assert x_args[1]["allocation"].keys() == {"lx"}
    assert not x_args[0]["has_tile_advance"]

    bmms = [op for op in loop_ops if base._op_name(op) == "batchmatmul"]
    assert len(bmms) == MODE_SPECS[mode]["bmms"]
    streamed_weights = []
    for op in bmms:
        hbm_inputs = [
            tensor
            for tensor in base._tensor_args(op)
            if tensor["is_input"] and tensor["allocation"].keys() == {"hbm"}
        ]
        assert len(hbm_inputs) == 1
        assert hbm_inputs[0]["has_tile_advance"]
        streamed_weights.append(hbm_inputs[0]["arg_index"])

    hbm_writes = [
        (op, tensor)
        for op in all_ops
        for tensor in base._tensor_args(op)
        if not tensor["is_input"] and tensor["allocation"].keys() == {"hbm"}
    ]
    assert len(hbm_writes) == 1
    assert any(
        "coarse_tile_reduce_copy" in item
        for item in base._debug_chain(hbm_writes[0][0])
    )

    fills = [
        op
        for op in top_ops
        if any("coarse_tile_fill" in item for item in base._debug_chain(op))
    ]
    combines = [
        op
        for op in loop_ops
        if any("coarse_tile_combine" in item for item in base._debug_chain(op))
    ]
    assert len(fills) == len(combines) == 1
    assert all(
        tensor["allocation"].keys() == {"lx"}
        for tensor in base._tensor_args(combines[0])
    )

    alpha_copies = [
        op
        for op in loop_ops
        if base._op_name(op) == "identity"
        and any(
            tensor["is_input"]
            and tensor["allocation"].keys() == {"hbm"}
            and tensor["has_tile_advance"]
            for tensor in base._tensor_args(op)
        )
    ]
    assert len(alpha_copies) == int(MODE_SPECS[mode]["alpha"])

    return {
        "one_source_bundle": True,
        "one_flat_expert_loop": True,
        "one_input_hbm_to_lx_preheader": True,
        "bmm_count": len(bmms),
        "streamed_weight_arg_indices": streamed_weights,
        "runtime_alpha_copy": bool(alpha_copies),
        "all_internal_compute_lx": True,
        "hbm_pool_allocations": 0,
        "restickify_ops": 0,
        "final_hbm_outputs": 1,
    }


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    diff = actual - reference
    return {
        "max_abs": float(diff.abs().max()),
        "rel_l2": float(
            torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference)
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual.flatten(), reference.flatten(), dim=0
            )
        ),
    }


def main() -> None:
    global E
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_SPECS), required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    if args.experts < 2:
        raise ValueError("--experts must be at least two")
    E = args.experts
    mode = args.mode

    output_dir = pathlib.Path(args.output_dir)
    cache_dir = pathlib.Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch_spyre._autoload()
    pnd.reset()
    for name, size in {"E": E, "T": T, "H": H, "F": F, "K": 1}.items():
        pnd.declare_tensor_dim(name, size)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    # Hidden-add has a much larger numeric range than the real GELU-times-up
    # graph.  Use the reduced gate's stable scale only for that control; device
    # work, layouts, addresses, and control flow are unchanged by values.
    scale = 0.01 if mode == "hidden_add" else 0.1
    x = torch.randn((T, H), dtype=torch.float16, generator=generator) * scale
    gate_w = torch.randn((H, E, F), dtype=torch.float16, generator=generator) * scale
    up_w = torch.randn((H, E, F), dtype=torch.float16, generator=generator) * scale
    down_w = torch.randn((F, E, H), dtype=torch.float16, generator=generator) * scale
    alpha = torch.rand((E, T, 1), dtype=torch.float16, generator=generator) * 0.8 + 0.1

    cpu_inputs = [x, gate_w, up_w, down_w, alpha]
    dim_names = [
        ["T", "H"],
        ["H", "E", "F"],
        ["H", "E", "F"],
        ["F", "E", "H"],
        ["E", "T", "K"],
    ]
    gate = torch.einsum("th,hef->etf", x.float(), gate_w.float())
    up = torch.einsum("th,hef->etf", x.float(), up_w.float())
    activated = (
        gate
        if mode == "no_gelu"
        else torch.nn.functional.gelu(gate, approximate="tanh")
    )
    hidden = activated + up if mode == "hidden_add" else activated * up
    down = torch.einsum("etf,feh->eth", hidden, down_w.float())
    routed = down + alpha.float() if mode == "route_add" else down * alpha.float()
    reference = routed.sum(dim=0)

    device_inputs = [tensor.to("spyre") for tensor in cpu_inputs]
    for tensor, names in zip(device_inputs, dim_names):
        pnd.name_tensor_dims(tensor, names)

    compiled = torch.compile(MODE_SPECS[mode]["fn"], dynamic=False, fullgraph=True)
    with config.patch(
        {
            "sencores": CORES,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    ):
        actual, source_codes = run_and_get_code(compiled, *device_inputs)
    assert len(source_codes) == 1
    source = source_codes[0]
    source_path = output_dir / "generated_module.py"
    source_path.write_text(source)
    structure = _analyze(source, mode)

    actual_cpu = actual.cpu().float()
    numeric = _metrics(actual_cpu, reference)
    torch.testing.assert_close(actual_cpu, reference, rtol=0.03, atol=0.05)
    assert numeric["rel_l2"] <= 0.03 and numeric["cosine"] >= 0.999

    for _ in range(5):
        compiled(*device_inputs)
    torch.spyre.synchronize()
    samples = []
    for round_index in range(3):
        for block_index in range(20):
            torch.spyre.synchronize()
            start = time.perf_counter_ns()
            for _ in range(5):
                compiled(*device_inputs)
            torch.spyre.synchronize()
            samples.append(
                {
                    "round": round_index,
                    "block": block_index,
                    "calls": 5,
                    "per_call_ms": (time.perf_counter_ns() - start) / 5_000_000,
                }
            )

    bundle_dirs = sorted({path.parent for path in cache_dir.rglob("bundle.mlir")})
    assert len(bundle_dirs) == 1
    bundle = bundle_dirs[0] / "bundle.mlir"
    result = {
        "mode": mode,
        "shape": {"E": E, "T": T, "H": H, "F": F, "cores": CORES},
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "structure": structure,
        "correctness": numeric,
        "protocol": {"warmups": 5, "rounds": 3, "blocks": 20, "block_iters": 5},
        "samples": samples,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
