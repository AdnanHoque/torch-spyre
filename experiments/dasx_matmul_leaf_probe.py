#!/usr/bin/env python3
"""Measure one projection with the D-AS-X M32 row ownership.

The standalone leaf includes HBM reads for both operands and an HBM output, so
it is a conservative proxy rather than the exact in-loop BMM cost.  The real
expert loop keeps the activation and result in LX and streams only the expert
weight.  Results from this probe must therefore be reported with the extra
HBM byte delta, never as an exact per-SDSC timestamp.
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
CORES = 32
SHAPES = {
    "gate": (2816, 704),
    "down": (704, 2816),
}


def projection(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    with spyre_hint(work_div={"T": CORES}):
        return torch.mm(x, weight)


def _analyze(source: str, kind: str) -> dict:
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
    assert len(sdsc_calls) == 1
    specs = sdsc_calls[0].args[1]
    assert isinstance(specs, ast.List)
    ops = [
        node
        for node in specs.elts
        if isinstance(node, ast.Call) and base._call_name(node) == "OpSpec"
    ]
    assert len(ops) == 1
    op = ops[0]
    assert base._op_name(op) == "batchmatmul"
    tensors = base._tensor_args(op)
    assert len(tensors) == 3
    assert all(tensor["allocation"].keys() == {"hbm"} for tensor in tensors)
    splits = sorted(base._iteration_splits(op).values())
    assert splits == [1, 1, 32], splits

    k, n = SHAPES[kind]
    input_bytes = T * k * 2
    weight_bytes = k * n * 2
    output_bytes = T * n * 2
    return {
        "one_bmm": True,
        "work_division": "M32,N1,K1",
        "allocations": "HBM inputs and HBM output",
        "logical_bytes": {
            "activation_read": input_bytes,
            "weight_read": weight_bytes,
            "output_write": output_bytes,
            "extra_vs_dasx_loop": input_bytes + output_bytes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=sorted(SHAPES), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    cache_dir = pathlib.Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=True)

    k, n = SHAPES[args.kind]
    torch_spyre._autoload()
    pnd.reset()
    for name, size in {"T": T, "K": k, "N": n}.items():
        pnd.declare_tensor_dim(name, size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    x = torch.randn((T, k), dtype=torch.float16, generator=generator) * 0.1
    weight = torch.randn((k, n), dtype=torch.float16, generator=generator) * 0.1
    x_device = x.to("spyre")
    weight_device = weight.to("spyre")
    pnd.name_tensor_dims(x_device, ["T", "K"])
    pnd.name_tensor_dims(weight_device, ["K", "N"])

    compiled = torch.compile(projection, dynamic=False, fullgraph=True)
    with config.patch({"sencores": CORES, "lx_planning": True}):
        actual, source_codes = run_and_get_code(compiled, x_device, weight_device)
    assert len(source_codes) == 1
    source = source_codes[0]
    (output_dir / "generated_module.py").write_text(source)
    structure = _analyze(source, args.kind)

    reference = torch.mm(x.float(), weight.float())
    actual_cpu = actual.cpu().float()
    diff = actual_cpu - reference
    correctness = {
        "max_abs": float(diff.abs().max()),
        "rel_l2": float(
            torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference)
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_cpu.flatten(), reference.flatten(), dim=0
            )
        ),
    }
    torch.testing.assert_close(actual_cpu, reference, rtol=0.03, atol=0.05)
    assert correctness["rel_l2"] <= 0.03 and correctness["cosine"] >= 0.999

    for _ in range(5):
        compiled(x_device, weight_device)
    torch.spyre.synchronize()
    samples = []
    for round_index in range(3):
        for block_index in range(30):
            torch.spyre.synchronize()
            start = time.perf_counter_ns()
            for _ in range(5):
                compiled(x_device, weight_device)
            torch.spyre.synchronize()
            samples.append(
                {
                    "round": round_index,
                    "block": block_index,
                    "calls": 5,
                    "per_call_ms": (time.perf_counter_ns() - start) / 5_000_000,
                }
            )

    bundles = list(cache_dir.rglob("bundle.mlir"))
    assert len(bundles) == 1
    result = {
        "kind": args.kind,
        "shape": {"M": T, "K": k, "N": n, "cores": CORES},
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "bundle_sha256": hashlib.sha256(bundles[0].read_bytes()).hexdigest(),
        "structure": structure,
        "correctness": correctness,
        "protocol": {"warmups": 5, "rounds": 3, "blocks": 30, "block_iters": 5},
        "samples": samples,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
