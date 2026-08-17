#!/usr/bin/env python3
"""Measure the exact PR293 Ec32 expert path as one logical E128 FFN call.

This benchmark intentionally excludes router-logit computation, matching the
accepted AS-GEMM kernel boundary.  A logical baseline call is Antoni and
Swagath's exact compiled ``_moe_expert_chunk`` invoked four times over offset-0
32-expert banks while threading the device accumulator between calls.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
import types

import torch
import torch.nn.functional as F
import torch_spyre
import torch_spyre.execution.async_compile as spyre_async_compile


HF_REV = "672b2fc8b5f017a08c6b43b928deb3ccd0560761"
HF_FILE_SHA = "29fd989f1ca7f4c1fdb946d9b657934edba149920b376607ab7115098dcc5412"
TORCH_REV = "65508a025f557663c5694e3596c49b814d87517a"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(source: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True
    ).strip()


def _load_pr293(path: pathlib.Path) -> types.ModuleType:
    package = types.ModuleType("hf_adapters")
    package.__path__ = []  # type: ignore[attr-defined]
    common = types.ModuleType("hf_adapters.hf_common")
    common.text_config = lambda config: config  # type: ignore[attr-defined]
    gemma = types.ModuleType("hf_adapters.hf_gemma4")
    for name in (
        "_gemma4_attention",
        "_gemma4_backbone",
        "_run_blocks_over_embeds",
        "_setup_gemma4_text_decoder",
    ):
        setattr(gemma, name, lambda *args, **kwargs: None)
    sys.modules["hf_adapters"] = package
    sys.modules["hf_adapters.hf_common"] = common
    sys.modules["hf_adapters.hf_gemma4"] = gemma
    name = "hf_adapters.hf_gemma4_moe_pr293_timing"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _balanced_weights(tokens: int, experts: int, top_k: int) -> torch.Tensor:
    ids = torch.empty((tokens, top_k), dtype=torch.int64)
    for token in range(tokens):
        base = (token * top_k) % experts
        ids[token] = torch.arange(base, base + top_k) % experts
    weights = torch.zeros((tokens, experts), dtype=torch.float16)
    weights.scatter_(1, ids, 1.0 / top_k)
    return weights


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    delta = actual_f - reference_f
    return {
        "max_abs": float(delta.abs().max()),
        "rel_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference_f)),
        "cosine": float(F.cosine_similarity(actual_f.reshape(1, -1), reference_f.reshape(1, -1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--torch-source", required=True)
    parser.add_argument("--hf-source", required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--singles", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-iters", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    T, H, Fdim, E, Ec, K = 512, 2816, 704, 128, 32, 8
    output = pathlib.Path(args.output)
    cache = pathlib.Path(args.cache)
    torch_source = pathlib.Path(args.torch_source)
    hf_source = pathlib.Path(args.hf_source)
    hf_file = hf_source / "hf_adapters" / "hf_gemma4_moe.py"
    if output.exists():
        raise SystemExit("output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if any(cache.glob("inductor-spyre/*/bundle.mlir")):
        raise SystemExit("cache already contains a compiled bundle")
    if _git(torch_source, "rev-parse", "HEAD") != TORCH_REV:
        raise AssertionError("unexpected Torch-Spyre revision")
    if _git(hf_source, "rev-parse", "HEAD") != HF_REV:
        raise AssertionError("unexpected hf-adapters revision")
    if _sha256(hf_file) != HF_FILE_SHA:
        raise AssertionError("unexpected hf_gemma4_moe.py")

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    torch_spyre._autoload()
    spyre_async_compile._COMPILE_TIMEOUT_S = 600.0
    module = _load_pr293(hf_file)
    torch._dynamo.reset()
    compiled = torch.compile(module._moe_expert_chunk, dynamic=False)

    generator = torch.Generator().manual_seed(17)
    x_cpu = torch.randn((T, H), dtype=torch.float16, generator=generator)
    router_cpu = _balanced_weights(T, E, K)
    sample_rows = [0, 1, 127, 255, 511]
    reference = torch.zeros((len(sample_rows), H), dtype=torch.float32)
    x_sample = x_cpu[sample_rows].float()
    x = x_cpu.to("spyre")
    router = router_cpu.to("spyre")
    zero = torch.zeros((T, H), dtype=torch.float16, device="spyre")
    chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for lo in range(0, E, Ec):
        gate_cpu = torch.randn((Ec, H, Fdim), dtype=torch.float16, generator=generator)
        gate_cpu.mul_(1.0 / math.sqrt(H))
        up_cpu = torch.randn((Ec, H, Fdim), dtype=torch.float16, generator=generator)
        up_cpu.mul_(1.0 / math.sqrt(H))
        down_cpu = torch.randn((Ec, Fdim, H), dtype=torch.float16, generator=generator)
        down_cpu.mul_(1.0 / math.sqrt(Fdim))
        onehot_cpu = torch.eye(E, dtype=torch.float16)[lo : lo + Ec]
        for local in range(Ec):
            hidden = F.gelu(
                x_sample @ gate_cpu[local].float(), approximate="tanh"
            ) * (x_sample @ up_cpu[local].float())
            projected = hidden @ down_cpu[local].float()
            alpha = router_cpu[sample_rows, lo + local].float().unsqueeze(1)
            reference.add_(projected * alpha)
        chunks.append(
            (
                gate_cpu.to("spyre"),
                up_cpu.to("spyre"),
                down_cpu.to("spyre"),
                onehot_cpu.to("spyre"),
            )
        )

    def logical_call() -> torch.Tensor:
        accumulator = zero
        for gate, up, down, onehot in chunks:
            accumulator = compiled(x, router, accumulator, gate, up, down, onehot)
        return accumulator

    first = logical_call()
    torch.spyre.synchronize()
    actual = first.cpu()[sample_rows]
    before = _metrics(actual, reference)
    if not torch.allclose(actual.float(), reference, rtol=0.03, atol=0.05):
        raise AssertionError(f"pre-timing correctness failed: {before}")

    for _ in range(args.warmups):
        logical_call()
        torch.spyre.synchronize()

    singles: list[float] = []
    blocks: list[float] = []
    for _round in range(args.rounds):
        for _sample in range(args.singles):
            started = time.perf_counter_ns()
            logical_call()
            torch.spyre.synchronize()
            singles.append((time.perf_counter_ns() - started) / 1e6)
        for _sample in range(args.blocks):
            started = time.perf_counter_ns()
            for _ in range(args.block_iters):
                logical_call()
            torch.spyre.synchronize()
            blocks.append(
                (time.perf_counter_ns() - started) / 1e6 / args.block_iters
            )

    final = logical_call()
    torch.spyre.synchronize()
    after_actual = final.cpu()[sample_rows]
    after = _metrics(after_actual, reference)
    if not torch.allclose(after_actual.float(), reference, rtol=0.03, atol=0.05):
        raise AssertionError(f"post-timing correctness failed: {after}")

    bundles = sorted(cache.glob("inductor-spyre/*/bundle.mlir"))
    if len(bundles) != 1:
        raise AssertionError(f"expected one compiled chunk bundle, found {len(bundles)}")
    sdsc_files = sorted(bundles[0].parent.glob("sdsc_*.json"))
    batchmatmuls = sum(
        any(key.endswith("_batchmatmul") for key in json.loads(path.read_text()))
        for path in sdsc_files
    )
    if batchmatmuls != 96:
        raise AssertionError("expected 96 static BMM calls in the Ec32 program")

    result = {
        "status": "passed",
        "scope": "exact PR293 expert path; router logits excluded; one logical call is four Ec32 invocations",
        "shape": {"T": T, "H": H, "F": Fdim, "E": E, "Ec": Ec, "K": K, "cores": 32},
        "source": {
            "hf_revision": HF_REV,
            "hf_file_sha256": HF_FILE_SHA,
            "torch_spyre_revision": TORCH_REV,
            "pod": os.environ.get("HOSTNAME"),
            "pci": os.environ.get("AIU_WORLD_RANK_0"),
        },
        "structure": {
            "compiled_programs": 1,
            "runtime_invocations_per_logical_call": 4,
            "static_bmms_per_program": 96,
            "bundle_sha256": _sha256(bundles[0]),
            "bundle_path": str(bundles[0]),
        },
        "correctness": {"before": before, "after": after, "sample_rows": sample_rows},
        "protocol": {
            "warmups": args.warmups,
            "singles_per_round": args.singles,
            "blocks_per_round": args.blocks,
            "block_iters": args.block_iters,
            "rounds": args.rounds,
        },
        "timing": {
            "single_median_ms": statistics.median(singles),
            "block_median_ms": statistics.median(blocks),
            "single_samples_ms": singles,
            "block_samples_ms": blocks,
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
