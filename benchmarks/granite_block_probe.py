#!/usr/bin/env python3
"""Probe FMS GraniteBlock compile/runtime behavior on Spyre.

This is a focused diagnostic benchmark, derived from the Granite cost-model
probe.  It intentionally sits below perf-suite so we can bisect full-block
lowering issues by submodule while still using the real FMS GraniteBlock code.

Examples:

    python benchmarks/granite_block_probe.py --part mlp_core --regime prefill
    python benchmarks/granite_block_probe.py --part mlp --regime prefill
    python benchmarks/granite_block_probe.py --part attn_core --regime decode

By default parameters are materialized with ``to_empty(device="spyre")``.  This
keeps the probe useful for compiler/performance work without paying the cost of
copying random CPU weights to the AIU.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch_spyre

if hasattr(torch_spyre, "_autoload"):
    torch_spyre._autoload()

from fms.models.granite import GraniteBlock, GraniteConfig
from fms.modules.positions import RotaryEmbedding


EMB = 4096
NHEADS = 32
KVHEADS = 8
HEAD_DIM = 128
HIDDEN = 12800
REGIME_M = {"prefill": 512, "decode": 64}


def _strip_hash(name: str) -> str:
    return re.sub(r"_[0-9a-z_]{8}$", "", name)


def _inventory(cache_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory in sorted(glob.glob(str(cache_dir / "inductor-spyre" / "*"))):
        if not os.path.isdir(directory):
            continue
        name = _strip_hash(os.path.basename(directory))
        for sdsc_path in sorted(glob.glob(os.path.join(directory, "sdsc_*.json"))):
            with open(sdsc_path, encoding="utf-8") as handle:
                data = json.load(handle)
            for key, value in data.items():
                work_slices = value.get("numWkSlicesPerDim_")
                if work_slices:
                    rows.append(
                        {
                            "name": name,
                            "key": key,
                            "work_slices": dict(work_slices),
                            "path": sdsc_path,
                        }
                    )
    return rows


def _make_block(args: argparse.Namespace) -> tuple[GraniteBlock, GraniteConfig, RotaryEmbedding]:
    config = GraniteConfig(
        src_vocab_size=49155,
        emb_dim=EMB,
        nheads=NHEADS,
        kvheads=KVHEADS,
        nlayers=1,
        hidden_grow_factor=HIDDEN / EMB,
        norm_eps=1e-5,
        p_dropout=0.0,
        fused_weights=args.fused_weights,
        attention_multiplier=0.0078125,
        residual_multiplier=0.22,
        max_expected_seq_len=args.max_seq_len,
    )
    rotary = RotaryEmbedding(dim=HEAD_DIM, max_seq_len=config.max_expected_seq_len)
    block = GraniteBlock(config, rotary)
    block = block.to(torch.float16)
    if args.device == "spyre" and args.empty_weights:
        block = block.to_empty(device=torch.device("spyre"))
    else:
        block = block.to(args.device)
    block.requires_grad_(False)
    block.eval()
    return block, config, rotary


def _selected_freqs(
    rotary: RotaryEmbedding,
    position_ids_cpu: torch.Tensor,
    max_seq_len: int,
    *,
    device: str,
) -> torch.Tensor:
    alpha = rotary.compute_freqs_cis(torch.device("cpu"), max_seq_len)
    selected = rotary.cached_freqs[None][alpha][position_ids_cpu].contiguous()
    return selected.to(device)


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _make_attention_kwargs(
    rotary: RotaryEmbedding,
    args: argparse.Namespace,
    sequence_len: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    if args.regime == "prefill":
        position_ids_cpu = torch.arange(sequence_len, dtype=torch.long).unsqueeze(0)
        max_seq_len = sequence_len + 1
        return {
            "position_ids": position_ids_cpu,
            "past_key_value_state": None,
            "use_cache": True,
            "attn_name": args.attn_name,
            "contiguous_cache": True,
            "max_seq_len": max_seq_len,
            "selected_freqs": _selected_freqs(
                rotary, position_ids_cpu, max_seq_len, device=args.device
            ),
        }

    cache_len = args.decode_cache_len
    position_ids_cpu = (
        torch.arange(sequence_len, dtype=torch.long) + cache_len
    ).unsqueeze(0)
    key_cache = torch.randn(
        1, KVHEADS, cache_len, HEAD_DIM, dtype=torch.float16, generator=generator
    ).to(args.device)
    val_cache = torch.randn(
        1, KVHEADS, cache_len, HEAD_DIM, dtype=torch.float16, generator=generator
    ).to(args.device)
    max_seq_len = cache_len + sequence_len
    return {
        "position_ids": position_ids_cpu,
        "past_key_value_state": (key_cache, val_cache),
        "use_cache": True,
        "attn_name": args.attn_name,
        "contiguous_cache": True,
        "max_seq_len": max_seq_len,
        "selected_freqs": _selected_freqs(
            rotary, position_ids_cpu, max_seq_len, device=args.device
        ),
    }


def _unwrap(value: Any) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def _make_function(
    block: GraniteBlock,
    config: GraniteConfig,
    rotary: RotaryEmbedding,
    args: argparse.Namespace,
    sequence_len: int,
    generator: torch.Generator,
):
    residual_multiplier = config.residual_multiplier
    attention_kwargs = _make_attention_kwargs(rotary, args, sequence_len, generator)

    if args.part == "mlp_core":
        return lambda x: block.ff_sub_layer(x)
    if args.part == "mlp_norm":
        return lambda x: block.ff_sub_layer(block.ff_ln(x))
    if args.part == "mlp_residual":
        return lambda x: block.ff_sub_layer(x) * residual_multiplier + x
    if args.part == "mlp":
        return lambda x: block.ff_sub_layer(block.ff_ln(x)) * residual_multiplier + x
    if args.part == "attn_core":
        return lambda x: _unwrap(block.attn(q=x, **attention_kwargs))
    if args.part == "attn_norm":
        return lambda x: _unwrap(block.attn(q=block.ln(x), **attention_kwargs))
    if args.part == "attn_residual":
        return lambda x: _unwrap(block.attn(q=x, **attention_kwargs)) * residual_multiplier + x
    if args.part == "attn":
        return (
            lambda x: _unwrap(block.attn(q=block.ln(x), **attention_kwargs))
            * residual_multiplier
            + x
        )
    return lambda x: _unwrap(block(x, **attention_kwargs))


def _sync_to_cpu(value: Any) -> None:
    _unwrap(value).cpu()


def run(args: argparse.Namespace) -> int:
    generator = _cpu_generator(args.seed)
    sequence_len = args.seq_len or REGIME_M[args.regime]
    cache_dir = Path(
        args.cache_dir
        or os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_adnan")
    )
    block, config, rotary = _make_block(args)
    x = torch.randn(
        1, sequence_len, EMB, dtype=torch.float16, generator=generator
    ).to(args.device)
    fn = _make_function(block, config, rotary, args, sequence_len, generator)
    compiled_fn = fn if args.eager else torch.compile(fn, backend="inductor")

    out = compiled_fn(x)
    _sync_to_cpu(out)

    times_ms = []
    for _ in range(args.iters):
        start = time.time()
        out = compiled_fn(x)
        _sync_to_cpu(out)
        times_ms.append((time.time() - start) * 1000.0)
    times_ms.sort()

    result = {
        "part": args.part,
        "regime": args.regime,
        "seq_len": sequence_len,
        "device": args.device,
        "empty_weights": args.empty_weights,
        "fused_weights": args.fused_weights,
        "attn_name": args.attn_name,
        "iters": args.iters,
        "median_ms": statistics.median(times_ms),
        "all_ms": [round(value, 3) for value in times_ms],
        "cache_dir": str(cache_dir),
    }
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
    for row in _inventory(cache_dir):
        print("SDSC " + json.dumps(row, sort_keys=True), flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=[
            "mlp_core",
            "mlp_norm",
            "mlp_residual",
            "mlp",
            "attn_core",
            "attn_norm",
            "attn_residual",
            "attn",
            "block",
        ],
        default="block",
    )
    parser.add_argument("--regime", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--decode-cache-len", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--attn-name", default="sdpa_causal")
    parser.add_argument("--device", default="spyre")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--real-weights",
        action="store_false",
        dest="empty_weights",
        help="Copy initialized CPU weights to the target device instead of using to_empty.",
    )
    parser.set_defaults(empty_weights=True)
    parser.add_argument(
        "--fused-weights",
        action="store_true",
        help="Use FMS fused QKV/FFN weights. The cost-model probe default is unfused.",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Run eager instead of torch.compile. This is mostly useful for CPU checks.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
