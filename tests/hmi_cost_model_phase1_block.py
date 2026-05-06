# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase 1 of Project B: transformer-decoder-block wall-time prediction.

Composes the per-op HMI cost model into a serial sum across one
decoder block of a chosen LLM. For each op in the block (projections
+ attention compute + norms + activations) we predict wall time under
the planner's natural pick (pure-M (32, 1, 1)) and report:

  - per-op wall + classification
  - end-to-end serial wall (what today's runtime delivers)
  - which ops dominate the block

This is the qualitative output Project B needs: which ops are
HMI-bound, where in the block does HMI dominate, and what fraction
of the block is launch-floor / HMI / compute.

Usage:
    python tests/hmi_cost_model_phase1_block.py
    python tests/hmi_cost_model_phase1_block.py --model llama_70b --m 128
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.hmi_cost_model import predict, label, LAUNCH_FLOOR_MS, HMI_BW_GBS


# ---- model configurations --------------------------------------------
# Compact decoder-block specs for the popular LLMs we measured against.
# Sources: model card configs. GQA collapses k_proj+v_proj into kv_proj.

@dataclass
class ModelConfig:
    name: str
    hidden: int          # hidden_size / d_model
    intermediate: int    # MLP intermediate / FFN inner
    kv_dim: int          # combined K+V projection out_features
    n_heads: int
    n_kv_heads: int
    head_dim: int


MODELS = {
    "llama_8b":   ModelConfig("Llama 3.1 8B",   4096, 14336, 1024,  32,  8, 128),
    "llama_70b":  ModelConfig("Llama 3.1 70B",  8192, 28672, 1024,  64,  8, 128),
    "llama_405b": ModelConfig("Llama 3.1 405B", 16384, 53248, 1024, 128, 8, 128),
    "mixtral":    ModelConfig("Mixtral 8x7B",   4096, 14336, 1024,  32,  8, 128),
    "dsv3":       ModelConfig("DeepSeek V3",    7168, 18432, 1536,  128, 128, 128),
    # DSv3 uses MLA so the kv structure differs; we approximate the
    # absorbed q_a_proj here. o_proj has out_features=hidden (7168);
    # attention head dims are non-trivial — represented best-effort.
}


# ---- block op list ---------------------------------------------------
# Each op: (name, kind, shape_fn). Matmuls have shape_fn returning
# (M, N, K). Non-matmul ops have shape_fn returning a simple param tuple.

@dataclass
class OpSpec:
    name: str
    kind: str           # "matmul" | "norm" | "softmax" | "elementwise"
    shape: tuple        # for matmul: (M, N, K); for others: dim params


def block_ops(cfg: ModelConfig, M: int) -> list[OpSpec]:
    """Build the op list for one decoder block at batch-token count M."""
    H = cfg.hidden
    I = cfg.intermediate
    KV = cfg.kv_dim
    return [
        # ── pre-attention norm ────────────────────────────────────────
        OpSpec("input_rmsnorm", "norm", (M, H)),
        # ── attention projections ─────────────────────────────────────
        OpSpec("q_proj",  "matmul", (M, H, H)),
        OpSpec("kv_proj", "matmul", (M, KV, H)),
        # ── attention compute (softmax(QKᵀ/√d)V) ──────────────────────
        # Approximated as one matmul per head ≈ M^2 ops. Size-light at
        # decode M; heavy at long-prefill M. Treated as a single op.
        OpSpec("attn_qkt_softmax_v", "softmax", (M, cfg.n_heads, cfg.head_dim)),
        OpSpec("o_proj",  "matmul", (M, H, H)),
        # ── post-attention residual + norm ────────────────────────────
        OpSpec("post_attn_residual", "elementwise", (M, H)),
        OpSpec("post_attn_rmsnorm", "norm", (M, H)),
        # ── MLP (SwiGLU) ──────────────────────────────────────────────
        OpSpec("gate_proj", "matmul", (M, I, H)),
        OpSpec("up_proj",   "matmul", (M, I, H)),
        OpSpec("silu_mul",  "elementwise", (M, I)),
        OpSpec("down_proj", "matmul", (M, H, I)),
        # ── post-MLP residual ─────────────────────────────────────────
        OpSpec("post_mlp_residual", "elementwise", (M, H)),
    ]


# ---- non-matmul cost models ------------------------------------------
# Norms / softmax / elementwise are roughly memory-bound at the rates
# we see, so model them as HMI bytes / BW with launch floor.

def _predict_nonmatmul(op: OpSpec) -> dict:
    """Coarse cost model for non-matmul ops.

    Norms: read M·H, write M·H, plus reduction. Memory-bound.
    Softmax (attention): O(M·n_heads·head_dim) for QKᵀ + softmax + V.
                         At decode (M small), tiny. At prefill (M big),
                         can be significant.
    Elementwise (residual, silu_mul): read 2·bytes, write 1·bytes.
    """
    name = op.name
    kind = op.kind
    if kind == "norm":
        M, H = op.shape
        bytes_in_out = 2 * M * H * 2  # read + write, fp16
        t_hmi = bytes_in_out / (HMI_BW_GBS * 1e9) * 1e3
        wall = max(t_hmi + LAUNCH_FLOOR_MS, LAUNCH_FLOOR_MS)
        return dict(name=name, kind=kind, t_compute=0.0, t_hmi=t_hmi,
                    t_wall=wall, hmi_bytes=bytes_in_out, label="HMI/LF")
    elif kind == "elementwise":
        M, D = op.shape
        bytes_io = 3 * M * D * 2
        t_hmi = bytes_io / (HMI_BW_GBS * 1e9) * 1e3
        wall = max(t_hmi + LAUNCH_FLOOR_MS, LAUNCH_FLOOR_MS)
        return dict(name=name, kind=kind, t_compute=0.0, t_hmi=t_hmi,
                    t_wall=wall, hmi_bytes=bytes_io, label="HMI/LF")
    elif kind == "softmax":
        # QKᵀ + softmax + V·sm. Heads are independent so they
        # parallelize across cores (one core per head, batched if
        # n_heads > 32). Per-head: 2·M²·head_dim macs (QKᵀ + V mul) +
        # M² softmax. We divide total flops by 32 cores.
        M, n_heads, head_dim = op.shape
        macs_per_head = 2 * M * M * head_dim
        flops = 2 * macs_per_head * n_heads
        per_core_flops = flops / 32
        pt_util = min(1.0, M / 8) * min(1.0, head_dim / 64)
        t_compute = per_core_flops / (1e12 * max(pt_util, 1e-3)) * 1e3
        bytes_io = (M * n_heads * head_dim) * 4 * 2  # Q, K, V, output
        t_hmi = bytes_io / (HMI_BW_GBS * 1e9) * 1e3
        wall = max(t_compute, t_hmi + LAUNCH_FLOOR_MS)
        return dict(name=name, kind=kind, t_compute=t_compute, t_hmi=t_hmi,
                    t_wall=wall, hmi_bytes=bytes_io, label="attention")
    raise ValueError(f"unknown op kind: {kind}")


def _predict_matmul(op: OpSpec) -> dict:
    """Wrap predict() with the planner-natural split (32, 1, 1)."""
    M, N, K = op.shape
    cb = predict((M, N, K), (32, 1, 1), dtype="fp16", k_fast=False)
    return dict(name=op.name, kind=op.kind,
                t_compute=cb.t_compute_ms, t_hmi=cb.t_hmi_ms,
                t_wall=cb.t_wall_ms, hmi_bytes=cb.hmi_bytes,
                label=label(cb))


def predict_op(op: OpSpec) -> dict:
    if op.kind == "matmul":
        return _predict_matmul(op)
    return _predict_nonmatmul(op)


# ---- main ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama_70b",
                        choices=list(MODELS.keys()))
    parser.add_argument("--m", type=int, default=128,
                        help="batch-token count M (decode=32-128, prefill=2048+)")
    args = parser.parse_args()

    cfg = MODELS[args.model]
    ops = block_ops(cfg, args.m)
    rows = [predict_op(op) for op in ops]

    print(f"# Phase 1 cost-model: {cfg.name} decoder block at M={args.m}\n")
    print(f"# Per-op breakdown (planner-natural split (32, 1, 1), kf off)\n")
    print("| op | kind | shape | wall ms | compute ms | hmi ms | hmi MB | class |")
    print("|---|---|---|---:|---:|---:|---:|---|")
    total_wall = 0.0
    total_hmi_ms = 0.0
    total_compute_ms = 0.0
    total_hmi_bytes = 0
    for op, r in zip(ops, rows):
        shape_str = "×".join(str(d) for d in op.shape)
        print(f"| {r['name']} | {r['kind']} | {shape_str} | "
              f"{r['t_wall']:.3f} | {r['t_compute']:.3f} | {r['t_hmi']:.3f} | "
              f"{r['hmi_bytes'] / 1e6:.1f} | {r['label']} |")
        total_wall += r['t_wall']
        total_hmi_ms += r['t_hmi']
        total_compute_ms += r['t_compute']
        total_hmi_bytes += r['hmi_bytes']

    print()
    print(f"## Block totals\n")
    print(f"  serial wall:      {total_wall:.2f} ms")
    print(f"  total HMI demand: {total_hmi_bytes / 1e6:.1f} MB")
    print(f"  Σ t_compute:      {total_compute_ms:.2f} ms")
    print(f"  Σ t_hmi:          {total_hmi_ms:.2f} ms")
    print(f"  Σ launch floor:   {len(rows) * LAUNCH_FLOOR_MS:.2f} ms")

    # Top dominators
    print()
    print(f"## Top-5 walltime contributors\n")
    sorted_rows = sorted(rows, key=lambda r: -r['t_wall'])
    for r in sorted_rows[:5]:
        print(f"  {r['name']:<24} {r['t_wall']:6.2f} ms "
              f"({r['t_wall'] / total_wall * 100:4.1f}%)  [{r['label']}]")

    # Class breakdown
    print()
    print(f"## Class breakdown\n")
    by_class = {}
    for r in rows:
        by_class.setdefault(r['label'], []).append(r)
    for cls, grp in sorted(by_class.items(),
                           key=lambda kv: -sum(r['t_wall'] for r in kv[1])):
        cls_wall = sum(r['t_wall'] for r in grp)
        print(f"  {cls:<22} {cls_wall:6.2f} ms "
              f"({cls_wall / total_wall * 100:4.1f}%)  "
              f"{len(grp)} op{'s' if len(grp) > 1 else ''}")

    print()
    print(f"## Phase 1 readouts\n")
    hmi_dominant = [r for r in rows if r['label'] == 'HMI-bound']
    print(f"  HMI-bound ops: {len(hmi_dominant)} of {len(rows)}")
    if hmi_dominant:
        hmi_dom_wall = sum(r['t_wall'] for r in hmi_dominant)
        print(f"  HMI-bound wall fraction: {hmi_dom_wall / total_wall * 100:.0f}%")
        print(f"  → scheduling headroom is bounded by overlapping these HMI ops")
        print(f"    with non-HMI ops in the block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
