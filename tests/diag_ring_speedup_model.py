# Copyright 2026 The Torch-Spyre Authors.
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

"""Theoretical speedup model for FUNDAMENTAL restickify replacement.

Compares today's `ReStickifyOpHBM` (2x HBM round trip) against a hypothetical
`STCDPOpLx`-based on-chip ring shuffle, for the workloads in the 1H 2026
roadmap (GPT-OSS-20B, Granite-4 Hybrid 30B, Mistral-small, Qwen2.5-VL-7B,
Llama-3.1-8B-instruct, Ministral 8B/14B).

Three ring cost models are presented because the AIU RIU ring's effective
bandwidth for an all-to-all stick shuffle depends on routing/pipelining:

  A. **Bisection-bound** — half the tensor crosses the cut.
        T = (tensor_bytes / 2) / (2 * link_bw)
  B. **Uniform all-to-all** — `byte-hops needed / byte-hops available`,
        average hop ≈ N/4 across N cores. T = tensor_bytes / (4 * link_bw)
  C. **Aggregate parallel** — every core uses its full link bidirectionally.
        T = tensor_bytes / (2 * N * link_bw)

(A) is pessimistic; (C) is the theoretical aggregate. (B) is the most
defensible for a transpose-style shuffle on a uniform ring. Empirical
microbench (proposed below) is the way to pick which is right for AIU RIU.

Run: python3 tests/diag_ring_speedup_model.py
"""

from __future__ import annotations
from dataclasses import dataclass


# -- hardware constants ------------------------------------------------------

# AIU RIU ring bandwidth per the IBM Spyre spec: 166 GB/s per direction
# per link (i.e. each bidirectional link carries 166 GB/s in each direction
# simultaneously, for 332 GB/s aggregate per-link).
RING_LINK_BW_PER_DIR = 166e9  # bytes/s, one direction
NUM_CORES = 32
MAX_HOP = NUM_CORES // 2  # bidirectional → worst-case distance N/2

# Measured effective HBM bandwidth for `ReStickifyOpHBM` on torch-spyre
# (from earlier session probes — single-shot, single-op).
HBM_EFFECTIVE_BW = 107e9  # bytes/s


# -- cost models -------------------------------------------------------------

def hbm_restickify_time(tensor_bytes: float, hbm_bw: float = HBM_EFFECTIVE_BW) -> float:
    """Today's `ReStickifyOpHBM`: read + write through HBM, sequentially."""
    return 2.0 * tensor_bytes / hbm_bw


def ring_time_bisection(
    tensor_bytes: float, link_bw_per_dir: float = RING_LINK_BW_PER_DIR
) -> float:
    """Model A: half the data crosses the bisection.
    A ring has 2 links at any cut, each bidirectional → 4 channels at
    link_bw_per_dir. Bisection capacity = 4 * link_bw_per_dir.
    """
    bisection_bw = 4.0 * link_bw_per_dir
    return (tensor_bytes / 2.0) / bisection_bw


def ring_time_all_to_all(
    tensor_bytes: float, num_cores: int = NUM_CORES,
    link_bw_per_dir: float = RING_LINK_BW_PER_DIR,
) -> float:
    """Model B: uniform all-to-all. byte-hops / byte-hop-bw.
    Each core sends tensor/N to N-1 others, avg hop N/4 (bidirectional ring).
    Total byte-hops = N * (tensor/N) * N/4 = tensor * N / 4.
    Aggregate byte-hop bw = N links × 2 directions × link_bw_per_dir.
    """
    total_byte_hops = tensor_bytes * num_cores / 4.0
    aggregate_bw = num_cores * 2.0 * link_bw_per_dir
    return total_byte_hops / aggregate_bw


def ring_time_aggregate(
    tensor_bytes: float, num_cores: int = NUM_CORES,
    link_bw_per_dir: float = RING_LINK_BW_PER_DIR,
) -> float:
    """Model C: aggregate ring bandwidth, no hop penalty (best case).
    All N cores use both directions of their links concurrently.
    """
    return tensor_bytes / (num_cores * 2.0 * link_bw_per_dir)


def per_op_speedup(tensor_bytes: float) -> dict[str, float]:
    """Per-FUNDAMENTAL-restickify speedup vs today's ReStickifyOpHBM."""
    t_hbm = hbm_restickify_time(tensor_bytes)
    return {
        "bisection (A)": t_hbm / ring_time_bisection(tensor_bytes),
        "all-to-all (B)": t_hbm / ring_time_all_to_all(tensor_bytes),
        "aggregate (C)": t_hbm / ring_time_aggregate(tensor_bytes),
    }


# -- workload modeling -------------------------------------------------------

@dataclass
class TransformerAttn:
    """Per-transformer-block FUNDAMENTAL restickify cost at given seq length."""
    name: str
    H: int           # hidden dim
    num_heads: int
    head_dim: int    # H == num_heads * head_dim
    inter_dim: int   # MLP intermediate
    n_layers: int
    n_attn_layers: int  # for hybrid models (Mamba+Transformer)
    dtype_bytes: int = 2  # fp16

    def attn_fundamental_bytes(self, M: int) -> int:
        """Two FUNDAMENTAL restickifies per attention layer:
        1) on Q before Q@K^T (matmul→transposed-matmul, [B,H,M,D] = M*H bytes)
        2) on attn output before reshape (matmul→transposed-pointwise, same shape)
        """
        per_layer = 2 * M * self.H * self.dtype_bytes
        return per_layer * self.n_attn_layers

    def layer_total_hbm_bytes(self, M: int, materialize_scores: bool = False) -> int:
        """Rough total HBM bytes per attention layer (weights + activations).

        If `materialize_scores` (today's reality without flash attention),
        adds the round-tripped [B, H, M, M] attention-score tensor traffic,
        which grows as O(M^2) and dominates at long context.
        """
        # 3 QKV proj (H*H each) + 1 O proj (H*H) -> 4 H^2 weight reads.
        # MLP: 3 * H * inter_dim (gate, up, down).
        weights = (4 * self.H * self.H + 3 * self.H * self.inter_dim) * self.dtype_bytes
        # Residual stream activations (in + out).
        acts = 2 * M * self.H * self.dtype_bytes
        # Without flash attention, the Q@K^T scores are materialized:
        # write once, softmax round-trips (read+write), then attn@V reads.
        # ~3 round trips of B * H * M * M.
        scores = (3 * self.num_heads * M * M * self.dtype_bytes
                  if materialize_scores else 0)
        return self.n_attn_layers * (weights + acts + scores)


# -- representative configs from the 1H 2026 roadmap ------------------------

WORKLOADS = [
    TransformerAttn("Llama-3.1-8B", H=4096, num_heads=32, head_dim=128,
                    inter_dim=14336, n_layers=32, n_attn_layers=32),
    TransformerAttn("Ministral-8B", H=4096, num_heads=32, head_dim=128,
                    inter_dim=14336, n_layers=36, n_attn_layers=36),
    TransformerAttn("Mistral-small-24B", H=5120, num_heads=32, head_dim=128,
                    inter_dim=32768, n_layers=40, n_attn_layers=40),
    # Granite-4 30B Hybrid: ~9 attention layers, ~31 mamba layers (rough),
    # plus MoE (8 experts/layer). FUNDAMENTAL restickify is in the attention
    # layers; mamba SSM contributes little because the SSM scan stays along
    # one (sequence) dim. MoE routing may add its own pattern, modeled
    # separately — not included here.
    TransformerAttn("Granite-4-Hybrid 30B (attn-only)", H=4096, num_heads=32,
                    head_dim=128, inter_dim=16384, n_layers=40, n_attn_layers=9),
    TransformerAttn("GPT-OSS-20B", H=4096, num_heads=32, head_dim=128,
                    inter_dim=11008, n_layers=40, n_attn_layers=40),
]
SEQ_LENGTHS = [128, 512, 2048, 8192, 32768, 131072]


# -- main --------------------------------------------------------------------

def main():
    print("\n=== Per-op FUNDAMENTAL restickify speedup vs ReStickifyOpHBM ===")
    print(f"  HBM effective bw: {HBM_EFFECTIVE_BW/1e9:.1f} GB/s "
          f"(measured, restickify single-shot)")
    print(f"  Ring link bw:     {RING_LINK_BW_PER_DIR/1e9:.1f} GB/s "
          f"per direction (spec)")
    print(f"  Ring cores:       {NUM_CORES}\n")
    print(f"  {'tensor size':<14} {'A bisection':>14} {'B all-to-all':>14} "
          f"{'C aggregate':>14}")
    print("  " + "-" * 60)
    for mb in (1, 4, 16, 64):
        nbytes = mb * 1024 * 1024
        sp = per_op_speedup(nbytes)
        print(
            f"  {mb:>3} MB         "
            f"{sp['bisection (A)']:>14.2f}x"
            f"{sp['all-to-all (B)']:>14.2f}x"
            f"{sp['aggregate (C)']:>14.2f}x"
        )
    print("  (per-op speedup is shape-invariant under these models)")

    per_op = per_op_speedup(1024*1024)["all-to-all (B)"]

    for materialize, label in [
        (False, "FLASH ATTENTION (scores NOT in HBM)"),
        (True, "MATERIALIZED SCORES (no flash, today's torch-spyre)"),
    ]:
        print(f"\n\n=== Layer-level share + speedup — {label} ===")
        print("  share of layer HBM:")
        print(f"  {'Model':<32} " + "  ".join(f"M={M:<8}" for M in SEQ_LENGTHS))
        for w in WORKLOADS:
            shares = []
            for M in SEQ_LENGTHS:
                fund = w.attn_fundamental_bytes(M)
                tot = w.layer_total_hbm_bytes(M, materialize_scores=materialize)
                shares.append(100 * fund / tot)
            print(
                f"  {w.name:<32} "
                + "  ".join(f"{s:7.2f}%" for s in shares)
            )

        print(f"\n  per-attention-layer speedup (per_op={per_op:.2f}x):")
        print(f"  {'Model':<32} " + "  ".join(f"M={M:<8}" for M in SEQ_LENGTHS))
        for w in WORKLOADS:
            speedups = []
            for M in SEQ_LENGTHS:
                fund = w.attn_fundamental_bytes(M)
                tot = w.layer_total_hbm_bytes(M, materialize_scores=materialize)
                share = fund / tot
                layer = 1.0 / (1.0 - share * (1.0 - 1.0 / per_op))
                speedups.append(layer)
            print(
                f"  {w.name:<32} "
                + "  ".join(f"{s:8.3f}x" for s in speedups)
            )

    print("\n=== Notes ===")
    print("  - Per-op speedup is shape-invariant under each model because")
    print("    both T_hbm and T_ring are proportional to tensor_bytes.")
    print("  - Layer-level speedup grows with M (long-context): at M=128 the")
    print("    weight-read mass dominates HBM, restickify is small; at M=8192")
    print("    the activation restickify scales linearly while weights stay flat.")
    print("  - Granite-4 hybrid: FUNDAMENTAL restickify is only in the 9 attn")
    print("    layers (estimate); 31 mamba layers contribute ~zero. Whole-model")
    print("    speedup is roughly 9/40 × per-attn-layer speedup + (31/40)×1 .")


if __name__ == "__main__":
    main()
