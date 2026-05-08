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

"""Falsification probe for M4 (liveness-aware op reordering for LX peak).

M4's premise: a scheduler pass that reorders ops to minimize peak per-core
LX occupancy is worth building. The premise only holds if peak LX
*currently* approaches the 2 MB/core hard cap on production transformer
blocks. If it doesn't, the reordering search space has nothing to chew on
and the project closes.

Static-analysis falsification. We don't need hardware to answer the
binary question "does peak LX approach 2 MB/core?" — only:

  - The op sequence in a transformer block (we have it: block_ops)
  - Which ops produce LX-residency-eligible outputs (look at
    `torch_spyre._inductor.scratchpad.OP_OUTPUT_GOOD_FOR_LX_REUSE`)
  - Per-tensor sizes (transformer activation shapes, dtype = fp16)
  - Liveness intervals (each tensor lives from producer to last consumer)
  - The per-core split (planner's pure-M default = (32, 1, 1))

Two policies modelled:

  Model A — current torch_spyre behaviour: only outputs of `max`, `sum`,
            `clone` get pinned. In a Llama-style transformer block these
            are the softmax intermediates plus any explicit clones —
            tiny tensors (M × n_heads scalars).

  Model B — hypothetical expanded LX residency: all matmul outputs get
            pinned (this is the regime in which M4 would be deployed,
            since reordering only matters when a meaningful fraction of
            LX is being held cross-op).

For each (model, M), we walk the block, compute peak per-core LX
under both policies, and report whether the cap is approached.

Verdict criteria:

  - If max(Model A peak) < 200 KB across the suite: M4 has nothing to do
    in current torch_spyre. Project blocked on first expanding the
    eligible-op set, which is a separate project.
  - If max(Model B peak) < 1.6 MB across the suite: even with hypothetical
    expansion, peak LX doesn't bind. M4 closes.
  - If max(Model B peak) ≥ 1.6 MB: M4 has a real target *if* the eligible-
    op set is expanded. Probe should also estimate reordering headroom
    (peak under default order vs. theoretical-min peak).

Usage:
    python tests/diag_m4_peak_lx_falsification.py
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- model configs + block-op enumeration (vendored from
#      tests/hmi_cost_model_phase1_block.py on the
#      AdnanHoque/diag-core-ordering branch lineage; this probe is
#      cut from main and self-contained) ----------------------------

@dataclass
class ModelConfig:
    name: str
    hidden: int
    intermediate: int
    kv_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int


MODELS = {
    "llama_8b":   ModelConfig("Llama 3.1 8B",   4096, 14336, 1024,  32,  8, 128),
    "llama_70b":  ModelConfig("Llama 3.1 70B",  8192, 28672, 1024,  64,  8, 128),
    "llama_405b": ModelConfig("Llama 3.1 405B", 16384, 53248, 1024, 128, 8, 128),
    "mixtral":    ModelConfig("Mixtral 8x7B",   4096, 14336, 1024,  32,  8, 128),
    "dsv3":       ModelConfig("DeepSeek V3",    7168, 18432, 1536,  128, 128, 128),
}


@dataclass
class OpSpec:
    name: str
    kind: str           # "matmul" | "norm" | "softmax" | "elementwise"
    shape: tuple


def block_ops(cfg: ModelConfig, M: int) -> list[OpSpec]:
    H = cfg.hidden
    I = cfg.intermediate
    KV = cfg.kv_dim
    return [
        OpSpec("input_rmsnorm",       "norm",        (M, H)),
        OpSpec("q_proj",              "matmul",      (M, H, H)),
        OpSpec("kv_proj",             "matmul",      (M, KV, H)),
        OpSpec("attn_qkt_softmax_v",  "softmax",     (M, cfg.n_heads, cfg.head_dim)),
        OpSpec("o_proj",              "matmul",      (M, H, H)),
        OpSpec("post_attn_residual",  "elementwise", (M, H)),
        OpSpec("post_attn_rmsnorm",   "norm",        (M, H)),
        OpSpec("gate_proj",           "matmul",      (M, I, H)),
        OpSpec("up_proj",             "matmul",      (M, I, H)),
        OpSpec("silu_mul",            "elementwise", (M, I)),
        OpSpec("down_proj",           "matmul",      (M, H, I)),
        OpSpec("post_mlp_residual",   "elementwise", (M, H)),
    ]


# ---- constants -----------------------------------------------------

LX_BYTES_PER_CORE = 2 * 1024 * 1024
LX_USABLE_PER_CORE = int(LX_BYTES_PER_CORE * 0.8)  # 1.6 MB after backend reserve
PURE_M_CORES = 32
DTYPE_BYTES = 2   # fp16
DTYPE_PSUM_BYTES = 4  # fp32 accumulator

# Current torch_spyre OP_OUTPUT_GOOD_FOR_LX_REUSE
CURRENT_LX_ELIGIBLE_OPS = {"max", "sum", "clone"}


# ---- per-op tensor accounting --------------------------------------

@dataclass
class TensorRecord:
    name: str
    size_bytes_total: int        # full tensor across all cores
    size_bytes_per_core: int     # divided by pure-M split (= total / 32)
    eligible_lx: bool            # would the LX allocator pin this?
    produced_at: int             # op index that produces it
    last_used_at: int            # op index that last consumes it


def _tensor_size_for_op(op_name: str, op_kind: str, op_shape, M: int):
    """Estimate output tensor size in bytes (full tensor, before splitting)."""
    if op_kind == "matmul":
        # output is M × N elements at fp16
        _, N, _ = op_shape
        return M * N * DTYPE_BYTES
    if op_kind == "norm":
        M_dim, H = op_shape
        return M_dim * H * DTYPE_BYTES
    if op_kind == "elementwise":
        M_dim, D = op_shape
        return M_dim * D * DTYPE_BYTES
    if op_kind == "softmax":
        # softmax output is M × n_heads × head_dim (matches input)
        M_dim, n_heads, head_dim = op_shape
        return M_dim * n_heads * head_dim * DTYPE_BYTES
    return 0


def _build_block_with_residency(cfg, M, residency_policy: str):
    """Build a sequence of (op_idx, op_name, op_kind, output_record) plus
    the producer→consumer liveness for each output.

    residency_policy is "current" (only max/sum/clone outputs pinned) or
    "expanded" (all matmul outputs pinned plus the current set).

    Liveness is approximated by transformer-block structure: each op's
    output is consumed by the next op (the residual connection makes
    the residual op consume the post-norm input, but that's a small
    correction we ignore for this conservative estimate).
    """
    ops = block_ops(cfg, M)
    records: list[TensorRecord] = []

    for idx, op in enumerate(ops):
        # softmax kind in block_ops uses (M, n_heads, head_dim) but the
        # output shape is the same as input (matmul of attention).
        # Approximate output size as M × hidden_size.
        if op.kind == "matmul":
            # Output is (op.shape[0], op.shape[1]) i.e. (M, N).
            size_total = M * op.shape[1] * DTYPE_BYTES
        elif op.kind == "norm":
            size_total = op.shape[0] * op.shape[1] * DTYPE_BYTES
        elif op.kind == "elementwise":
            size_total = op.shape[0] * op.shape[1] * DTYPE_BYTES
        elif op.kind == "softmax":
            # Output is M × hidden (post attn_qkt_softmax_v projection)
            size_total = op.shape[0] * cfg.hidden * DTYPE_BYTES
        else:
            size_total = 0

        size_per_core = size_total // PURE_M_CORES

        # Determine eligibility based on policy.
        #   "current" — uses the OP_OUTPUT_GOOD_FOR_LX_REUSE set
        #   "expanded" — also pins matmul outputs
        eligible = False
        if residency_policy == "current":
            # Approximate: softmax has internal max/sum reductions —
            # treat its output as eligible (close to truth).
            if op.kind == "softmax":
                eligible = True
            # For norms, an embedded clone might happen — treat the
            # post-residual or post-norm clone as eligible (heuristic).
            # Conservative: NO non-softmax tensors pinned under current.
        elif residency_policy == "expanded":
            if op.kind == "matmul":
                eligible = True
            # Plus the current set
            if op.kind == "softmax":
                eligible = True

        # Liveness — assume consumed by the very next op.
        # The residual connections in a transformer block create longer
        # liveness for some tensors (input to a block lives until the
        # post-attn residual, etc.). Conservative: extend liveness by
        # one op for "residual"-named ops to capture this.
        last_used = min(idx + 1, len(ops) - 1)
        if "residual" in op.name:
            # the residual consumes a tensor produced earlier
            pass  # we don't attribute the longer liveness to *this* op
        # approximate: long-range residual liveness on input_rmsnorm
        # output would extend to post_attn_residual. We don't model
        # this exactly — see caveats in the writeup.

        records.append(TensorRecord(
            name=op.name,
            size_bytes_total=size_total,
            size_bytes_per_core=size_per_core,
            eligible_lx=eligible,
            produced_at=idx,
            last_used_at=last_used,
        ))
    return records, ops


def _peak_lx_per_core(records: list[TensorRecord]) -> tuple[int, int]:
    """Walk op timeline; track currently-resident tensors. Return
    (peak_per_core_bytes, peak_op_idx).
    """
    n = len(records)
    peak = 0
    peak_idx = 0
    for t in range(n):
        # Tensors resident at time t = those eligible_lx, produced ≤ t,
        # not yet last-consumed (last_used_at > t means still needed).
        live_bytes = 0
        for r in records:
            if not r.eligible_lx:
                continue
            if r.produced_at <= t and r.last_used_at >= t:
                live_bytes += r.size_bytes_per_core
        if live_bytes > peak:
            peak = live_bytes
            peak_idx = t
    return peak, peak_idx


def _print_block_summary(label: str, cfg_name: str, M: int,
                          records: list[TensorRecord], ops):
    # Per-record summary
    n_eligible = sum(1 for r in records if r.eligible_lx)
    total_eligible_bytes = sum(r.size_bytes_per_core for r in records
                                if r.eligible_lx)
    peak, peak_idx = _peak_lx_per_core(records)
    peak_op = ops[peak_idx].name
    return dict(
        label=label,
        model=cfg_name,
        M=M,
        n_eligible=n_eligible,
        total_eligible_bytes_per_core=total_eligible_bytes,
        peak_bytes_per_core=peak,
        peak_op=peak_op,
        n_ops=len(ops),
    )


# ---- main ----------------------------------------------------------

def _fmt(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.2f} MB"
    if n >= 1024:
        return f"{n // 1024} KB"
    return f"{n} B"


def main() -> int:
    print("# M4 falsification probe — peak LX occupancy on transformer blocks\n")
    print(f"LX hard cap: {_fmt(LX_BYTES_PER_CORE)}/core")
    print(f"LX usable:   {_fmt(LX_USABLE_PER_CORE)}/core "
          "(after DXP_LX_FRAC_AVAIL=0.2 backend reserve)\n")
    print("Policies:")
    print("  Model A (current): softmax outputs LX-eligible "
          "(approximates max/sum/clone behaviour)")
    print("  Model B (M4 hypothetical): matmul outputs ALSO LX-eligible")
    print()
    print("Per-core sizes assume planner pure-M (32, 1, 1).\n")

    suite = []
    for model_key, cfg in MODELS.items():
        for M in (32, 128, 512, 2048):
            for policy in ("current", "expanded"):
                records, ops = _build_block_with_residency(cfg, M, policy)
                suite.append(_print_block_summary(
                    "A" if policy == "current" else "B",
                    cfg.name, M, records, ops))

    print("## Per-(model, M, policy) peak LX usage\n")
    print("| model | M | policy | peak LX/core | "
          "% of 1.6 MB cap | peak op | total resident bytes/core |")
    print("|---|---:|---|---:|---:|---|---:|")
    for r in suite:
        pct = r["peak_bytes_per_core"] / LX_USABLE_PER_CORE * 100
        print(f"| {r['model']} | {r['M']} | {r['label']} | "
              f"{_fmt(r['peak_bytes_per_core'])} | {pct:.0f}% | "
              f"{r['peak_op']} | "
              f"{_fmt(r['total_eligible_bytes_per_core'])} |")

    # Verdict
    a_rows = [r for r in suite if r["label"] == "A"]
    b_rows = [r for r in suite if r["label"] == "B"]

    a_max = max(r["peak_bytes_per_core"] for r in a_rows)
    b_max = max(r["peak_bytes_per_core"] for r in b_rows)
    a_max_row = max(a_rows, key=lambda r: r["peak_bytes_per_core"])
    b_max_row = max(b_rows, key=lambda r: r["peak_bytes_per_core"])

    print()
    print("## Verdict\n")
    print(f"Model A (current behaviour): max peak LX/core = "
          f"{_fmt(a_max)} on {a_max_row['model']} M={a_max_row['M']}")
    print(f"Model B (expanded matmul):   max peak LX/core = "
          f"{_fmt(b_max)} on {b_max_row['model']} M={b_max_row['M']}")
    print()
    if a_max < 200 * 1024:
        print(f"  Under current behaviour (A): peak {_fmt(a_max)} is well below "
              "the 1.6 MB usable cap.")
        print("  M4 reordering has no target with the current OP_OUTPUT_GOOD_FOR_LX_REUSE "
              "set.")
    if b_max < LX_USABLE_PER_CORE:
        print(f"  Under expanded matmul-residency (B): peak {_fmt(b_max)} is "
              "still under the 1.6 MB cap.")
        print("  M4 reordering doesn't help even in the hypothetical expanded "
              "regime — peak LX never approaches the cap on these shapes.")
        print()
        print("  VERDICT: M4 closes. Reordering for LX peak occupancy has no "
              "production target; the cap is rarely binding because per-core "
              "tensor sizes are small at typical M values under pure-M split.")
    else:
        print(f"  Under expanded matmul-residency (B): peak {_fmt(b_max)} "
              "approaches/exceeds the cap.")
        print("  M4 has a real target IF the OP_OUTPUT_GOOD_FOR_LX_REUSE set is "
              "expanded. The current narrow set provides no target.")
        print()
        print("  VERDICT: M4 is conditionally viable. Sequence: first expand the "
              "eligible-op set (separate project), then reordering.")

    print()
    print("## Caveats\n")
    print("- Per-core peak under pure-M (32, 1, 1) divides total tensor by 32.")
    print("  Other splits (e.g., (1, 16, 2)) put more bytes on each core.")
    print("- Liveness model is approximate: tensors are assumed consumed by")
    print("  the very next op, missing residual-connection long-range liveness.")
    print("  This UNDER-estimates peak LX. Real peak could be ~2× this.")
    print("- Static analysis doesn't account for the LX allocator's")
    print("  fragmentation behaviour, which can effectively reduce usable bytes.")
    print("- Softmax intermediate handling is approximated as one resident")
    print("  output; the actual max/sum tensors are MUCH smaller. So Model A is")
    print("  an OVER-estimate of current peak LX.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
