# Inductor-only on-chip MLP — work-division co-assignment + LX-flip

Supersedes the data-op reshard (A2) for landing the win **without any deeptools
change**. The reshard was blocked by the dxp import assert
(`SdscTree.cpp:152`, no bundle-imported data-ops). This path emits **no data-op**
at all, so the gate is never touched.

## The reframe

The cross-division `matmul→pointwise` HBM round-trip exists only because the
matmul `(m4,n8)` and the pointwise (pure-M) are divided **independently**. Moving
data between those divisions needs an `STCDPOpLx` data-op → dxp-blocked. Instead,
**make the consumer inherit the producer's division** so the edge is
**same-division same-core** (each consumer core reads exactly the tile its own
core produced) → nothing to move → persist with a base-pointer flip.

## Two pure-Inductor passes

1. **Co-assignment** — after the cost model sets the matmul split, propagate the
   matmul's output `(b,m,n)` split to its **element-wise** consumers by setting
   `consumer.op_it_space_splits` (the hook `span_reduction` uses; `apply_splits`
   writes it at `work_division.py:503`, `work_distribution_pass` reads it at
   `:608`). Element-wise ops are division-agnostic → always correctness-safe. The
   core mapping must match (same `coreIdToWkSlice`) so the edge is same-**core**,
   not just same-division.
2. **LX-flip** — reuse `onchip_softmax_chain.apply_lx_flip` (the value-correct
   1.88× mechanism) to flip the now-same-division producer-output + consumer-input
   to LX-resident. Pure dict surgery on the existing DL labeledDs: `memOrg_→lx`,
   HBM cleared, per-core LX base. **Adds no datadsc** → SDSC stays
   `dscs_=[DL], datadscs_=[]` → passes the dxp import assert. No deeptools.

## Why it wins where A1 steer (matmul→pure-M) lost

A1 forced the **producer** to pure-M → matmul slow (27.8 vs 19.8 ms). This forces
the cheap, memory-bound **consumer** to the matmul's `(m4,n8)` → matmul keeps its
fast split, the pointwise barely cares, and the hand-off goes on-chip. Captures
the reshard's goal (keep m×n, kill the hand-off) with zero data-ops.

## Unfused vs fused

- **Unfused** (separate gate/up matmuls): both `(m4,n8)` on all 32 cores →
  `silu(gate)×up` mul is local → whole chain on-chip. **Target this** (also the
  faster baseline: 13.9 vs 19.8 ms).
- **Fused** (combined gate+up matmul + `split_with_sizes`): gate=out-bands 0–3,
  up=out-bands 4–7 → different cores → `gate×up` mul stays cross-core (one
  residual edge). Lower priority.

## Implementation plan

1. **Co-assignment pass** (new, Inductor work-division). After cost-model +
   before/within `work_distribution`: for each matmul whose consumers are
   element-wise, set each consumer's `op_it_space_splits` to the matmul's output
   split (mapped through the consumer's index/coeff), preserving the core mapping.
   Gate on: consumer is element-wise (`Pointwise`), the split divides the
   consumer's iteration space, and the per-core tile fits the LX window.
2. **LX-flip**: enable `onchip_softmax_chain` (already on this branch,
   value-correct, no dxp gate); it now finds the matmul→pointwise edges
   same-division and persists them. (Extend its detection to the matmul-output →
   pointwise-input edge if it currently scopes only the softmax tail.)
3. **Harness**: `run_ab.py --lever coassign` — monkeypatch the co-assignment +
   enable the LX-flip pass. No checkout edit.
4. **Validation**:
   - **CPU dxp-accept** (no device): compile the co-assigned + LX-flipped bundle,
     run `dxp_standalone --bundle` → expect exit 0 (no data-op, gate not hit).
   - **Device** (solo, long timeout): `max_err` vs CPU + kernel time vs A0
     (unfused 13.9 ms; the win = the eliminated cross-division hand-off, with the
     matmul kept at `(m4,n8)`).

## Status
Architected here; implementing the co-assignment pass next. This is the
"Track A work-distribution steering" the prior thread flagged as the
pure-Inductor lever (vs the dxp-blocked data-op reshard, now parked as a
deeptools RFC in `STATUS.md`).
