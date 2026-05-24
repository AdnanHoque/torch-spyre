# A/B Mamba-2 on-chip same-stick handoff — classification + projection

OFFLINE (2026-05-24). No device run, no torch.compile. Classifies the activation
handoffs in a Mamba-2 (SSD) block by Tier-1 same-stick eligibility and projects
the on-chip speedup from the empirical anchor. Scripts under `/tmp/ab_mamba2/`.

## Empirical anchor (MEASURED)

From `CoreToCoreDataMovementRecipe.md` §12 / §9, proven on silicon:
**~0.029 ms saved per MB of eliminated same-stick handoff, minus ~0.005 ms STCDP
setup; net-positive above ~1 MB handoff.** Relative speedup peaks mid-range
(1.22x @1024) and tapers as matmul O(N³) dwarfs the O(N²) handoff. Same-stick
same-shard = same-core copy (HBM-elim, no ring, fits all sizes); same-stick
diff-shard = real cross-core ring (proven `i->31-i->i`); layout-changing (stick
flips) = blocked on the Compute-CB transpose fault.

## 1. Per-handoff Tier-1 classification (PROJECTED by analogy to real edges)

stick = stickDimOrder_; tensor sizes for d_model=2048, seq=256, fp16.

| handoff | prod stick | cons stick | same-stick? | shard | tensor MB | verdict |
|---|---|---|---|---|---|---|
| in-proj -> {x,B,C,z,dt} slices | out | out | YES | mb-resharded | 4.2 fused | STCDP-today (cross-core) |
| x slice -> conv1d | out | mb (channel) | NO (flip) | — | 1.05 | needs-transpose (blocked) |
| conv -> SiLU | out | out | YES | mb | 1.05 | STCDP-today same-core |
| SiLU -> scan in (C@B) | out | in | NO (flip) | — | 1.05 | needs-transpose (blocked) |
| C@B -> decay mul (intra-SSD) | out | out | YES | out16 -> mb | 0.13 | STCDP-today cross-core |
| decay-mul -> scan @x (intra-SSD) | out | in | NO (flip) | — | 0.13 | needs-transpose (blocked) |
| scan -> gate (z·SiLU) | out | out | YES | mb | 1.05 | STCDP-today same-core |
| gate -> RMSNorm | out | out | YES | mb | 1.05 | STCDP-today same-core |
| RMSNorm mean/rsqrt | out | x | NO (flip) | — | tiny | needs-transpose (blocked) |
| RMSNorm -> out-proj | out | out | YES | mb | 1.05 | STCDP-today cross-core |

x/B/C/z/dt slices on the d_model axis stay **same-stick** (out): the split is
re-ownership, not a stick flip -> STCDP-eligible. The blocked ones are the
matmul-input transposes (out->in) and the RMSNorm reduction reshape (out->x),
matching the 8/40 layout-changing cluster in `/tmp/real_edge_analysis.md`.

## 2. Pieces microbenched + why (highest Tier-1 benefit)

- `micro_ssd_matmul.py` — intra-SSD C@B -> decay-mul -> @x. bmm-out -> mul is
  same-stick cross-core (proven QK^T->softmax class), recurs per block. Highest
  recurrence matmul handoff.
- `micro_gate_norm_out.py` — scan->gate->RMSNorm->out-proj. Long same-stick
  same-shard chain = same-core HBM-elim, fits at seq=1 (decode). Bandwidth-bound.
- `micro_inproj_split.py` — fused in-proj output (4.2 MB) -> x/z consumers, the
  biggest single handoff and 4× the d_model width.

`mamba2_workload.py` = whole-block baseline+correctness; A/B via SPLICED_DIR.

## 3. Projected speedups (anchor MEASURED, applied PROJECTED)

| handoff | MB | net = 0.029·MB − 0.005 |
|---|---|---|
| prefill seq=256: in-proj fused | 4.2 | **+0.117 ms** |
| prefill seq=256: scan/gate/out | 1.05 | **+0.025 ms** |
| decode seq=1: in-proj | 0.016 | −0.005 (below crossover) |
| decode seq=1: scan/gate | 0.004 | −0.005 (below crossover) |
| batched decode B=64: in-proj | 1.05 | **+0.025 ms** |
| batched decode B=256: scan | 1.05 | **+0.025 ms** |

## 4. Honest assessment (decode emphasis)

Per-block STCDP-today edges: ~4 same-core HBM-elim + 3 cross-core ring. Blocked:
3 transposes (conv-in, scan-in, RMSNorm reduce). Single-step decode (seq=1) is
well below the 1 MB crossover -> net-negative; the bandwidth win needs batch B≥64
(in-proj) / B≥256 (scan). Prefill clears crossover easily but matmul dwarfs it.
Sweet spot: batched decode at d_model 2k-4k.

## 5. Simplifications

- SSD scan = single-chunk; inter-chunk recurrence dropped (handoffs intact).
- dt/A folded into decay; conv1d via F.conv1d (MAMBA_SKIP_CONV=1 if blocked).
- Anchor MEASURED on (a+b.t+c.t)@d; Mamba edges PROJECTED, not yet device-spliced.
