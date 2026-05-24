# On-Chip Core-to-Core Projection for SDPA / Flash Attention (Spyre AIU)

Offline projection (2026-05-24). No device run. Projects the on-chip (LX->ring->LX)
speedup for the attention block's eligible activation handoffs, anchored on the
device-measured same-stick STCDP numbers and applied to the SDPA QK^T -> softmax
edge spliced and offline-verified by `splice_attention_qk_softmax.py`.

## Empirical anchor (MEASURED on device)

From the proven `(a + b.t() + c.t()) @ d` cross-core round trip
(`CoreToCoreDataMovementRecipe.md` section 9; `bench_onchip_*.txt`):

- **~0.029 ms saved per MB** of eliminated same-stick handoff tensor (one HBM
  round trip removed). [MEASURED anchor]
- **~0.005 ms fixed STCDP setup** cost per spliced edge. [MEASURED anchor]
- **Net-positive above ~1 MB** of handoff tensor; below that the setup dominates
  and the edge should stay on HBM (the size-crossover seen at S=512 -> 0.95x).
- Relative speedup measured 0.95x (512) / **1.22x (1024)** / 1.19x (2048) /
  1.13x (4096): it peaks mid-range and tapers as the matmul O(N^3) dwarfs the
  O(N^2) handoff. [MEASURED]

Per-edge net saving model (PROJECTED from the anchor):
`net_ms = 0.029 * handoff_MB - 0.005`.

## The targeted edge and its handoff size

**Edge:** SDPA `batchmatmul(QK^T)[4].OUTPUT -> softmax(sub)[6].INPUT`, stick
`['out']` on both endpoints (SAME-STICK), producer shards `{mb:32}`, consumer
shards `{x:32}` (genuine cross-core ring). Verified STCDP-eligible and the ring
signature was reproduced offline (all 32 cores emit `L3_LDU`/`L3_STU` to mirror
core `31-i`). The handoff tensor is the **attention score matrix**
`[B*H, seq_q, seq_k]` in fp16 (2 B/elem):

```
handoff_MB = (B*H) * seq_q * seq_k * 2 / 2^20
```

Each attention layer writes this matrix to HBM from the QK^T matmul and reads it
straight back into the softmax max/sub. Eliminating that round trip is exactly
one `net_ms` saving, per layer, per forward.

## Projected savings (PROJECTED from the MEASURED anchor)

| config | score handoff (MB) | net_ms / layer | status |
|---|---|---|---|
| cached bundle anchor (B*H=32, sq=sk=64, hd=128) | 0.250 | +0.0023 | marginal (sub-MB) |
| decode (B=1, H=32, sq=1, sk=2048) | 0.125 | -0.0014 | gate to HBM (sub-MB) |
| decode (B=1, H=32, sq=1, sk=4096) | 0.250 | +0.0023 | marginal (sub-MB) |
| prefill (B=1, H=32, sq=sk=512) | 16.0 | +0.459 | NET-POSITIVE |
| MoE-ish (B*H=256, sq=sk=128) | 8.0 | +0.227 | NET-POSITIVE |
| prefill (B=1, H=32, sq=sk=2048) | 256.0 | +7.42 | NET-POSITIVE |
| Llama-7B prefill (H=32, sq=sk=4096) | 1024.0 | +29.69 | NET-POSITIVE |

Whole-model: multiply `net_ms / layer` by the layer count (e.g. 32 for a 7B-class
model) -- the QK^T->softmax handoff recurs in every attention layer of every
roadmap model (Llama/Mistral/Granite/GPT-OSS).

### Reading the table -- two regimes, opposite to the naive intuition

The score matrix grows **O(seq^2)**, so the handoff size is the inverse of the
usual "decode is bandwidth-bound" framing for *activations*:

- **Prefill (long seq):** the score matrix is large (16 MB at seq=512, 1 GB at
  seq=4096), so the QK^T->softmax handoff is squarely net-positive and the
  absolute saving is large. This is where the attention-score edge pays off most.
- **Decode (seq_q=1):** the score row is tiny (`[B*H, 1, seq_k]`, 0.125-0.25 MB),
  below the ~1 MB crossover -> this *specific* edge should stay on HBM in decode.
  (Decode's bandwidth pressure is on the KV-cache and the projection GEMMs, not on
  the score handoff -- a different set of edges.)
- **MoE / batched attention:** large `B*H` lifts the score size back over 1 MB
  even at short seq, restoring net-positive.

Caveat (consistent with the anchor): even where net-positive, the *relative*
end-to-end speedup tapers in compute-bound prefill because the two attention
matmuls grow O(seq^2 * head_dim) while the handoff grows O(seq^2). The `net_ms`
above is the correct *absolute* per-edge saving; end-to-end percentage must weight
it by the layer's total time.

## Which attention edges are addressable vs blocked

From `real_edge_analysis.md` (the SDPA bundle, traced offline):

**Same-stick -> STCDP-eligible TODAY (this projection applies):**

| edge | stick | shard change | type |
|---|---|---|---|
| `bmm(QK^T)[4] -> max[5]` | out | {mb:32}->{x:32} | cross-core ring (TARGETED) |
| `bmm(QK^T)[4] -> sub[6]` | out | {mb:32}->{x:32} | cross-core ring (SPLICED + verified) |
| `bmm(PV)[10] -> identity[11]` | out | {mb:32}->{x:32} | cross-core ring |
| `max[5] -> sub[6]` | out | same-shard | same-core (HBM-elim only, no ring) |
| `sub[6] -> exp[7]`, `exp -> sum/realdiv`, `sum -> realdiv` | out | same-shard | same-core (HBM-elim only) |
| `identity[0] -> mul[1]`, `mul[2] -> restickify[3]` | out / x | (see analysis) | same-stick |

The whole softmax interior (max/sub/exp/sum/div) is same-stick `['out']`; most of
it is same-shard (degenerate same-core LX copy = HBM-elimination, no ring), while
the three **matmul-output -> elementwise/softmax** edges are the genuine cross-core
ring cases. The targeted `QK^T -> softmax` edge is the heaviest of these.

**Layout-changing -> BLOCKED (needs the Compute-CB-faulting transpose):**

| edge | stick change | why blocked |
|---|---|---|
| `mul(Q-scale)[1] -> bmm(QK^T)[4].in` | out -> in | stick flips for the matmul contraction axis |
| `realdiv(softmax)[9] -> bmm(PV)[10].in` | out -> in | stick flips for the PV matmul contraction axis |

These are the **Q/K/V / score-into-matmul transposes**: the stick dim flips
(`out -> in`) entering a matmul's contracted axis, requiring `ReStickifyOpWithPTLx`,
which currently faults Compute-CB on device (recipe section 11a). The Q/K/V
prelayout `ReStickifyOpHBM[3]` is a weight/activation restickify in the
prelayout/transpose bucket, not addressable by the same-stick STCDP.

## Bottom line

The single heaviest cross-core same-stick attention edge -- QK^T -> softmax -- is
spliced and offline-verified (cross-core ring signature, HBM-free, dxp exit 0).
Projected per-layer saving is net-positive for prefill and batched/MoE attention
(seq or B*H large enough to push the score matrix over ~1 MB), and correctly gates
to HBM for single-token decode where the score row is sub-MB. The two attention
edges that flip the stick (into the QK^T and PV matmuls) remain blocked on the
deeptools/hardware transpose primitive.

*MEASURED = device-measured anchor from the `(a+b.t+c.t)@d` proof. PROJECTED =
derived by applying that anchor's per-MB rate to the attention score-matrix sizes;
not independently measured on attention. INFERENCE: the per-MB rate is assumed
shape-independent for same-stick STCDP, which the anchor supports across
512-4096 but was not measured on the attention bundle itself.*
