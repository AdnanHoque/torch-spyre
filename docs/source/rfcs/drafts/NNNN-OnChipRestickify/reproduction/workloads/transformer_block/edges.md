# On-chip-eligible activation handoffs in the transformer block (OFFLINE analytic)

Identification of producer->consumer activation handoffs in
`transformer_block_workload.py` that are eligible for the proven same-stick
core-to-core `STCDPOpLx` primitive, vs blocked (layout-changing) vs prelayout.

**No device, no `torch.compile`, no dxp run.** This is derived analytically from
(1) the block's op structure, (2) the granite reference bundle
`/tmp/granite_inductor/.../sdsc_fused_add_linear_mul_rms_norm_6_m56h1rzb` (whose
RMSNorm+linear sub-structure is identical to this block's), and (3) Agent C's
per-edge classification method in `/tmp/real_edge_analysis.md`. The reusable
parser `edge_classifier.py` (built on `/tmp/edge_analyze.py`) lets the
orchestrator **confirm** every row below on the real compiled bundle.

> **How to confirm on the real bundle (orchestrator step):**
>
> ```
> PYTHONPATH=/tmp /home/adnan/dt-inductor/.venv/bin/python \
>     edge_classifier.py <compiled_block_bundle_dir>
> ```
>
> It reproduces the granite reference classification exactly (verified offline:
> 7 same-stick same-shard, 2 cross-core ring, 5 layout-changing -- matching
> `/tmp/real_edge_analysis.md`'s granite row of 6 same-core + 2 cross-core,
> modulo one address-aliased double-count flagged below).

## Method note (inherited, load-bearing)

- Edges are traced by matching a producer OUTPUT's per-core HBM base to a
  consumer INPUT's HBM base via the `scheduleTree_` allocate node
  (`startAddressCoreCorelet_.data_["[0, 0, 0]"]`), **not** `hbmStartAddress_`
  (absent from cached SDSCs -- it is materialized only post-dxp). Latest-prior
  producer wins (handles buffer reuse).
- Stick orientation: `primaryDsInfo_[role].stickDimOrder_`. Sharding: SuperDSC
  `numWkSlicesPerDim_`.
- **Same-stick** (stickDimOrder identical on both endpoints) => `STCDPOpLx`
  applies. **Layout-changing** (stick dim flips) => needs the
  `ReStickifyOpWithPTLx` transpose, which faults Compute-CB on device (BLOCKED).
- Among same-stick: **same-shard** => degenerate same-core LX->LX copy (HBM-elim
  only, no ring); **diff-shard** => genuine cross-core RIU-ring move.

## Block op structure -> SDSC bundle (analytic)

The block is:

```
h = x + Attn(RMSNorm(x));   y = h + MLP(RMSNorm(h))
```

After Spyre fusion the bundle is a sequence of fused SDSCs of these shapes
(grounded in the granite cache, which contains the *same* RMSNorm+linear and
fused-SDPA building blocks):

| Stage | Fused SDSC group (op family) | Spyre kernel evidence |
|---|---|---|
| RMSNorm(x) | `mul/mean/add/rsqrt/mul` (RMSNorm tail) + leading `linear` | granite `sdsc_*_add_linear_mul_rms_norm` |
| Q/K/V proj | `linear` (= `batchmatmul`) x3, each via a weight `ReStickifyOpHBM` | granite `[0]restickify->[1]bmm`, `[3]restickify->[4]bmm` |
| SDPA | **one** fused `_scaled_dot_product_fused_attention_overrideable` kernel | granite `sdsc_fused__scaled_dot_product_..._0_*` |
| O proj | `linear` | granite bmm |
| residual add | `add` | granite `[6]add` |
| RMSNorm(h) | RMSNorm tail again | granite RMSNorm tail |
| MLP gate/up/down | `linear` x3 + `silu` + `mul` (SwiGLU) | granite `..._silu_*` cache |
| residual add | `add` | granite `[6]add` |

**Crucial scope point (flagged inference):** the attention-score ->
softmax handoff (`bmm(QK^T) -> max/sub/exp`) -- the single best cross-core target
in `/tmp/real_edge_analysis.md` -- lives **inside** the fused SDPA kernel in this
workload (PyTorch's `scaled_dot_product_attention` lowers to one fused Spyre
attention SDSC). It is therefore an *intra-SDSC* edge here, **not** a
cross-SDSC HBM handoff, so the block-level A/B does not capture it. It only
becomes an addressable HBM handoff when SDPA is *not* fused (the manual
`q@k.T -> softmax -> @v` form, as in `test__simple_attn`). I note this so the
projection does not double-count the attention-score saving.

## Per-handoff eligibility list (analytic, at hidden=2048, n_heads=16, seq=512)

Handoff tensor sizes are computed from the workload shape (fp16, 2 B/elem):

- hidden activation `[B=1, S=512, H=2048]` = **2.000 MB**
- MLP intermediate `[B, S=512, I=5504]`   = **5.375 MB**
- per-row norm reduction scratch `[B, S]` = tiny (< 1 stick/core)

| # | Producer -> Consumer (block-level) | prod stick | cons stick | same-stick? | shard delta | bucket | ring? | handoff size |
|---|---|---|---|---|---|---|---|---|
| E1 | RMSNorm(x) tail `mul` -> Q/K/V `linear` input | `out` | `in` | **no** | resplit `out`->`in` | layout-changing (BLOCKED) | - | 2.000 MB |
| E2 | weight `ReStickifyOpHBM` -> Q/K/V/O/gate/up/down `linear` (KERNEL) | `mb` | `out` | no | weight | prelayout/marker | - | (weight, n/a) |
| E3 | Q/K/V `linear` out -> SDPA input | `out` | `out` | **YES** | bmm`{out,in}`->attn`{mb}` | same-stick **diff-shard** (cross-core RING) | YES | 2.000 MB (per Q,K,V) |
| E4 | SDPA out -> O `linear` input | `out` | `in` | **no** | attn`{x}`->bmm`{in}` | layout-changing (BLOCKED) | - | 2.000 MB |
| E5 | O `linear` out -> residual `add` | `out` | `out` | **YES** | bmm`{out,in}`->add`{mb}` | same-stick **diff-shard** (cross-core RING) | YES | 2.000 MB |
| E6 | residual `add` out -> RMSNorm(h) `mul` | `out` | `out` | **YES** | `{mb}`==`{mb}` | same-stick **same-shard** (same-core) | no | 2.000 MB |
| E7 | RMSNorm(h) internal `mul->mean->add->rsqrt` chain | `out`/`x` | `out`/`x` | **YES** (mostly) | `{mb}`==`{mb}` | same-stick **same-shard** (same-core) | no | ~2.0 MB + tiny scratch |
| E8 | RMSNorm(h) `mean` out -> reduction `add` | `out` | `x` | **no** | `{mb}`->`{out}` | layout-changing (BLOCKED, small) | - | tiny (`[B,S]`) |
| E9 | RMSNorm(h) `rsqrt` out -> final `mul` | `x` | `out` | **no** | `{out}`->`{mb}` | layout-changing (BLOCKED, small) | - | tiny |
| E10 | RMSNorm(h) tail `mul` -> gate/up `linear` input | `out` | `in` | **no** | resplit `out`->`in` | layout-changing (BLOCKED) | - | 2.000 MB |
| E11 | gate `linear` out -> `silu`/`mul` | `out` | `out` | **YES** | bmm`{out,in}`->mul`{mb}` | same-stick **diff-shard** (cross-core RING) | YES | 5.375 MB |
| E12 | up `linear` out -> SwiGLU `mul` | `out` | `out` | **YES** | bmm`{out,in}`->mul`{mb}` | same-stick **diff-shard** (cross-core RING) | YES | 5.375 MB |
| E13 | SwiGLU `mul` out -> down `linear` input | `out` | `in` | **no** | resplit `out`->`in` | layout-changing (BLOCKED) | - | 5.375 MB |
| E14 | down `linear` out -> residual `add` | `out` | `out` | **YES** | bmm`{out,in}`->add`{mb}` | same-stick **diff-shard** (cross-core RING) | YES | 2.000 MB |

(E3 "per Q,K,V" = three structurally identical edges, one per projection.)

## Eligibility summary (this block, analytic)

| Bucket | Edges | Status |
|---|---|---|
| **same-stick same-shard** (same-core LX->LX, HBM-elim, no ring) | E6, E7 (+ granite shows ~6-7 such per RMSNorm tail) | addressable today, simplest win |
| **same-stick diff-shard** (genuine cross-core RIU ring) | E3 (x3), E5, E11, E12, E14 | addressable today (proven primitive), needs per-size LX |
| **layout-changing** (needs transpose) | E1, E4, E8, E9, E10, E13 | BLOCKED (Compute-CB fault) |
| **prelayout / weight** | E2 (all 7 `linear` weight restickifies) | prelayout RFC, no runtime primitive |

**Addressable-today same-stick handoffs (the A/B target set):** E3(x3), E5, E11,
E12, E14 cross-core RING + E6, E7 same-core. The big-byte ones on the critical
path are the **MLP** edges E11/E12 (5.375 MB each) and the linear-output ->
elementwise/residual edges E5/E14 (2.0 MB). These are exactly the granite
`[4]batchmatmul -> [5]mul` class proven cross-core in `/tmp/real_edge_analysis.md`.

**Blocked (honest):** every `activation -> linear-input` edge (E1, E4, E10, E13)
is layout-changing -- the stick flips `out`->`in` entering the matmul's contracted
axis. These are high-byte and on the critical path but need the transpose
primitive that faults Compute-CB today. The RMSNorm reduction flips (E8, E9) are
also layout-changing but move negligible bytes.

## Confirming on the real bundle

The granite reference already validates the parser end-to-end (run above). When
the orchestrator compiles this block, `edge_classifier.py <bundle>` will:

1. list the actual fused SDSC order (confirming whether SDPA fused as assumed),
2. emit each traced handoff with measured per-core HBM base + size, and
3. bucket-count them, which should land in the ranges above (the exact per-stage
   split depends on how inductor fuses the RMSNorm tail and linears).

### Known parser caveats (carried from `/tmp/real_edge_analysis.md`)

- **Address-aliased double-count.** A consumer that reads the same buffer as both
  operands (e.g. `mul(z, z)`) is reported twice (seen on granite `[6]add->[7]mul`).
  De-dup on `(prod_i, cons_i, addr)` when counting.
- **`size=?` on cached SDSCs.** The cache labeledDs `dimToLayoutSize_` is null at
  the DL level, so the parser prints `size=?`; the sizes in the table above are
  computed from the workload shape instead. A freshly compiled bundle may carry
  concrete sizes; if not, fall back to the shape formula.
- **Graph-segment markers** (addresses that are exact 16-GiB multiples) are graph
  inputs/weights, bucketed prelayout, not intra-block activations.
