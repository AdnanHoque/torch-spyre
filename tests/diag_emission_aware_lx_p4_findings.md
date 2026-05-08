# Probe 4 findings — n=1 triggers a streaming-output fast path

Companion to `diag_emission_aware_lx_p4_streaming_path.py`. Purpose:
discriminate two hypotheses for why DSv3 o_proj M=2048 (1, 1, 32)+kf
runs at 30 ms despite C_psum = 56 MB (28× LX overage) — well outside
the catastrophic regime that consumes (1, n, k>1) splits at much
smaller overage.

## TL;DR

**H2 confirmed: `n = 1` triggers a streaming-output fast path** in
the kernel template. The fast path absorbs C_psum overage that is
catastrophic at n > 1.

The smoking gun (same shape, DSv3 o_proj M=2048):

| split | n | C_psum overage | wall ms |
|---|---:|---:|---:|
| (8, 1, 4)+kf | 1 | 3.50× | **18.14** |
| (1, 8, 4)+kf | 8 | 3.50× | **125.04** |

Same C_psum, same chain length, same emission, same shape. The only
difference is whether the work-division split divides N or M. **n=1
runs 7× faster.**

This is a real, production-actionable mechanism. The cost model and
planner both need to know about it.

## Full data

DSv3 o_proj M=2048 (2048, 7168, 16384), kf emission unless noted:

| split | description | C_psum/core | overage | wall ms |
|---|---|---:|---:|---:|
| (32, 1, 1) identity | pure-M baseline | 1.75 MB | 0.88× | 13.29 |
| (16, 1, 2)+kf | n=1, chain=2 | 3.50 MB | 1.75× | 18.57 |
| (8, 1, 4)+kf | n=1, chain=4 | 7.00 MB | 3.50× | 18.14 |
| (4, 1, 8)+kf | n=1, chain=8 | 14.00 MB | 7.00× | 56.93 |
| (2, 1, 16)+kf | n=1, chain=16 | 28.00 MB | 14.00× | 59.07 |
| (1, 1, 32)+kf | pure-K, single chain | 56.00 MB | 28.00× | 30.38 |
| **(1, 8, 4)+kf control** | n=8, chain=4 | 7.00 MB | 3.50× | **125.04** |

## Mechanism (best inference)

The kernel template emits a streaming-output path when `n = 1`. Under
this path:

- The output of each chain is a single column tile, M_per × N
  elements wide.
- The chain head doesn't need to hold the full output resident; it
  can write each accumulated tile to HMI as the chain reduces past
  it.
- This bypasses the per-core LX residency requirement for the PSUM
  accumulator that breaks (1, n>1, k>1) splits.

This is consistent with the AIU's hardware: the data ring and HMI
write-back path can absorb sustained chain-head output if the chain
head is producing one continuous tile. With n > 1, the chain head
holds multiple output tiles (one per N-slice), and the kernel
template can't stream them out independently because they're
interleaved in K-loop iteration order.

## Secondary observation — n=1 family is not perfectly flat

The (m, 1, k) walls are not constant across m:

| split | m | k | wall |
|---|---:|---:|---:|
| (32, 1, 1) | 32 | 1 | 13.3 |
| (16, 1, 2)+kf | 16 | 2 | 18.6 |
| (8, 1, 4)+kf | 8 | 4 | 18.1 |
| (4, 1, 8)+kf | 4 | 8 | **56.9** |
| (2, 1, 16)+kf | 2 | 16 | **59.1** |
| (1, 1, 32)+kf | 1 | 32 | 30.4 |

There's a hump at chain length 8-16 (walls 3× the chain=2-4 region)
that drops back at chain=32. The streaming-output path likely has
two regimes:

- **Pipeline regime** (chain ≤ 4): the path absorbs C_psum overage
  for free; wall scales with compute + HMI.
- **Sync regime** (chain 8-16): the streaming pipeline can't keep
  up with the longer chain; some serialisation cost shows up.
- **Allreduce regime** (chain = 32, single chain): a separate
  fast path likely activates for "all cores in one chain", probably
  using a tree-reduction primitive in the SFP ring.

This three-regime structure is informative but not yet fully
characterised. The probe data is consistent with it; we'd need
more shapes / chain-length combinations to confirm.

## What's actionable

### For the cost model

- **Suppress the C_psum overflow penalty when n = 1.** The
  streaming-output path absorbs overage at low chain lengths.
- **Add a chain-length-dependent factor for n = 1, m > 1**: roughly,
  +30 ms wall when 4 ≤ chain ≤ 16 on wide-N shapes (calibrated on
  this single shape; needs more data).
- **Pure-K (1, 1, 32) is its own regime**: probably model it
  separately, or treat it as a special case the planner can
  consider for wide-N shapes.

### For the planner

The planner today picks pure-M (32, 1, 1) for nearly every matmul.
For wide-N prefill shapes where pure-M itself has C_psum > LX
(14 of 120 production matmuls per the LX-Phase-1 diagnostic), the
planner currently has no good option in its search space.

After this finding, the planner can also consider:

- `(m, 1, k>1) + kf` with chain length 2 or 4 — the streaming-output
  fast path. C_psum overage doesn't matter here.
- `(1, 1, 32) + kf` — pure-K all-reduce path for shapes where the
  full-output single-chain approach wins.

For DSv3 o_proj M=2048 specifically, pure-M (13 ms) still wins
against the best n=1 alternative (18 ms). But for shapes where pure-M
is *not* a winner — for example, MLP gate/up at M=2048 on Llama 70B
— the (m, 1, k) family becomes a real option that hasn't been
explored.

### For research narrative

This is the "novel finding" the project was looking for, just not
through the mechanism we hypothesised:

> **The work-division planner's split space has a structural
> asymmetry between dividing the M dimension and dividing the N
> dimension, mediated by the kernel template's streaming-output
> behaviour. The asymmetry is invisible in compute-cost models,
> hidden in operand-residency models, and only visible to a model
> that includes per-PE accumulator residency conditional on split
> structure.**

To my knowledge, no published auto-scheduler (Roller, Ansor, AKG,
TVM) models per-PE accumulator residency conditional on
output-dimension splits. Per-PE *operand* residency is sometimes
modelled; per-PE *accumulator* residency at all is rare; conditional
on which dim is split, I can't find prior art.

This is publishable research. The contribution is concrete:
characterise the n=1 streaming-output fast path on AIU 1.0, show
the implied planner extension, and quantify the wins on shapes
where pure-M doesn't win.

## Next steps

Concrete tractable work:

1. **Probe 5: extend the n=1 family across more shapes.** Run
   (m, 1, k)+kf on L3-70B gate_proj M=2048 (a wide-N shape where
   pure-M overflows C_psum), DSv3 down_proj M=2048, and a few
   more candidates. Goal: confirm n=1 fast path is general, not
   o_proj-specific.

2. **Probe 6: characterise the chain-length hump.** Sweep more
   combinations of (m, 1, k) to map the wall surface — is the
   wall ≈ const + chain-len-dependent term?

3. **Cost-model fix D**: incorporate the n=1 streaming-output path
   into hmi_cost_model.predict. Compare to validation set rows.

4. **Planner integration**: add (m, 1, k)+kf to the candidate search
   for shapes where pure-M C_psum > LX. Quantify production wins.

The lever is real. The next milestone is to find the production
shapes where it pays out.
