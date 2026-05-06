# Hardware calibration of the FA prototype — findings

## TL;DR

Real attention wall on AIU is **10-15× slower than the FA prototype
predicted**. The headline reason is structural, not a tunable
parameter: **AIU's torch_spyre attention path materializes the full
M×M attention matrix and does not appear to use FA-style tiling**,
even via `torch.nn.functional.scaled_dot_product_attention`.

This means the joint SWP+WS proposal's per-op speedup numbers
(1.36× to 1.83×) apply to **a workload AIU isn't running today**. The
project would actually require shipping FA-tiled attention as a
prerequisite before joint scheduling could deliver its predicted wins.

## What we measured

Two attention forms compared on AIU at fp16, varying (n_heads, M, d):

- **bmm-form**: explicit `torch.bmm(Q, K.T) → softmax → torch.bmm(P, V)`.
  Materializes the M×M attention matrix.
- **SDPA-form**: `torch.nn.functional.scaled_dot_product_attention`.
  Routes through `torch.ops.aten._scaled_dot_product_fused_attention_overrideable`,
  which torch_spyre overrides via `spyre__sdpa_overrideable` in
  `decompositions.py:507`.

Results:

| n_heads | M | head_dim | bmm-form ms | SDPA-form ms | ratio |
|---:|---:|---:|---:|---:|---:|
| 8 | 256 | 128 | 9.93 | 14.91 | 0.67× |
| 8 | 512 | 128 | 10.54 | 15.71 | 0.67× |
| 8 | 1024 | 128 | 12.47 | 17.04 | 0.73× |
| 32 | 512 | 128 | 13.17 | 18.85 | 0.70× |
| 32 | 1024 | 128 | 21.41 | 27.95 | 0.77× |
| 64 | 1024 | 128 | 32.51 | 45.88 | 0.71× |

**SDPA is consistently slower than bmm** by ~30-50%. That's the opposite
of what FA-tiling would deliver. **It strongly suggests SDPA on AIU
isn't FA-tiled at all** — the spyre override decomposes back to bmm-like
ops with similar overhead.

## How prototype predictions compare

For Llama 70B-style shape (64 heads × M=2048 × d=128) the prototype
predicted decoupled wall = **5.79 ms** at SFP=3000 cyc/tile.

Extrapolating measured walls to M=2048:
- 64 heads × M=1024 measured = 32.5 ms (bmm-form)
- M=2048 = 4× the M·M work → ~120-130 ms scaled
- Prototype prediction at M=2048: 5.79 ms

**Ratio: real wall is ~22× slower than prototype prediction.**

This isn't a "cycle counts off by 30%" miss. It's structural — the
prototype assumed FA tiling and the real workload isn't tiled.

## Why the prototype assumed FA tiling

The Twill paper (and this proposal generalizing it) is built around the
assumption that the workload IS FA-tiled, because that's what makes
joint SWP+WS valuable: the per-iteration ping-pong of PT and SFP only
makes sense in a tiled inner loop.

If the workload materializes the full M×M matrix:
1. Bottleneck is HMI (writing/reading the 64M-element matrix per layer).
2. PT and SFP run sequentially within the matrix-shaped op, with HMI
   between.
3. There's no per-tile inner loop to ping-pong over.

The joint SWP+WS optimization simply doesn't apply to non-tiled
attention. Prerequisite: ship FA tiling first.

## What this means for the project

The honest verdict updates substantially:

### Updated project chain
1. **Phase 0.A (codebase analysis)**: scheduler IS decoupled. ✓
2. **Phase 0.B (generic ILP)**: tractable with horizon decomposition,
   7-9% gain on compute-balanced shapes. ✓
3. **Phase 0.C (FA prototype)**: 1.36-1.83× per-op speedup — but
   **predicated on FA tiling that doesn't exist in production today**.
4. **Phase 0.D (end-to-end block)**: 1-5% block savings — but again,
   contingent on FA tiling.
5. **Phase 0.E (this calibration)**: AIU runs full-attention bmm-form,
   not FA-tiled. The prototype's predictions are 10-15× off because
   they model a different algorithm.

### Revised verdict

**The proposal as scoped (joint SWP+WS only) doesn't deliver wins on
AIU's current attention path** because the workload doesn't have the
right tile structure. The proposal would need to be **expanded to
include FA tiling itself as a prerequisite** — that's a much bigger
project than 12-16 weeks of joint scheduling work.

Two paths forward:

1. **Re-scope to "FA tiling + joint scheduling for AIU"**. Now a much
   larger project (~6 months). The win is also bigger because FA
   tiling alone would deliver 2-3× speedup on attention compute (FA
   typically saves 2-4× over materialized attention).

2. **Pivot the joint scheduling project to a different workload**
   that already has FA-style structure on AIU. Hard to identify what
   that is — need to find an op the AIU compiler already tiles with
   per-iter PT/SFP overlap potential. May not exist.

3. **Close the project**. The premise (joint scheduling has wins on
   today's AIU workloads) doesn't hold without FA tiling.

## Even bigger finding: SDPA on AIU is slower than bmm

Independently of the joint scheduling question, the calibration
revealed that **AIU's `torch.nn.functional.scaled_dot_product_attention`
is 30-50% slower than the manual bmm form**. That's surprising and
worth raising independently — typically SDPA fuses operations and
should be faster, not slower.

Possible causes:
- The spyre override (`spyre__sdpa_overrideable`) might decompose
  inefficiently
- Padding, masking, or causal-mask handling adds overhead even when
  not used
- The fused-attention path generates worse SDSC than three explicit
  ops

**This is a separate, easier project**: investigate why SDPA is slower
than bmm on AIU and either fix the override or document the finding so
production code uses bmm.

## Methodological lesson

I should have done this calibration in Phase 0 BEFORE running the
generic-matmul, FA-prototype, and end-to-end analyses. The cycle-count
estimates I used were derived from cost-model intuitions and never
grounded in real measurement. **This is the kind of failure mode the
"verify before commit" pattern is supposed to catch.**

Two cheap signals would have caught this earlier:
1. Run real attention on AIU at one shape (this calibration). Would
   have shown the prototype's predicted wall is 10-15× off.
2. Re-examine existing diag-branch measurements. The k_fast
   popular-models sweep had attention-related shapes whose walls
   were 30-50% off from the cost model — but I didn't audit those
   for FA-relevance specifically.

Both are <1 hour of work and gate the project's premise. Including
them in any future Phase 0 template.

## Note on the calibration's environment

The compile path on the joint scheduling branch (off upstream/main)
crashes with "File does not end with .json or .cbor" in the kernel
deserializer. Tracked to upstream commit `ba9274d "Compute nograph"`
which appears to break compatibility with the local sendnn runtime.
Calibration was run on the `AdnanHoque/hmi-cost-model-simulator`
branch (which is off the diag base, pre-`ba9274d`) where compile
works correctly, then results were copied back.

## Files

- `joint_swp_ws_attn_calibration.py` — benchmark script (bmm vs SDPA)
- `joint_swp_ws_attn_calibration_results.txt` — raw measurements
- This doc — calibration findings
