# Softmax-chain LX-persistence pass — evidence record

Self-contained pure-persistence on-chip pass
(`torch_spyre/_inductor/onchip_softmax_chain.py`), gated by
`config.onchip_softmax_chain` (`SPYRE_ONCHIP_SOFTMAX_CHAIN=1`, default off).

## Mechanism

Same-shard same-core chain detection: walks the realized SDSC bundle and, for
each HBM-backed producer output, finds every later SDSC input with an
**identical `numWkSlicesPerDim_` split** and **byte-identical per-core HBM bases**
across all 32 cores (every core reads exactly the slot it wrote). Each such
producer-output buffer is one chain intermediate (one producer write, one or
more consumer reads). For each intermediate it flips the producer output and
every consumer input LX-resident at one coordinated per-core base, assigned by a
liveness-aware first-fit packer over the usable LX window (~1.5 MB / core: lower
1.5 MB of the 2 MB per-core LX; the upper 512 KB is reserved for each op's
auto-assigned working buffers). A buffer that does not fit stays HBM-backed and
its edge is left untouched (fail-closed, correct but not accelerated). Pure
persistence: no data-op, no mixed SDSC, stock dxp. When the flag is off, output
is byte-identical to before.

## Measurement

- **1.88×** end-to-end on production SDPA, seq512 (2.557 → 1.358 ms).
- Re-confirmed **1.86×** in an isolated re-measurement (2.458 → 1.322 ms).
- **Value-correct** on standard `torch.randn` fp16, tight max_err (~7.6e-5).
- Measured on the **materialized-scores SDPA decomposition** — today's Spyre
  main (`q*s, k*s, matmul, softmax → max/sub/exp/sum/realdiv, matmul`), **NOT**
  the flash form.

## Faithfulness check (compile + JSON inspection; no device, no C++ build)

Compiled the materialized-scores SDPA (B1/H32/M512/D128) at seq512 and seq2048
on the current main `.venv` (torch 2.11, materialized decomp) with cache-bust per
run, then ran this self-contained pass over the emitted SDSCs:

- **(a) Fires:** YES at both sizes — detects the softmax-tail same-core chain
  (`max → sub → exp → sum/realdiv`), 4 intermediates each.
- **(b) Flips the expected edges:** every flipped labeledDs has
  `hbmStartAddress_ = -1`, `memOrg_.lx.isPresent = 1`, allocate
  `component_ = "lx"`, and all 32 per-core bases equal to the coordinated LX
  base. Placements are in-budget and have no live-overlapping regions.
  - seq512: all 4 intermediates placed, **9 endpoints flipped** (incl. the
    `exp → {sum, realdiv}` two-consumer fan-out, and `sum`'s region reusing the
    base freed by the dead `max/sub` region — liveness-aware reuse exercised).
  - seq2048: the 2 MB scores/exp tensors exceed the 1.5 MB usable window, so
    those 2 intermediates correctly fail-closed to HBM; the two small
    max/sub and sum/realdiv vector edges flip (**4 endpoints**). The pass is
    structurally valid; the full softmax-tail LX win is partial at seq2048
    because the scores/exp tensors do not fit the usable window.
- **(c) Byte-identical to the original (935fd62):** ran the original
  attention-overlap pass logic (genuine branch helpers + genuine 935fd62
  detect/plan/apply bodies) over the same SDSCs; the post-pass bundles are
  **byte-identical** at both seq512 and seq2048. The extraction preserves
  behavior exactly. (The executable detect/plan/apply/realize bodies are
  verbatim from 935fd62; only docstrings were trimmed to drop references to the
  cross-shard `onchip_realize` module that does not exist on `main`.)

This module imports **nothing** from `onchip_realize` / `onchip_bridge` (those
modules do not exist on `main`); the four pure-Python dict-surgery helpers
(`LxFlip`, `_dl_op`, `_core_state_init_entry`, `apply_lx_flip` — the safe shared
base-pointer flip) and the `LX_CAPACITY_BYTES` constant are inlined verbatim.

## Enable

`SPYRE_ONCHIP_SOFTMAX_CHAIN=1` (default off → byte-identical output).

## E2E projection (Granite 3.3 8B, dense) — PROJECTION, not measured

The win is **sequence-length-dependent**, because the per-core scores/exp tensor
grows with prefill length and the pass only persists what fits the ~1.5 MB usable
LX window (it fail-closes the rest to HBM):

- **Short prefill** (scores/exp fit LX — at this shape, seq ≲ ~768): the full
  softmax-tail chain flips → ~1.88× attention kernel → **~10–15% e2e prefill**,
  via an Amdahl argument on attention's estimated ~25–30% of prefill time-share
  (attention is ~8% of FLOPs but runs at ~4% of PE peak / memory-bound, so it
  takes a disproportionate share of wall time).
- **Long prefill** (seq ≳ 2048): the scores/exp tensors exceed the LX window and
  fall back to HBM; only the small `max/sub` and `sum/realdiv` vector edges
  persist → much smaller kernel win (est. ~1.05–1.2×, **unmeasured**) →
  **~2–4% e2e**. The 1.88× headline does **not** carry to long context as-is.
- Recovering the large win at long context would require **tiling/streaming** the
  persisted scores chain (keep LX-resident tiles rather than the whole tensor) —
  a follow-on, not part of this pass.

**These are projections** — attention's full-layer time-share is estimated, not
measured, and the long-prefill kernel win is not yet measured on device.

## What this is NOT

- **Not the flash form** (PR2363): rebasing this onto the flash decomp is
  **value-incorrect** (device max_err 2.2–2.7 on standard `torch.randn`, ~3
  orders above the fp16 noise floor) at seq512 and seq2048, MHA and GQA.
- **Not warp-spec overlap** (break-even, not a win).
- **Not the asymmetric cross-shard handoff** (the corrupt 2.14× path).

This is the clean **SPMD elimination** pass: same-shard same-core persistence,
value-correct, separated from the broken machinery.

## Open items

- Confirm the **1.88×** holds at seq2048 (it was measured at seq512; at seq2048
  the 2 MB scores/exp tensors do not fit the 1.5 MB usable window and fall back
  to HBM, so the win there is expected to be smaller — needs device
  measurement).
- Full Granite-3.3-8B-layer prefill profile to firm up the e2e % (currently a
  projection), at the actual target prefill length.
- Tiling/streaming the persisted scores chain to recover the full win at long
  context (seq ≳ 2048), where the whole-tensor flip currently fail-closes.
- Merge path.
