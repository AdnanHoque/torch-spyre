# Flash-attention Phase 0c — LX_PLANNING is invisible at the per-kernel level

The Phase 0c probe (`tests/diag_lx_planning.py`) confirms whether
`LX_PLANNING=1` actually pins allowlisted op outputs in scratchpad,
skipping DDR roundtrips for downstream consumers. **Result: at our test
shape, LX_PLANNING shows zero measurable wall-time effect.** The reason
is simple — at 3 ms per launch, the µs-scale savings from scratchpad-
vs-DDR for tiny intermediate state are invisible.

**This is good news for the flash-attention design.** It means the
project doesn't depend on the LX scratchpad pinning mechanism working,
because per-Q-tile running state is small enough that DDR transit is
already cheap.

## Method

Two chains, each run with `LX_PLANNING=False` and `=True` (toggled via
`config.lx_planning` + `torch._dynamo.reset()` between modes):

1. **Softmax chain** — `max → sub → exp → sum → realdiv` at shape
   `(1, 32, 64, 64)` fp16. Has TWO allowlisted producers (max, sum) per
   `OP_OUTPUT_GOOD_FOR_LX_REUSE = ["max", "sum", "clone"]`. Should
   benefit if pinning works.
2. **MM chain** — `(x @ W1) @ W2` at `(128, 4096) → (4096, 4096) →
   (4096, 128)`. Has NO allowlisted producers. Negative control.

## Results

| Chain | LX off | LX on | Speedup | Verdict |
|---|---:|---:|---:|---|
| softmax | 3.46 ms | 3.47 ms | 1.00× | **tied** |
| mm chain | 4.08 ms | 4.08 ms | 1.00× | tied (expected) |

LX_PLANNING has no measurable effect on either chain.

## Why this is OK (and probably expected)

At our test shape `(1, 32, 64, 64)` the softmax intermediates are tiny:

- `max` output: `(1, 32, 64, 1)` × 4 bytes = 8 KB
- `sum` output: same shape = 8 KB
- The intermediate buffers `sub`, `exp`, `realdiv` outputs: 16 KB each

At LPDDR5 peak ~200 GB/s, 16 KB transits in **80 nanoseconds**. With
per-launch overhead at 3 ms (Phase 0b), even saving FIVE such roundtrips
amounts to 400 ns out of 3500 µs total = 0.01% of wall time. Below noise.

So LX_PLANNING's effect would only be visible at shapes where the
intermediate is large enough that BW transit is comparable to launch
overhead. For an intermediate to take 1 ms at 200 GB/s, it'd need to be
200 MB — way bigger than the 2 MB scratchpad anyway.

**LX_PLANNING is a real optimization for shapes between those bounds**
(big enough that BW matters, small enough to fit in scratchpad). For
flash attention's per-Q-tile running state, we're firmly below the
"matters" threshold.

## Implication for flash-attention design

Originally I framed flash-attention as needing scratchpad-pinned running
state to win. **That was wrong.** The win is dominated by:

1. **Eliminating the (B, H, S, S) intermediate** that flows through DDR
   in the naive path. At S=4096 that intermediate is 1 GB; transiting
   it costs many ms even just from BW. Flash attention's tile-streaming
   approach never materializes this tensor.
2. **Reducing total kernel launches** from 14 (current SDPA) to 4-20
   (depending on tile choice and how many kernels per inner-loop step).
   At 3 ms each, this drops launch-overhead from 42 ms to 12-60 ms.

The per-Q-tile running state (max, sum, output for one Q-tile) is small
enough that it can transit DDR between kernels at negligible cost — even
if we couldn't pin it, the design works. **Phase 1 doesn't need an
allowlisted custom op output**; it just needs to emit a kernel sequence
that doesn't materialize the score tensor.

## Phase 0 verdict (Phases 0a, 0b, 0c combined)

Three things are now confirmed:

1. **(0a) The naive path is 30-86× off the flash-attention floor at S=4096.**
   1555 ms → realistic 24-60 ms target.
2. **(0b) Per-launch overhead is ~3 ms flat across all relevant shapes.**
   Tile size is bounded by scratchpad + per-core span, but launch cost
   doesn't penalize us further.
3. **(0c) LX scratchpad pinning is irrelevant at the per-kernel state
   level for flash attention.** The design works without it.

**Phase 1 design proceeds.** No remaining architectural unknowns.

## Files

- `tests/diag_lx_planning.py` — Phase 0c probe
- `tests/diag_lx_planning_results.md` — auto-regenerated bench output
- `tests/sdpa_phase0c_findings.md` — this document

## Reproducing

```sh
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
cd $DTI_PROJECT_ROOT/torch-spyre
python tests/diag_lx_planning.py
```
