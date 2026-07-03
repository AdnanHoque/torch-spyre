# Flash Layout-All-Gather Relayout Probe, 2026-07-03

## Summary

This probe explored the remaining flash-attention activation spill class that is not covered by PR1 scatter relayout:

```text
ReStickifyOpLx output in LX -> batchmatmul KERNEL operand in LX
```

The edge is not a pure same-layout scatter. It is a layout-aware grouped all-gather:

- producer/restickify layout: `x,out,mb`, stick dim `x`
- consumer batchmatmul KERNEL layout: `out,in,x`, stick dim `out`
- logical rename: `restickify.x -> batchmatmul.out`, `restickify.out -> batchmatmul.in`, `restickify.mb -> batchmatmul.x`
- communication class: `layout_allgather_restickify`

Torch emitted 32 `layout_allgather_restickify` backend plans for the flash bundle. DXP compiled and the kernels launched, but all tested materializations were value-wrong with the same output signature:

```text
Mismatched elements: ~166469xx / 16777216 (99.2%)
Greatest absolute difference: inf at index (0, 0, 0, 0)
```

## Variants Tested

| Variant | DXP compile | Runtime launch | Correctness | Notes |
| --- | --- | --- | --- | --- |
| Staged `ReStickifyOpLx -> STCDPOpLx`, logical source coords | pass | pass | fail | First staged implementation after fixing empty-consumer STCDP construction. |
| Same staged path with data-op schedule flag fixed | pass | pass | fail | Corrected standalone schedule rows to mark them as data-op rows. |
| Staged path with source layout corrected to `out,in,x` | pass | pass | fail | Removed likely wrong `in,out,x` source interpretation. |
| Direct grouped `STCDPOpLx` all-gather with data-op schedule flag fixed | pass | pass | fail | Removed scratch/restickify stage; still value-wrong. |

See [flash_layout_allgather_attempts_20260703_excerpts.md](flash_layout_allgather_attempts_20260703_excerpts.md) for run directories and log excerpts.

## Standalone DataOp Follow-Up

A standalone `DataOpStandalone` sample was added after the full flash runs to isolate one grouped all-gather edge from `test_flash.py`.

That standalone sample passes descriptor generation/DCG lowering and emits the same core transfer shape as the full flash debug descriptor:

```text
pSubPiece rows: 32
cSubPiece rows: 256
dtTable rows: 32
fanout per producer row: 8
maxConsumers: 8
```

See [standalone_flash_grouped_allgather_20260703.md](standalone_flash_grouped_allgather_20260703.md).

The deeper executable check is still failing: `senpcfg` and `dcc-opt` pass, but `senulator -v store` reports `LX Store verification failed`. That makes this a backend executable-lowering/value issue for grouped LX all-gather, not just a full-flash integration issue.

## Current Interpretation

This is no longer a basic DXP import/routing failure. DXP accepts the inserted movement, and the AIU runtime launches the affected bundle. The failure is now a value-correctness issue in physical materialization:

- the grouped all-gather may still describe source or destination piece coordinates incorrectly;
- the consumer operand may not be rebound to the post-relayout LX allocation the way the generated batchmatmul expects;
- the STCDP lowering may require additional per-subpiece metadata for this layout-renamed all-gather case;
- the current full-flash test is too coarse to distinguish those cases.

## Recommendation

Stop iterating with full flash smokes as the primary debug loop. The next useful step is a small patterned data-op harness that checks the bytes immediately after the inserted relayout and before downstream flash math. That harness should cover:

1. One source group with eight producer chunks.
2. One destination consumer core receiving the full grouped operand slice.
3. Deterministic per-chunk patterns so swapped `out/in/x` dimensions are obvious.
4. Both direct `STCDPOpLx` grouped gather and staged `ReStickifyOpLx -> STCDPOpLx`.

Only after this passes should we re-enable the full flash bundle.

## Captured Files

- [deeptools_current_direct_allgather_experiment.patch](deeptools_current_direct_allgather_experiment.patch)
- [deeptools_current_direct_allgather_experiment_diff_stat.txt](deeptools_current_direct_allgather_experiment_diff_stat.txt)
- [flash_layout_allgather_attempts_20260703_excerpts.md](flash_layout_allgather_attempts_20260703_excerpts.md)
- [standalone_flash_grouped_allgather_20260703.md](standalone_flash_grouped_allgather_20260703.md)
