# Flash Baseline Value Oracle Check, 2026-07-06

This records a CDX check of the upstream-style flash attention script with LX
relayout disabled.

## Purpose

The previous `flash_gather_restickify_clean_cdx_20260706` archive proves the
structural communication transformation:

- activation-side `ReStickifyOpHBM` rows are replaced by `ReStickifyOpLx`;
- 32 `matmul_operand_broadcast` backend plans are emitted;
- the plans lower as `all_gather_replicate` with `gather_then_restickify`.

That run intentionally skipped host-to-device data copies and CPU comparison,
so it was a compile/runtime structural probe rather than a value-correctness
claim. This directory records the follow-up baseline oracle: run the same flash
script with relayout disabled and with the CPU comparison enabled.

## Result

- Pod: `adnan-cdx-spyre-dev-pf`
- Clean root: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236`
- Run: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_clean_baseline_value_oracle_20260706_122337`
- Return code: `1`

The baseline value check fails before any relayout comparison is meaningful:

```text
AssertionError: Tensor-likes are not close!
Mismatched elements: 12602561 / 16777216 (75.1%)
Greatest absolute difference: inf at index (0, 0, 1, 1) (up to 0.1 allowed)
Greatest relative difference: nan at index (0, 0, 1, 1) (up to 0.1 allowed)
```

## Interpretation

This is not evidence that the DLDSC gather/restickify path is value-wrong. The
same script fails with relayout disabled, which matches the known independent
flash baseline issue around broadcast/zero-stride view handling. Until that
baseline issue is fixed or bypassed, this flash script can be used for
structural compile/lowering evidence, but not for relayout value correctness or
performance claims.

## Archived Files

- `env.txt`: pinned environment used for this check.
- `returncode.txt`: process return code.
- `stdout.log`: raw stdout.
- `stderr.log`: raw stderr with the `assert_close` failure.
- `summary.json`: compact parsed failure summary.

## Next Useful Validation

Use a smaller synthetic harness that emits the same
`matmul_operand_broadcast -> all_gather_replicate -> gather_then_restickify`
backend plan but has a clean CPU oracle, or wait for the flash baseline
zero-stride/broadcast issue to be fixed.
