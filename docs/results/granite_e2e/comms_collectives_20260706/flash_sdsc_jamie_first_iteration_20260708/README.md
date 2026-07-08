# Flash Attention SDSC Before/After, First Iteration

Generated on 2026-07-08 from the archived CDX run:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
```

Jamie requested summarizing only the first iteration because the flash SDSC
contains a repeated 17-row pattern. The full main flash kernel has 544 SDSCs:

```text
32 iterations * 17 SDSCs/iteration = 544 SDSCs
```

The first repeated iteration is `sdsc_0.json` through `sdsc_16.json` from the
main fused flash kernel:

```text
sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_*
```

## Source Runs

| Variant | Source run | Full SDSC count | First-iteration SDSCs |
|---|---|---:|---:|
| before, relayout off | `/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/before_relayout_off` | 550 | 17 |
| after, relayout on | `/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/after_relayout_on` | 550 | 17 |

## Headline Delta

| First-iteration row | Before | After |
|---|---|---|
| relayout row | `sdsc_2: ReStickifyOpHBM` | `sdsc_2: ReStickifyOpLx` |
| relayout tensors | `INPUT (lx), OUTPUT (hbm)` | `INPUT (lx), OUTPUT (lx)` |
| following QK/V-style matmul input | `batchmatmul: INPUT (hbm), INPUT (hbm), OUTPUT (hbm)` | `batchmatmul: INPUT (hbm), INPUT (lx), OUTPUT (hbm)` |
| full-run repeated count | 32 HBM restickifies | 32 LX restickifies |

This is the structural proof that the flash activation handoff no longer
round-trips through HBM in the relayout-enabled run. It does not claim numeric
correctness; the flash value path has a known independent baseline issue.

## Generated Jamie-Style Reports

- `flash_first_iter_before_summarize_sdsc.md`
- `flash_first_iter_after_summarize_sdsc.md`

The full 550-SDSC summaries are one directory up:

- `../flash_before_relayout_off_summarize_sdsc.md`
- `../flash_after_relayout_on_summarize_sdsc.md`
