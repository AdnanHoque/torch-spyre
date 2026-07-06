# Granite S512 gather/restickify 4-slot replay

Date: 2026-07-06
Pod: adnan-spyre-dev-pf
Torch root: /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/torch-spyre
Deeptools root: /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/deeptools

## Result

DXP replay passed with the experimental rank-grouping + partial-stick + chunked gather/restickify patch.

- returncode: 0
- backend plans: 2
- logical transfers: 1536
- 16_batchmatmul: matmul_operand_broadcast, all_gather_replicate, gather_then_restickify, lowered_gather_then_restickify, 1024 logical transfers
- 8_batchmatmul: matmul_operand_broadcast, all_gather_replicate, gather_then_restickify, lowered_gather_then_restickify, 512 logical transfers

## Why this matters

Earlier variants established the failure sequence:

1. Equal-rank assumption failed for Granite attention operand relayout.
2. Rank-grouping fixed the logical coordinate mapping, then DCG rejected sub-stick output pieces.
3. Partial-stick padding/valid-gap fixed piece legality, then full temp materialization ran out of LX.
4. One-piece chunks fixed LX temp pressure, but DCC failed IBUFF 152 > 128.
5. Two-piece chunks improved IBUFF to 137 > 128.
6. Four-piece chunks passed replay.

The current shape is bounded-temp staged all-gather + local ReStickifyOpLx. It avoids HBM fallback for this Granite attention operand class in DXP replay.

## Validation

Focused Deeptools tests also pass at:
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/deeptools_tests_chunked_4slot_20260706_071233

- LayoutAllgatherRestickify.*: rc 0, 27/27 passed
- CoreWorkDivIncomptLxRelayout*: rc 0, 2/2 passed

## Files

- chunked_4slot_replay_summary.json
- experimental_rank_grouping_chunked_4slot_patch.diff
- stdout.log
- stderr.log
- backend_plans/*.json
