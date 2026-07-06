# Granite S512 gather-restickify replay runbook

This artifact records the first DEV replay where the Granite attention matmul operand all-gather/restickify path lowered without HBM fallback and without DXP/DCC failure.

## Environment

- Pod: adnan-spyre-dev-pf
- Torch branch: ah/comms-collectives
- Torch SHA: 3f13d2f9fd8b14a9efa986cabd9b1de038faf122
- Deeptools branch: ah/comms-collectives
- Deeptools SHA: a4930be14b6e7d01f7447b7692a79a20487c09c3
- Source run: /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_staged_rank_group_chunked_4slot_replay_20260706_071147

## Command shape

Set DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1, DXP_LX_FRAC_AVAIL=1, DXP_BACKEND_LX_FRAC_AVAIL=1, DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR=<run>/backend_plans, and DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR=<run>/backend_plans, then replay the archived Granite SuperDSC bundle through dxp_standalone --bundle -d <granite_attention_bundle> -b SENTIENT.

## Result

- returncode: 0
- backend plans: 2
- logical transfers total: 1536

- 16_batchmatmul: matmul_operand_broadcast / all_gather_replicate / gather_then_restickify / lowered_gather_then_restickify / logical transfers 1024
- 8_batchmatmul: matmul_operand_broadcast / all_gather_replicate / gather_then_restickify / lowered_gather_then_restickify / logical transfers 512

## What changed technically

The new Deeptools patch handles the Granite attention operand case that needs source-rank to target-rank grouping, partial-stick valid/gap modeling, and bounded temporary storage. The successful configuration uses staged all-gather into four reusable temp slots per destination core followed by local ReStickifyOpLx into the matmul operand layout.

Earlier failures proved the shape of the gap:

1. Equal-rank source/target mapping was too narrow.
2. Sub-stick output pieces needed physical stick padding plus valid/gap metadata.
3. Full temp materialization ran out of LX.
4. One-slot and two-slot chunking lowered but exceeded DCC IBUFF.
5. Four-slot chunking passed.

## Scope

This proves DXP/DCC replay for the Granite attention matmul operand all-gather/restickify communication class. It does not prove value correctness; the known upstream flash zero-stride/broadcast issue remains separate from the relayout carrier.
