# Flash Attention DLDSC Probe Artifacts 2026-07-04

Source pod: `adnan-cdx-spyre-dev-pf`

Run root on pod:

`/home/adnan-cdx/codex-isolated/flash_attention_verify_reuse_20260704_023339`

Script:

`test_flash.py` at commit `afda166e58b23519d0b4ca871350b011b56d91a3`

## Current Main Baseline

- return code: `1`
- generated SDSCs: `5`
- backend plans: `0`
- failure class: DXP LX capacity failure before relayout metadata; no non-empty `lxRelayoutClassifications_` observed.

Representative error is in `current_main_baseline/error_excerpt.txt`:

```text
Unable to map graph within architecture constraints: The initial chunk parameters must fit in LX for SuperDSC: 0_identity
```

## DLDSC Collectives Path

- return code: `1`
- generated SDSCs: `549`
- backend plans: `1`
- backend plan: `dldsc_ah_comms_relayout/backend_plans/3_batchmatmul_Tensor1_0_layout_allgather_restickify_plan.json`
- plan kind: `unknown`

Plan summary:

- `group_count`: `4`
- `producer_chunks_per_group`: `8`
- `consumer_cores_per_group`: `8`
- `logical_transfer_count`: `256`

The optimized path progresses much further than current main and emits DLDSC relayout metadata, but it still fails because this flash case selects dense `layout_allgather_restickify`, which tries to allocate a full replicated operand in LX:

```text
layout_allgather_restickify could not allocate 1048576 bytes in LX for consumer core 0
```

## Interpretation

This is the same remaining communication gap seen in Granite diagnostics: direct scatter and staged matmul operand broadcast/all-gather-replicate are useful, but dense resident layout all-gather is not viable. The next backend target is a staged layout-allgather/restickify carrier that binds shards directly into the matmul transfer loop instead of materializing a full per-consumer LX operand.

Full raw logs remain on the source pod run root if deeper forensic inspection is needed.
