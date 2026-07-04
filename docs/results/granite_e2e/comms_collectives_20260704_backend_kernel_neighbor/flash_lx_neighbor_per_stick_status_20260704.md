# Flash LX Neighbor Broadcast Diagnostic - 2026-07-04

## Latest Update

The blocker described below was resolved in the diagnostic branch by changing the destination-allocation contract:

- Do not copy producer coordinate folds into the synthetic `matmul_operand_broadcast` destination allocation.
- Seed destination ownership from the consumer core map.
- Let DDC derive the consumer-sized LX allocation from the normal transfer/compute coordinates.
- Disable the earlier diagnostic cache/peer seeding that reintroduced the producer fold.

With backend `DXP_LX_FRAC_AVAIL=1`, the full `test_flash.py` compile/runtime smoke now passes in probe mode:

- Run: `/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/flash_after_collectives_backend1_20260704_085439`
- Return code: `0`
- SDSC files: `550`
- Backend `matmul_operand_broadcast` plans: `32`
- Sample plan: `all_gather_replicate`, `group_count=4`, `replication_factor=8`, `logical_transfer_count=256`
- Caveat: the run used `no_h2d,skip_cpu_ref`, so it proves compile/runtime smoke but not numeric correctness.

See `flash_after_collectives_backend1/README.md` for the current checkpoint and exact Deeptools diagnostic diff.

## Workspace

- Pod: `adnan-cdx-spyre-dev-pf`
- Root: `/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507`
- Torch branch: `ah/comms-collectives`
- Deeptools branch: `ah/comms-collectives`
- Primary run dir: `/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/flash_lx_neighbor_per_stick_guarded_20260704_073513`
- Failing bundle: `cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_fuo7dz5t`

## What Was Tested

This diagnostic tried to physically realize the `matmul_operand_broadcast` / grouped all-gather class in Deeptools for the flash attention RHS operand. The prototype keeps the DLDSC marker path, creates a synthetic LX-neighbor transfer, attaches exact logical source/destination core pairs, and lowers those pairs to ring transfer nodes.

The ring lowering was intentionally diagnostic-only: ranges were expanded to one ring node per 128-byte stick to avoid ambiguity in DCC's existing burst-loop ring path.

## Changes Exercised

- `dsc/dsc2.h`: added `TransferNode::lxNeighborRingTransfers_` metadata.
- `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp`:
  - classified `matmul_operand_broadcast`;
  - created synthetic LX-neighbor transfer nodes;
  - guarded physical realization on chunkable LX extents;
  - moved physical transfer population until after allocation start addresses are assigned;
  - registered newly created LX-local allocations with the memory tracker.
- `dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp`: emitted per-stick `SenPcfgRingDtNode` send/receive nodes from `lxNeighborRingTransfers_`.

## Failure Chain

1. Initial replay aborted with:
   `cannot derive per-producer-chunk LX byte size`

   Cause: the classifier opted in on metadata that had no concrete LX extent. The scheduler now requires a valid movement plan and nonzero, evenly chunkable `lxSize_`.

2. Next replay segfaulted in `fillFinalStartAddressAndOffset`.

   Cause: the synthetic LX destination allocation had no start-address fold data. A diagnostic check now reports the actual tensor/allocation instead of segfaulting:
   `Missing LX start address coordinates for Tensor1 allocation allocate_lds1_lx`

3. The transfer population was moved after allocation address assignment.

   This exposed that ordinary LX-local allocations created by the scheduler were not registered with the memory tracker. Registering them removed the missing-start-address failure.

4. Direct DXP replay with backend `DXP_LX_FRAC_AVAIL=0.2` then failed capacity:
   `The initial chunk parameters must fit in LX for SuperDSC: 3_batchmatmul`

   Direct replay with backend `DXP_LX_FRAC_AVAIL=1` progressed further.

5. Direct DXP replay with backend `DXP_LX_FRAC_AVAIL=1` now fails in DDC coordinate propagation:
   `std::out_of_range` in `ddc::Ddc::buildFoldForAllocation`.

   This is the current blocker.

## Current Read

The backend can identify the grouped all-gather/broadcast edge, and the allocator can now assign physical LX addresses to the synthetic allocation. The remaining blocker is coordinate metadata for backend-created LX relayout destinations.

The generated SDSC says `Tensor1` is LX-resident and carries `matmul_operand_broadcast` metadata, but the backend-created allocation has no coordinate reference. `LabeledDsInfo` has address/size fields, but allocation coordinates live on schedule-tree `AllocateNode`s. Existing coordinate propagation starts from HBM allocations and follows HBM/LX transfer edges; the synthetic `NO_COMPONENT -> LX` neighbor transfer is not a usable coordinate source.

So the missing contract is not just "move bytes from core A to core B." Deeptools also needs a coordinate reference for the relayout destination allocation: either emitted by Torch in the DLDSC/SDSC contract, or derived by Deeptools from explicit producer/consumer distribution metadata.

## Next Decision

Two viable directions remain:

1. Torch emits an explicit relayout-destination coordinate contract for `matmul_operand_broadcast` edges.
2. Deeptools builds allocation coordinates from the existing `producer_core_id_to_device_slice` and `consumer_core_id_to_device_slice` classification metadata.

The current prototype is blocked until one of those is implemented.
