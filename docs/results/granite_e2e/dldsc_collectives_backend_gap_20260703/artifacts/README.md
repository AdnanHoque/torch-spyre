# Artifact Index

This directory contains the compact proof artifacts for `cdx_collectives_update_20260703.md`.

## Passing RHS Broadcast / All-Gather

- `min_matmul_auto_relayout_nosplit_scale1_backend02_20260703_142134/run.log`
  - `ALLCLOSE True`
  - `MAX_DIFF 0.03125`
- `min_matmul_auto_relayout_nosplit_scale1_backend02_20260703_142134/1_batchmatmul_relayout_sdsc.json`
  - Generated backend relayout SDSC for the passing 4-way RHS all-gather/broadcast case.
- `min_matmul_auto_relayout_nosplit_scale2_backend02_20260703_142049/run.log`
  - `ALLCLOSE True`
  - `MAX_DIFF 0.0625`
  - Scale-2 source check that proves the transfer reads the producer LX output instead of stale input.
- `min_matmul_auto_relayout_nosplit_scale2_backend02_20260703_142049/1_batchmatmul_relayout_sdsc.json`
  - Generated backend relayout SDSC for the passing scale-2 check.

## Remaining Backend Contract Gap

- `min_matmul_auto_relayout_N8_backend02_unionfold_20260703_142428/run.log`
  - Fails with `DtException: query fold dimension with higher fold factor`.
  - This is the source-core-set-wider-than-consumer-core-set case.
  - The backend needs source LX address metadata for producer cores not present in the consumer SDSC fold, or it needs to synthesize/widen that metadata before lowering.

## Pure Gather Diagnostic

- `gather_20260703/min_lx_gather_common_refinement_bad_dynamic_run.log`
  - Pure gather lowers and runs, but the dynamically allocated destination base corrupts row 64.
  - This isolates the remaining issue to destination LX allocation/address ownership, not high-level coordinate classification.
- `gather_20260703/min_lx_gather_srcsplit2_forcedbase_run.log`
  - Same pure gather with `DEEPTOOLS_RELAYOUT_FORCE_DST_BASE=65536`.
  - `ALLCLOSE True`, no large-diff rows.
- `gather_20260703/min_lx_gather_forcedbase_relayout_sdsc.json`
  - Backend relayout SDSC for the value-correct forced-base gather run.
  - The output side is split into common-refinement cells that match the input pieces.

## Attention Layout-AllGather / Restickify Gap

- `flash_4_head_relayout_backend02_20260703_143107/run.log`
  - Fails with `layout_allgather_restickify could not allocate 1048576 bytes in LX for consumer core 0`.
- `flash_4_head_relayout_backend02_20260703_143107/3_batchmatmul_relayout_sdsc.json`
  - Generated relayout SDSC for the failing attention edge.
  - Shows the edge class: layout all-gather plus form-changing restickify, not simple scatter.
- `flash_4_head_relayout_backend02_no_layout_allgather_20260703_143335/run.log`
  - Passing control when layout-allgather/restickify insertion is disabled.
- `flash_4_head_relayout_backend02_noforcedbase_20260703_143157/run.log`
  - Same attention capacity failure without a forced destination base.

## Readout

The current DLDSC path has evidence for simple scatter and compatible-core-set broadcast/all-gather. Pure gather can be value-correct when the destination LX base is frontend-safe, but backend-only dynamic destination allocation is not yet a production contract. The remaining attention spill is not removed by PR1 scatter because materializing the consumer's full dense tile would exceed the per-core LX budget. That edge needs streaming/WSR or an on-chip restickify form that does not require the full destination tile at once.

## Granite Sidecar

- `granite_sidecar_20260703/sidecar_granite_prefill_sanity_report_20260703_145128.md`
  - Disabled one-layer Granite causal prefill control passed.
  - Enabled collectives emitted backend plans but failed before timing on custom `coreIdToWkSlice` fold propagation.

## Diagnostic Patch Diffs

- `../patches/cdx_torch_gather_source_address_diagnostic.diff`
  - Torch-side diagnostic contract extension for source-core LX start addresses.
- `../patches/cdx_deeptools_gather_diagnostic.diff`
  - CDX Deeptools diagnostic diff used during this exploration.
  - This is not a clean PR patch; it records the local state needed to reproduce the gather lowering/runtime findings.
