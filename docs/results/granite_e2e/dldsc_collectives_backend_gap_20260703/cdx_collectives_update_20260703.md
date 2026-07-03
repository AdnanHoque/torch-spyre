# CDX DLDSC Collectives Update - 2026-07-03

This note records the CDX-side evidence for the `ah/comms-collectives` exploration. The goal is to identify which on-chip communication classes work through the DLDSC coordinate contract and which backend/runtime gaps remain before we can remove non-weight HBM spills from Granite and attention.

## Environment

- Pod: `adnan-cdx-spyre-dev-pf`
- AIU: `/dev/vfio/80`
- Torch checkout: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/torch-spyre`
- Torch branch/SHA: `ah/comms-collectives` / `0572df5`
- Deeptools checkout: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/deeptools`
- Deeptools branch/SHA before local diagnostics: `ah/comms-collectives` / `3d54e87eb`
- Runtime wrapper: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/tools/dxp-split-wrapper/dxp_standalone`
- Important split-env convention:
  - Torch sees `DXP_LX_FRAC_AVAIL=0`, meaning full frontend LX planning.
  - The wrapper maps `DXP_BACKEND_LX_FRAC_AVAIL` to DXP's `DXP_LX_FRAC_AVAIL`.
  - For value-correct producer-LX persistence, backend `DXP_BACKEND_LX_FRAC_AVAIL=0.2` was required in the synthetic RHS broadcast probe. Backend `1.0` allowed DXP scratch to corrupt the producer LX output.

## What Works Now

### 4-Way Matmul RHS All-Gather / Broadcast

Synthetic probe:

- Script: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/min_matmul_operand_broadcast_onehot.py`
- Shape: `M=512, K=128, N=512`
- Producer work division: RHS pointwise under `work_div={"N": 4}`
- Consumer work division: matmul under `work_div={"M": 4}`
- Communication class: source RHS split by output/N across 4 cores, destination matmul consumers need full RHS on each M-split core. This is all-gather/broadcast from LX to LX.

Passing run:

- Run dir: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_auto_relayout_nosplit_scale1_backend02_20260703_142134`
- Env highlights: `DXP_LX_FRAC_AVAIL=0`, `DXP_BACKEND_LX_FRAC_AVAIL=0.2`, `SPYRE_LX_PLANNER_RELAYOUT=1`, `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1`
- Result: `ALLCLOSE True`, `MAX_DIFF 0.03125`

Scale-2 stale-source check:

- Script: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/min_matmul_operand_broadcast_onehot_scale2_rows.py`
- Passing run: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_auto_relayout_nosplit_scale2_backend02_20260703_142049`
- Result: `ALLCLOSE True`, `MAX_DIFF 0.0625`
- Why this matters: with `scale=2`, stale input RHS and producer output differ. The passing result proves the relayout reads the producer output when backend scratch reservation is sane.

## Important Runtime Finding

Backend `DXP_BACKEND_LX_FRAC_AVAIL=1` produced value-wrong output for the same RHS all-gather probe:

- Run dir: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_auto_relayout_split_input_sticks_scale2_20260703_141824`
- Result: `ALLCLOSE False`, `MAX_DIFF 94.0`
- Symptom: low rows read stale or offset source values; high rows were partly correct.
- Interpretation: producer LX output at address 0 can be stomped by backend internal staging when the backend is allowed to consume the whole LX budget. This is a runtime/compiler environment contract issue, not an STCDP logical-transfer issue.

## Remaining Gaps

### Source Core Set Larger Than Consumer Core Set

Synthetic stress:

- Script: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/min_matmul_operand_broadcast_onehot_N8_scale1.py`
- Producer split: `work_div={"N": 8}`
- Consumer split: `work_div={"M": 4}`
- Run dir: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_auto_relayout_N8_backend02_unionfold_20260703_142428`
- Failure: `DtException: query fold dimension with higher fold factor`

Reason:

The consumer SDSC has `coreFoldProp=4`, but the pinned input tensor metadata says the source distribution includes cores `0..7`. The consumer-side LX start-address fold also only has entries for cores `0..3`. Backend cannot legally query source LX bases for cores `4..7`.

Required contract/fix:

- Frontend must carry source-core address metadata for every producer core referenced by a consumer input tensor, or backend must widen and synthesize those source address folds before lowering.
- This is not needed for the 4-way compatible-core-set case, but it is needed for true broader all-gather.

### Attention Layout-Allgather/Restickify Capacity

4-head flash script:

- Source: `github.ibm.com/aviros/test-spyre-scripts`, commit `05deb9702654f73781b457ed052a3ff69316670f`, file `test_flash_4_head.py`
- Local copy: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/test_flash_4_head.py`

Passing control with layout-allgather disabled:

- Run dir: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/flash_4_head_relayout_backend02_no_layout_allgather_20260703_143335`
- Env: relayout enabled, but `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=0`
- Result: `SUCCESS`

Failing layout-allgather attempt:

- Run dir: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/flash_4_head_relayout_backend02_noforcedbase_20260703_143157`
- Failure: `layout_allgather_restickify could not allocate 1048576 bytes in LX for consumer core 0`

Edge shape from artifact:

- Artifact: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/flash_4_head_relayout_backend02_20260703_143107/backend_plans/3_batchmatmul_3_batchmatmul-Relayout_auto_relayout_sdsc.json`
- Consumer: `3_batchmatmul`
- Source LDS: `Tensor0-LxRelayout-inp`
- Source pieces: approximately `{in:16, mb:1024, x:1}` across 32 producer cores
- Destination pieces: approximately `{in:128, mb:128, x:1}` across 32 consumer cores
- Communication class: layout all-gather plus restickify/form change, not simple scatter.

Why this is not PR1 scatter:

The destination matmul wants a dense full-`in` tile per consumer core. Materializing that full destination requires a large per-core LX buffer. The existing fission switch only slices the movement program; it still allocates the full destination because the downstream matmul consumes the full dense tile.

Required next feature:

- Either WSR/streaming must make the downstream matmul consume fissioned chunks, or backend needs a true `ReStickifyOpLx`/streaming relayout path that does not materialize the full dense tile at once.
- This is a follow-on communication plus scheduling problem, not a bug in basic DLDSC scatter.

## Flash No-Scalar Control

The smaller local `test_flash_no_scalar.py` is not useful for validating relayout because it fails even with relayout disabled:

- Relayout-enabled failure: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/flash_no_scalar_relayout_backend02_autoload_20260703_142838`
- Baseline failure: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/flash_no_scalar_baseline_backend02_autoload_20260703_142942`
- Common failure: `DtException: out_reuse_dim.size() == 1` in `L3DlOpsScheduler.cpp`

This is baseline DXP behavior for that script and should not be counted against relayout.

## Current Classification

| Class | Current status | Evidence |
| --- | --- | --- |
| Scatter / same cardinality remap | Works in PR1 path | Earlier PR1 unit/device evidence |
| 1:many broadcast / all-gather with compatible source/consumer core set | Works with backend `0.2` | `min_matmul_auto_relayout_nosplit_scale*_backend02` runs |
| Source core set wider than consumer core set | DXP lowering unblocked by local diagnostics; runtime correctness requires frontend-safe destination allocation | pure gather source-address/fold reset plus forced-base run |
| Attention layout-allgather/restickify | Not yet supported without streaming/full-buffer capacity | 4-head flash allocation failure |
| Reduction / all-reduce | Not exercised in this CDX slice | Needs separate probes |

## Pure Gather Update

After the first CDX update, I chased a pure many-source-to-one-consumer gather:

- Script: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/min_lx_gather_probe.py`
- Shape: `M=128, N=64`
- Producer: `x + 1` under `work_div={"M": 4}` or `work_div={"M": 2}`
- Consumer: `mul` under `work_div={"M": 1}`
- Communication class: gather. Multiple producer cores own disjoint `M` shards, one consumer core needs the full `M` slice.

The path exposed three backend/frontend-contract gaps:

1. The frontend consumer input carried producer coordinates, but not source-core LX start addresses for all source cores. A local Torch diagnostic patch added `lx_residency_core_id_to_lx_start_address`.
2. Deeptools JSON import assumed an allocate-node start-address fold used the caller SDSC fold cardinality. A local importer diagnostic patch allowed tensor-local start-address fold cardinality.
3. Deeptools relayout insertion then had to reset the post-relayout consumer allocation back to the consumer fold. This used `FoldManager::clone()`; plain assignment preserves the old fold shape and fails when source and consumer core cardinalities differ.

After those fixes, DXP replay passed for the generated pure-gather SuperDSC bundle.

The exact local diagnostic diffs are archived under `patches/`. They are intentionally recorded as evidence, not as clean PR-ready patches.

Runtime status:

- Dynamic backend destination allocation:
  - Run log: `artifacts/gather_20260703/min_lx_gather_common_refinement_bad_dynamic_run.log`
  - Result: runtime completes, but `ALLCLOSE False`
  - True movement corruption: only row `64` has a large error; row `64` starts at destination offset `8192` bytes.
- Forced frontend-safe destination base:
  - Run log: `artifacts/gather_20260703/min_lx_gather_srcsplit2_forcedbase_run.log`
  - Relayout SDSC: `artifacts/gather_20260703/min_lx_gather_forcedbase_relayout_sdsc.json`
  - Env addition: `DEEPTOOLS_RELAYOUT_FORCE_DST_BASE=65536`
  - Result: `ALLCLOSE True`, `MAX_DIFF 0.25`, no large-diff rows.

Interpretation:

The high-level gather plan is now correct when the destination address is safe. The remaining issue is not the coordinate classification itself; it is ownership of the destination LX buffer. Backend-only dynamic allocation picked a high backend-reserved address that corrupted the gathered tensor at the 8KB offset. Forcing a lower frontend-safe LX base made the same movement plan value-correct.

Architectural implication:

For gather/all-gather that materializes a new consumer-shaped LX tensor, the cleaner production direction is for the frontend LX planner to allocate the destination buffer and emit the tensor distribution/address contract. Deeptools should synthesize the physical ring movement into that provided destination. Letting Deeptools invent an untracked destination address is fragile because the consumer compute row is still part of the frontend-planned LX lifetime.

## Granite Sidecar Update

Volta ran the one-layer Granite causal prefill sidecar on `adnan-spyre-dev-pf`:

- Report: `artifacts/granite_sidecar_20260703/sidecar_granite_prefill_sanity_report_20260703_145128.md`
- Disabled control:
  - Wall median: `26.672 ms`
  - Trace kernel: `14.671 ms/iter`
- Enabled collectives:
  - Failed before timing with `Can not propagate coordinates for coreletSplit dimensionmb from allocateNode allocate-Tensor0_lx with custom coreIdToWkSlice`.
  - Backend plans were emitted for `10_batchmatmul_Tensor1_0_layout_allgather_restickify` and `18_batchmatmul_Tensor1_0_matmul_operand_broadcast`.

This failure is consistent with the pure-gather findings: broader collectives need a stronger tensor-address/distribution contract and frontend-planned destination buffers before Granite can remove the remaining non-weight HBM spills end to end.

## Practical Next Steps

1. Treat backend `DXP_BACKEND_LX_FRAC_AVAIL=0.2` as required for current DLDSC relayout correctness tests until the frontend/backend LX budget contract is cleaned up.
2. Add a frontend/backend contract for source LX address metadata when a consumer input references source cores outside the consumer SDSC core fold.
3. Add frontend LX planner support for destination buffers for gather/all-gather materialization. Deeptools should realize movement into those provided addresses instead of allocating opaque backend scratch for tensors consumed by frontend-planned compute.
4. For attention, do not try to force full layout-allgather/restickify into PR1. The current edge needs either WSR/streaming consumer scheduling or an explicit on-chip restickify primitive that avoids full dense materialization.
5. Add reduction/all-reduce probes separately; do not infer them from scatter/all-gather success.
