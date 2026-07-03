# Granite DLDSC Collectives Update - 2026-07-03

This captures the CDX diagnostic run that separated three cases:

1. direct synthetic LX gather works;
2. Granite can emit matmul-operand all-gather metadata;
3. Granite cannot yet realize that metadata because the current backend path tries to materialize the full post-relayout RHS in LX.
4. the same mechanism can produce a measurable small-shape win when the live set is small enough for the current backend realization.
5. the standalone flash-attention script exposes the same `matmul_operand_broadcast` class as Granite attention.

## Environment

- Pod: `adnan-cdx-spyre-dev-pf`
- AIU: `/dev/vfio/80`
- Root: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525`
- Torch branch: `ah/comms-collectives`
- Deeptools branch: `ah/comms-collectives`
- Split LX convention:
  - Torch sees `DXP_LX_FRAC_AVAIL=0`
  - DXP wrapper rewrites the backend subprocess to `DXP_LX_FRAC_AVAIL=$DXP_BACKEND_LX_FRAC_AVAIL`
  - Current backend value: `DXP_BACKEND_LX_FRAC_AVAIL=0.2`

## Key Findings

### 1. Synthetic gather is value-correct

Artifacts:

- `synthetic_gather/srcsplit2_run.log`
- `synthetic_gather/srcsplit2_sdsc_1.json`
- `synthetic_gather/srcsplit4_run.log`
- `synthetic_gather/srcsplit4_sdsc_1.json`

Both `SRC_SPLIT=2` and `SRC_SPLIT=4` pass with:

- `ALLCLOSE True`
- `MAX_DIFF 0.25`
- `LARGE_DIFF_ROWS []`

The important contract field is present:

```json
"consumer_core_id_to_lx_start_address": {"0": 8192}
```

This proves the basic many-to-one gather path can work when the destination is small enough and Torch owns the destination LX address.

### 2. Dirty backend diagnosis

The CDX backend originally failed even for Granite disabled control. The generated SDSCs had empty `lxRelayoutClassifications_`, so the failure was not caused by relayout metadata.

The culprit was an experimental global import change in `dsc/dsc2.cpp` for `startAddressCoreCorelet_`. Restoring the original import path by default made Granite disabled control pass again. The experimental path is now guarded by:

```bash
DEEPTOOLS_USE_JSON_START_ADDR_FOLD_PROPS=1
```

This is recorded in `diffs/cdx_deeptools_diagnostic.diff`.

### 3. Granite disabled control passes

Artifact:

- `disabled_control/result.json`
- `clean_clc_disabled_control/result.json`
- `clean_clc_disabled_control/trace_summary.json`

Run:

```bash
SPYRE_LX_PLANNING=1
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=0.2
```

Result:

- `returncode: 0`
- `median_ms: 44.371 ms` in the profiled single-iteration run
- later non-profile paired control stabilized near `30.57 ms`

The profiled run has empty Kineto kernel timing in this environment, so wall timing is only used as a local sanity signal here.

A separate clean CLC lane reproduced the disabled S=512 control from clean pushed branches:

- Torch SHA: `fef3c8916484e846a394d2a20b0d521345d41338`
- Deeptools SHA: `3d54e87eb404b54c0ba74b98d6caa83945b2ef5b`
- FMS SHA: `b4f36b5af526b938db506a17dcd32d468a7a91d8`
- `returncode: 0`
- wall `median_ms: 27.223`
- Kineto `kernel_ms_per_iter: 14.734`
- all `lxRelayoutClassifications_` entries were empty.

### 4. Metadata-only Granite collectives do not speed up the block

Artifact:

- `metadata_only/result.json`
- `metadata_only/sdsc_8_attention_rhs_contract.json`
- `metadata_only/sdsc_16_attention_rhs_contract.json`

Run:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
```

Result:

- `returncode: 0`
- `median_ms: 30.680 ms`
- two attention matmul RHS contracts are emitted:
  - `communication_class: all_gather`
  - `communication_pattern: all_gather_replicate`
  - `kind: matmul_operand_broadcast`
- no HBM restickify rows are removed.

Why no speedup: the producer operand is not LX-pinned, so Deeptools skips relayout before it can act on the metadata. This is a contract-only artifact.

### 5. Pinning computed restickify output exposes the next real backend gap

Artifact:

- `matmul_operand_full_materialization_failure/result.json`
- `matmul_operand_full_materialization_failure/stderr.log`
- `matmul_operand_full_materialization_failure/sdsc_8_attention_rhs_contract.json`
- `matmul_operand_full_materialization_failure/sdsc_16_attention_rhs_contract.json`

Run adds:

```bash
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
```

Result:

```text
DtException: matmul_operand_broadcast could not allocate 4194304 bytes in LX for consumer core 0
```

This is the important failure. Once the computed restickify producer is LX-resident, Deeptools tries to realize the matmul RHS all-gather, but the current implementation materializes the full post-relayout RHS buffer per consumer core. That is too large for LX.

### 6. Smaller Granite S=128 proves the path can realize on-chip movement

Artifacts:

- `s128_disabled/result.json`
- `s128_enabled/result.json`
- `s128_enabled/example_attention_relayout_sdsc_8.json`
- `s128_enabled/backend_plans/*.json`
- `clean_clc_s128_enabled/result.json`
- `clean_clc_s128_enabled/trace_summary.json`

Both runs used `B=1, S=128, H=4096`, causal prefill, two measured iterations, one warmup, and `DEEPTOOLS_PATH` pointed at the Deeptools source tree. That last part matters because the source checkout has `ddc/ddl_templates/restickify_lx.ddl`, while the installed `/opt/ibm/spyre/deeptools/share` path on this pod did not.

Disabled control:

- `returncode: 0`
- `all_ms: [36.025, 31.452]`
- `median_ms: 33.739`
- attention rows include `ReStickifyOpHBM: 5`

Enabled relayout:

- `returncode: 0`
- `all_ms: [17.523, 17.264]`
- `median_ms: 17.394`
- attention rows include `ReStickifyOpHBM: 4`
- attention rows include `ReStickifyOpLx: 1`
- backend emitted three relayout plans:
  - `7_ReStickifyOpLx_7_ReStickifyOpLx-Relayout_auto_relayout_sdsc.json`
  - `8_batchmatmul_8_batchmatmul-Relayout_auto_relayout_sdsc.json`
  - `16_batchmatmul_16_batchmatmul-Relayout_auto_relayout_sdsc.json`

The key `sdsc_8` contracts are:

```text
kind=all_gather
communication_pattern=many_to_many
transfer_count=128
realized=true

kind=matmul_operand_broadcast
communication_pattern=all_gather_replicate
transfer_count=1024
requires_staged_realization=true
realized=false
```

This is not a final performance claim: the run is short and wall-time only. It is still an important feasibility point. For a smaller sequence length, the current DLDSC + backend prototype can replace one HBM activation restickify with an LX restickify, emit backend relayout plans, and run end to end.

The same smaller-shape enabled run was reproduced on the clean CLC lane from clean pushed branches:

- `returncode: 0`
- wall `median_ms: 15.133`
- Kineto `kernel_ms_per_iter: 10.725`
- SDSC JSON files: `44`
- `ReStickifyOpHBM: 4`
- `ReStickifyOpLx: 1`
- relayout classification entries: `6`
- all six entries are `kind=matmul_operand_broadcast`, `communication_class=all_gather`, `realized=false`

This clean-lane result means the branch can compile and run the smaller case with an LX restickify present. It does not by itself prove the full S=512 Granite target is solved.

### 7. Chunked fission avoids full allocation but still cannot feed matmul

Artifacts:

- `matmul_operand_chunked_fission_failure/result.json`
- `matmul_operand_chunked_fission_failure/error.txt`
- `matmul_operand_chunked_fission_failure/backend_plans/10_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
- `matmul_operand_chunked_fission_failure/backend_plans/fission0_inserted_sdsc.json`
- `matmul_operand_chunked_fission_failure/backend_plans/fission31_inserted_sdsc.json`

The backend worker added an env-gated diagnostic path:

```bash
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_FISSION_ROWS=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_CHUNKED_FISSION_DST=1
```

This path gets past the earlier full-buffer allocation failure. It emits one compact movement SDSC per source-core chunk, so the movement side can be represented as 32 fissioned transfers. The representative plan records:

```text
kind=matmul_operand_broadcast
communication_pattern=all_gather_replicate
source_core_count=32
producer_chunks_per_group=32
replication_factor=32
logical_transfer_count=1024
realization_strategy=loop_scoped_input_fetch
physical_lowering_status=blocked
```

The final failure is intentional:

```text
matmul_operand_broadcast chunked fission emitted compact movement SDSCs
but cannot redirect the matmul operand: current DL matmul lowering has
one resident LDS operand pointer and no valid staged partial matmul
transfer-loop path
```

This proves the next missing piece is not just "generate more ring transfers." The backend can describe the chunked movement. The missing implementation is a matmul-consumer path that accepts staged operand chunks instead of requiring a complete resident RHS tensor.

The clean CLC S=512 enabled run shows the same area is not production-safe yet:

- `clean_clc_s512_enabled/stdout.log`
- `clean_clc_s512_enabled/stderr.log`

It emitted the same class of metadata before aborting at runtime:

```text
ReStickifyOpHBM: 4
ReStickifyOpLx: 1
classification entries: 6
kind: matmul_operand_broadcast
communication_class: all_gather
realized: false
```

The runtime failure was:

```text
RuntimeStream::synchronize() still waiting after 60000ms
PCIe bus master fence, code 0xa35e
Signal Received: 6 (Aborted)
```

So the clean branch does not fail closed soon enough for S=512. The dirty CDX diagnostic branch was useful because it converts that unsafe runtime behavior into an earlier explicit backend error.

### 8. Flash attention probe shows the same communication class

Artifacts:

- `flash_attention_probe/flash_attention_probe_summary.json`
- `flash_attention_probe/baseline_first_restickify_hbm.json`
- `flash_attention_probe/baseline_first_following_batchmatmul.json`
- `flash_attention_probe/metadata_first_restickify_hbm.json`
- `flash_attention_probe/metadata_first_following_batchmatmul.json`
- `flash_attention_probe/baseline_env.txt`
- `flash_attention_probe/metadata_env.txt`

The tested script was `github.ibm.com/aviros/test-spyre-scripts/test_flash.py` at `afda166`, using compile probes with `no_h2d,skip_cpu_ref`. That means these runs verify compile/backend/runner success, not full numerical correctness.

Summary:

```text
baseline:
  returncode=0
  total_sdsc_json=550
  ReStickifyOpHBM=32
  nonempty_lxRelayoutClassifications_=0

metadata:
  returncode=0
  total_sdsc_json=550
  ReStickifyOpHBM=32
  nonempty_lxRelayoutClassifications_=32
  classification_kinds={"matmul_operand_broadcast": 32}
```

Every HBM restickify in the flash main kernel is immediately followed by a `batchmatmul` SDSC carrying:

```text
kind=matmul_operand_broadcast
communication_class=all_gather
communication_pattern=all_gather_replicate
operand_read_index=1
operand_role=rhs
```

So flash attention does not introduce a separate primitive for this spill. It hits the same unresolved staged RHS all-gather/replicate problem as Granite attention.

## Interpretation

PR1-style scatter and small direct gather are not enough for Granite attention. The relevant attention communication class is a staged matmul-operand all-gather/replicate:

```text
producer shards in LX
  -> grouped all-gather / replicate
  -> matmul RHS consumption
```

The destination should not be a full resident RHS tensor. It needs to be fissioned or streamed into the matmul transfer loop, so only a tile/chunk is live at a time.

The S=128 result shows the on-chip path is real when capacity is favorable. The S=512 failure shows why the next feature cannot be naive full-buffer materialization.

The chunked fission diagnostic narrows the S=512 gap further: compact movement SDSCs are feasible, but the DL matmul consumer cannot yet consume those chunks.

The flash-attention probe reinforces that this backend feature is shared. Solving staged matmul-operand all-gather should address both the Granite attention spill class and the standalone flash attention spill class.

## Backend Seam For Staged Matmul RHS

The current backend has useful pieces, but they do not yet compose into a valid staged matmul operand.

Existing pieces:

- `InputFetchNeighbor` / STCDP can describe LX-to-LX movement.
- The scheduler has some `KERNEL`-operand awareness, not only `INPUT`.
- The relayout diagnostic can emit compact fission movement SDSCs for every producer-core chunk.

Missing piece:

- DL matmul still consumes its RHS through one resident `DataInfo` / LDS base pointer.
- Separate chunked movement SDSCs do not become a live per-chunk RHS value for the matmul compute.

Relevant files from the CDX checkout:

```text
dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp
  Partial KERNEL/RHS IFN awareness.

dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
  Detects LX-neighbor only when data-op and DL op are paired in one schedule step.
  Has the likely schedule-tree hook for chunked transfer before compute.

dcg/dcg_manager/dcg_manager.cpp
  Classifies paired datadsc+dldsc steps as IFN and calls generatePcfgIRForDataOpInpFetch.

dsc/dsc2.h
  DataInfo has one myLdsIdx_ / startAddr_ model.

dcc/src/Conversion/DSC2ToDataflowIR/V3/SNComputeLowering.cpp
  Lowers compute inputs from the resident DataInfo address.

dxp/SdscRelayoutInsertion.cpp
  Current prototype emits movement SDSCs, but the staged-matmul path fails closed.
```

The likely production direction is to reuse the existing LX-neighbor scheduling machinery, but represent the matmul RHS relayout as a paired IFN+DL schedule step or a fused schedule tree. The STCDP movement and the batchmatmul RHS chunk loop need to share the same schedule context. A separate pre-matmul movement SDSC is not enough for Granite-sized attention because it either materializes the whole RHS or leaves matmul pointing at a stale/static LDS base.

This matches the current North Star:

1. Torch should classify the edge and decide that this is an all-gather/replicate communication.
2. Torch should avoid weight restickifies and only consider computed activation edges here.
3. Deeptools should synthesize the physical ring movement.
4. The next backend step is staged/fissioned matmul operand realization, not full-buffer LX materialization.

## Next Backend Work

The next prototype should change `matmul_operand_broadcast` realization so that it does not allocate `matmulDstBytes` for the whole RHS on every consumer core.

Promising direction:

- use fission rows/chunks as the allocation unit;
- emit one movement SDSC per chunk or per source-core group;
- redirect the matmul operand LDS only for the current staged chunk;
- avoid requiring the full post-relayout RHS to exist in LX at once.

The current full-materialization path is useful as a proof that the metadata routes to the right backend code, but it is not production-usable for Granite-sized attention.
