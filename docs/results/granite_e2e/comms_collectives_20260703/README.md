# Granite DLDSC Collectives Update - 2026-07-03

This captures the CDX diagnostic run that separated three cases:

1. direct synthetic LX gather works;
2. Granite can emit matmul-operand all-gather metadata;
3. Granite cannot yet realize that metadata because the current backend path tries to materialize the full post-relayout RHS in LX.

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

## Interpretation

PR1-style scatter and small direct gather are not enough for Granite attention. The relevant attention communication class is a staged matmul-operand all-gather/replicate:

```text
producer shards in LX
  -> grouped all-gather / replicate
  -> matmul RHS consumption
```

The destination should not be a full resident RHS tensor. It needs to be fissioned or streamed into the matmul transfer loop, so only a tile/chunk is live at a time.

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

