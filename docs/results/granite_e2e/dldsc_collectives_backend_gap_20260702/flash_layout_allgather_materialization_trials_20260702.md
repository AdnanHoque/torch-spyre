# Flash DLDSC Layout-Allgather Materialization Trials - 2026-07-02

## Context

Branch direction is DLDSC. The older `ah/comms-collectives-dldsc-agent` branch was reviewed and is useful context, but it is superseded by the current `ah/comms-collectives` branch: the later branch already has guarded LX restickify emission and broader DLDSC collective metadata.

The failing flash attention edge is not PR1 scatter. It is:

```text
mul -> ReStickifyOpLx -> batchmatmul Tensor1 KERNEL input
```

Communication class: grouped layout-allgather/restickify.

For the representative failing edge:

```text
producer/restickify work split: mb=4, x=8, out=1
consumer batchmatmul split:   x=4, mb=8, out=1, in=1
rename:
  restickify.x   -> batchmatmul.out
  restickify.out -> batchmatmul.in
  restickify.mb  -> batchmatmul.x
```

Each producer chunk should only fan out to the 8 consumer cores in the same group. Example:

```text
source core 4 = group 0, producerChunk 1
expected destinations = 0,4,8,12,16,20,24,28
```

## Runs

All runs below were on `adnan-cdx-spyre-dev-pf` under:

```text
/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525
```

### Baseline DLDSC Runtime Failure

```text
runs/test_flash_lrfimm_split_autoload_20260702_151321
```

Result:

```text
ReStickifyOpHBM removed for this edge.
32 layout_allgather_restickify backend plans emitted.
Runtime reaches kernels but fails correctness.
Mismatch: 16222516 / 16777216 (96.7%)
```

### Torch Consumer Coordinate Probe

Patch archived in:

```text
runs/failed_probe_diffs_20260702_162125/torch_consumer_coordinate_probe.diff
```

Run:

```text
runs/test_flash_lrfimm_split_torchcoordfix_20260702_160426
```

Result:

```text
SDSC Tensor1 LX residency map becomes consumer-labelled:
  core0 -> out=0,in=0,x=0
  core4 -> out=1,in=0,x=0
DXP accepts it, but runtime remains value-wrong.
Mismatch: ~99.2%
```

Conclusion: frontend coordinate labeling alone is not enough.

### Generic Deeptools Relayout Probe

Patch archived in:

```text
runs/failed_probe_diffs_20260702_162125/deeptools_generic_relayout_probe.diff
```

Run:

```text
runs/test_flash_lrfimm_split_genericrelayout_20260702_161303
```

Result:

```text
DXP falls through to generic LxRelayout path.
Runtime remains value-wrong.
Mismatch: ~99.2%
```

Conclusion: generic relayout is not a correct shortcut for this renamed layout-allgather class.

### Grouped Fanout Prototype

Patch is included in `deeptools_layout_allgather_experiment_current.diff` in this artifact directory.

Run:

```text
runs/test_flash_lrfimm_split_groupedfanout_fresh_20260702_163038
```

Result:

```text
Uses layoutAllgatherRestickifyLogicalTransfers() to restrict fanout to same-group destinations.
Runtime remains value-wrong.
Mismatch: ~99.2%
```

Conclusion: all-core fanout was a real bug, but not the only bug.

### Expanded Destination Allocation Prototype

Run:

```text
runs/test_flash_lrfimm_split_groupedalloc_20260702_164006
```

Result:

```text
Allocates a 1 MB destination operand per consumer core.
Updates the consumer input LX base to the destination allocation.
Runtime remains value-wrong.
Mismatch: ~99.2%
```

Conclusion: missing resident destination capacity was also a real gap, but not sufficient.

### Materialization Debug / Local-Range Prototype

Run:

```text
runs/test_flash_lrfimm_split_materialization_debug_20260702_164606
runs/test_flash_lrfimm_split_localrange_20260702_165446
```

Representative materialization debug:

```text
out_piece_size=1.04858e+06
source=4 src_base=131072 dests=0:131072,4:131072,...
global_dim_start=in:0,out:512,x:0
local_dim_start=in:0,out:0,x:0
dim_size=in:128,out:512,x:1
chunk_index=1 chunk_bytes=131072
```

Result:

```text
Runtime remains value-wrong.
Mismatch: ~99.2%
```

Conclusion: encoding destination chunk placement through byte address offsets did not fix the issue either.

## Current Read

The invariant ~99.2% mismatch across materially different movement descriptions suggests the inserted mixed `dataOpdscs_` rows may not actually be realized/executed in this scheduled path, or STCDPOpLx cannot express this translated copy through the current piece-overlap interface.

The backend gap is now narrower:

1. DLDSC sees the incompatible tensor distribution.
2. Torch emits the classification and dimension rename.
3. Deeptools recognizes the class and emits plan artifacts.
4. The physical materialization is not value-correct.

Likely missing backend capability:

```text
translated LX->LX copy / grouped all-gather materialization
source: local producer shard coordinates and source LX base
destination: consumer-local/full operand placement and destination LX base/range
schedule: movement before consuming batchmatmul
```

The existing STCDPOpLx overlap model appears naturally suited to same-coordinate overlap, but this class needs source-local to destination-global or destination-local translation.

## Side Results

### dev-pf Granite Attempt

Run:

```text
/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/granite_prefill_profile_current_dldsc_20260702_162328
```

Result:

```text
Exit 255 during warmup.
RAS::PCI::BusFence code 0xa35e.
No profile timings.
SDSC evidence: attention activation relayout is ReStickifyOpLx, while remaining ReStickifyOpHBM rows are weight-shaped.
```

### CLC Unit-Test Gap

Existing tests validate parser/logical transfer shape, but not DXP physical piece materialization.

Missing assertions should check:

```text
source core 4 -> dest cores 0,4,8,12,16,20,24,28 only
source core 31 -> dest cores 3,7,11,15,19,23,27,31 only
producer chunk offsets preserved into consumer out ranges
no all-32 fanout
piece start/size or translated byte ranges match expected materialization
```

A disabled local gtest sketch was added on CLC only under:

```text
/home/adnan/codex-isolated/dldsc_clc_refresh_20260702_154247/deeptools/util/test/LayoutAllgatherRestickify_unit_test.cpp
```

## Next Step

Stop trying to prove this with mixed `dataOpdscs_` inside the consumer SDSC until Deeptools confirms that path is executable for this shape.

Next backend prototype should be one of:

1. A standalone relayout SDSC/DLDSC row that Deeptools already schedules reliably, but with explicit translated source/destination ranges.
2. A proper extension to STCDPOpLx that supports source-local to destination-local translated ranges directly.
3. A loop-scoped input-fetch path for the matmul operand that does not require resident full operand materialization.

The production direction remains DLDSC: frontend classifies and costs the mismatch; backend synthesizes the movement.
