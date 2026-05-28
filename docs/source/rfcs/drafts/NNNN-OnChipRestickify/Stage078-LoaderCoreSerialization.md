# Stage078 - Loader-Core Serialization

## Question

Does the same-row K/V HBM prefetch failure come from global overlap, or from
the loader core overlapping its own current attention compute slice?

## Lead-In

Stage178 showed that widening L3/LX sync to include corelet 1 did not fix the
full-tile overlap mismatch. Stage179 and Stage180 then moved the loader LX
source away from the current consumer range and preserved LXLU/LXSU node
corelet selection, but both still produced the same mismatch pattern.

Stage181 moved the loader core from core 0 to core 31:

```text
variant: onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_core31_probe
shape: B1 H8 L128 D64
env:
  SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_CORELET1=1
  DT_DSC_L3_LX_SYNC_BOTH_CORELETS=1
  DT_DCC_LXLUSUFIFO_USE_NODE_CORELET=1
  CODEGEN_DUMP_IRS=1
status: failed
mismatched: 505 / 65536
max abs: 0.361083984375 at (0, 6, 127, 23)
max rel: 163.0 at (0, 1, 127, 49)
```

The worst row moved to row 127, which is in core 31's current compute slice for
L128 with 32 cores. This strongly suggested a core-local overlap hazard.

## Implementation

Added a default-off torch-spyre schedule knob:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIALIZE_LOADER_CORE=1
```

The new schedule only changes cores that run the loader HBM prefetch dataop.
Those cores now run the prefetch load and current compute in separate rows.
All other cores keep the same overlapped compute row.

For the core31 full-tile loader-fanout shape, the current-prefetch schedule
now becomes:

```text
core0:  [[0, -1, 0, 1], [-1, 0, 1, 1], [2, -1, 1, 1], [3, -1, 1, 0]]
core31: [[0, -1, 0, 1], [1, -1, 1, 1], [-1, 0, 1, 1], [2, -1, 1, 1], [3, -1, 1, 0]]
ops:    ["nop", "STCDPOpHBM", "nop", "STCDPOpLx"]
```

The sweep variant is:

```text
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_core31_serialize_loader_probe
```

A shorter alias for this first correct loader-specialized path is also
available:

```text
onchip_warpspec_kv_hbm_prefetch_loader_core31
```

## Results

Stage182 kept the corelet-1 and lower-stack diagnostic sync/routing envs:

```text
shape: B1 H8 L128 D64
status: ok
median: 0.575630 ms
max abs error: 0.00341796875
mixed SDSCs: 10
```

Stage183 removed the lower-stack diagnostic envs and kept only corelet1:

```text
shape: B1 H8 L128 D64
status: ok
median: 0.532854 ms
max abs error: 0.00341796875
mixed SDSCs: 10
```

Stage184 removed corelet1 as well, using the default corelet-0 dataop route:

```text
shape: B1 H8 L128 D64
status: ok
median: 0.536190 ms
max abs error: 0.00341796875
mixed SDSCs: 10
```

Stage185 expanded the same default-corelet path to L256:

```text
shape: B1 H8 L256 D64
status: ok
median: 0.724677 ms
max abs error: 0.004638671875
mixed SDSCs: 20
```

Stage186 verified the shorter alias:

```text
variant: onchip_warpspec_kv_hbm_prefetch_loader_core31
shape: B1 H8 L128 D64
status: ok
median: 0.528784 ms
max abs error: 0.00341796875
mixed SDSCs: 10
```

The focused logic tests on the pod also passed:

```text
313 passed in 2.34s
```

## Interpretation

The passing Stage184/Stage185 path means the practical correctness invariant is
not "use corelet 1" and not "add broader L3/LX sync." The invariant is:

```text
Do not overlap the loader core's HBM prefetch data movement with that same
core's current attention compute slice.
```

Other cores can still compute while the loader core performs the HBM prefetch.
This is a usable AIU analogue of warp specialization, but it is currently a
partially reserved-loader-core schedule rather than a fully redistributed
31-compute-core attention schedule.

## Next Step

Promote this from a probe into the main experimental warp-specialized path:

- pick a stable public knob or variant name for "loader core serialized"
- keep the lower-stack corelet diagnostics default-off
- benchmark more lengths and batch/head cases
- decide whether to redistribute the loader core's compute slice onto the
  remaining cores, or accept the one-core local serialization as the first
  correct warp-specialized AIU schedule
