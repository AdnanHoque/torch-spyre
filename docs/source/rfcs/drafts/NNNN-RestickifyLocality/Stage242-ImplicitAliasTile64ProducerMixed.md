# Stage 242: Implicit-Alias Tile64 Producer-Mixed Probe

## Goal

Move the implicit same-bundle PT-LX prototype closer to the production-shaped
contract:

```text
producer DLDsc
  -> streaming 64x64 PT-LX bridge data ops
  -> consumer DLDsc reading the bridge output from LX
```

The previous implicit-alias prototype chose the largest tile that fit in LX
workspace. For a 512x512 tensor this became one 512x512 tile, which did not
exercise the desired bounded streaming contract. This stage makes implicit
streaming default to 64x64 tiles and packages the bridge data ops with the
producer DLDsc rather than as a standalone bridge or consumer-mixed SDSC.

## Changes

- Added `SPYRE_RESTICKIFY_PTLX_STREAMING_TILE_SIZE`.
  - Default: `64`.
  - `auto` keeps the old largest-fitting-tile behavior for debugging.
- Changed implicit-alias streaming to default to producer-mixed packaging:
  - producer DLDsc remains in `dscs_`;
  - PT-LX bridge data ops are attached under `datadscs_`;
  - `coreIdToDscSchedule` runs producer DLDsc before bridge data ops;
  - patched consumer remains the next SDSC and reads the bridge output from LX.
- Preserved a diagnostic split mode:
  - `SPYRE_RESTICKIFY_PTLX_IMPLICIT_ALIAS_SPLIT_BRIDGE=1`
  - This emits a bridge-only SDSC, but DXP rejects pure data-op SDSCs.
- Fixed mixed-SDSC metadata to preserve DLDsc opfunc names in `opFuncsUsed_`
  alongside bridge data-op names.

## Validation

Pod unit/static:

```sh
python -m py_compile torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
26 passed
```

## Same-Bundle Implicit-Alias Result

Command shape:

```sh
LX_PLANNING=1
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=20
SPYRE_RESTICKIFY_USE_SPECIFIC_INSERT=1
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
python tools/restickify_scenario_probe.py \
  --case computed_self_transpose_join \
  --size 512 \
  --skip-correctness \
  --skip-kernel-launch \
  --output-dir /tmp/stage242-producer-mixed-opfuncs \
  --fail-on-error
```

The compiler emitted a valid 64x64 LX-only bridge contract:

```text
kind: ptlx-implicit-alias-producer-streaming
tile_size: 64
total_tiles: 64
datadsc_count: 192
max_fan_in: 4
max_fan_out: 4
has_hbm_restickify: false
hbm_placements: 0
value_flow_contract.valid: true
```

But DXP still rejected the generated producer-mixed SDSC:

```text
DtException: There must be at least one valid candidate.
file L3DlOpsScheduler.cpp line 1075
```

The diagnostic split bridge mode changed the failure to:

```text
DtException: Datadsc not allowed without dldsc schedule.
file SdscTree.cpp line 155
```

That confirms pure data-op bridge SDSCs are not a viable bundle item.

## Cross-Bundle Control

The existing cross-bundle producer-mixed PT-LX path still compiles for the
known high-signal family:

```sh
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=7
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E=1
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=1048576
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 512 --size 1024 --size 2048 \
  --skip-correctness \
  --skip-kernel-launch \
  --output-dir /tmp/stage242-cross-bundle-control \
  --fail-on-error
```

Result:

```text
ok size=512  restickifies=2 bytes=1048576  byte_hops=0
ok size=1024 restickifies=2 bytes=4194304  byte_hops=0
ok size=2048 restickifies=2 bytes=16777216 byte_hops=0
Completed 3 rows with 0 errors
```

## Interpretation

This stage narrows the production gap:

- The 64x64 streaming descriptor and LX value-flow contract exist.
- Producer-mixed packaging is the right bundle shape; standalone data-op SDSCs
  are rejected.
- Deeptools accepts producer-mixed PT-LX bridges for the cross-bundle
  `adds_then_matmul` family across 512, 1024, and 2048.
- The same-bundle implicit-alias family still fails L3 candidate selection when
  the bridge requires fragmented gather/scatter around a local SFP consumer.

The next production-shaped step should not be another packaging variant. It
should either:

1. coalesce the same-bundle implicit-alias fragmented tiles into a scheduler
   shape closer to the successful cross-bundle row/stripe contracts; or
2. fail closed for this implicit-alias family and focus first on the
   cross-bundle restickify family that already compiles without HBM for
   multiple sizes.

