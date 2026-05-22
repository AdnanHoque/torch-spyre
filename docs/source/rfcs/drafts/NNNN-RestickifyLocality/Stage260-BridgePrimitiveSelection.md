# Stage 260: Bridge Primitive Selection

## Goal

Fix the Stage259 endpoint-contract blocker by selecting the bridge primitive
from the actual source and destination descriptors.

Stage259 showed that the direct PT-LX tile sidecar was wrong for the current
generated `computed_transpose_adds_then_matmul_tuple` edge:

```text
bridge output:       layout out,mb  stick mb
destination expects: layout mb,out  stick out
```

The compiler evidence says this edge does not need a PT-LX layout/stick
transform at the bridge boundary.  It needs same-layout LX ownership
materialization: producer-owned tiles must move to destination-owned tiles while
preserving `layout mb,out` and `stick out`.

## Code Changes

Added a sidecar and mixed-schedule bridge selector:

```text
same source/destination layout+stick -> tiled STCDPOpLx LX remap
otherwise                         -> direct tiled ReStickifyOpWithPTLx path
```

The new same-layout bridge emits one tiled `STCDPOpLx` movement per logical
64x64 tile:

```text
coalescing: same-layout-lx-ownership-remap-64x64-tiles
semantic_transform_certified: true
fallback: ReStickifyOpHBM
```

This is still default-off and keeps the stock HBM fallback.

## Sidecar Validation

Probe:

```text
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size <size> \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code \
  --output-dir /tmp/stage260-bridge-selector-<size> \
  --fail-on-error
```

### Size 512

```text
bridge_kind:                    same-layout-lx-ownership-remap
bridge_endpoint_contract_valid: true
bridge layout/stick:            mb,out / out
destination layout/stick:       mb,out / out
tile records:                   64
bridge data ops:                64
ops used:                       STCDPOpLx
contains ReStickifyOpHBM:       no
```

### Size 2048

```text
bridge_kind:                    same-layout-lx-ownership-remap
bridge_endpoint_contract_valid: true
bridge layout/stick:            mb,out / out
destination layout/stick:       mb,out / out
tile records:                   1024
bridge data ops:                1024
ops used:                       STCDPOpLx
contains ReStickifyOpHBM:       no
```

## Mixed-Schedule Validation

With explicit non-overlapping prototype endpoints:

```text
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=262144
```

the 512 no-launch compiler probe now patches the generated bundle:

```text
status: patched
kind: ptlx-streaming-mixed-schedule
replacement_sdsc: 1_StreamingMixedReStickifyOpWithPTLxConsumer
coalescing: same-layout-lx-ownership-remap-64x64-tiles
datadsc_count: 64
has_hbm_restickify: false
endpoint_contract_valid: true
semantic_transform_certified: true
consumer_descriptor_valid: true
value_preservation_valid: true
valid: true
```

The replacement SDSC contains:

```text
ops: STCDPOpLx, add
streamingLXRemapFull_: present
streamingPTLXFull_: absent
```

For 2048 with explicit endpoints, the existing non-streaming mixed path still
patches first because the full-tensor bridge is representable:

```text
status: patched
kind: ptlx-mixed-schedule
replacement_sdsc: 1_MixedReStickifyOpWithPTLxConsumer
```

The streaming selector is therefore most important for non-2048/small shapes
where the old PT-LX path could not certify a valid direct transform.

## Tests

In the pod:

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_tile_ownership_probe.py \
  tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
60 passed in 6.86s
```

## Interpretation

This stage fixes the semantic mismatch discovered in Stage259.  The compiler no
longer blindly treats every restickify boundary as a PT-LX transform.  It first
checks whether source and destination descriptors already agree; when they do,
it emits a tiled LX ownership remap that preserves the descriptor and satisfies
the producer-bridge-consumer contract.

What remains before production enablement:

- replace forced endpoint bases with allocator-proven endpoint allocations;
- hardware-run the patched mixed bundle for 512 and verify values;
- decide whether the 2048 path should also be forced through streaming tiles or
  continue using the existing full mixed bridge when representable;
- add final legality gates and default-off rollout controls.

## Artifacts

```text
artifacts/stage260_bridge_selector/
```
