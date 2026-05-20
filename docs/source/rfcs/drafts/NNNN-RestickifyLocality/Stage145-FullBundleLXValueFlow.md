# Stage 145: Full-Bundle LX Value-Flow Contract

## Goal

The current goal is stricter than proving an isolated LX-to-LX movement program:

```text
producer output LX
  -> restickify movement
  -> consumer input LX
```

must live inside the normal Torch-Spyre fused bundle shape, with producer,
restickify, and consumer agreeing on the same internal value-flow.

## Compile-Only Contract Probe

I used an isolated pod workspace:

```text
/tmp/torch-spyre-stage145
```

and generated no-launch artifacts for the same-bundle fixture:

```python
def fn(a, b, c, d):
    u = a + (b + c).t()
    return u, u @ d
```

The useful compile-only run used the real LX restickify bridge:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1
SPYRE_RESTICKIFY_DDL_BRIDGE_SOURCE_ADDRESS=compact-lxlu
SPYRE_RESTICKIFY_DDL_BRIDGE_BOUNDARY_PATCH=1
SPYRE_RESTICKIFY_DDL_BRIDGE_OPFUNC=ReStickifyOpLx
SPYRE_RESTICKIFY_DDL_BRIDGE_ALLOW_MULTI_SPLIT=1
SPYRE_RESTICKIFY_DDL_SHIM_SKIP_CORELET_NAMES=0_add,1_ReStickifyOpLx_ddl_bridge,2_add
SPYRE_RESTICKIFY_DDL_SHIM_SKIP_L3_NAMES=1_ReStickifyOpLx_ddl_bridge
SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=1
SPYRE_ALIGN_CORE_MAPPING_CONTINUITY=1
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1
DXP_LX_FRAC_AVAIL=1
```

Command:

```bash
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --copy-kernel-code \
  --kernel-launch-log \
  --skip-kernel-launch \
  --output-dir /tmp/stage145-rslx-2048-continuity \
  --fail-on-error
```

## What Worked

The first generated bundle had the intended three-SDSC shape:

```text
sdsc_0_add.json
sdsc_1_ReStickifyOpLx_ddl_bridge.json
sdsc_2_add.json
bundle.mlir
segment_size.json
```

It was not the stock HBM restickify path. The middle SDSC used:

```text
opFuncName = ReStickifyOpLx
```

and both restickify labeled data spaces were LX-only:

```text
hbmStartAddress_ = -1
memOrg_ = lx
```

The endpoint aliases also matched exactly:

```text
producer output LX start == bridge input LX start
bridge output LX start == consumer input LX start
```

The consumer's restickified input was LX-only:

```text
sdsc_2_add Tensor1: memOrg_ = lx, hbmStartAddress_ = -1
```

The compiler telemetry also reported the modeled producer-to-restickify edge as
zero-hop:

```text
bytes_moved = 8,388,608
ring_total_byte_hops = 0
producer_splits = d0:32
restickify_splits = d0:32
```

So compile-time packaging is now past the earlier "can this be represented in a
normal bundle?" blocker.

## What Failed

The hardware correctness run launched cleanly but produced wrong values:

```text
Mismatched elements: 2,373,297 / 4,194,304 (56.6%)
Greatest absolute difference: 1.044921875
```

The device stayed healthy after the failed correctness check:

```text
health [1.0, 2.0]
```

This means the problem is not currently a device crash. It is a semantic
contract mismatch.

## Diagnosis

The bundle now agrees on physical LX endpoints, but it still does not fully
agree on the logical source view that lives at those endpoints.

For the successful no-launch 2048 artifact:

```text
producer split:  mb:32,out:1
bridge split:    mb:1,out:32
consumer split:  mb:1,out:32
```

The bridge and consumer are aligned with each other, and the LX addresses alias.
But the producer writes the tensor in its own physical output ownership/layout,
while the restickify bridge reads the source as the restickify logical source
view. In the HBM path, HBM is the global exchange point that makes that legal.
With local LX aliases, the bridge can only read the local region actually
present on that core unless the bridge explicitly fetches remote-core regions.

So "LX endpoint aliasing" is necessary but not sufficient. A value-correct
LX-to-LX restickify needs one of:

1. a producer output contract that writes the exact source view the LX
   restickify consumes;
2. an explicit remote-LX movement/fetch step that materializes the source view;
3. a locality certificate strong enough to prove the producer physical view and
   restickify source view are the same per core.

This run had the endpoint alias but not the full source-view proof.

## Negative Matrix

I also checked the interslice route again:

| Variant | Size | Result |
|---|---:|---|
| `interslicetranspose_fp16`, reference contract | 512 | DXP cardinality mismatch |
| `interslicetranspose_fp16`, no reference contract | 512 | DDC allocation failure |
| `ReStickifyOpLx` bridge | 512 | compile-only pass, but split contract remains mismatched |
| `ReStickifyOpLx` bridge | 2048 | compile-only pass, hardware launch wrong values |

The `ReStickifyOpLx` bridge is still the most production-shaped path, but
blind LX aliasing is not value-correct.

## Next Step

The next implementation should make the source-view contract first-class:

1. Add a verifier that compares producer physical output view, restickify
   logical source view, bridge input view, and consumer input view.
2. Gate any LX bridge launch on that verifier, so endpoint aliasing alone cannot
   pass as "locality".
3. For the general transposed source case, use the descriptor-driven
   `InputFetchNeighbor`/remote-LX movement path or a Deeptools-native bridge
   that explicitly fetches the missing remote regions before the consumer reads
   LX.

The immediate blocker is now:

```text
make producer physical source view == restickify LX source view,
or explicitly materialize that source view with remote LX movement.
```
