# Stage 62: LX Bridge Runtime Fixture

## Summary

Stage 62 found a cleaner runtime fixture for the LX-local restickify bridge and
used it to separate three facts that were previously tangled together:

1. stock `ReStickifyOpHBM` still runs correctly with the stock Deeptools
   templates;
2. the DDL-shaped LX bridge still produces an HBM/L3-free generated program;
3. that generated program does not retire on hardware yet.

So the current status is: we have a stronger compiler-artifact proof of an
LX-local restickify shape, but not an end-to-end runtime proof.

## New Fixture

The old positive Stage 3B case, `adds_then_matmul`, contains two restickifies:
one graph-input restickify and one in-graph restickify. That makes it noisy for
LX bridge work because the graph-input row is intentionally outside the bridge's
scope.

Stage 62 added three probe cases to make the fixture space more precise:

```text
plain_adds_then_matmul
computed_transpose_adds_then_matmul
computed_transpose_join
```

The useful case is:

```python
def computed_transpose_adds_then_matmul(a, b, c, d):
    return (a + (b + c).t()) @ d
```

At size `2048`, this emits exactly one restickify, and its source is
`in_graph_computed`:

```text
restickify_count: 1
source_kind: in_graph_computed
bytes moved: 8,388,608
producer split: d0:32
restickify split: d0:32
modeled byte-hops: 0
```

This is the best current fixture for the LX bridge because it removes the
unoptimizable graph-input restickify from the test.

## Stock Control

With the stock template path:

```text
DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
```

`computed_transpose_adds_then_matmul`, size `2048`, runs through both generated
bundles:

```text
sdsc_fused_add_t_0:
  sdsc_0_add.json
  sdsc_1_ReStickifyOpHBM.json
  sdsc_2_add.json

sdsc_fused_mm_1:
  sdsc_0_batchmatmul.json
```

The launch log reaches `after_sync` for both bundles. This confirms that the
fixture itself is healthy and that stock HBM restickify is not the failure mode.

## DDL Bridge Candidate

With the DDL bridge enabled:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1
DEEPTOOLS_PATH=/tmp/stage50-template-share
```

Torch-Spyre emits the DDL bridge for the in-graph restickify:

```json
{"status":"emitted","op":"ReStickifyOpHBM","source_kind":"in_graph_computed","source_name":"buf0","work_slices":{"mb":1,"out":32}}
```

The add bundle becomes:

```text
sdsc_fused_add_t_0:
  sdsc_0_add.json
  sdsc_1_ReStickifyOpHBM_ddl_bridge.json
  sdsc_2_add.json
```

The bridge SDSC uses LX allocations and preserves the runtime tensor segments
for this fixture:

```text
input segment: stack
output segment: heap
schedule: allocate lx, allocate lx, local lx transfer, loop nest, local lx transfer
```

The generated bridge op still names the op function `ReStickifyOpHBM`, but its
SDSC is DDL-shaped and LX-only.

## Generated Program Evidence

`DUMP_SPYRE_CODE=1` shows the DDL bridge lowers to an HBM/L3-free restickify
program, while the stock restickify uses the normal L3/LX path.

Instruction-prefix summary for the restickify SDSC:

| mode | HBM refs | L3 load units | L3 store units | LX load units | LX store units | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stock `ReStickifyOpHBM` | 0 | 128 | 128 | 64 | 64 | normal stock lowering |
| DDL bridge | 0 | 0 | 0 | 0 | 32 | compact LX-local bridge lowering |

The full DDL bridge `senprog.txt` still contains SFP/PE/PT programs. The table
above is focused on the memory-movement instruction prefixes because the
question is whether this path goes through HBM/L3. It does not, at least in the
generated artifact.

## Hardware Result

The DDL bridge bundle launches, but it does not retire:

```text
before_launch sdsc_fused_add_t_0
  [sdsc_0_add.json, sdsc_1_ReStickifyOpHBM_ddl_bridge.json, sdsc_2_add.json]
after_launch sdsc_fused_add_t_0
before_sync sdsc_fused_add_t_0
... about 280 seconds ...
after_sync sdsc_fused_add_t_0
before_launch sdsc_fused_mm_1
launch_exception: stream in error state
```

The runtime reports a compute control-block timeout and marks the stream in an
error state before the next matmul bundle can run:

```text
PipelineId(COMPUTE)
Compute CB hardware error detected
Fail on RB time-out
Cannot schedule operation on stream in error state
```

Interpretation: the DDL bridge has moved past DXP compilation and kernel launch,
but the generated program/control-block contract is still not correct enough to
retire on hardware.

## `ReStickifyOpLx` Variant

Stage 62 also added an experimental probe knob:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_OPFUNC=ReStickifyOpLx
```

With a matching template variant where `restickify.ddl` uses
`opFuncName="ReStickifyOpLx"`, Torch-Spyre emits:

```text
sdsc_1_ReStickifyOpLx_ddl_bridge.json
computeOp_.opFuncName: ReStickifyOpLx
```

This variant fails earlier, inside DXP:

```text
DtException: sizeIdx >= 0
/project_src/deeptools/dsc/dsc2.cpp line 2684
```

That is still useful: `ReStickifyOpLx` is probably not a drop-in opfunc rename
for the current DDL bridge shape. It needs a different Deeptools-side contract,
not only a Torch-Spyre name change.

## Current Conclusion

We have not yet proven end-to-end LX-to-LX restickify execution without an HBM
round trip.

What we have proven:

- a clean in-graph restickify fixture exists with no graph-input restickify;
- stock HBM restickify runs for that fixture;
- the DDL bridge can emit an LX-only restickify SDSC for that fixture;
- DXP can lower the DDL bridge into a generated program with no HBM/L3 movement;
- the generated DDL bridge launches on hardware.

What remains unproven:

- the DDL bridge retires on hardware;
- the DDL bridge preserves tensor values;
- the bridge moves data across core LX memories correctly;
- the bridge improves runtime.

## Recommendation

Continue, but keep this as a separate prototype branch. We are close enough that
this is worth one more focused blocker pass, but not close enough to fold into a
production Stage 3B PR.

The next blocker is lower than the Torch-Spyre pass:

1. inspect the generated DDL bridge control-block/status/fence behavior against
   the stock `ReStickifyOpHBM` program;
2. compare DDL bridge `scheduleTree_` transfers against a known-good DDL data-op
   that retires;
3. find why the bridge generated program has no LXLU-side memory unit activity
   in the summary while stock restickify has both LXLU and LXSU;
4. decide whether the bridge needs a real Deeptools `ReStickifyOpLx` contract
   rather than reusing `ReStickifyOpHBM` inside an LX-only DDL envelope.

Until that is solved, Stage 3B should remain the production-facing path, and the
DDL/LX bridge should be treated as the proof-of-physical-locality sidequest.
