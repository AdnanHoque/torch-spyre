# Stage 256: Interslice Contract Refresh

## Goal

Move the PT-LX prototype back toward the value-correct inter-slice route and
test it against current-main restickify behavior.

Stage255 showed that the consumer-shaped `ReStickifyOpWithPTLx`/`validGap_`
diagnostic can remove `ReStickifyOpHBM` but still returns wrong values.  This
stage revisited the older `interslicetranspose_fp16` route because it is the
only route that previously produced value-correct LX-to-LX results.

## Code Changes

The compact interslice bridge now matches the DDL template more closely:

- reference interslice SDSCs use one corelet, matching the earlier
  value-correct artifact;
- the compact reference layout no longer adds a synthetic `y` dimension;
- the reference bridge output now uses the consumer/output stick only instead
  of carrying both input and output sticks into the consumer.

The output-stick change matters because the old generated bridge described the
consumer input as a two-stick PT-packed layout:

```text
layout: mb,out
stick:  out,mb
stick sizes: 8,8
```

The following `add` consumer rejected that through `broadcast_ops.ddl` with a
stick-size mismatch.  The refreshed bridge instead produces:

```text
layout: mb,out
stick:  out
stick size: 64
```

## Unit Validation

In the pod:

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_ddl_bridge.py \
  tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
63 passed in 8.40s
```

## Hardware Probes

The current-main 512 tuple case now gives the restickify a multi-dimensional
split:

```text
work_slices: mb:4,out:8
```

The conservative DDL bridge correctly skips this shape unless
`SPYRE_RESTICKIFY_DDL_BRIDGE_ALLOW_MULTI_SPLIT=1` is enabled.

For the high-signal 2048 tuple case, the bridge is emitted:

```text
work_slices: mb:1,out:32
status: emitted
opfunc: interslicetranspose_fp16
```

With the refreshed 2D/single-stick contract, the no-HBM bridge compiles and
launches at 2048.  The stock HBM restickify is absent from the first bundle, but
correctness still fails:

```text
Mismatched elements: 1783604 / 4194304 (42.5%)
Greatest absolute difference: 0.70703125 at index (864, 1207)
```

Running with `SPYRE_RESTICKIFY_DDL_BRIDGE_VALUE_FLOW_ASSERT=1` explains the
failure before launch:

```text
producer->bridge mismatches=[
  "num_work_slices_per_dim",
  "core_id_to_work_slice",
  "primary"
]
bridge->consumer mismatches=[]
```

The generated 2048 bundle makes the problem concrete:

```text
producer output:
  layout mb,out
  stick  out
  split  mb:32,out:1

bridge input:
  layout mb,out
  stick  mb
  split  mb:1,out:32

bridge output:
  layout mb,out
  stick  out
  split  mb:1,out:32

consumer input:
  layout mb,out
  stick  out
  split  mb:1,out:32
```

The bridge and consumer now agree.  The remaining mismatch is producer to
bridge: the producer writes one physical ownership view, while the bridge reads
the transposed/restickified source view from a different set of cores.

## Interpretation

This stage removes two false blockers:

1. the current interslice path can compile and launch at 2048 after the compact
   contract fix;
2. the bridge output can be made compatible with the following consumer.

The real blocker is now exactly the production-shaped one:

```text
producer-owned LX fragments
  -> remote 64x64 gather/scatter over RIU
  -> consumer-owned LX tile
```

Compact LX aliasing is not enough, because bridge core `k` cannot assume the
needed producer fragment also lives on core `k`.  For the 2048 case, producer
ownership is row/`mb` based while bridge/consumer ownership is column/`out`
based.

Trying to patch the producer output layout and propagate the bridge core mapping
back into the producer is too invasive: it breaks normal producer scheduling.
That reinforces the production direction: the bridge must consume producer
ownership records and explicitly materialize the consumer view, rather than
rewriting the producer to fit the bridge.

## Next Step

Use the Stage70 `InputFetchNeighbor`/source-fragment direction as the production
path.  The next prototype should generate a real internal-edge descriptor from
the producer and consumer SDSCs:

```text
producer real LX output pieces
  -> STCDPOpLx/InputFetchNeighbor gather
  -> local PT/interslice restickify tile
  -> STCDPOpLx/InputFetchNeighbor scatter or consumer LX write
```

The fallback remains the stock `ReStickifyOpHBM` unless this descriptor proves:

- source is in-graph computed;
- producer, bridge, and consumer views are all known;
- every 64x64 tile has bounded LX workspace;
- required remote fragments are explicitly represented;
- bridge output matches the consumer input view.

## Artifacts

```text
artifacts/stage256_interslice_contract/
```
