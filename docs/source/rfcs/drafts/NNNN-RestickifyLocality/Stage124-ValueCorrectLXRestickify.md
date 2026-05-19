# Stage 124: Value-Correct LX Restickify Goal

## Goal

The active target is now stricter than the earlier Stage 3B byte-hop result:

```text
producer -> restickify -> consumer
```

should execute as an LX-to-LX restickify path, avoid the normal
`ReStickifyOpHBM` internal round trip, retire on hardware, and match the CPU
reference values.

## Smallest Useful Fixture

The high-signal `adds_then_matmul` case is still the best telemetry case, but
it is not the best first integration fixture because the producer and
restickify are split across runtime bundles. The current value-correctness work
therefore uses the same-bundle fixture:

```python
def computed_transpose_adds_then_matmul(a, b, c, d):
    return (a + (b + c).t()) @ d
```

The first fused add bundle has:

```text
sdsc_0_add
sdsc_1_ReStickifyOpHBM
sdsc_2_add
```

That gives us an adjacent producer/restickify/consumer triple inside one
runtime artifact.

## What The Current Prototype Does

With:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1
SPYRE_RESTICKIFY_DDL_BRIDGE_SOURCE_ADDRESS=compact-lxlu
```

Torch-Spyre can replace the middle `ReStickifyOpHBM` SDSC with a compact DDL
bridge. Deeptools lowers that bridge to a program containing LX/SFP/PT work and
no visible L3/HBM tokens for the bridge frame.

For `computed_transpose_adds_then_matmul`, size `2048`, the bridge emits and
retires, but correctness still fails:

```text
mismatched elements: about 83% to 84%
```

## Key Finding

The restickify bridge itself writes an LX result, but the following consumer
SDSC is still lowered as a normal tensor boundary. DXP reintroduces an
HBM-backed load for the consumer input:

```text
bridge:
  transfer_lds2_src:ptrow*_dst:lxsu   # writes restickified data into LX

consumer:
  transfer_lds1_src:hbm_dst:lx        # reloads the same logical tensor from HBM
  transfer_lds1_src:lxlu_dst:sfp      # compute consumes the reloaded value
```

So the current integration is not yet a real internal LX edge. The bridge can
compute a local result, but the consumer does not consume that local result.
It consumes the ordinary HBM-backed SDSC boundary instead.

## Transition Sweep

Default bridge eligibility requires one dominant split dimension. Smaller
same-bundle cases pass because the bridge is skipped:

| Size | Bridge Status | Result |
|---:|---|---|
| `64` | skipped, no dominant split | value-correct HBM path |
| `128` | skipped, multi-split | value-correct HBM path |
| `256` | skipped, multi-split | value-correct HBM path |
| `512` | skipped, multi-split | value-correct HBM path |
| `768` | skipped, multi-split | value-correct HBM path |
| `1024` | skipped, multi-split | value-correct HBM path |
| `1536` | emitted | value mismatch |
| `2048` | emitted | value mismatch |

I added a default-off probe flag:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_ALLOW_MULTI_SPLIT=1
```

This lets us force the DDL bridge on multi-split cases. It is intentionally not
production behavior. With the flag enabled, the bridge emits for sizes that
normally skip, but the same consumer-boundary issue remains:

| Size | Work Slices | Result |
|---:|---|---|
| `128` | `mb:2,out:2` | DXP stitch failure |
| `256` | `mb:4,out:4` | DXP stitch failure |
| `512` | `mb:4,out:8` | value mismatch, about `79%` |
| `768` | `mb:2,out:12` | DXP stitch failure |
| `1024` | `mb:2,out:16` | value mismatch, about `83%` |

This confirms the correctness problem is not specific to the single-split
`1536/2048` shape. It is the missing internal LX consumer contract.

## Consumer Transplant Probe

I also tried a lower-level manual transplant:

1. start from the generated `512` DDL-bridge bundle;
2. replace `sdsc_2_add.json` with the already-scheduled consumer debug JSON;
3. remove only the `Tensor1` HBM allocation and `transfer_lds1_src:hbm_dst:lx`;
4. keep the consumer's `transfer_lds1_src:lxlu_dst:sfp`;
5. rerun DXP while skipping generic corelet splitting and L3 scheduling for
   that consumer.

The first attempt failed import because the LX allocation still listed the
removed HBM transfer as an allocation user. After fixing that, DXP advanced but
DDC rejected the already-scheduled consumer:

```text
DtException: External transfer node improperly set:
transfer_lds2_src:sfp_dst:lxsu
```

This says the direction is plausible but the current hook point is wrong. We
need either a first-class internal-edge schedule object before DXP finalizes the
consumer, or a lower-level packaging path that can splice a patched scheduled
consumer frame without re-running DDC on it.

## Current Conclusion

We have not yet achieved value-correct LX-to-LX restickify in a Torch-Spyre
compiled graph.

What we have proven:

- the DDL bridge can emit an LX/SFP/PT-only restickify frame;
- the frame can be packaged and retired on hardware;
- the remaining correctness failure is now localized to the SDSC boundary
  after the bridge, where the consumer still reloads the tensor from HBM;
- forcing more shapes into the bridge does not fix correctness, which rules out
  "just find a smaller shape" as the solution.

## Next Implementation Step

The next prototype should stop replacing only the restickify SDSC. It must
replace the whole adjacent triple's boundary contract:

```text
producer output LX allocation
  aliases restickify input

restickify output LX allocation
  aliases consumer input

consumer input
  must not get a generated HBM load for that logical tensor
```

The most likely implementation path is a scheduled-consumer frame splice:

1. run normal DXP once to discover producer output and consumer input LX maps;
2. generate the DDL bridge frame using the producer output map and consumer
   input map;
3. generate or patch a consumer frame where the restickified input's HBM load is
   removed and its LX allocation aliases the bridge output;
4. splice both frames into `loadprogram_to_device/.../init.txt`;
5. validate the same `computed_transpose_adds_then_matmul` fixture with
   correctness enabled.

If that proves too brittle, the cleaner compiler direction is a first-class
Schedule IR / Deeptools internal-edge descriptor rather than post-DXP JSON
patching.

## Validation

Pod tests:

```text
python3 -m py_compile torch_spyre/_inductor/codegen/restickify_ddl_bridge.py
python3 -m pytest tests/inductor/test_restickify_ddl_bridge.py -q
```

Result:

```text
13 passed
```
