# Stage 318: Native Valid-Gap Endpoint DeeRT Export

## Summary

Stage 317 proved that the three-dataop native PT-LX + valid-gap endpoint bundle
passes DCG. Stage 318 sends that same SDSC through the DeeRT data-op export
path used by prior runtime-frame probes:

```text
STCDPOpLx gather
ReStickifyOpWithPTLx native local tile transform
ReStickifyOpWithPTLx valid-gap consumer endpoint adapter
  -> DeeRT export
  -> senprog.txt / init.txt
```

This is still not a hardware value run. It proves the artifact can reach
runtime-frame generation without `ReStickifyOpHBM`.

## Commands

```sh
/tmp/stage65-deeprt-dataop-probe \
  /tmp/stage317-native-validgap-endpoint-tile-2048-sparse-schedule/\
sdsc_native_ptlx_validgap_endpoint_tile_2048_0.json \
  /tmp/stage318-native-validgap-endpoint-export-2048 \
  sentient

/tmp/stage65-deeprt-dataop-probe \
  /tmp/stage317-native-validgap-endpoint-tile-512-sparse-schedule/\
sdsc_native_ptlx_validgap_endpoint_tile_512_0.json \
  /tmp/stage318-native-validgap-endpoint-export-512 \
  sentient
```

## Results

| Size | Input Data-Ops | Cores | Export RC | Init Bytes | HBM | L3LU | L3SU | LXLU | LXSU | SFP | PT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 3 | 1 | 0 | 6425 | 0 | 0 | 0 | 2 | 2 | 29 | 136 |
| 512 | 3 | 4 | 0 | 8481 | 0 | 3 | 9 | 2 | 2 | 29 | 136 |

For 2048, the exporter log shows the full vertical path:

```text
input_dataops=3 input_dldscs=0 cores=1
Calling DataOp Scheduler
Calling DCG
Calling Code Generator
Calling DCC
Calling Init Generator
Calling DIP
Calling Program FramePtr Filler
Calling Export
rc=0
```

For 512, export also succeeds. The `L3LU/L3SU` tokens are expected for this
tile shape because the sampled tile has multi-core gather activity. The
important invariant for this stage is that both exports have `HBM=0`.

## Interpretation

This moves the prototype one layer closer to a runtime value check:

- the combined native PT-LX + valid-gap endpoint SDSC compiles;
- DeeRT emits `senprog.txt` and `init.txt`;
- exported programs contain no textual `HBM` traffic;
- the PT/SFP tokens show that the native PT-LX transform path is present.

What remains unproven:

- value correctness on hardware;
- whether the valid-gap endpoint adapter correctly reinterprets native PT-LX
  workspace values;
- whether this bundle can safely replace an in-bundle `ReStickifyOpHBM` in a
  normal Torch-Spyre generated runtime artifact.

Next step: build a hardware value harness for this exact exported frame. It
should compare one output tile against the stock HBM restickify result before
attempting full tensor replacement.
