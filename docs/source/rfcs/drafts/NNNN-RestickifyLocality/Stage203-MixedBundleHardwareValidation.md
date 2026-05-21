# Stage 203: Mixed Bundle Hardware Validation

## Summary

This stage rebuilt a fresh Torch-Spyre stack with the mixed PT/LX bridge experiment and validated the high-signal `computed_transpose_adds_then_matmul_tuple` case on hardware.

The important result is that the normal stock path still emits `ReStickifyOpHBM`, while the mixed PT/LX bridge path emits `MixedReStickifyOpWithPTLxConsumer`, runs value-correctly at size `2048`, and shows a stable speedup in the isolated probe.

## Environment

- Pod: `adnan-cdx-spyre-dev-pf`
- Project root: `/home/adnan-cdx/dt-inductor-mixed`
- Torch-Spyre branch: `AdnanHoque/rfc-restickify-first-principles`
- Torch-Spyre commit: `aa349e5`
- PyTorch: `2.11.0+cpu`
- Python: `3.12.13`
- Deeptools patch: local-only mixed-SuperDSC DXP patch in `dxp/SdscTree.cpp` and `dxp/dxp.cpp`

The rebuilt stack completed LLVM, Deeptools, Flex, libaiupti, torch_sendnn, and torch-spyre. Torch-Spyre's extension build used C++20.

## Device Smoke

A tiny stock Torch-Spyre smoke passed:

```text
torch 2.11.0+cpu
tensor([2., 4.], dtype=torch.float16)
```

## Compile-Only Guardrail

The 2048 mixed bridge path compiled without launching hardware:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple restickifies=1 bytes=8388608 byte_hops=0 device_events=0
```

The audit row showed the expected mixed bundle rewrite:

```text
replacement_sdsc: 1_MixedReStickifyOpWithPTLxConsumer
mixed_schedule: [[0, -1, 0, 1], [1, -1, 1, 1], [-1, 0, 1, 0]]
producer_pieces_patched: 32
consumer_pieces_patched: 32
num_dataops: 2
```

## Hardware Correctness

Size `512` ran value-correctly, but did not receive the locality override:

```text
ok size=512 restickifies=1 bytes=524288 byte_hops=4194304 device_events=0
locality_assertion: skipped
locality_skip_reason: no-core-mapping-override
producer_splits: {"d0": 32}
restickify_splits: {"d0": 8, "d1": 4}
```

This is expected conservative behavior. The smaller shape does not expose the same full 32-way producer/restickify ownership match.

Size `2048` ran value-correctly and passed the locality certificate:

```text
ok size=2048 restickifies=1 bytes=8388608 byte_hops=0 device_events=0
locality_assertion: passed
locality_certified: true
certified_byte_hops: 0
producer_splits: {"d0": 32}
restickify_splits: {"d0": 32}
```

## Kernel Path Check

The stock baseline emitted:

```text
sdsc_0_add.json
sdsc_1_ReStickifyOpHBM.json
sdsc_2_add.json
```

The mixed PT/LX bridge emitted:

```text
sdsc_0_add.json
sdsc_1_MixedReStickifyOpWithPTLxConsumer.json
```

So the benchmark comparison is not just a different telemetry mode. It replaces the stock HBM restickify SDSC with the mixed PT/LX bridge SDSC.

## Timing

Probe command used `--time --warmup 5 --iters 50` in fresh Python processes, alternating baseline and candidate order.

| Mode | Rep 1 median ms | Rep 2 median ms | Rep 3 median ms | Median of medians |
| --- | ---: | ---: | ---: | ---: |
| Stock `ReStickifyOpHBM` | 1.313992 | 1.322222 | 1.316610 | 1.316610 |
| Mixed PT/LX bridge | 1.014610 | 1.010250 | 1.009161 | 1.010250 |

Median-of-medians speedup:

```text
1.303251x
```

## Interpretation

This is the strongest evidence so far that the HBM restickify path is avoidable for the isolated high-signal in-graph case. The compiler can emit a mixed bundle whose bridge keeps the logical value flow internal to the bundle, passes value correctness at `2048`, and avoids the generated `ReStickifyOpHBM` SDSC.

The result is still a prototype, not production-ready:

- The Deeptools mixed-SuperDSC patch is local-only.
- The Torch-Spyre path is guarded by prototype env flags.
- The measurement is an isolated probe, not a model workload.
- The timing is end-to-end probe timing, not fabric-specific counter proof.
- Direct RIU-vs-HBM attribution still needs reliable hardware counters or profiler support.

## Next Step

Turn the splice into normal lowering by making Torch-Spyre generate the producer/restickify/consumer LX value-flow contract before final SDSC/codegen, then remove the launch-time patching path.
