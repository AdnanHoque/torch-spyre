# Stage 197: Normal Lowering Attempt And Quick Benchmark

## Summary

This stage tried to move the proven same-artifact PT-aware LX restickify bridge
from a post-codegen frame splice into normal Torch-Spyre bundle generation.

The quick runtime benchmark still uses the validated same-artifact splice path,
because the first normal-lowering attempt exposed a Deeptools packaging blocker.
The benchmark result remains useful as the performance target for the normal
lowering path.

## Quick Benchmark

Case:

```text
computed_transpose_adds_then_matmul_tuple, size=2048
```

Settings:

```text
warmup=5, iters=20, correctness skipped for timing
stock HBM restickify vs PT-aware LX bridge splice
```

Results:

| Path | Median ms | p10 ms | p90 ms | Notes |
|---|---:|---:|---:|---|
| Stock `ReStickifyOpHBM` | 1.3526 | 1.3266 | 1.3652 | normal HBM restickify frame |
| PT-LX bridge splice | 1.1743 | 1.1621 | 1.1858 | `ReStickifyOpWithPTLx -> STCDPOpLx`, HBM-free frame |

Observed speedup:

```text
1.1519x, about 0.178 ms saved for the fused add/restickify/add bundle
```

The bridge frame was HBM-free in `senprog.txt`:

```text
HBM=0, L3LU=96, L3SU=96, LXLU=64, LXSU=64, PT=4352, SFP=928
```

## Normal-Lowering Attempt

Added a default-off prototype hook:

```text
SPYRE_RESTICKIFY_PTLX_BRIDGE_E2E=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_AUDIT_JSONL=/tmp/.../audit.jsonl
```

The hook runs inside `generate_bundle()` after normal SDSC JSON payloads are
generated and before files are written. For an eligible adjacent
producer/restickify/consumer triple it:

1. Requires an in-graph producer and a zero-byte-hop locality certificate.
2. Marks the producer output as LX-resident.
3. Replaces the `ReStickifyOpHBM` payload with a two-step PT-LX data-op bridge.
4. Marks the consumer input as LX-resident.

Audit confirmed the intended rewrite fired for the 2048 case:

```json
{
  "status": "patched",
  "replacement_sdsc": "1_TwoStepReStickifyOpWithPTLxStcdp",
  "direction": "kernel-to-output",
  "restickify_logical_direction": "output-to-kernel",
  "producer_lx_unique_starts": [16384],
  "consumer_lx_unique_starts": [8192],
  "bridge_endpoint_patch": {
    "producer_pieces_patched": 32,
    "consumer_pieces_patched": 32,
    "num_dataops": 2
  }
}
```

## Blocker

DXP rejected the generated bundle before hardware launch:

```text
DtException: Datadsc not allowed, use dldsc
```

This is an important boundary: the standalone/same-artifact path can compile a
`datadscs_` bridge into a frame, but normal `sdscbundle.sdsc_execute` import
does not accept a top-level data-op-only SDSC. The production-shaped solution
therefore cannot be "drop a datadsc bridge in as the restickify SDSC." It needs
the bridge represented through Deeptools' DL/data-op contract, most likely the
InputFetchNeighbor/dldsc path or an equivalent first-class mixed DL+data-op
artifact.

## Conclusion

We have a measured performance target: about `1.15x` for the high-signal 2048
synthetic bundle when replacing the HBM restickify frame with the HBM-free PT-LX
bridge.

We do not yet have normal Torch-Spyre lowering for that bridge. The next
implementation step is to generate the bridge through the dldsc/InputFetchNeighbor
contract instead of emitting top-level `datadscs_` in the bundle.
