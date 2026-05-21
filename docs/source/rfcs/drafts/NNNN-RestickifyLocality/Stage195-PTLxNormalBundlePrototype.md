# Stage 195: PT-Aware LX Restickify In A Normal Bundle

## Summary

This stage produced the first value-correct normal Torch-Spyre bundle prototype
where an in-graph producer-to-restickify-to-consumer edge avoids the stock
`ReStickifyOpHBM` frame.

The working prototype replaces the original restickify program frame with an
HBM-free data-op bridge using:

```text
ReStickifyOpWithPTLx -> STCDPOpLx
```

for the `kernel-to-output` materialization direction. The producer and consumer
SDSCs are patched/recompiled so the edge value lives in LX:

```text
producer output LX base: 16384
consumer input LX base: 8192
```

This is still a prototype. It is not production lowering, and it remains behind
explicit diagnostic flags.

## Why The Previous Bridge Was Wrong

The earlier HBM-free bridge used plain `ReStickifyOpLx` stages:

```text
ReStickifyOpLx -> STCDPOpLx -> ReStickifyOpLx
```

That launched without HBM traffic, but deterministic value probes showed a tile
permutation bug. The bridge moved ownership between cores but did not perform
the required in-stick/in-tile transpose. Its `senprog.txt` had:

```text
PT=0, SFP=0
```

The PT-aware bridge fixes that missing class of work. A compile-only export for
the 2048 case produced:

```text
HBM=0, L3LU=96, L3SU=96, LXLU=64, LXSU=64, PT=4352, SFP=928
```

## Negative Paths

Not every PT-aware spelling worked:

- `ReStickifyOpWithPTLx` without its corelet fields failed DCG with
  `coreLetWorkDs.cl0ToLxOffsetLU != -1`.
- Adding the minimal one-corelet contract made the target bridge legal:
  `numClToUse=1`, `defaultClId=0`, `cl0ToLxOffsetLU=0`,
  `cl0ToLxOffsetSU=0`, `useARF=1`, `doInPlace=0`.
- `restickify-stcdp-restickify` exported and launched, but values were still
  wrong for random inputs. A deterministic probe showed it fixed the coarse
  64-column block but still used `row % 64` where the correct coordinate needed
  `col % 64`.
- `output-to-kernel` exported but was clearly semantically wrong for this edge.
- `stcdp-then-restickify` failed DeeRT export for this shape.

The working direction for this edge is `kernel-to-output`.

## Passing Validation

Command shape:

```sh
SPYRE_RESTICKIFY_RING_TELEMETRY=1 \
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_DATAOP_DIRECTION=kernel-to-output \
SPYRE_RESTICKIFY_LX_DATAOP_RESTICKIFY_OP=ReStickifyOpWithPTLx \
SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_CONSUMER_SDSC=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --copy-kernel-code \
  --kernel-launch-log \
  --lx-bridge-same-artifact-splice \
  --output-dir /tmp/stage194-ptlx-kernel-to-output-full-correctness-2048 \
  --fail-on-error
```

Result:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple
restickifies=1 bytes=8388608 byte_hops=0
Completed 1 rows with 0 errors
```

Telemetry confirmed the compiler-side locality certificate:

```text
source_kind=in_graph_computed
bytes_moved=8388608
byte_hops=0
locality_certified=true
locality_assertion=passed
producer_splits={"d0": 32}
restickify_splits={"d0": 32}
```

The spliced bridge frame was HBM-free:

```text
HBM=0, L3LU=96, L3SU=96, LXLU=64, LXSU=64, PT=4352, SFP=928
```

The full tuple was validated, not only the intermediate bridge output, so the
downstream matmul result also matched CPU within the existing probe tolerance.

## Interpretation

This proves a narrow but important point:

> Torch-Spyre can replace an eligible in-graph `ReStickifyOpHBM` edge with an
> HBM-free LX-to-LX bridge inside the normal fused bundle, provided the bridge
> uses a PT-aware restickify contract and the consumer reads the LX-resident
> materialized value.

It does not yet prove a production implementation. The current mechanism is a
late artifact splice plus SDSC patch/recompile path. Productionization should
move the same contract earlier into normal Torch-Spyre lowering:

1. Generate the producer/restickify/consumer LX value-flow contract before final
   SDSC generation.
2. Select the PT-aware `kernel-to-output` bridge only when source and
   destination layout/stick contracts match the proven case.
3. Replace the HBM restickify frame generation path without binary splicing.
4. Keep graph-input, weight, constant, and persistent-state restickifies out of
   scope until their layout ownership can be planned explicitly.

## Artifacts

Pod artifacts from this stage:

```text
/tmp/stage187-ptlx-bridge-frame-2048
/tmp/stage192-ptlx-kernel-to-output-deterministic-2048
/tmp/stage193-ptlx-kernel-to-output-correctness-2048
/tmp/stage194-ptlx-kernel-to-output-full-correctness-2048
```

The AIU health check after validation passed with `aiu-query-devices`.
