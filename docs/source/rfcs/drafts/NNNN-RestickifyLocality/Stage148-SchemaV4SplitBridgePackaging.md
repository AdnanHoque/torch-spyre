# Stage 148: Schema-v4 split bridge packaging

## Summary

This stage moved the LX-to-LX restickify prototype one step closer to normal
Torch-Spyre bundle integration.  The runtime split prototype now refuses the old
HBM-address alias path by default and only packages an LX data-op bridge when the
generated bundle has a schema-v4 `lx_materialization_contract`.

The target contract is still:

```text
producer compute -> ReStickifyOpLx/STCDPOpLx data-op bridge -> consumer compute
```

The stock generated bundle still contains `sdsc_1_ReStickifyOpHBM.json`; the
prototype recognizes that producer/restickify/consumer triplet, splits it into
producer/data-op/consumer stages, and materializes the restickified value through
LX endpoints instead of running the stock HBM restickify.

## Code Changes

- Added `SPYRE_RESTICKIFY_LX_SPLIT_REQUIRE_MATERIALIZATION_CONTRACT=1`
  behavior to `tools/restickify_scenario_probe.py`.
- The split data-op generator now records the descriptor contract source,
  materialization kind, intended Deeptools sequence, and patched producer/consumer
  piece counts in launch JSONL.
- Added `SPYRE_RESTICKIFY_LX_SPLIT_PREPARE_ONLY=1` so the path can compile and
  package the split bridge without launching hardware kernels.
- Added `SPYRE_RESTICKIFY_LX_SPLIT_STOP_AFTER_BRIDGE=1` and
  `--validate-tuple-prefix` for the next clean-device value check.  This lets the
  tuple probe validate the first output, the bridge materialized tensor, while
  intentionally skipping the later matmul kernel.

## Validation

The DeePRT data-op exporter was rebuilt in the pod:

```sh
c++ -std=c++17 -O0 -g \
  -I/opt/ibm/spyre/deeptools/include \
  /tmp/torch-spyre-stage145/tools/deeprt_dataop_export_probe.cpp \
  -L/opt/ibm/spyre/deeptools/lib \
  -Wl,-rpath,/opt/ibm/spyre/deeptools/lib \
  -ldeeprt -ldsc -lsharedtools -lsgr -lcommon -lg3logger -lutil -ljson11 \
  -o /tmp/stage65-deeprt-dataop-probe
```

Static validation:

```sh
python3 -m py_compile tools/restickify_scenario_probe.py
python3 -m py_compile /tmp/torch-spyre-stage145/tools/restickify_scenario_probe.py
```

Prepare-only validation used:

```sh
SPYRE_RESTICKIFY_RING_TELEMETRY=1 \
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_SPLIT_PREPARE_ONLY=1 \
SPYRE_RESTICKIFY_LX_SPLIT_REQUIRE_MATERIALIZATION_CONTRACT=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --copy-kernel-code \
  --kernel-launch-log \
  --lx-split-dataop-prototype \
  --output-dir /tmp/stage148-materialization-split-prepare-2048 \
  --fail-on-error
```

Result:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple
restickifies=1 bytes=8388608 byte_hops=0
```

The launch log reported:

```text
dataop_contract_source=schema-v4-lx-materialization-contract
dataop_materialization_kind=torch_spyre.restickify_lx_materialization_contract
dataop_intended_sequence=["ReStickifyOpLx", "STCDPOpLx"]
dataop_producer_pieces_patched=32
dataop_consumer_pieces_patched=32
dataop_export_returncode=0
producer_lx_unique_starts=[16384]
consumer_lx_unique_starts=[8192]
```

The generated data-op `senprog.txt` for the 2048 materialization had no HBM
instructions:

```text
HBM=0, LXLU=64, LXSU=64
```

The program also contained L3 load/store instructions emitted by the DeeRT data-op
pipeline.  That means the useful property at this stage is "no HBM path in the
generated bridge program"; it is not yet a pure-LX-only program.

## Negative Checks

- `computed_transpose_adds_then_matmul_tuple` at size `512` generated a
  schema-v4 descriptor file with zero materialization edges.  The new guard
  rejected the old fallback:

  ```text
  LX split data-op requires schema-v4 materialization contract; got legacy-hbm-base-match
  ```

- `computed_transpose_join` at size `2048` was not an in-graph producer case.
  The descriptor classified the restickify source as `graph_input_or_weight`, so
  it is out of scope for this path.

## Hardware Run

After prepare-only succeeded, one non-prepare 2048 run was attempted with
correctness skipped.  The producer, data-op bridge, and consumer stages reached
`lx_split_dataop_launch_done`, but the following matmul bundle hit a stream error:

```text
RAS::RUNTIMESCHEDULER::ComputeHardwareError
RAS::RUNTIMESCHEDULER::StreamInErrorState
```

The log ordering matters: the bridge sequence reached `launch_done` before the
matmul attempted to launch.  However, because the runtime reported a prior compute
hardware error, this is not accepted as a value-correct or hardware-clean proof.
Further hardware launches should wait for a clean device.

## Current Conclusion

We now have a stricter, production-shaped packaging contract for the remote-LX
materialization path:

- only schema-v4 materialization descriptors are accepted;
- legacy HBM-address aliasing is rejected by default;
- the high-signal 2048 case can be compiled and packaged as a producer/data-op/
  consumer split without launching hardware;
- the generated data-op bridge has no HBM instructions.

The remaining blocker is a clean-device value check of the bridge output itself.
The next run should use `SPYRE_RESTICKIFY_LX_SPLIT_STOP_AFTER_BRIDGE=1` with
`--validate-tuple-prefix 1` on the tuple probe so the matmul tail is skipped and
only the materialized join tensor is compared against CPU.

