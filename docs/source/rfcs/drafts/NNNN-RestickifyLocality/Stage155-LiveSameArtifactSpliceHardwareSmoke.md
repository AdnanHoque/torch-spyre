# Stage 155: Live Same-Artifact Splice Hardware Smoke

## Summary

This stage moved the Stage 154 same-artifact splice from a compile-only
package check into the live Torch-Spyre launch path.

The prototype is still default-off and still experimental.  It does not change
normal Torch-Spyre lowering unless the probe is run with:

```sh
--lx-bridge-same-artifact-splice
```

The important result is that the same fused bundle shape:

```text
sdsc_0_add.json
sdsc_1_ReStickifyOpHBM.json
sdsc_2_add.json
```

was patched immediately before launch so the runtime program frame for
`sdsc_1_ReStickifyOpHBM.json` was replaced by the HBM-free LX bridge frame, and
the hardware launch retired without a stream hardware error.

Correctness was intentionally skipped in this stage.  This was a packaging and
device-safety smoke, not yet a value-correctness proof.

## Code Change

Two small probe changes were added.

First, `tools/restickify_scenario_probe.py` gained:

```text
--lx-bridge-same-artifact-splice
```

When enabled, the kernel launch debug hook checks each generated code directory
for both:

```text
restickify_lx_neighbor_edges.json
sdsc_*ReStickify*.json
```

If both are present, it:

1. generates a Stage 152 LX bridge frame for that exact code directory;
2. splices the bridge frame into the normal fused artifact in place;
3. writes `.stage154_lx_bridge_same_artifact_spliced`;
4. logs the splice result into the kernel launch JSONL;
5. launches the normal Torch-Spyre artifact.

Second, `tools/restickify_lx_bridge_frame.py` now retries DeeRT export once by
default.  This avoids a flaky exporter path where the first invocation can
return `-11` after producing partial files, while a retry succeeds and produces
the needed `init.txt` and `senprog.txt`.

## Pre-Hardware Inspection

Before launching, the same live hook was run with:

```sh
--skip-kernel-launch
```

Result:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple
restickifies=1 bytes=8388608 byte_hops=0
```

The launch log showed:

```json
{
  "phase": "lx_bridge_same_artifact_splice",
  "status": "patched",
  "patched_bytes": 29184,
  "patched_flits_128b": 228,
  "restickify_start_flit": 45,
  "restickify_original_bytes": 7296,
  "bridge_frame_bytes": 17664,
  "bridge_hbm_free": true,
  "bridge_tokens": {
    "HBM": 0,
    "L3LU": 96,
    "L3SU": 96,
    "LXLU": 64,
    "LXSU": 64,
    "PT": 0,
    "SFP": 0
  }
}
```

The copied runtime-active files were checked for stale text references to:

```text
ReStickifyOpHBM
dataop
lx_split
```

inside:

```text
loadprogram_to_device/
execute_dsg.txt
pagi.json
segment_size.json
```

No matches were found in those runtime-active files.  Source-level references
still exist in files such as the SDSC JSON and descriptor files, but those were
not the runtime files being executed by this smoke.

## Hardware Smoke

Command shape:

```sh
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --copy-kernel-code \
  --kernel-launch-log \
  --lx-bridge-same-artifact-splice \
  --output-dir /tmp/stage155-same-artifact-hw-smoke-2048-retry \
  --fail-on-error
```

Result:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple
restickifies=1 bytes=8388608 byte_hops=0 device_events=0
```

The launch log recorded:

```text
event 1: lx_bridge_same_artifact_splice, status=patched
event 2: before_launch sdsc_fused_add_t_0
event 3: after_launch  sdsc_fused_add_t_0
event 4: before_launch sdsc_fused_mm_1
event 5: after_launch  sdsc_fused_mm_1
```

The copied marker file confirmed the live artifact was patched:

```json
{
  "status": "patched",
  "patched_bytes": 29184,
  "patched_flits_128b": 228,
  "restickify_start_flit": 45,
  "restickify_original_bytes": 7296,
  "bridge_frame_bytes": 17664,
  "bridge_hbm_free": true,
  "bridge_tokens": {
    "HBM": 0,
    "L3LU": 96,
    "L3SU": 96,
    "LXLU": 64,
    "LXSU": 64,
    "PT": 0,
    "SFP": 0
  }
}
```

Ring telemetry for the logical compiler model also reported a certified local
in-graph edge:

```json
{
  "source_kind": "in_graph_computed",
  "bytes_moved": 8388608,
  "byte_hops": 0,
  "locality_certified": true,
  "locality_assertion": "passed",
  "producer_splits": {"d0": 32},
  "restickify_splits": {"d0": 32}
}
```

## Interpretation

This is the first successful hardware smoke for:

```text
producer add -> same-artifact LX bridge frame -> consumer add -> matmul
```

inside the normal Torch-Spyre fused bundle shape, without launching a separate
consumer harness.

That matters because the earlier split-consumer hardware path was unsafe: it
created a runtime scheduler hardware error.  This stage avoids that known bad
path by patching the replacement frame into the original fused artifact before
the normal launch.

## What This Proves

This proves:

1. the live Torch-Spyre artifact can be patched immediately before launch;
2. the replacement frame can be HBM-free by token inspection;
3. the fused add bundle can launch and return after the splice;
4. the downstream matmul bundle can also launch and return;
5. the device does not hit the previous stream hardware error for this
   same-artifact path.

## What This Does Not Prove

This does not yet prove value correctness.

It also does not yet prove that this should be a production implementation.
The prototype is still a launch-time artifact splice, not a proper compiler
contract between producer output planning, restickify lowering, and consumer
input planning.

The `aiu-smi` log did not provide useful fabric-level traffic numbers in this
container run.  It should not be interpreted as proof of RIU/HBM traffic either
way.

## Next Step

Run one bounded correctness attempt on the same case:

```sh
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --copy-kernel-code \
  --kernel-launch-log \
  --lx-bridge-same-artifact-splice \
  --validate-tuple-prefix 1 \
  --output-dir /tmp/stage155-same-artifact-correctness-2048 \
  --fail-on-error
```

If that passes, the prototype has crossed from "same-artifact hardware smoke"
to "same-artifact value-correct LX bridge" for the high-signal 2048 case.

If it fails, the failure is likely now a value-flow or view-contract issue,
not the earlier split-launch scheduler issue.
