# Stage 154: Same-Artifact Bridge Splice

## Summary

This stage moved the Stage 152 LX bridge frame into the normal fused
Torch-Spyre runtime artifact shape without launching hardware.

The target fused bundle remained:

```text
sdsc_0_add.json
sdsc_1_ReStickifyOpHBM.json
sdsc_2_add.json
```

The splice replaces the program frame corresponding to
`sdsc_1_ReStickifyOpHBM.json` inside:

```text
loadprogram_to_device/*/init.txt
```

with the Stage 152 LX bridge frame:

```text
producer LX -> ReStickifyOpLx/STCDPOpLx data-op -> consumer LX
```

This is still a packaging prototype.  The SDSC JSON file names are intentionally
unchanged in this stage; the experiment is about replacing the already-generated
runtime program frame in the same artifact, not about introducing a production
compiler lowering.

## Code Change

Added:

```text
tools/restickify_lx_bridge_same_artifact_splice.py
```

The tool:

1. copies a generated fused Torch-Spyre code directory;
2. finds the `ReStickify` SDSC position;
3. infers program frame boundaries from `init.txt` frame headers, or accepts
   explicit `--frame-sizes`;
4. reads `init_binary_sentinel_cleared.bin` from a Stage 152 bridge frame
   directory;
5. replaces only the restickify frame bytes;
6. updates `segment_size.json`;
7. updates `loadprogram_to_device_dsg.txt`;
8. writes a `splice_summary.json`;
9. launches no hardware.

This avoids the DXP-debug rerun used by the older
`restickify_lx_neighbor_frame_splice.py`.  That rerun currently fails on this
artifact with:

```text
std::out_of_range: basic_string::replace
```

inside Deeptools DCC for `sdsc_0_add`.  The header-scan route gives us a
compile-only package artifact without hitting that Deeptools failure.

## Validation

Command:

```sh
python tools/restickify_lx_bridge_same_artifact_splice.py \
  --code-dir /tmp/stage152-frame-prepare-2048/kernel_code/computed_transpose_adds_then_matmul_tuple_2048/0001_sdsc_fused_add_t_0 \
  --bridge-frame-dir /tmp/stage152-lx-bridge-frame-2048-ok \
  --output-dir /tmp/stage154-same-artifact-splice-2048 \
  --summary /tmp/stage154-same-artifact-splice-2048/splice_summary.json \
  --require-hbm-free
```

Result:

```json
{
  "status": "patched",
  "frame_sizes": [5760, 7296, 5760],
  "frame_starts": [0, 5760, 13056],
  "restickify_position": 1,
  "restickify_start_flit": 45,
  "original_bytes": 18816,
  "patched_bytes": 29184,
  "original_flits_128b": 147,
  "patched_flits_128b": 228,
  "restickify_original_bytes": 7296,
  "bridge_frame_bytes": 17664
}
```

The Stage 152 bridge summary was required to be HBM-free:

```text
HBM=0
L3LU=96
L3SU=96
LXLU=64
LXSU=64
```

Metadata was updated consistently:

```text
loadprogram_to_device_dsg.txt: A size 6 18816 -> 29184
segment_size.json const:      147 -> 228
```

The final package check verified:

```text
runtime init bytes = 29184
runtime init flits = 228
header at old restickify offset = bridge header
```

## Interpretation

This stage achieves the safe compile/package-only goal from Stage 153:

```text
normal fused runtime artifact
  with the original restickify program-frame slot replaced by
  the HBM-free Stage 152 LX bridge frame
```

The important difference from the unsafe split-launch path is that producer,
replacement bridge frame, and consumer now stay in the original fused artifact
ordering.  No separate consumer launch is involved in the packaged artifact.

## What This Does Not Prove

This does not yet prove hardware correctness.  The package has not been
launched.  It also does not prove that the consumer's logical view contract is
correct; it only proves the replacement bridge frame can be placed into the
same runtime `init.txt` and that the surrounding metadata can be made internally
consistent.

## Next Step

The next safety gate before hardware should inspect the patched package for the
consumer-side contract:

1. confirm the patched artifact has no split producer/data-op/consumer harness;
2. confirm the first bundle still has one fused `SenProgSend` init;
3. compare the old restickify frame header and new bridge frame header at the
   same offset;
4. inspect whether any runtime metadata still names `ReStickifyOpHBM` in a way
   that would cause the runtime to expect the old frame semantics;
5. only then run one bounded hardware attempt with `aiu-smi` logging.

If hardware still fails, the failure would be much more informative than Stage
153 because it would be testing same-artifact packaging rather than the known
bad split-consumer harness.
