# Latest-Head Validation Payload - 2026-07-07

This directory records the post-auth validation run for the DLDSC comms
collectives work.

## Source Heads

Torch prototype:

```text
AdnanHoque/torch-spyre:gather-restickify
102520820da890d6a62f781e86573f38dcc6f244
```

Deeptools source:

```text
Adnan-Hoque1/deeptools:ah/comms-collectives
a5ff55eee627c5c2bd4b7b0518bb0cbaad385952
```

The later Torch head fixes a partial-view gather fixture symbol mismatch so the
full relayout test file can run cleanly.  The later Deeptools head reverts a
diagnostic source-core chunking experiment that regressed the saved flash replay
into an IBuff failure.

## Torch Validation

Artifacts:

```text
torch_gather_restickify_validation_summary_devpf.md
torch_full_lx_relayout_test_20260707.txt
torch_full_lx_relayout_test_20260707.rc
```

Current result on CDX:

```text
38 passed in 3.82s
```

Older DEV result:

- branch reset to `7a188395295947e7cfe51619f958df712e676c6f`;
- `py_compile` passed;
- `compileall torch_spyre/_inductor` passed;
- lightweight relayout import test passed: `2 passed`;
- full `tests/inductor/test_lx_relayout_dldsc.py` collection is blocked on the
  DEV pod by a local `_C.so` / `libspyre_comms.so.1` ABI mismatch, not by a test
  assertion.

## Deeptools Focused Tests

Artifact:

```text
dxp_unit_focused_320630da_equiv.log
```

Current focused result:

```text
7 focused DXP tests passed.
```

The passing set includes:

- bounded broadcast gather-restickify compile;
- bounded multicast gather-restickify compile;
- fail-closed broadcast/multicast controls;
- core-work-division LX relayout tests.

## Full Flash Saved SuperDSC Replay

Artifacts:

```text
full_flash_replay_320630da_equiv_fixed_20260707_184815.tgz
full_flash_plan_summary_320630da_equiv.json
```

Result:

```text
rc = 0
backend plan artifacts = 64
communication_pattern = all_gather_replicate: 64
realization_strategy = gather_then_restickify: 64
physical_lowering_status = lowered_gather_then_restickify: 64
stale_loop_stage_with_gather = 0
```

This closes the earlier latest-head replay gap for the saved flash SuperDSC
bundle. The replay proves the current all-gather/replicate flash path lowers to
the staged gather/restickify carrier without the stale loop-scoped diagnostic
stage.

## Bounded Broadcast Artifact

Artifacts:

```text
bounded_broadcast_artifact_320630da_equiv_20260707_185136.tgz
bounded_broadcast_plan_320630da_equiv.json
bounded_broadcast_plan_summary_320630da_equiv.txt
```

Result:

```text
communication_pattern = broadcast
source_core_count = 1
consumer_core_count = 2
group_count = 1
consumer_replicas_per_group = 2
logical_transfer_count = 2
realization_strategy = gather_then_restickify
physical_lowering_status = lowered_gather_then_restickify
stages = source_operand_shards, grouped_broadcast, local_layout_conversion, gather_then_restickify, bind_matmul_kernel_operand
```

This supersedes the older stale/mislabeled broadcast JSON that showed a
multicast/loop-scoped path.
