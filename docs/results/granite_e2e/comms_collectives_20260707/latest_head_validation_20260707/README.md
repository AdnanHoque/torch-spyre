# Latest-Head Validation Payload - 2026-07-07

This directory records the post-auth validation run for the DLDSC comms
collectives work.

## Source Heads

Torch prototype:

```text
AdnanHoque/torch-spyre:gather-restickify
7a188395295947e7cfe51619f958df712e676c6f
```

Deeptools source:

```text
Adnan-Hoque1/deeptools:ah/comms-collectives
320630da56beb2bb12e6c96ae5b016127962353c
```

The CDX pod did not have private-key auth for `github.ibm.com`, so the two
diagnostic commits from `320630da` were applied as patches on top of the local
`3a4349e62` checkout. The resulting local head was:

```text
9c191c4ae9f273f5e0dcdf98413176c644f5fbb0
```

The patch content is the same as:

```text
2ccd5cefbf638e4d7fb04c88ed56a26c93a4459c
320630da56beb2bb12e6c96ae5b016127962353c
```

Both commits are diagnostic/test-only for plan artifact consistency. They do
not change the physical transfer lowering.

## Torch Validation

Artifact:

```text
torch_gather_restickify_validation_summary_devpf.md
```

Result:

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

Result:

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

