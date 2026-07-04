# Flash Attention DLDSC Collective Gap - 2026-07-04

This checkpoint records the current state of the flash-attention side of the
`ah/comms-collectives` exploration. The Granite S512 scatter path is working and
speeding up the block, but the broader flash-attention collective path is not yet
value-correct.

## Branch State Used

CDX pod run root:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
```

Value investigation root:

```text
/home/adnan-cdx/codex-isolated/flash_attention_value_mismatch_investigate_20260704_050448
```

Code checkouts for the value runs:

```text
Torch branch: ah/comms-collectives
Torch SHA: 65cf3c2af02ac98e8a7bb1470fb8b3a5a961696d
Deeptools branch: ah/comms-collectives
Deeptools SHA: fa36c57166ed4541216c55cf97527f96819d7d5e
test-spyre-scripts SHA: afda166e58b23519d0b4ca871350b011b56d91a3
Script: repos/test-spyre-scripts/test_flash.py
```

Note: `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=0` still enables the
kernel-neighbor path because Deeptools currently checks whether the variable is
present, not whether its value is truthy. Use unset vs. set for controlled tests.

## Passing Controls

Baseline, relayout disabled:

```text
runs/value_correct_relayout_off_20260704_050448
returncode: 0
stage: runtime_success
ReStickifyOpHBM files: 32
ReStickifyOpLx files: 0
backend plans: 0
```

Relayout enabled, but matmul/layout collective lowering disabled so no backend
movement fires:

```text
runs/value_correct_relayout_on_collectives_generic_no_matmul_no_layout_ag_20260704_051626
returncode: 0
stage: runtime_success
ReStickifyOpHBM files: 32
ReStickifyOpLx files: 0
backend plans: 0
```

This proves the value issue is not caused by the Python test, generic runtime
setup, or boundary-clone plumbing alone.

## First Failing Edge

The earliest backend plan appears at:

```text
sdsc prefix: 3
root op: 3_batchmatmul
input: Tensor1
program step: 5
classification: matmul_operand_broadcast
communication class: all_gather
communication pattern: all_gather_replicate
```

Source SDSC in the failing run:

```text
runs/value_correct_relayout_on_kernel_neighbor_off_20260704_050448/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_fvwsc2qs/sdsc_3.json
```

Backend plan artifact:

```text
runs/value_correct_relayout_on_kernel_neighbor_off_20260704_050448/backend_plans/3_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
```

The SDSC classification contains the actual producer and consumer coordinate maps.
For group coordinate `0 == 0`, the expected source and destination core family is
`0, 4, 8, ..., 28`. The current count-based backend plan instead groups cores as
`0, 1, 2, ..., 7`. That is the key semantic mismatch: count/group metadata alone
is too weak for this collective.

## Physical Lowering Results

Kernel-neighbor matmul operand path:

```text
runs/value_correct_relayout_on_kernel_neighbor_off_20260704_050448
returncode: 1
