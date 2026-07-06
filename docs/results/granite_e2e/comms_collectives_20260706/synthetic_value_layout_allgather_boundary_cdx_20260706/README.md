# Synthetic Layout-Allgather Boundary Run

This directory archives a negative control from the clean `gather-restickify`
branches on the CDX pod.

## What Was Tested

The test is a small attention-shaped synthetic graph:

```python
scores = torch.matmul(q * scale, k.transpose(-1, -2) * scale)
v2 = v * 1.0
out = torch.matmul(scores, v2)
```

The intent was to create a value-correct matmul-operand all-gather case before
returning to full Granite and flash attention. The run used the clean
`gather-restickify` Torch and Deeptools branches with the LX relayout flags
enabled.

## Result

The run failed during DXP lowering with:

```text
layout_allgather_restickify metadata was classified, but physical lowering is
blocked: dense resident all-gather materialization is unsafe on Granite
attention shapes. Use matmul_operand_broadcast for loop-scoped matmul operands,
or set DEEPTOOLS_ENABLE_UNSAFE_LAYOUT_ALLGATHER_RESTICKIFY=1 only for diagnostic
replay experiments.
```

Return code: `1`

Backend plans emitted: `1`

Plan file:

```text
backend_plans/3_batchmatmul_Tensor1_0_layout_allgather_restickify_plan.json
```

## Interpretation

This is not a value-correctness failure in the relayout implementation. It is a
boundary artifact showing that not every all-gather-like mismatch should be
materialized as a dense resident all-gather.

The path that passed structural validation for Granite and flash attention is
the narrower matmul operand contract:

```text
matmul_operand_broadcast -> all_gather_replicate -> gather_then_restickify
```

That path keeps the gather loop-scoped to the matmul operand and avoids unsafe
full-resident materialization.

## Why This Artifact Matters

It documents the current split:

- Supported: matmul RHS operand all-gather/replicate with staged
  gather-then-ReStickifyOpLx lowering.
- Intentionally blocked: dense resident layout-allgather materialization for
  Granite/attention-sized tensors.

The next synthetic value test should mimic the Granite/flash matmul operand
metadata closely enough to hit `matmul_operand_broadcast`, not the dense
`layout_allgather_restickify` fallback path.

## Reproduction Notes

Original run root on CDX:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/synthetic_value_matmul_operand_broadcast_20260706_124347
```

The archived files include:

- `env.sh`
- `synthetic_value_matmul.py`
- `run_summary.json`
- `stdout.log`
- `stderr.log`
- `returncode.txt`
- backend plan JSON

