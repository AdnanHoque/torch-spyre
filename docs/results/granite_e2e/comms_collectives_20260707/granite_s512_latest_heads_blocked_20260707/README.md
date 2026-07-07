# Granite S512 Current-Head Attempt - Clean FMS Block

This directory archives the first current-head Granite S512 attempt using the
clean Codex FMS checkout:

```text
/home/adnan-cdx/dt-inductor-codex-clean/foundation-model-stack
```

Both relayout-disabled and relayout-enabled variants failed before SDSC
generation with the same frontend validation error:

```text
Unsupported: Spyre backend does not support: All inputs to an op must have same
element arrangement, op: mul, args: "buf0": ElementArrangement.DL16_TO_FP32,
"buf4": ElementArrangement.STANDARD
```

Because this failure happens before SDSCs are emitted, this directory is not used
for HBM-spill classification. The pinned-FMS run in the sibling directory is the
current-head structural evidence:

```text
../granite_s512_latest_heads_pinned_fms_20260707/
```

