# Antoni Minimal Flags Handoff

Generated: 2026-07-06

For the current `gather-restickify` Torch patch plus matching rebased Deeptools patch, the relayout feature is enabled with one public flag:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

That one flag covers both sides of the prototype:

- Torch LX relayout planning;
- Torch collective/matmul-operand DLDSC metadata;
- Torch ReStickifyOp output LX eligibility;
- Deeptools staged gather/restickify lowering for matmul operand relayout.

Do not require this older backend flag for new repros:

```bash
# legacy compatibility only
# export DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
```

For full-LX Granite or flash experiments, the split DXP capacity setup is still needed when using the wrapper. This is runtime capacity plumbing, not a second relayout feature flag:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

If Antoni is applying portable patches to his own environment, give him both patches and the single feature flag above. If his environment already uses the split DXP wrapper, the capacity split is the only extra setup detail.
