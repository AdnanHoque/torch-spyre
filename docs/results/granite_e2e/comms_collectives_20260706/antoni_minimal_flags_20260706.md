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

If Antoni applies both portable patches, normal Torch-launched Granite or flash runs should not need the old split-LX wrapper. The Torch patch defaults frontend LX planning to full LX and passes backend workspace to the `dxp_standalone` subprocess under the same umbrella flag.

Only direct manual SDSC replay that bypasses Torch needs an explicit backend workspace setting:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1 DXP_LX_FRAC_AVAIL=1 dxp_standalone -d path/to/sdsc_bundle
```

If Antoni is applying portable patches to his own environment, give him both patches and the single feature flag above.
