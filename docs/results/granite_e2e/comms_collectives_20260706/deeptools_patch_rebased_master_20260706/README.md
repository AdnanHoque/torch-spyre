# Deeptools gather/restickify patch with one public gate

This directory archives the portable Deeptools patch for the experimental Granite/flash LX communication work.

## Source

- Source branch: Adnan-Hoque1/deeptools ah/comms-collectives
- Head: 5faca11015753999da1d26b87ecccdb08e4975ce
- Upstream master observed at generation: 949cfeea885e05cb12dd37ea07d480d82f1ee27c
- Branch merge-base with upstream master: 0a9da5eb19d08712383312bb7dec18fbd7caf711
- Generated: 2026-07-06

## Gate

Normal Torch-launched runs should use one feature flag:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

For direct manual SDSC replay that bypasses Torch, also provide backend LX workspace:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1 DXP_LX_FRAC_AVAIL=1 dxp_standalone -d path/to/sdsc_bundle
```

DXP_LX_FRAC_AVAIL=1 is a direct-replay/backend workspace setting, not a feature gate.

## One-gate path selection

SPYRE_LX_PLANNER_RELAYOUT=1 selects the staged gather_then_restickify path for matmul operand relayout. The older direct kernel-neighbor experiment is intentionally left behind the explicit debug flag DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1; it is not enabled by the umbrella flag.

## Included changes

- Gate the experimental Deeptools relayout paths behind SPYRE_LX_PLANNER_RELAYOUT.
- Keep older DEEPTOOLS_* toggles only as hidden debug compatibility aliases.
- Add coordinate-overlap grouping for matmul operand relayouts where producer and consumer splits differ.
- Add staged gather/restickify and mixed-neighbor carrier support for the Granite/flash matmul operand case.
- Support multiple LX input-neighbor tensors in one DSC by inserting uniquely named soft sync nodes for each LX neighbor transfer.
- Lower the flash matmul operand all-gather/replicate path through gather_then_restickify without requiring the old diagnostic flags.

## Focused validation

- LayoutAllgatherRestickify.*: 32 passed
- DxpTestFixture.CoreWorkDivIncomptLxRelayout*: 2 passed
- Direct replay of the previously failing flash SDSC with only SPYRE_LX_PLANNER_RELAYOUT=1 plus DXP_LX_FRAC_AVAIL=1: return code 0, 64 backend plans
- Full test_flash.py structural probe with only SPYRE_LX_PLANNER_RELAYOUT=1: return code 0, ReStickifyOpHBM=0, ReStickifyOpLx=64, 64 gather_then_restickify plans

See ../flash_onegate_gate_path_fix_20260706/ for the latest flash proof artifacts.
