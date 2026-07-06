# Deeptools gather/restickify patch with one public gate

This directory archives the portable Deeptools patch for the experimental Granite/flash LX communication work.

## Source

- Source branch: Adnan-Hoque1/deeptools ah/comms-collectives-mixed-neighbor-probe
- Head: da52ebdee91a5cee0a933e0adba9cb8f689fe213
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

## Included changes

- Gate the experimental Deeptools relayout paths behind SPYRE_LX_PLANNER_RELAYOUT.
- Keep older DEEPTOOLS_* toggles only as hidden debug compatibility aliases.
- Add coordinate-overlap grouping for matmul operand relayouts where producer and consumer splits differ.
- Add staged gather/restickify and mixed-neighbor carrier support for the Granite/flash matmul operand case.
- Lower the Granite S512 matmul operand all-gather/replicate replay through loop_scoped_input_fetch without requiring the old diagnostic flags.

## Focused validation

- LayoutAllgatherRestickify.*: 32 passed
- DxpTestFixture.CoreWorkDivIncomptLxRelayout*: 2 passed
- Granite S512 SDSC replay with only SPYRE_LX_PLANNER_RELAYOUT=1 plus DXP_LX_FRAC_AVAIL=1: return code 0, one backend plan

See ../mixed_neighbor_granite_s512_onegate_replay_20260706/ for replay logs and backend plan JSON.
