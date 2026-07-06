# Deeptools comms-collectives patch rebased on upstream master

Generated: 2026-07-06

This directory contains the portable Deeptools patch for the DLDSC LX relayout communication-collectives prototype, rebased cleanly on current upstream Deeptools master.

## Branches

- Upstream master: `949cfeea885e05cb12dd37ea07d480d82f1ee27c`
- Rebased branch: `Adnan-Hoque1/deeptools:ah/comms-collectives-master`
- Rebased branch head: `f3e1f6c04cd209c63d3746e889e1e64b91dd6fd2`

## Patch file

- `deeptools_ah_comms_collectives_rebased_on_master.patch`

Apply from a current upstream Deeptools master checkout with:

```bash
git apply deeptools_ah_comms_collectives_rebased_on_master.patch
```

## Focused validation

The rebased tree built `dxp_standalone`, `util_unit_test`, `dxp_unit_test`, and `dcg_unit_test` on CDX.

Passing focused tests:

- `LayoutAllgatherRestickify.*`: 31/31
- `DxpTestFixture.CoreWorkDivIncomptLxRelayout*`: 2/2
- `stcdpLibtest.multicast*`: 2/2

See the copied test logs in this directory.
