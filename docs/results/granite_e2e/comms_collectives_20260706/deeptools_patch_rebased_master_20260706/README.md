# Deeptools collectives patch rebased on master

Generated: 2026-07-06

This directory contains the portable Deeptools-side patch for the DLDSC LX relayout gather/restickify and collectives prototype.

## Patch context

- Upstream Deeptools master: `0a9da5eb19d08712383312bb7dec18fbd7caf711`
- Patch branch/source: `Adnan-Hoque1/deeptools:ah/comms-collectives`
- Patch head: `b594b3afc725b693d074a64fc027ac7a6024d5fd`
- Ahead/behind upstream master: `0	43`

## Patch file

- `deeptools_ah_comms_collectives_rebased_on_master.patch`

Apply from a current Deeptools checkout with:

```bash
git apply deeptools_ah_comms_collectives_rebased_on_master.patch
```

Then rebuild the DXP binary used by Torch:

```bash
cmake --build build-deeptools --target dxp_standalone dxp_unit_test -j8
```

## Runtime flag

Normal Torch-launched runs should use the same single feature flag as Torch:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

Deeptools treats that flag as enabling the gather/restickify relayout prototype gates. The older `DEEPTOOLS_ENABLE_*` flags remain as narrow diagnostic aliases.

For direct manual SDSC replay that bypasses Torch, also provide backend LX workspace explicitly:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1 DXP_LX_FRAC_AVAIL=1 dxp_standalone -d path/to/sdsc_bundle
```

## Validation

Focused Deeptools gate after this flag update:

```text
DxpTestFixture.CoreWorkDivIncomptLxRelayout*: 2 passed
```
