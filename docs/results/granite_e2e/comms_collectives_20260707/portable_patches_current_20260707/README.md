# Current Portable Prototype Patches - 2026-07-07

These patches are for applying the current communication-collectives prototype
onto another engineer's local checkouts. They are not lean PR diffs.

## Torch Patch

File:

```text
torch_spyre_gather_restickify_vs_upstream_main_20260707.patch
```

Generated from:

```text
repo:   https://github.com/AdnanHoque/torch-spyre
branch: gather-restickify
head:   7a188395295947e7cfe51619f958df712e676c6f
base:   upstream/main merge-base 7e45168f1d56ca1cec4889a3e19b14719dcdd23f
```

Patch size:

```text
18 files changed, 3308 insertions(+), 26 deletions(-)
```

## Deeptools Patch

File:

```text
deeptools_comms_collectives_vs_upstream_master_20260707.patch
```

Generated from:

```text
repo:   https://github.ibm.com/Adnan-Hoque1/deeptools
branch: ah/comms-collectives
head:   2ccd5cefbf638e4d7fb04c88ed56a26c93a4459c
base:   upstream/master merge-base 0a9da5eb19d08712383312bb7dec18fbd7caf711
```

Patch size:

```text
38 files changed, 11135 insertions(+), 219 deletions(-)
```

## Feature Flag

For normal Torch-launched runs, the intended single public gate is:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

For manual DXP replay that bypasses Torch, also set:

```bash
export DXP_LX_FRAC_AVAIL=1
```

For historical full Granite/full-LX reproduction, the split-capacity setup is:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

Treat the LX fraction variables as runtime capacity setup, not feature gates.

## Status

Use these patches to reproduce or extend the prototype, especially for flash
attention structural spill removal and bounded communication-class tests.

Do not use these as the proposed final PR shape. The expected production path
still needs:

- a refreshed latest-head full-flash replay artifact;
- a clean regenerated bounded broadcast artifact;
- fresh Granite S512 validation at current Torch/Deeptools heads;
- explicit WSR boundary notes for oversized activations.
