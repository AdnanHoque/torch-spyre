# Torch gather/restickify patch rebased on upstream main

Generated: 2026-07-06

This directory contains the portable Torch-side patch for the DLDSC LX relayout gather/restickify flash/Granite-spill prototype.

## Patch context

- Upstream Torch main: `7e45168f1d56ca1cec4889a3e19b14719dcdd23f`
- Patch branch/source: `AdnanHoque/torch-spyre:gather-restickify`
- Patch head: `cc29ba259ec190e62f760f032ffd3355581a57df`
- Ahead/behind upstream main: `0	9`

## Patch file

- `torch_spyre_gather_restickify_rebased_on_main.patch`

Apply from a current upstream Torch checkout with:

```bash
git apply torch_spyre_gather_restickify_rebased_on_main.patch
```

## Runtime flag

Normal Torch-launched runs should only need the top-level relayout flag:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

That flag enables the collectives/restickify sublanes and makes Torch pass backend LX workspace to the `dxp_standalone` subprocess. No external split-LX wrapper is required for normal Torch compilation.

For direct manual SDSC replay that bypasses Torch, set backend workspace explicitly because Torch is not launching the subprocess:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1 DXP_LX_FRAC_AVAIL=1 dxp_standalone -d path/to/sdsc_bundle
```

The narrower `SPYRE_LX_PLANNER_RELAYOUT_*` flags still exist only as debugging overrides.

## Validation

The source branch config probe passed:

```text
default False 0.2 0.2
relayout True 0.0 1.0
override True 0.3 0.7
```

Focused source-branch test after the same one-flag split-LX update:

```text
tests/inductor/test_lx_relayout_dldsc.py: 28 passed
```
