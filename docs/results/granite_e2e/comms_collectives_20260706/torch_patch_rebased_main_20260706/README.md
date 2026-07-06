# Torch gather/restickify patch rebased on upstream main

Generated: 2026-07-06

This directory contains the portable Torch-side patch for the DLDSC LX relayout gather/restickify flash-spill prototype.

## Patch context

- Upstream Torch main: `7e45168f1d56ca1cec4889a3e19b14719dcdd23f`
- Patch branch/source: `AdnanHoque/torch-spyre:gather-restickify`
- Patch head: `c80af0d688a477c4d67b62d3090c2258cfd9adbe`
- Ahead/behind upstream main: `0	6`

## Patch file

- `torch_spyre_gather_restickify_rebased_on_main.patch`

Apply from a current upstream Torch checkout with:

```bash
git apply torch_spyre_gather_restickify_rebased_on_main.patch
```

## Validation

The isolated patch checkout cannot collect the full test file without a matching local `torch_spyre/_C.so`; see `isolated_patch_checkout_binary_mismatch.log` for that environment issue.

The same patch head was validated in the source branch environment with the matching extension:

```bash
python -m pytest tests/inductor/test_lx_relayout_dldsc.py tests/inductor/test_layout_allgather_restickify_import_light.py
```

Result: 27/27 passed. See `torch_source_branch_focused_tests.log`.

## Runtime flags

Torch side now only needs:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

With the matching rebased Deeptools patch, Deeptools honors the same flag; no separate Deeptools feature flag is required. The old `DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1` flag remains only as a compatibility alias.

For Granite-style full-LX runs using the split DXP wrapper, keep this runtime capacity setup:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```
