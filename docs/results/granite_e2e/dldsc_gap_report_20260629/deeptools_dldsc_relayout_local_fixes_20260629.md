# Deeptools Local Fixes Used For dldsc Granite Profiling - 2026-06-29

The successful dldsc Granite prefill profiles were run with Deeptools worktree:

`/home/adnan-cdx/codex-worktrees/deeptools-master-relayout`

Base commit:

`20fb48ac5a4b7a5abaae24f7355f6239a64102c2 [dxp] hardcode to 512`

This worktree was dirty during the run. The exact local diff is archived in:

`deeptools_dldsc_relayout_local_fixes_20260629.patch`

High-level contents:

- `dxp/SdscRelayoutInsertion.cpp`: resident relayout piece-size/capacity fixes and tensor-dimension filtering.
- `ddc/ddc_fold.cpp`: coordinate-consistency fix for empty allocation maps defaulting to the SDSC compute map, plus diagnostics.
- `dsc/dsc2.cpp`: diagnostic output for HBM dim-gap failures.

These patches improve the resident `scatter` relayout path used in the measured `1.136x` run. They do not implement the missing `matmul_operand_broadcast` / `all_gather_replicate` class required for `buf21 -> buf22`.
