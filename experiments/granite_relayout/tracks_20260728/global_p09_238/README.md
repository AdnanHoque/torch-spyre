# Granite global-P09 prototype

This directory archives the Torch-Spyre source overlay and run contract used by
the Granite B1/S512 prefill investigation that reported a 238.480700 ms median.
It is an experimental, graph-specific policy layer on top of the earlier
Granite relayout archive, not a production API.

The source snapshot was recovered from the preserved global-P09 control used by
the later P05 ablation. Relative to commit `36804f23`, the overlay changes:

- `torch_spyre/_inductor/config.py`
- `torch_spyre/_inductor/work_division.py`
- `torch_spyre/_inductor/lx_relayout.py`
- `torch_spyre/_inductor/scratchpad/allocator.py`
- `torch_spyre/_inductor/spyre_kernel.py`
- `torch_spyre/_inductor/codegen/superdsc.py`
- `tests/inductor/test_lx_relayout_dldsc.py`

`run_global_p09_p05_ablation.sh` preserves the later matched runner. Its P09-on,
P05-off arm is selected with:

```text
<run-name> 5 1 16 2 1 1 0
```

The script intentionally retains the original pod paths and environment names;
it is provenance, not a portable launcher. The measured program used
`DXP_LX_FRAC_AVAIL=0.2`, enabled the LX relayout planner, and enabled the P09
Q/K/V ownership treatment with
`SPYRE_RELAYOUT_ORACLE_PREFILL_QKV_INPUTS=1`.

`REALIZED_WD_CHAIN_LEDGER.md` records the emitted-bundle audit. The 238.480700
ms number is preserved evidence from that run; it has not been remeasured on
this branch.
