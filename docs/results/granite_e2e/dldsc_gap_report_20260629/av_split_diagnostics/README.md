# AV Matmul Split Diagnostics - 2026-06-29

Purpose: test whether the remaining Granite prefill gap could be recovered by changing the attention value/PV matmul split, instead of adding a new matmul_operand_broadcast / all_gather_replicate communication class.

Baseline comparison point:

| Variant | Kernel ms/iter | Median wall ms |
|---|---:|---:|
| Best dldsc relayout, boundary clones, full Torch LX | 10.9780 | 17.7715 |

Diagnostic runs used the same dldsc relayout settings and added a temporary env knob:

SPYRE_FORCE_TRUE_BMM_VALUE_SPLIT=b_m_n_k

## Results

| Forced AV split | Run directory | Kernel ms/iter | Median wall ms | Result |
|---|---|---:|---:|---|
| 4_4_2_1 | dldsc_force_av_split_4_4_2_1_20260629_144425 | 11.3459 | 18.2439 | Regressed |
| 2_8_2_1 | dldsc_force_av_split_2_8_2_1_20260629_144659 | 11.5269 | 18.0908 | Regressed |

The 4_4_2_1 run confirmed the AV SDSC changed from mb:32 to x:4, mb:4, out:2, so the diagnostic exercised the intended split path.

## Readout

Changing the AV split is not the path to the missing approximately 0.58 ms/iter. It does not beat the best current resident scatter dldsc relayout run and it increases trace memory time.

This reinforces the current conclusion: reproducing the old approximately 1.2x Granite block result requires a real backend communication class for the value operand, not another work-division override.
