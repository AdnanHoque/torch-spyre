# Codex Pod Generic Structural Cost Model Fit

## Summary

This artifact replaces the earlier shape-specific QK / long-K tuning with generic structural terms only. The goal was to keep the e2e `rhs_loaded_once` fix while removing branches that looked like Granite-op recognition.

## Code Branch

- Branch: `AdnanHoque/torch-spyre:cost-model-tuned`
- Commit: `2518925` (`inductor: use generic structural matmul cost terms`)

## Generic Terms Kept

- `rhs_loaded_once`: folded no-batch projection matmuls are costed as unbatched/shared RHS loads.
- Per-core PSUM: K-split cost is charged against each core's output tile, not the whole output tensor.
- Sublinear HBM cohort pressure: broadcast contention grows with split fanout but not as a hard linear cliff.
- Reduction-width-scaled true-BMM batch split cost: batch splitting is cheaper for short reductions and more expensive for wide reductions.
- Generic PT fill, M-split, N-tile width, and core-underuse terms.

## Removed

- No `K <= 128 and N >= 256` QK branch.
- No `K >= 8192` long-K shared-weight branch.
- No negative shape-specific N-split discount.

## Fit Result

```text
searched 70000 in 92.8s; best first seen at 17769
score=5177.70 total_us=5176.95 exact=7/12 max_gap=16.6%
params={"b_bonus_k_ref": 512, "b_split_bonus": 5, "batch_k_power": 0.5, "batch_k_ref": 16, "batch_penalty": 150, "cohort_limit": 8, "cohort_power": 0.75, "core_penalty": 150, "m_penalty": 50, "n_split_penalty": 0, "n_wide_penalty": 75, "psum_bmm": 0.0003, "psum_shared": 0.003, "pt_power": 0.5, "target_n_tile": 512, "target_passes": 5, "tie_passes": 3}
decode  K/V proj  pred=1_4_8_1       67.80 best=1_8_4_1       66.78 gap=  1.5%
decode  MLP down  pred=1_4_8_1      689.43 best=1_4_4_1      689.20 gap=  0.0%
decode  MLP up    pred=1_4_8_1      673.44 best=1_4_8_1      673.44 gap=  0.0%
decode  Q/O proj  pred=1_4_8_1      231.84 best=1_4_8_1      231.84 gap=  0.0%
decode  QK^T      pred=1_4_3_2      104.89 best=8_2_1_2       89.93 gap= 16.6%
decode  attn@V    pred=1_4_2_3       55.04 best=1_4_2_3       55.04 gap=  0.0%
prefill K/V proj  pred=1_8_4_1      117.55 best=1_8_4_1      117.55 gap=  0.0%
prefill MLP down  pred=1_4_8_1      926.95 best=1_4_8_1      926.95 gap=  0.0%
prefill MLP up    pred=1_4_8_1     1037.61 best=1_4_8_1     1037.61 gap=  0.0%
prefill Q/O proj  pred=1_4_8_1      340.14 best=1_8_4_1      331.07 gap=  2.7%
prefill QK^T      pred=1_4_8_1      734.54 best=4_1_8_1      731.06 gap=  0.5%
prefill attn@V    pred=1_16_2_1     197.72 best=1_16_2_1     197.72 gap=  0.0%
```

Interpretation: the generic model is within `28.76 us` of the forced-split device best over the 12 Granite shapes (`5176.95 us` vs `5148.19 us`) and within about `9 us` of the prior shape-specific fit (`5167.88 us`). The remaining notable miss is decode `QK^T`, where the generic model picks `1_4_3_2` instead of device-best `8_2_1_2`; this is left visible rather than hidden behind an op-shaped heuristic.

## Emitted Split Probe

```csv
op,phase,shape,shared_weight,compact_split,sdsc_split,device_us_median
QK^T,prefill,512x32x512x128,False,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
attn@V,prefill,32x512x128x512,False,1_16_2_1,"{'x': 1, 'mb': 16, 'out': 2, 'in': 1}",
Q/O proj,prefill,1x512x4096x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
K/V proj,prefill,1x512x1024x4096,True,1_8_4_1,"{'x': 1, 'mb': 8, 'out': 4, 'in': 1}",
MLP up,prefill,1x512x12800x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
MLP down,prefill,1x512x4096x12800,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
QK^T,decode,64x32x576x128,False,1_4_3_2,"{'x': 1, 'mb': 4, 'out': 3, 'in': 2}",
attn@V,decode,32x64x128x576,False,1_4_2_3,"{'x': 1, 'mb': 4, 'out': 2, 'in': 3}",
Q/O proj,decode,1x64x4096x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
K/V proj,decode,1x64x1024x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
MLP up,decode,1x64x12800x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
MLP down,decode,1x64x4096x12800,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
```
