# Codex Pod Per-Core PSUM Joint Fit

## Summary

This artifact evaluates Claude's per-core PSUM idea against the Codex-pod 12-shape Granite forced-split timing sweep. The already-pushed tuned cost model remains the production candidate. The per-core PSUM variant is promising offline, but the first implementation probe did not reproduce the offline MLP-down split, so it should stay experimental until the planner metadata path is understood.

## Baselines

- Device-best total from the forced-split sweep: `5148.19 us`.
- Upstream/main selected-split total measured on this pod: `7575.95 us` (`1.47x` device-best).
- Pushed tuned model selected-split total: `5244.05 us` (`1.019x` device-best).
- One-line per-core PSUM hybrid selected-split total: `5251.11 us` (`1.020x` device-best), slightly worse than the pushed tuned model because it moved prefill `attn@V` to a slower K-split.

## Joint Fit Result

The joint search over per-core PSUM and the other cost terms found this offline candidate:

```text
searched 12001 candidates in 13.2s; best first seen at 7521
best score=5168.88 selected_total_us=5167.88 exact=8/12 max_gap=16.6%
params={"batch_penalty": 150, "core_penalty": 150, "long_k_m_penalty": 0, "long_k_n_discount": 150, "m_penalty": 50, "n_penalty": 150, "psum_bmm_per_core": 0.001, "psum_shared_per_core": 0.0001, "pt_power": 0.25, "qk_batch_discount": 10, "qk_m_penalty": 50, "qk_n_penalty": 25, "qk_target_m_split": 3, "target_n_tile": 1024, "target_passes": 3, "tie_passes": 4}
decode  K/V proj  pred=1_4_8_1       67.80 best=1_8_4_1       66.78 gap=  1.5%
decode  MLP down  pred=1_4_8_1      689.43 best=1_4_4_1      689.20 gap=  0.0%
decode  MLP up    pred=1_4_8_1      673.44 best=1_4_8_1      673.44 gap=  0.0%
decode  Q/O proj  pred=1_4_8_1      231.84 best=1_4_8_1      231.84 gap=  0.0%
decode  QK^T      pred=1_4_3_2      104.89 best=8_2_1_2       89.93 gap= 16.6%
decode  attn@V    pred=1_4_2_3       55.04 best=1_4_2_3       55.04 gap=  0.0%
prefill K/V proj  pred=1_8_4_1      117.55 best=1_8_4_1      117.55 gap=  0.0%
prefill MLP down  pred=1_4_8_1      926.95 best=1_4_8_1      926.95 gap=  0.0%
prefill MLP up    pred=1_4_8_1     1037.61 best=1_4_8_1     1037.61 gap=  0.0%
prefill Q/O proj  pred=1_8_4_1      331.07 best=1_8_4_1      331.07 gap=  0.0%
prefill QK^T      pred=1_4_8_1      734.54 best=4_1_8_1      731.06 gap=  0.5%
prefill attn@V    pred=1_16_2_1     197.72 best=1_16_2_1     197.72 gap=  0.0%
```

This is close to device-best (`5167.88 us`, `1.004x`), but it depends on two extra shape-aware corrections beyond the clean pushed model:

- QK balance terms to keep decode QK from overusing the wrong M/N/batch shape.
- A long-K shared-weight term to prefer more N splitting for prefill MLP-down.

## Code Probe

I applied the fitted per-core PSUM candidate locally in the isolated `cost-model-tuned` worktree and ran the no-joint current-pick probe. It emitted:

```csv
op,phase,shape,shared_weight,compact_split,sdsc_split,device_us_median
QK^T,prefill,512x32x512x128,False,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
attn@V,prefill,32x512x128x512,False,1_16_2_1,"{'x': 1, 'mb': 16, 'out': 2, 'in': 1}",
Q/O proj,prefill,1x512x4096x4096,True,1_8_4_1,"{'x': 1, 'mb': 8, 'out': 4, 'in': 1}",
K/V proj,prefill,1x512x1024x4096,True,1_8_4_1,"{'x': 1, 'mb': 8, 'out': 4, 'in': 1}",
MLP up,prefill,1x512x12800x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
MLP down,prefill,1x512x4096x12800,True,1_8_4_1,"{'x': 1, 'mb': 8, 'out': 4, 'in': 1}",
QK^T,decode,64x32x576x128,False,1_4_3_2,"{'x': 1, 'mb': 4, 'out': 3, 'in': 2}",
attn@V,decode,32x64x128x576,False,1_4_2_3,"{'x': 1, 'mb': 4, 'out': 2, 'in': 3}",
Q/O proj,decode,1x64x4096x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
K/V proj,decode,1x64x1024x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
MLP up,decode,1x64x12800x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
MLP down,decode,1x64x4096x12800,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
```

The important mismatch: direct `_matmul_split_cost(..., shared_weight=True)` prefers `1_4_8_1` for prefill MLP-down under the fitted candidate, but the compiled probe still emitted `1_8_4_1`. That means the offline fit is not enough; the real planner/lowering path is not applying the fitted shared-weight interpretation for that case, or the probe path is observing a different row-dimension identity than the scalar cost call.

## Conclusion

Do not replace the pushed tuned cost model with the per-core PSUM variant yet. The per-core formulation is worth continuing, but it needs a source-level audit of the MLP-down planner path before it can be considered production-ready. The clean tuned model is still the safer PR candidate because it is implemented, tested, pushed, and verified to emit the intended Granite splits.
