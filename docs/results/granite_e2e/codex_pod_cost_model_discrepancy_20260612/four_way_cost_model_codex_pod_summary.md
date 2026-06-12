# Four-way cost-model A/B on Codex pod

All picked splits are real emitted SDSC picks for each branch, scored against the same Codex-pod forced-split timing oracle.

## Inputs

- Oracle: `/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/device_timing_repro_20260612_040740/all_splits/rows.json`
- pr2407: `/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/cost_model_2407_vs_tuned_20260612_173630/current_picks_pr2407/current_picks.csv`
- latest-main: `/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/four_way_cost_model_codex_pod_20260612_190017/current_picks_latest_main/current_picks.csv`
- tuned: `/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/cost_model_tuned_branch_20260612_161303/generic_structural_cost_probe/current_picks/current_picks.csv`
- claude-7fb4e55: `/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/claude_cost_model_min_cores_vs_tuned_20260612_174428/current_picks_claude/current_picks.csv`

## Totals

| model | selected us | gap vs device-best | speedup vs latest-main | speedup vs PR2407 |
|---|---:|---:|---:|---:|
| latest-main | 7575.95 | 47.2% | 1.000x | 0.742x |
| pr2407 | 5617.99 | 9.1% | 1.349x | 1.000x |
| tuned | 5176.96 | 0.6% | 1.463x | 1.085x |
| claude-7fb4e55 | 5263.80 | 2.2% | 1.439x | 1.067x |
| device-best | 5148.19 | 0.0% | 1.472x | 1.091x |

## Phase totals

| phase | best us | latest-main gap | PR2407 gap | tuned gap | Claude gap |
|---|---:|---:|---:|---:|---:|
| prefill | 3341.96 | 13.6% | 9.5% | 0.4% | 2.3% |
| decode | 1806.24 | 109.3% | 8.5% | 0.9% | 2.2% |

## Per shape

| op | phase | shape | best | PR2407 | latest-main | tuned | Claude |
|---|---|---|---:|---:|---:|---:|---:|
| QK^T | prefill | 512x32x512x128 | 4_1_8_1 731 | 1_4_8_1 735 (+0%) | 32_1_1_1 989 (+35%) | 1_4_8_1 735 (+0%) | 1_4_8_1 735 (+0%) |
| attn@V | prefill | 32x512x128x512 | 1_16_2_1 198 | 1_32_1_1 327 (+66%) | 1_32_1_1 327 (+66%) | 1_16_2_1 198 (+0%) | 2_8_2_1 204 (+3%) |
| Q/O proj | prefill | 1x512x4096x4096 | 1_8_4_1 331 | 1_8_4_1 331 (+0%) | 1_4_8_1 340 (+3%) | 1_4_8_1 340 (+3%) | 1_4_8_1 340 (+3%) |
| K/V proj | prefill | 1x512x1024x4096 | 1_8_4_1 118 | 1_8_4_1 118 (+0%) | 1_4_8_1 175 (+48%) | 1_8_4_1 118 (+0%) | 1_4_8_1 175 (+48%) |
| MLP up | prefill | 1x512x12800x4096 | 1_4_8_1 1038 | 1_8_4_1 1140 (+10%) | 1_4_8_1 1038 (+0%) | 1_4_8_1 1038 (+0%) | 1_4_8_1 1038 (+0%) |
| MLP down | prefill | 1x512x4096x12800 | 1_4_8_1 927 | 1_8_4_1 1008 (+9%) | 1_4_8_1 927 (+0%) | 1_4_8_1 927 (+0%) | 1_4_8_1 927 (+0%) |
| QK^T | decode | 64x32x576x128 | 8_2_1_2 90 | 32_1_1_1 203 (+126%) | 32_1_1_1 203 (+126%) | 1_4_3_2 105 (+17%) | 4_8_1_1 124 (+37%) |
| attn@V | decode | 32x64x128x576 | 1_4_2_3 55 | 1_32_1_1 94 (+72%) | 1_32_1_1 94 (+72%) | 1_4_2_3 55 (+0%) | 2_8_2_1 60 (+9%) |
| Q/O proj | decode | 1x64x4096x4096 | 1_4_8_1 232 | 1_4_8_1 232 (+0%) | 1_32_1_1 623 (+169%) | 1_4_8_1 232 (+0%) | 1_4_8_1 232 (+0%) |
| K/V proj | decode | 1x64x1024x4096 | 1_8_4_1 67 | 1_4_8_1 68 (+2%) | 1_32_1_1 143 (+114%) | 1_4_8_1 68 (+2%) | 1_4_8_1 68 (+2%) |
| MLP up | decode | 1x64x12800x4096 | 1_4_8_1 673 | 1_4_8_1 673 (+0%) | 1_4_8_1 673 (+0%) | 1_4_8_1 673 (+0%) | 1_4_8_1 673 (+0%) |
| MLP down | decode | 1x64x4096x12800 | 1_4_4_1 689 | 1_4_8_1 689 (+0%) | 1_32_1_1 2044 (+197%) | 1_4_8_1 689 (+0%) | 1_4_8_1 689 (+0%) |

## Read

On this Codex pod, latest-main is worse than PR2407 because it regresses decode shared-weight projections to pure-M splits. Both tuned models restore those projection splits. The Codex tuned model wins on projection/prefill coverage; Claude 7fb4e55 wins on attention BMMs, especially attn@V decode. The remaining common miss is QK^T prefill, where all non-best models pick 1_4_8_1 on this oracle.
