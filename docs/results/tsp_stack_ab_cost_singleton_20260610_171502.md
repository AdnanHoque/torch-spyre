# TSP stack A/B: cost model and singleton sequence

Run root: `/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/tsp_stack_ab_cost_singleton_20260610_171502`

## Commits
- `cost_before 937aa1c722bb83c9abc9b5d317dfb067a8d0451e`
- `cost_after 20978b92ebe7b97d729937d0155de5fec99c2ff8`
- `singleton_before f520b5e670a74bfd2622cf16344836a8b39319b3`
- `singleton_after 9035fb8ec579567201e505e638808cadcf5b9fb4`

Environment: py211 clean compile/timing lane, `SPYRE_COST_MODEL_MATMUL_PLANNER=1`, `LX_PLANNING=0`, `SENCORES=32`. Timings are resident-device wall timings with synchronization, not Spyre profiler `kernel_ms`.

## PR2407 Standalone Matmul A/B
| shape | before ms | after ms | speedup |
|---|---:|---:|---:|
| prefill_kv | 0.426 | 0.209 | 2.04x |
| prefill_qo | 0.948 | 0.411 | 2.31x |
| prefill_mlp_proj | 2.371 | 1.259 | 1.88x |
| decode_bs4_kv | 0.148 | 0.145 | 1.02x |
| decode_bs4_qo | 0.311 | 0.319 | 0.97x |
| decode_bs4_mlp_proj | 0.924 | 0.823 | 1.12x |
| decode_bs1_kv | 0.146 | 0.144 | 1.01x |

## PR2407 Antoni/FMS-Style MLP A/B
| tokens | variant | before ms | after ms | speedup |
|---:|---|---:|---:|---:|
| 512 | single_gate | 5.690 | 4.762 | 1.20x |
| 512 | res_gate | 5.811 | 4.924 | 1.18x |
| 512 | full_glu_chain | 15.424 | 11.989 | 1.29x |
| 1 | single_gate | 3.215 | 3.247 | 0.99x |
| 1 | res_gate | 3.223 | 3.232 | 1.00x |
| 1 | full_glu_chain | 8.614 | 8.555 | 1.01x |

## Singleton Sequence Antoni/FMS-Style MLP A/B
| tokens | variant | before ms | after ms | after/before |
|---:|---|---:|---:|---:|
| 512 | single_gate | 4.796 | 4.683 | 0.976 |
| 512 | res_gate | 4.910 | 4.768 | 0.971 |
| 512 | full_glu_chain | 11.933 | 11.622 | 0.974 |
| 1 | single_gate | 3.237 | 3.212 | 0.992 |
| 1 | res_gate | 3.213 | 3.218 | 1.001 |
| 1 | full_glu_chain | 8.593 | 8.586 | 0.999 |

## Generated-Code Check
For singleton before/after, decode (`tokens=1`) generated Python hashes were identical for all three FMS-style variants and no generated code contained `shared_weight_unit_bmm`. Prefill hashes differ across commits because the source changed, but also did not contain `shared_weight_unit_bmm` in either case.

## Read
- PR2407/cost model gives the large prefill matmul win and also improves the Antoni/FMS-style prefill GLU block.
- Decode is neutral in the PR2407 A/B and neutral in the singleton-sequence A/B.
- The singleton sequence does not reproduce the e2e decode regression in this Antoni/FMS-style isolated block; it also does not appear to be the source of the FMS-style prefill win.
