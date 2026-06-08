# Prefill bs1 sl512 main-vs-PR/sendnn sweep

```text
runroot: /tmp/prefill-bs1-sl512-ab-20260608_072036
lx_planning: 0
main_ref: upstream/main
main_commit: c7861a0
main_extra_patch: 824ad4b (benchmark compatibility only: patches.py)
fix_ref: origin/pr-mlp-prefill-explained
fix_commit: e544a05
suite_commit: 7450624
flex_commit: 2457d3fc
deeptools_commit: 60b12999e4
sendnn_reference_perf: /home/adnan-cdx/codex-worktrees/pr-mlp-fix/torch-spyre/docs/pr-mlp-prefill-explained/artifacts/outputs/full_suite_attempts/pr_mlp_fix_full_suite_20260607_095404/perf
```

Notes:
- Main and PR torch-spyre columns are fresh measurements from this runroot.
- Upstream main required the compat-only `824ad4b` no-dispatch bypass to run this external `run_benchmark.py` path; the baseline performance code is otherwise upstream main.
- Fresh sendnn attempts in this runtime lane produced CPU-only zero-kernel traces, so the sendnn column uses the archived valid full-suite sendnn reference at the same `spyre-perf-suite` commit (`7450624`). Raw failed sendnn logs are archived separately for transparency.

| op | main tsp kernel ms | PR tsp kernel ms | PR speedup vs main | sendnn ref kernel ms | PR tsp/sendnn ref | main PT% | PR PT% | main spyre ms | PR spyre ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul KV | 0.127 | 0.090 | 1.411x | 0.106 | 0.849x | 47.038 | 65.956 | 0.295 | 0.295 |
| matmul QO | 0.560 | 0.318 | 1.761x | 0.358 | 0.888x | 42.522 | 74.831 | 1.081 | 0.829 |
| matmul MLP-proj | 3.729 | 1.020 | 3.656x | 0.959 | 1.064x | 29.958 | 73.014 | 5.723 | 3.003 |
| mlp | 21.724 | 6.962 | 3.120x | 5.747 | 1.211x | 11.213 | 36.188 | 26.883 | 12.232 |

## Raw JSON

```json
[
  {
    "op": "matmul KV",
    "main_kernel": 0.127,
    "fix_kernel": 0.09,
    "sendnn_kernel_ref": 0.106,
    "main_spyre": 0.295,
    "fix_spyre": 0.295,
    "sendnn_spyre_ref": 0.808,
    "main_mem": 0.168,
    "fix_mem": 0.204,
    "sendnn_mem_ref": 0.702,
    "main_pt": 47.038,
    "fix_pt": 65.956,
    "speedup_kernel": 1.4111111111111112,
    "fix_sendnn_ratio_ref": 0.8490566037735849
  },
  {
    "op": "matmul QO",
    "main_kernel": 0.56,
    "fix_kernel": 0.318,
    "sendnn_kernel_ref": 0.358,
    "main_spyre": 1.081,
    "fix_spyre": 0.829,
    "sendnn_spyre_ref": 2.514,
    "main_mem": 0.52,
    "fix_mem": 0.511,
    "sendnn_mem_ref": 2.156,
    "main_pt": 42.522,
    "fix_pt": 74.831,
    "speedup_kernel": 1.7610062893081762,
    "fix_sendnn_ratio_ref": 0.888268156424581
  },
  {
    "op": "matmul MLP-proj",
    "main_kernel": 3.729,
    "fix_kernel": 1.02,
    "sendnn_kernel_ref": 0.959,
    "main_spyre": 5.723,
    "fix_spyre": 3.003,
    "sendnn_spyre_ref": 7.103,
    "main_mem": 1.995,
    "fix_mem": 1.983,
    "sendnn_mem_ref": 6.144,
    "main_pt": 29.958,
    "fix_pt": 73.014,
    "speedup_kernel": 3.6558823529411764,
    "fix_sendnn_ratio_ref": 1.0636079249217936
  },
  {
    "op": "mlp",
    "main_kernel": 21.724,
    "fix_kernel": 6.962,
    "sendnn_kernel_ref": 5.747,
    "main_spyre": 26.883,
    "fix_spyre": 12.232,
    "sendnn_spyre_ref": 25.567,
    "main_mem": 5.159,
    "fix_mem": 5.27,
    "sendnn_mem_ref": 19.82,
    "main_pt": 11.213,
    "fix_pt": 36.188,
    "speedup_kernel": 3.120367710428038,
    "fix_sendnn_ratio_ref": 1.2114146511223247
  }
]
```
