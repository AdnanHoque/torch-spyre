# MLP Prefill Fix Explanation Artifacts

This directory collects the writeups, benchmark reports, helper scripts, and compact source snapshots used to explain and reproduce the MLP prefill/shared-weight unit-BMM fix.

## What This Branch Contains

- `artifacts/outputs/`
  - Markdown explainers:
    - `spyre_kb_theory_grounding.md`
    - `pr_mlp_fix_writeup.md`
    - `jamie_unit_bmm_probe.md`
    - `torch_spyre_mlp_gap_takeover.md`
    - `pr_mlp_fix_shape_aware_summary.md`
    - `static_unit_batch_bmm_canonicalization/first_principles_static_unit_bmm_writeup.md`
  - Shape-aware TSV summary:
    - `pr_mlp_fix_shape_aware_summary.tsv`
  - Official benchmark-suite style reports:
    - `official_suite_report/report.txt`
    - `official_suite_report/kernel_report.txt`
    - `official_suite_report/report.xml`
  - Full-suite attempts and perf traces:
    - `full_suite_attempts/pr_mlp_fix_full_suite_20260607_090651/`
    - `full_suite_attempts/pr_mlp_fix_full_suite_20260607_095404/`

- `artifacts/pr_mlp_fix_writeup.md`
  - Copy of the local working writeup from `work/`.

- `helper_scripts/`
  - `launch_pr_mlp_fix_full_suite.sh`
  - `resume_pr_mlp_bench.sh`
  - `shared_weight_mlp_matmul_op.py`
  - `shared_weight_mlp_op.py`

- `source_snapshots/`
  - `final_benchmark_summary/`: paired summary/report material for main-vs-PR focused measurements.
  - `pr_impl/`: compact source snapshot of the PR implementation area.
  - `pr_mlp_fix_files/`: focused file snapshot for the MLP fix.
  - `remote_patch/`: compact patch/source material copied from the remote pod workflow.

## Main Performance Read

The prefill/shared-weight projection fix changes the problematic MLP-proj matmul from the old slow torch-spyre layout/split to a sendnn-like shared-weight unit-BMM representation.

## Static Unit-Batch BMM Follow-Up

A follow-up probe extends the same idea to the full perf-suite prefill MLP when
the weights arrive as static unit-batch BMM operands:

```text
[1, M, K] @ [1, K, N] -> [1, M, N]
```

The result is archived in:

```text
artifacts/outputs/static_unit_batch_bmm_canonicalization/
```

Headline paired result with `LX_PLANNING=0`:

```text
mlp [[1, 512, 4096]]
torch-spyre kernel_ms: 6.169 ms
sendnn kernel_ms:      5.788 ms
tsp/sendnn:            1.066x
```

The standalone wide projection remains near parity:

```text
matmul [[1, 512, 4096], [4096, 12800]]
torch-spyre kernel_ms: 1.018 ms
sendnn kernel_ms:      0.960 ms
tsp/sendnn:            1.060x
```

See `artifacts/outputs/static_unit_batch_bmm_canonicalization/static_unit_batch_bmm_canonicalization_report.md`
for the codegen evidence, exact commands, and kernel breakdown.

See `artifacts/outputs/static_unit_batch_bmm_canonicalization/first_principles_static_unit_bmm_writeup.md`
for a first-principles explanation of why the static unit-batch BMM shape is
safe to canonicalize, how the metadata flows through lowering into `OpSpec`,
and how each isolated compiler chain maps to measured speedup.

## Fresh Main-vs-PR Prefill Sweep

A focused bs1/sl512 sweep was rerun on the three prefill matmul shapes and the
prefill MLP:

```text
artifacts/outputs/prefill_bs1_sl512_main_vs_pr_sendnn_20260608_072036/
```

Headline torch-spyre kernel changes from upstream main to this PR:

```text
matmul KV:       0.127 ms -> 0.090 ms, 1.41x
matmul QO:       0.560 ms -> 0.318 ms, 1.76x
matmul MLP-proj: 3.729 ms -> 1.020 ms, 3.66x
mlp:            21.724 ms -> 6.962 ms, 3.12x
```

The same summary logs the new torch-spyre/sendnn ratios. Fresh sendnn attempts
in this runtime lane produced CPU-only zero-kernel traces, so those ratios use
the archived valid sendnn perf files from the same `spyre-perf-suite` commit;
the failed fresh sendnn traces are archived next to the summary for provenance.

The key official-suite style result is in:

```text
artifacts/outputs/full_suite_attempts/pr_mlp_fix_full_suite_20260607_095404/report.txt
```

Important headline from that run:

```text
matmul [[1, 512, 4096], [4096, 12800]]
torch-spyre: ~1.020 ms
sendnn:      ~0.959 ms
ratio:       ~1.064x
```

Earlier focused A/B material in `source_snapshots/final_benchmark_summary/` shows the old upstream main behavior around `3.749 ms` for the same prefill projection, so the PR gives roughly a `3.7x` torch-spyre-side improvement for that shape and brings the production matmul gap to near parity.

## Known Non-Goal

The default-suite decode MLP shape with batched 3D weights still exercises the known unit/batched BMM path. That is a separate decode/MoE-style BMM issue and is not the prefill/shared-weight fix covered by this artifact branch.

## Reproduction Pointers

Use the benchmark suite's `run_benchmark.py` flow from the pod environment. The saved launch/resume scripts in `helper_scripts/` show the exact report/perf output pattern used for the archived runs.

For external review, start with:

1. `artifacts/outputs/pr_mlp_fix_writeup.md`
2. `artifacts/outputs/spyre_kb_theory_grounding.md`
3. `artifacts/outputs/jamie_unit_bmm_probe.md`
4. `artifacts/outputs/static_unit_batch_bmm_canonicalization/first_principles_static_unit_bmm_writeup.md`
5. `artifacts/outputs/prefill_bs1_sl512_main_vs_pr_sendnn_20260608_072036/prefill_bs1_sl512_main_vs_pr_sendnn_summary.md`
6. `artifacts/outputs/full_suite_attempts/pr_mlp_fix_full_suite_20260607_095404/summary.md`
7. `artifacts/outputs/full_suite_attempts/pr_mlp_fix_full_suite_20260607_095404/report.txt`
