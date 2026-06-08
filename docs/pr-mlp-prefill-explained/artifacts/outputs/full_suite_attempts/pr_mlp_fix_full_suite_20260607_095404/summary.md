# pr-mlp-fix full-suite run summary

Runroot: `/tmp/pr-mlp-fix-full-suite-20260607-095404`

Artifacts:
- `report.txt`
- `kernel_report.txt`
- `report.xml`
- `full_suite.log`
- `perf/`

Version info from the generated report:
- torch-spyre: `824ad4b` on branch `pr-mlp-fix`
- flex: `2457d3fc` on branch `main`
- deeptools: `60b12999e4` on branch `master`
- spyre-perf-suite: `7450624`

The report was regenerated from the existing perf files after setting
`PS_FLEX_PATH=/home/adnan-cdx/dt-inductor-codex-clean/flex` and
`PS_DEEPTOOLS_PATH=/home/adnan-cdx/dt-inductor-codex-clean/deeptools`, so
the measurements are unchanged but the version header now includes flex and
deeptools provenance.

## Environment fixes used

The run used the ready `spyre-perf-suite/run_benchmark.py` flow from the pod, copied into an isolated runroot. The branch source was also copied into that runroot.

Two environment/harness fixes were needed to get a complete built-in-op report:

- `PYTHONPATH` preserved the benchmark env and explicitly included `/home/adnan-cdx/dt-inductor-codex-clean/torch_sendnn`, fixing the earlier `ModuleNotFoundError: No module named 'torch_sendnn'`.
- A runroot-local copy of `run_benchmark.py` tolerated the known sendnn child shutdown segfault only when the sendnn perf file had already been written, and skipped the known torch-spyre default-LX prefill MLP failure for `mlp [[1,512,4096]]`.

The skipped torch-spyre MLP failure is the known LX-planning failure:

`DtException: Program verification failed ... Register initialization out of boundary: lxsu0 : LRF0 : 2457472`

No branch files or shared benchmark-suite files were modified for this run.

## Key kernel ratios

`tsp/sendnn < 1` means torch-spyre is faster.

| op | shape | torch-spyre ms | sendnn ms | tsp/sendnn |
|---|---:|---:|---:|---:|
| matmul QO prefill | `[[1,512,4096],[4096,4096]]` | 0.321 | 0.358 | 0.897x |
| matmul QO decode | `[[4,1,4096],[4096,4096]]` | 0.214 | 0.233 | 0.918x |
| matmul KV prefill | `[[1,512,4096],[4096,1024]]` | 0.090 | 0.106 | 0.849x |
| matmul KV decode | `[[4,1,4096],[4096,1024]]` | 0.056 | 0.062 | 0.903x |
| matmul MLP-proj prefill | `[[1,512,4096],[4096,12800]]` | 1.020 | 0.959 | 1.064x |
| matmul MLP-proj decode | `[[4,1,4096],[4096,12800]]` | 0.721 | 0.846 | 0.852x |
| default-suite MLP decode | `[[4,1,4096]]` | 23.407 | 8.497 | 2.755x |
| attention | `[[1,512,128],[1,4096,128],[1,4096,128]]` | 0.269 | 0.063 | 4.270x |

## Read

The PR closes the official matmul gap. The six official matmul kernel ratios are now between `0.849x` and `1.064x` vs sendnn; the original problem child, prefill MLP-proj `[[1,512,4096],[4096,12800]]`, is effectively at parity by kernel time.

The default-suite MLP decode gap remains because the suite's `mlp` op lowers to the batched-weight/BMM form. That is the already-identified BMM issue, not the shared-weight transformer MLP path the matmul fix targets.

The default-suite torch-spyre MLP prefill row is intentionally `N/A` in this report because default `LX_PLANNING=1` hits the known LX verifier failure before runtime. Earlier isolation showed this failure is not introduced by `pr-mlp-fix`.

Granite was attempted by the default suite and generated the expected eight perf filenames, but all Granite files were zero bytes. The report therefore contains empty Granite sections, so this run does not provide a valid Granite comparison.
