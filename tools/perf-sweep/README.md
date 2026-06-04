# Spyre perf-sweep scripts

Working scripts used in the MLP / matmul performance study on the Spyre AIU
(torch-spyre vs `torch_sendnn`). These are research harnesses, not production
tooling — they carry **hard-coded `/tmp` paths and dev-pod lib paths** (the
built worktrees, the venvs, `LD_LIBRARY_PATH`). Treat them as a record of what
was run and adjust the paths for your setup.

## Device rule

There is **one accelerator** per pod. Every script that runs on device must be
run **strictly serially** — one timing at a time, never concurrently — or the
numbers are confounded.

## Environment

| What | Value (dev pod) |
|---|---|
| torch-spyre python | `/home/adnan/dt-inductor/.venv/bin/python` (torch 2.11) |
| sendnn python | `/tmp/sendnn210-venv/bin/python` |
| `PYTHONPATH` (tsp) | `/tmp/cost_model_unified_shim` — makes `import torch_spyre` resolve to the built proxy worktree (see `cost_model_unified_shim/sitecustomize.py`) |
| Profiler | `USE_SPYRE_PROFILER=1` — **required** to get device events (`spyre_ms` / kernel times). A `_C.so` built without it silently reports zeros. |
| Other env | `SENCORES=32`, `DXP_LX_FRAC_AVAIL=1` |
| `LD_LIBRARY_PATH` | the `LD_TSP` / `LD_SENDNN` lists in `run_matmul_ab.sh`. For the **PR branch** binary, prepend `/opt/ibm/spyre/runtime/lib` (node-SDK libflex first) or the import fails with an `AllocationDirective … MemoryType` undefined symbol. |

`kernel_ms` everywhere below = **sum of the compute device events only**
(excludes `Memset`, `Memcpy`, and the harness `.sum()` reduction). It is device
execution time, not host launch / wall time.

## The canonical MLP measurement (most important)

Captures **two number sets** for the SwiGLU FFN (`(silu(x@Wg)*(x@Wu))@Wd`,
emb=4096, intermediate=16384), prefill (M=512) and decode (M=4):

- `mlp_probe_tsp.py` — torch-spyre probe. Weights are **resident device tensors**
  (loaded once), so `kernel_ms` is pure on-device compute with ~0 memcpy. Full
  per-event device dump.
- `mlp_probe_sendnn.py` — same probe for the sendnn backend.
- `mlp_harnessB_probe.py` — **Harness B**: replicates the production
  `benchmark.py --op mlp` behaviour (per-call `.to(device)` weight reload) with a
  full per-event dump, so you see the reload `Memcpy`/`Memset` separately from the
  compute kernel.
- `run_probe.sh` — driver for the probes across {prefill, decode} × {tsp, sendnn}.

Persistent (`mlp_probe_*`) vs production (`mlp_harnessB_probe`) is the bridge
between "our" resident-weight numbers and the production/profile numbers: the
gap is per-call weight-reload memory traffic.

> NOTE / open item: the `mlp` op reads `input_shapes[0][0]=M`, `[0][1]=emb`, so
> `[4, 4096]` is M=4 (a single `mm`), whereas the production profile's
> `[[4, 1, 4096]]` is **batch=4, M=1** (a `bmm`). These are different ops and the
> decode kernel times do **not** match (a batch-4 `bmm` may re-stream weights
> per batch element). Reconcile on identical shapes + build before quoting decode
> numbers.

## HBM bandwidth

- `hbm_bw_sweep.py` — pure data-movement sweep (`y = x*2`, 1 read + 1 write) over
  tensor sizes → the empirical bus-saturation bandwidth (~113 GB/s plateau over
  16–256 MB, ~55% of the 204.8 GB/s LPDDR5 spec; a codegen/tiling cliff above
  256 MB to exclude). No matmul, so nothing confounds the bandwidth read.
- `hbm_analysis.py` — offline arithmetic for the cost model's `hbm_us` /
  `cohort_penalty` term on the wide-N matmul.

## Force-split harnesses

Force a specific `(m, n, k)` work-division split on a matmul to measure each
candidate's kernel time (patches `work_division.multi_dim_iteration_space_split`).

- `force_split_mnk.py` — full 3D `(m, n, k)` forcing (use for prefill M-splits).
- `force_split_dbg.py` — 2D `(N, K)` forcing only (M folded out / decode). Cannot
  express an M-split, so it no-ops for M>1 shapes — use `force_split_mnk.py` there.
- `force_split_timing.py` — earlier timing variant.
- `run_fs_tsp.sh` — env wrapper for the force-split runs.

Required inductor knobs (already set in the harnesses): `compile_threads=1`,
`worker_start_method=fork`, `fx_graph_cache=False`, `fx_graph_remote_cache=False`,
plus a `sys.meta_path` EditableFinder strip so the patch fires.

## Planner A/B sweep

- `run_matmul_ab.sh` — planner **OFF-vs-ON** A/B over the 6 Priyanka matmul shapes
  plus the sendnn baseline, same profiler `_C.so` / harness for OFF and ON (so the
  tsp/tsp ratio is a clean same-harness improvement factor). OFF =
  `SPYRE_COST_MODEL_MATMUL_PLANNER=0` (≈ main heuristic), ON = `=1`. Holds the
  `LD_TSP` / `LD_SENDNN` lists.

## Cost model snapshot

- `cm_wd.py` — standalone copy of `work_division.py` (the cost model) for offline
  cost-function experiments. Verify against the live `work_division.py` before
  trusting (it is a point-in-time snapshot).
- `cost_model_unified_shim/sitecustomize.py` — put its directory on `PYTHONPATH`
  to make `import torch_spyre` resolve to the built proxy worktree instead of the
  venv's editable install.
