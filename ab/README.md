# Reshard A/B

Isolates the **cross-division `matmul→pointwise` edge** in the Granite SwiGLU MLP
and measures, by **profiler kernel time**, what eliminating it buys. Baseline
worksplits/perf and the edge taxonomy are in `../CORE_TO_CORE_SWIGLU_BASELINE.md`.

## The edge

The cost model splits the SwiGLU matmul `(m4,n8)` (`{mb:4,out:8,in:1}`); the
pointwise silu chain is pure-M (`{mb:32}`, prefill) / `out`-split (decode). Same
HBM tensor, **same stick (`out`), different shard** → an HBM round-trip with a
cross-shard re-read. Baseline PT-util is only **17–20%** (prefill) / **0.2%**
(decode) — the array is badly under-fed.

## Arms

| arm | edge treatment | matmul | hand-off | status |
|---|---|---|---|---|
| **A0 baseline** | as-is | (m4,n8) | cross-division HBM round-trip | measured (baseline doc) |
| **A1 steer** | matmul → pure-M (= consumer split) | pure-M | **same-division** (no reshard) | this harness (`--lever steer`) |
| **A2 reshard** | keep (m4,n8); move hand-off on-chip (Phase-0 map) | (m4,n8) | LX↔LX core-to-core, no HBM | next phase (STCDP substrate) |

## Decision

A0 vs A1 is the gate (cheap, no new runtime machinery):

- **A1 (pure-M) ≤ A0** → the cost model is **mis-ranking** SwiGLU's matmul; the
  fix is steering / a cost-model correction. **A2 reshard is unnecessary.**
- **A0 (m4,n8) < A1 but hand-off-bound** → A2 reshard is the prize: keep the fast
  matmul, eliminate the HBM round-trip on-chip. The **A0−A1 gap** quantifies what
  A2 must recover, and A1's matmul-time is the floor A2 can't beat.

Only build the (250-LOC + STCDP-substrate, historically value-incorrect) A2
reshard if A0 vs A1 says it's worth it.

## Files

- `run_ab.py` — runner. Reuses perf-suite `benchmark.run_tsp_stack` (identical
  kernel-time pipeline as the baseline). `--lever steer` monkeypatches
  `work_division._cost_model_divide_op → False` so the matmul takes the default
  pure-M split. Sets `compile_threads=1` + `fx_graph_cache=False` so the in-process
  patch reaches the work-division pass.
- `profenv.sh` — the locked profiler stack (source before running).
- `results/` — per-arm kernel-time outputs + the comparison.

## Run

```bash
source ab/profenv.sh
export PERF_SUITE=/tmp/spyre-perf-suite
P=/home/adnan/dt-inductor/.venv/bin/python
for lever in baseline steer; do
  for op in fms_granite_micro.swiglu fms_granite_micro.swiglu_unfused; do
    for shape in "1 512 4096" "4 1 4096"; do
      $P ab/run_ab.py --lever $lever --op $op --shape $shape --runs 3 \
        --out ab/results/${lever}_${op##*.}_$(echo $shape|tr ' ' x).txt
    done
  done
done
```

The device has an intermittent flex profiling-in-streams sync stall (≈60 s) — it
slows runs but does **not** corrupt `self_device_time_total` (device-side). Use a
long timeout; ignore wall-clock.
