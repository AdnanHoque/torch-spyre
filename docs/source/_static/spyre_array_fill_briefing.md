# Briefing: closing the torch-spyre ↔ sendnn PT-array under-fill gap

**Mission.** torch-spyre's wide-N MLP matmul runs **~3.3× slower** than the incumbent **sendnn** stack on
the IBM Spyre AIU, because it **under-fills the 8×8 PT systolic array** (PT-Util 29.5% vs 77.7%). A prior
investigation **proved the root cause** (§5). Your job is to explore the **fix** (§6): how to make
torch-spyre's matmul actually run DeepTools' weight-preload / double-buffer optimization. You have no prior
context — this brief gives you everything. The §5 diagnosis is device-proven, but you're welcome to
re-verify any step you doubt; the real value is in §6.

---

## 1. The one matmul that matters

Model: **Granite 3.3 8B**, prefill (512 tokens). The dominant gap is the SwiGLU FFN's gate/up projection:

```
C[512, 12800] = A[512, 4096] @ B[4096, 12800]
   M=512 (rows/tokens)   K=4096 (contraction, 64 sticks)   N=12800 (output, 200 sticks)
```

Measured (torch-spyre `main 5838b3b`, perf-suite owner's run): tsp **3.16 ms** / sendnn **0.96 ms** = **3.30×**;
**PT-Util 29.5% vs 77.7%**; tsp's HBM traffic is actually *lower*. This matmul is ~41% of whole-model prefill
kernel time (whole model 2.6×). **Decode is settled (parity, memory-bound) — out of scope; this is purely
prefill / array-fill.**

---

## 2. Hardware in one paragraph

Each AIU core has an **8×8 systolic PT array** (64 MACs) using a **weight-stationary (KG3)** dataflow: a K×N
weight tile loads into the per-corelet **weight register file (XRF, ~64 KB)** and stays stationary while 8
activation rows stream per "PT pass" (pipeline fills over ~8 passes). Tiling unit = a **stick = 64 fp16
elements**. 32 cores, LX scratchpad ~2 MB/core, HBM ~143 GB/s measured read. Lowering: torch-spyre →
TorchInductor → **SDSC** descriptor → **DeepTools** (`dxp_standalone`) → device binary. **sendnn is a separate
frontend (graph capture + DNNDaSher) that also bottoms out in the same DeepTools backend.**

---

## 3. What is MEASURED (trust these)

- The 3.30× kernel gap and PT-Util 29.5 vs 77.7 (same shape, same build).
- **Overlap signature (M-sweep, COMPUTE-only kernel_ms, INTER=16384):** tsp **0.95 / 0.99 / 2.72 ms** at
  M=1/64/512 (climbs — read+compute serialized) vs sendnn **0.94 / 0.99 / 1.18 ms** (flat — compute hidden
  under the weight read). HBM-read floor ≈ 0.81 ms.
- Forcing every legal work-division `(m,n,k)` split moves kernel time **<2%** → the split is not the lever.
- The MLP compiles to **0 restickifies**.

## 4. What is ESTABLISHED (don't re-derive)

- **Not the work-division split.** The planner already picks `(m=8,n=4,k=1)` → per-core M=64 = 8 PT passes =
  optimal row occupancy. (`work_division.py`: `_PT_ROWS=8`, `_TARGET_PT_PASSES=8`, `pt_eff=min(1,√(pt_passes/8))`.)
- **Not bandwidth.** Same 100 MB weight read; tsp's traffic is lower; sendnn finishes near the 0.70–0.81 ms
  read floor.
- **The cost model is blind to fill:** `pt_eff=1.0` (believes the array is full) at 29.5% realized — it models
  only per-core row count, not the weight-stream pipeline.
- **Geometry:** per-core weight slice at `(8,4)` ≈ **25 MB** ≫ 64 KB XRF and 2 MB LX, so weights must stream
  from HBM through a deep chunk loop during the matmul.

## 5. THE ROOT CAUSE — device-proven (this is settled)

A prior on-device investigation proved all four:

1. **Same DeepTools template.** Both run the matmul on the `pt` array (tsp `batchmatmul`, sendnn `MatMul`/`bmm`),
   same DeepTools libs (`/home/adnan/dt-inductor/sentient/deeptools/lib`). Not a different kernel.
2. **The difference is the SCHEDULE, not the template.** tsp's SDSC JSON has **zero** schedule fields (grep for
   `psum/buffer/doubleBuf/prefetch/unichain/...` → 0 hits; its `scheduleTree_` is just HBM allocations). When
   sendnn compiles the same matmul, DeepTools **explicitly** prints these optimizer stages:
   `Allocate double buffer space in LX` · `Weight Preload Optimization` · `LX Optimization` ·
   `Spatial Work Division and Pinning` · `Dynamic working set Act2/Act3` (banners from `libbaseOptimizer.so`),
   and tiles the weight `[16384,4096] → [64,4096,256]` (N-chunked, double-buffered).
3. **It's the DESCRIPTOR / ENTRY POINT, not `DT_OPT` flags.** torch-spyre's *only* DeepTools call is
   `subprocess.run(["dxp_standalone","--bundle","-d",out])` at **`torch_spyre/execution/async_compile.py:54`**.
   The `--bundle` path takes a pre-work-divided SDSC and compiles ~directly to a program, **skipping
   `libbaseOptimizer.so`**. PROVEN inert: recompiling the bundle bare vs with
   `DT_OPT=varsub=1,lxopt=1,opfusion=1,arithfold=1,dataopt=1` produced **byte-identical** output
   (segment_size / dsg / init), and on-device timing didn't move. The optimizer is *linked into* dxp_standalone
   but *not invoked* on the bundle path; sendnn's DeepRT/DSM pipeline *does* invoke it.
4. **"sendnn overlaps the weight stream, tsp doesn't" — proven** by the §3 M-sweep + the explicit double-buffer/
   weight-preload scheduler stages.

*Still inferred:* the exact `psum_algo` (unichain/bichain/singleshot) and per-chunk buffer depth (compiled into
the binary, not dumped).

**Net:** DeepTools *has* the optimization that fills the array; torch-spyre's `--bundle` entry point bypasses
it. This is a **torch-spyre ↔ DeepTools interface problem, not a missing-kernel problem.**

## 6. THE FRONTIER — what to explore (the actual fix)

How do we make torch-spyre's matmul run DeepTools' weight-preload + double-buffer passes? Two candidate paths —
investigate both, find which is viable:

- **(a) Entry point.** What DeepTools entry points exist besides `dxp_standalone --bundle`? What does
  **sendnn's** path call to get the DeepRT/DSM/BaseOptimizer pipeline (start at
  `/home/adnan/dt-inductor/torch_sendnn/` — `utils/graph_cache.py`, `env_state.py`, the libsendnn interface)?
  Can torch-spyre's `async_compile.py` route the matmul SDSC through an entry that runs `libbaseOptimizer.so`
  (double-buffer alloc + weight preload) instead of, or in addition to, `--bundle`? What does dxp_standalone's
  CLI/help expose (other than `--bundle`)? Is there a flag/subcommand that enables the optimizer passes on a
  bundle?
- **(b) Richer SDSC.** Could the Inductor planner emit an SDSC that *already encodes* the N-chunked,
  double-buffered weight tiling (the `[64,4096,256]` structure sendnn produces) so that even the `--bundle`
  path yields an overlapped program? What SDSC fields would express buffer count / chunk tiling / preload?
  (See `torch_spyre/_inductor/codegen/{superdsc.py,compute_ops.py}` for what the SDSC can currently express,
  and `schedule-ir-spec.md` in the KB for the buffering directives the schedule IR supports.)

Acceptance metric: **PT-Util on this matmul rising from ~29.5% toward sendnn's 77.7%** (or kernel time
3.16 → ~1.0 ms), measured on device.

## 7. Environment & artifacts (all on this dev pod)

| What | Where |
|---|---|
| torch-spyre repo | `/home/adnan/dt-inductor/torch-spyre` (`torch_spyre/execution/async_compile.py:54` = the DeepTools call; `_inductor/{work_division.py,codegen/superdsc.py,codegen/compute_ops.py}`) |
| sendnn source | `/home/adnan/dt-inductor/torch_sendnn/` (`utils/graph_cache.py`, `env_state.py`) |
| DeepTools libs | `/home/adnan/dt-inductor/sentient/deeptools/lib` (`libbaseOptimizer.so`, `libdsm.so`, `libdeeprt.so`, `libdsc.so`); binary `dxp_standalone` on PATH |
| torch-spyre python | `/home/adnan/dt-inductor/.venv/bin/python` (torch 2.11), `PYTHONPATH=/tmp/cost_model_unified_shim` |
| sendnn python | `/tmp/sendnn210-venv/bin/python` |
| **device probes** | `/tmp/spyre-perf-sweep-mod/tools/perf-sweep/run_probe.sh <tsp\|sendnn> <variant> <M>` — variants `mm1` (single front matmul), `ffn`, `front_real`, `mm_down`, `silu_only`. Prints COMPUTE kernel_ms + full per-event dump; has a device-free wait + correct env. |
| **dumped tsp SDSC + DeepTools artifacts** | `/tmp/sdsc-mlp/sdsc_dumps/mlp_M512_K4096_N12800_{heuristic,cost_model}/` (`sdsc_0_batchmatmul.json`, `execute_dsg.txt`, `pagi.json`, `segment_size.json`) |
| **sendnn evidence (from the proof run)** | `/tmp/sendnn_mm1_512.log` (full compile + optimizer-stage banners), `/tmp/sendnn_cache_probe/*.graph.cbor` (serialized program, shows the `[64,4096,256]` weight tiling), `/tmp/dt_parity_baseline/` (the DT_OPT byte-identity test) |
| Spyre KB | `/tmp/spyre-kb` (clone: `GH_HOST=github.ibm.com gh repo clone msrivats/spyre-knowledgebase /tmp/spyre-kb`). Key: `wiki/concepts/dataflow-architecture.md`, `wiki/artifacts/designs/schedule-ir-spec.md`, `wiki/stack/deeptools.md`. **KB omits sendnn by policy.** |

**Useful flags.** SDSC JSON is written every compile to `$TORCH_INDUCTOR_CACHE/inductor-spyre/<kernel>_<rand>/`.
Planning DEBUG dumps: `SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=DEBUG`. DeepTools verbosity:
`DT_DEEPRT_VERBOSE` (try 2–3), `DTLOG_LEVEL=debug`; sendnn graph cache `TORCH_SENDNN_CACHE_ENABLE=1` +
`TORCH_SENDNN_CACHE_DIR=...`.

## 8. Rules

- **ONE shared accelerator.** Run every device job **strictly serially** — never two timing jobs at once
  (parallel runs confound the numbers). The probe harness has a device-free wait; respect it.
- **Do NOT rebuild** the backend (`setup.py`) without a strong reason — it's a multi-minute build.
- **Distinguish measured from inferred** in everything you report.

## 9. The deliverable we'd value most

A **reproducible way to make torch-spyre's matmul run DeepTools' weight-preload/double-buffer passes** (entry
point or richer SDSC), with PT-Util measured rising toward 77.7% — or, if that's infeasible without DeepTools
changes, the **exact minimal DeepTools/interface change** required, scoped as a concrete RFC. A second-opinion
verification (or refutation) of the §5 root cause is also welcome.
