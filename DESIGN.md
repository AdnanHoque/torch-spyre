# Fused matmul + SiLU FEST inner-epilogue for Spyre (Option 3)

**Status:** implemented, device-validated, benchmarked. Auto-fusing on `torch-spyre` worktree `torch-spyre-prod` (branch `ah/fused-swiglu-prod`, commit `f51346c`). No forced env hook; a real detection gate drives it, default-on.

---

## 1. What was built

`torch.compile(lambda a, b: F.silu(a @ b))` on Spyre now auto-fuses `batchmatmul` + `silu` into a **single SDSC segment** whose `computeOp_ = [batchmatmul, silu]`. The SiLU runs as an **inner epilogue** of the matmul: each output tile is produced by the PE array, has SiLU applied while it is **still resident in the SFP LRF registers**, and only the SiLU'd result is written out. The intermediate (pre-activation gate) tensor is **never materialized** in on-chip LX SRAM or HBM.

This is "Option 3" of three possible fusion granularities:

| Option | Ops on device | Gate tensor lifecycle | What we did |
|---|---|---|---|
| **1** matmul → **decomposed** SiLU | 2 segments | written to + read from LX; SiLU expands to neg/exp/add/reciprocal/mul (~6 pointwise) | pre-existing default path (leaky) |
| **2** matmul → **single fused-SiLU** DDL op | 2 segments | written to + read from LX; SiLU is one op | intermediate step (silu-single-op wiring) |
| **3** matmul with **SiLU in the epilogue** | **1 segment** | **never materialized** — tile stays in SFP LRF across the matmul→silu boundary | **THIS WORK** |

Option 3 eliminates (a) one standalone device segment and its launch, and (b) the LX write **and** read of the full `d_ff`-wide gate tensor.

---

## 2. The FEST mode-6/7 recipe in `bmm.ddl` (deeptools)

The single deeptools change is in `bmm.ddl` (patch: `bmm_silu_festfix.patch`). SiLU is computed as `silu(x) = sigmoid(x) * x`, with `sigmoid` approximated by the PE array's **FEST** (Fast EXponential/Sigmoid Transcendental) unit as a piecewise-linear fast-sigmoid, then finished with SFP FMA/FMUL ComputeOps. Everything stays in-register; nothing spills to LX.

**Structure of the change:**

1. **Op binding + scratch tensors.** A new `%silu_op = ddl.operation_bind(..., opFuncName="silu", required=false)` is added, plus four fp16 internal scratch tensors (`%silu_ss` slope, `%silu_cs` offset, `%silu_t`, `%silu_sig`). These are threaded into `%bmm_fp16_op`'s aux-tensor list.

2. **Ordering + exclusivity constraints.** `%silu_op` joins the `relative_op_order` constraint set. It is made **mutually exclusive with `%relu_op`, `%bn_op`, `%bias_op`** via pairwise `max_num_valid = 1` constraints — a v1 register-pressure guard (single-epilogue only). `%silu_op` also joins `%is_any_aux_op` so the last-accumulation writeback path fires for it.

3. **Fast-sigmoid centering constant.** `%const_3C00 = ddl.define_constant(...){value=[0x3c00]}` = +0.5 in SEN169 FP16 encoding, allocated to `sfplrf` and staged into the SFP LRF (`fastsigmoid_connect`) under `ddl.if(%silu_op)`.

4. **The epilogue compute block** (`ddl.if(%silu_op)` in the writeback region). With the matmul result `x` present both in `%pe_fma_lrf` (PE) and `%sfp_output_lrf` (SFP):
   - **PE FEST mode 7** → `%silu_cs` (fast-sigmoid **offset/curve** term), **PE FEST mode 6** → `%silu_ss` (fast-sigmoid **slope** term). Both read the fp16 matmul result still in PE LRF (`%pesum`). *This is the 128×-wide PE_FEST path the deeptools fix enables — PE-LRF allocations added for the FEST outputs so the internal-tensor units resolve.*
   - Move `cs`/`ss` from the PE→SFP FIFO into SFP-LRF scratch.
   - `t = cs*1.0 + 0.5` via SFP `FMA16` (uses the staged `%const_3C00`).
   - `sig = x*ss + t` via SFP `FMA16` (x = `%sfp_output_lrf`) → the fast-sigmoid value.
   - `out = sig * x` via SFP `FMUL`, **written in place into `%sfp_output_lrf`** — reusing the existing writeback so no extra store path is needed.

Net: SiLU is four SFP ComputeOps + two PE FEST ops, entirely in LRF, folded into the matmul's last-accumulation iteration.

---

## 3. Frontend inner-epilogue emission (torch-spyre)

Patch: `torch_spyre_prod.patch` (9 files, +442/−29). Two concerns: **(A)** make Inductor recognize the pattern and emit one fused kernel, and **(B)** make the SDSC codegen emit `computeOp_ = [batchmatmul, silu]`.

### 3A. The detection gate — `scheduler.py`

Mirrors the existing `can_fuse_matmul_residual_add` gate. No env hook; a real vertical-fusion predicate:

- `_is_matmul_node(n)` — `n.is_reduction()` and `get_reduction_type() == BATCH_MATMUL_OP` (`"batchmatmul"`).
- `_is_silu_node(n)` — non-reduction pointwise whose IR-node `origins` target `aten.silu.default` / `aten.silu`.
- `_silu_consumes_full_matmul(n1, n2)` — the SiLU's only non-constant read is the matmul output buffer (unary, no aux tensor) **and** its output numel equals the matmul output numel (full-shape, no broadcast/reduction).
- `can_fuse_matmul_silu(n1, n2)` = `config.enable_matmul_silu_epilogue AND _is_matmul_node(n1) AND _is_silu_node(n2) AND _silu_consumes_full_matmul(n1, n2)` — wired into `can_fuse_vertical`. Horizontal fusion stays off.

When it fires, Inductor emits a `FusedSchedulerNode` → kernel `sdsc_fused_mm_silu_0`.

### 3B. Spec + emission — `superdsc.py`, `bundle.py`, `compute_ops.py`

- **`superdsc.py`**: `Epilogue` dataclass (`opfunc`, `execution_unit`/`exUnit`, `input_indices`, `output_index`) + `epilogues` field on `SDSCSpec`. `parse_fused_matmul_silu(matmul, silu)` parses the matmul normally (`computeOp_[0]`, exUnit `"pt"`), appends `Epilogue(opfunc="silu", exUnit="sfp", input_indices=[out], output_index=out)` (reads and writes the matmul output in place), retargets the matmul output to the SiLU's final buffer, and suppresses the standalone SiLU segment. `compile_fused_matmul_silu` matches latest-main's `compile_op_spec` symbol-threading ABI; adapted to `parse_op_spec` returning `(SDSCSpec, symbol_mapping)`. `is_matmul_silu_fusion` is mutually exclusive with relu/bn/bias (single-epilogue v1, unary only) — mirroring the DDL constraint.
- **`bundle.py`**: `_fold_matmul_silu` rewrites the spec tree (recursing into `LoopSpec`) so a matmul immediately followed by a fusible SiLU sibling absorbs it; `_compile_specs` emits ONE fused SDSC via `compile_fused_matmul_silu`. Both the compile pass and the emit pass walk the **identical folded tree** (lockstep preserved — a hard requirement of the bundle protocol).
- **`compute_ops.py`**: `_compute_ops(sdsc_spec, out_idx)` builds `computeOp_` = primary + epilogues. Length-1 when there is no epilogue (plain matmul is byte-for-byte unchanged); length-2 for the fused case. Each epilogue shares the primary's iteration space / PSUM output; `num_inputs` counts only the primary's real inputs so the epilogue-only aux tensor is excluded from `inputLabeledDs`.

### 3C. SiLU-as-single-op wiring (needed on the worktree)

latest-main decomposed SiLU into primitives; the fusion needs SiLU to survive as one recognizable op:
- **`constants.py`** — `silu` in `SPYRE_FP32_OPS`; `SILU_OP` / `SILU_EPILOGUE_OP` = `"silu"`; `BATCH_MATMUL_OP = "batchmatmul"`.
- **`decompositions.py`** — exclude `aten.silu` so it stays whole.
- **`lowering.py`** — `lower_silu` → single `silu` pointwise.
- **`spyre_kernel.py`** — `PointwiseOp("silu")`.

### 3D. Config — `config.py`

`enable_matmul_silu_epilogue` bound to `SPYRE_ENABLE_MATMUL_SILU_EPILOGUE`, **default-on** (`"1"`). Setting it to `0` restores the two-op baseline (used as the benchmark control arm).

---

## 4. Validation

### 4A. Bundle evidence (auto path, no forced flag)
- `F.silu(a @ b)`, M64/N256/K256 fp16 → kernel `sdsc_fused_mm_silu_0`, **1 SDSC segment**, `computeOp_ = ['batchmatmul','silu']`, exUnits `['pt','sfp']`, **LEAKED_DECOMP_OPS = []** (zero neg/exp/add/realdiv/sigmoid/mul/reciprocal leak).
- Plain `a @ b` → **1 segment, `computeOp_ = ['batchmatmul']`** (length-1) — behavior-preserving.

### 4B. Device numerics (detached, scored vs oracle)
Acceptance oracle is `silu(device_mm)` — it isolates the **epilogue** from the matmul's own fp16 quantization.

| Oracle | cosine | max_rel | max_abs | verdict |
|---|---|---|---|---|
| `silu(device_mm)` (acceptance) | **1.000000** | **0.21%** (≤2%) | **0.0084** (≤0.05) | **PASS** |
| `silu(fp32_mm)` (pure-fp32 math) | 0.999998 | 11.4% | 0.212 | fails thresholds |

`cos_vs_raw_matmul = 0.699` confirms SiLU actually applied (not raw matmul passed through). The `silu(fp32_mm)` "failure" is the **matmul's own fp16 quantization** folded in — not a silu-epilogue defect; against the correct acceptance oracle the epilogue is clean. Matches the prior validated forced-hook numbers.

Artifact: `prod_auto_numerics.json`.

---

## 5. Benchmark results

**Harness:** `spyre-granite-e2e-bench/benchmarks/granite_block_probe.py --part mlp_core` — the real FMS Granite-3-8B `GraniteBlock.ff_sub_layer` **SwiGLU** GLU at model-card dims (hidden 4096, d_ff 12800, fp16, empty weights): `down_proj( silu(gate_proj(x)) * up_proj(x) )`. Metric: median device wall-clock ms over 10 iters (torch.compile, first-op warmed). Both arms import-shadow `torch-spyre-prod`; they differ only by the gate (`SPYRE_ENABLE_MATMUL_SILU_EPILOGUE` 1 vs 0).

| Regime | Fused (ms) | Baseline (ms) | Delta | Fused faster |
|---|---|---|---|---|
| **Prefill** (seq 512) | 16.948 | 17.127 | −0.180 ms | **1.05%** |
| **Decode** (seq 1) | 7.996 | 8.000 | −0.004 ms | **0.05%** |

Prefill fused range [16.80, 17.03] vs baseline [17.08, 17.16] — essentially non-overlapping, so the ~1% prefill win is real but small. Decode ranges fully overlap → within noise.

**Fusion is device-verified to engage** (same segment change for prefill and decode):
- **FUSED:** `[batchmatmul, batchmatmul, mul, batchmatmul]` — **no standalone silu**; SiLU folded into the gate-projection batchmatmul (`sdsc_1.json` contains `silu`; device `execute_dsg.txt` references `silu` → the `bmm.ddl %silu_op` PE-FEST path fired).
- **BASELINE:** `[batchmatmul, silu, batchmatmul, mul, batchmatmul]` — SiLU is a separate `2_silu` segment.

**Note:** HPM / bandwidth / kernel-cycle counters were **not obtainable** — the perf-suite `kernel_ms` profiler hung >27 min CPU-bound with zero traces and was abandoned in favor of the lighter probe. Only device wall-clock median is reported.

Artifacts: `probe_bench_results.json`, `probe_bench_out/{prefill,decode}_{fused,baseline}.log`, `run_probe_bench.sh`.

---

## 6. Memory-hierarchy rationale — why Option 3 wins

Spyre matmul data path: **PE array** (systolic multiply-accumulate) → **SFP** (Scalar/Float Processor, does pointwise/activation over LRF registers) → **LX** (on-chip SRAM staging) → **HBM** (off-chip). Register (SFP/PE LRF) >> LX SRAM >> HBM in bandwidth and latency.

- **Option 1/2** both **materialize the gate tensor in LX**: the matmul segment writes the full `d_ff`-wide (12800-wide) pre-activation to LX, then a second segment reads it back to apply SiLU. That is one LX **write** + one LX **read** of a large tensor, plus a second segment launch.
- **Option 3** keeps each output tile in the **SFP LRF** across the matmul→silu boundary. SiLU is applied to the tile in-register (PE FEST + SFP FMA/FMUL) and only the activated result is written out. The gate tensor **never exists** as a materialized buffer — no LX round-trip, no second launch.

So the *structural* win is exact and guaranteed: **−1 segment, −1 LX write, −1 LX read** of the gate tensor. Its *magnitude* is shape-dependent. At Granite batch=1 the three 4096×12800 matmuls dominate, and the eliminated SiLU is a cheap elementwise op over a modest tensor, so the saving is ~1% (prefill) / noise (decode). The fusion matters more where the activated tensor is large relative to matmul cost, or where segment-launch overhead is a larger fraction of the total — not the batch=1 Granite SwiGLU MLP measured here.

---

## 7. Caveats

- **Single-epilogue v1.** Unary SiLU only; mutually exclusive with relu/bn/bias epilogues (enforced both in `bmm.ddl` via `max_num_valid=1` and in `superdsc.py`). No stacked/multi epilogues.
- **Forced-hook → real gate.** The prototype used `SPYRE_EXP_FORCE_MM_SILU`; that hook is **gone** (grep-confirmed absent). Production uses the `can_fuse_matmul_silu` detection gate, default-on.
- **Perf is shape-dependent and small** at the benchmarked Granite batch=1 shapes (~1% prefill, noise decode). Correctness and the structural segment/memory-traffic reduction are solid; the latency win is marginal here by design.
- **No HPM counters** captured (profiler hang); wall-clock only.
- **Not pushed.** Kept local per pre-publish convention (worktree `torch-spyre-prod` @ `f51346c`, no push attempted).

---

## 8. Exact file lists + patches

**deeptools (1 file):**
- `bmm.ddl` — patch `bmm_silu_festfix.patch` (`%silu_op` FEST mode-6/7 inner-epilogue, PE-LRF alloc unit=pe). Live + source synced.

**torch-spyre (9 files, +442/−29):** patch `torch_spyre_prod.patch`
```
torch_spyre/_inductor/codegen/bundle.py        +85   _fold_matmul_silu, one-SDSC emit
torch_spyre/_inductor/codegen/compute_ops.py   +65   _compute_ops → computeOp_ primary+epilogues
torch_spyre/_inductor/codegen/superdsc.py     +180   Epilogue, SDSCSpec.epilogues, parse/compile_fused_matmul_silu
torch_spyre/_inductor/config.py                 +8   enable_matmul_silu_epilogue (default-on)
torch_spyre/_inductor/constants.py             +10   silu in SPYRE_FP32_OPS, SILU_OP/SILU_EPILOGUE_OP
torch_spyre/_inductor/decompositions.py         +3   exclude aten.silu
torch_spyre/_inductor/lowering.py              +16   lower_silu → single pointwise
torch_spyre/_inductor/scheduler.py            +100   can_fuse_matmul_silu gate + helpers
torch_spyre/_inductor/spyre_kernel.py           +4   PointwiseOp("silu")
```

**Artifacts (pod `adnan-clc-spyre-dev-pf`):**
- `/home/adnan/dt-inductor/fest-bench/bmm_silu_festfix.patch`
- `/home/adnan/dt-inductor/fest-bench/torch_spyre_prod.patch`
- `/home/adnan/dt-inductor/fest-bench/prod_auto_numerics.json`
- `/home/adnan/dt-inductor/fest-bench/probe_bench_results.json`
- `/home/adnan/dt-inductor/fest-bench/probe_bench_out/`
- Worktree: `/home/adnan/dt-inductor/torch-spyre-prod` (branch `ah/fused-swiglu-prod`, commit `f51346c`)
