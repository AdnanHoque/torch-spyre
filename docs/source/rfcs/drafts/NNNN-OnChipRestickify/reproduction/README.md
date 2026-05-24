# On-chip core-to-core reproduction harness

Research/repro harnesses for the on-chip core-to-core data-movement work behind
`NNNN-OnChipRestickifyRFC.md` and `CoreToCoreDataMovementRecipe.md`. They splice
LX-resident on-chip data ops (`STCDPOpLx`) into compiled Spyre SDSC bundles,
validate them on device, benchmark on-chip vs HBM handoffs, and classify the
handoffs in real compiled fused kernels.

These are **not production code.** They monkey-patch the kernel runner, edit
dxp-resolved JSON in place, and assume a patched dxp plus a torch_spyre worktree
that carries `onchip_bridge.py`. They are preserved here so the proof is
reproducible. They were originally developed as throwaway scripts in `/tmp`.

## 1. Environment (`env.sh`)

Every machine-specific path is an overridable default in `env.sh`
(`: "${VAR:=default}"`). The defaults are the values from the machine where this
was developed, so it still runs as-is there. Override any of them in your shell
before invoking a script:

```bash
PYTHON=/my/python WORK_DIR=/scratch bash devval/devval_roundtrip_fix_512.sh
```

The shell scripts `source ../env.sh` themselves; the Python scripts read the
same names via `os.environ.get`.

| Variable | Default | Purpose |
|---|---|---|
| `PYTHON` | `/home/adnan/dt-inductor/.venv/bin/python` | venv with torch 2.11 + torch_spyre built |
| `PATCHED_DXP` | `/home/adnan/dt-inductor/build/deeptools-onchip/dxp/dxp_standalone` | dxp carrying the deeptools on-chip foundation patch |
| `ONCHIP_SRC` | `/tmp/tier-up` | torch_spyre worktree holding `onchip_bridge.py` (the tier0-tier1-onchip checkout) |
| `ONCHIP_BRIDGE` | `${ONCHIP_SRC}/torch_spyre/_inductor/codegen/onchip_bridge.py` | the bridge emitter the splice scripts load |
| `VAL_BOOT` | in-repo `reproduction/val-boot` | `sitecustomize.py` import shim dir (goes on `PYTHONPATH`) |
| `WORK_DIR` | `/tmp` | scratch root for `spliced-*/`, `*-cache/`, baseline dirs |
| `TORCHINDUCTOR_CACHE_ROOT` | `/tmp/torchinductor_adnan` | inductor cache root used by the analysis |
| `GRANITE_INDUCTOR` | `/tmp/granite_inductor` | Granite compiled-bundle root for the analysis |
| `EDGE_GRANITE_RMSNORM`, `EDGE_SDPA`, `EDGE_ATTN_RMSNORM` | derived from the two above | the three real fused-kernel `code_dir`s `gen_report.py` classifies |

`PATCHED_DXP` is not referenced by these scripts directly (the patched dxp must
be on `PATH`/used by the build that compiles the spliced bundle); it is recorded
here so the toolchain is captured in one place.

## 2. Pipeline order

1. **Compile baseline** — `bench/gen_baseline.py` compiles
   `(a + b.t() + c.t()) @ d` at a given `BENCH_SIZE`, runs it once to populate a
   `code_dir`, and reports the `code_dir` path plus the add->add edge sharding so
   the splice can be parameterized per size.
2. **Splice** — a `splice/` script copies that baseline `code_dir`, flips the
   producer output and consumer input to LX-resident, and installs the on-chip
   data op(s) (`STCDPOpLx` round trip, or the Tier-2 transpose). Output is a new
   spliced `code_dir` under `$WORK_DIR`.
3. **Patched-dxp compile** — recompile the spliced bundle with the patched dxp
   (`PATCHED_DXP`) so it resolves the spliced JSON into a runnable senprog. (Run
   separately; not scripted here.)
4. **Device-validate** — a `devval/` script redirects the fused kernel runner to
   the spliced `code_dir` and asserts the whole-graph result is value-correct,
   with a negative control that removes the senprog and must FAIL.
5. **Benchmark** — `bench/` scripts time baseline vs same-core vs round-trip.

## 3. Single-shared-accelerator rule

**There is one shared accelerator. Every device script (`devval/*`, `bench/*`)
must run SOLO and sequentially — never two device runs in parallel.** Parallel
device runs contend for the accelerator and confound both correctness and
timing. Each script already uses a fresh `TORCHINDUCTOR_CACHE_DIR` per run; the
caller is responsible for serializing the runs.

## 4. Scripts

### `splice/`

| Script | Role |
|---|---|
| `splice_onchip_stcdp.py` | Size-parameterized (`SPLICE_SIZE`) single same-layout `STCDPOpLx` (degenerate same-core copy; no transpose). Current version. |
| `splice_onchip_roundtrip.py` | Size-parameterized (`SPLICE_SIZE`) two-`STCDP` round trip — the genuine cross-core ring proof. Current version. |
| `splice_2048_stcdp.py` | Fixed-2048 predecessor of `splice_onchip_stcdp.py` (same-core STCDP). |
| `splice_2048_roundtrip.py` | Fixed-2048 predecessor of `splice_onchip_roundtrip.py` (cross-core round trip). |
| `splice_2048_bmm.py` | Tier-2 transpose bridge (`ReStickifyOpWithPTLx`). **FAULTS on device with the Compute-CB hardware error** — kept because it documents the fault. |

The `splice_onchip_*` pair supersedes the `splice_2048_*` pair (they are the same
methodology, generalized over `SPLICE_SIZE`). An earlier `splice_2048.py`
existed (it assumed the simpler 3-SDSC `[add, ReStickify, add]` layout with an
`add` consumer); it was superseded by `splice_2048_bmm.py` and is not included.

### `devval/`

| Script | Role |
|---|---|
| `devval_roundtrip_fix.py` + `devval_roundtrip_fix_{512,1024,2048}.sh` | Current per-size FIXED round-trip device validation (positive + negative control). |
| `devval_roundtrip.py` + `.sh` | 2048 cross-core round-trip validation. |
| `devval_direct.py` + `.sh` | 2048 Tier-2 transpose proof (loads the `splice_2048_bmm` bundle, which faults). |

Older `devval_stcdp.py` and `devval_correct.sh` existed but were superseded by the
above and are not included.

### `bench/`

| Script | Role |
|---|---|
| `bench_onchip.py` | One-config-per-process latency benchmark; `SPLICED_DIR` selects baseline vs a spliced bundle. |
| `bench_onchip_driver.sh` | 2048 baseline vs same-core vs round-trip, two reps. |
| `bench_onchip_multisize.sh` | Same across sizes 512/1024/2048/4096. |
| `gen_baseline.py` | Compile + report the baseline `code_dir` and edge sharding per size. |
| `results/bench_onchip_results.txt`, `results/bench_onchip_multisize.txt` | Recorded results (verbatim data). |

### `analysis/`

| Script | Role |
|---|---|
| `edge_analyze.py` | Offline producer->consumer HBM-handoff classifier (no device/compile/dxp). |
| `gen_report.py` | Runs the classifier over three real compiled fused kernels and writes the report. |
| `real_edge_analysis.md` | Recorded report (verbatim). |

### `val-boot/`

`sitecustomize.py` is a process-local import shim: put `val-boot` on
`PYTHONPATH` and it drops `torch_spyre*` from the editable-install finder and
prepends `$ONCHIP_SRC`, so the process imports `torch_spyre` from the on-chip
worktree without changing the global venv. Active only for processes that have
this dir on `PYTHONPATH`.

## 5. Gotchas (cross-ref recipe §7)

The device-validation methodology is load-bearing — see
`CoreToCoreDataMovementRecipe.md` §7 for the full treatment. Three things matter:

- **Cache-bust.** The per-process `g_artifact_cache` will silently reuse a
  program it has already seen. Every run uses a **fresh `TORCHINDUCTOR_CACHE_DIR`**
  (the `.sh` scripts set a unique one and `rm -rf` it first).
- **Fresh code_dir redirect.** The redirect points the runner at a `code_dir`
  path the process has never seen, forcing a real disk load of the spliced
  senprog (the trick in `devval_*.py` / `bench_onchip.py`).
- **Negative control.** Each `devval` script moves the spliced senprog aside and
  re-runs; the run **must FAIL**. This proves the positive run actually loaded
  from the spliced dir rather than a cached program.

## 6. Status

Research / reproduction harnesses, not production code. They are preserved so the
on-chip core-to-core proof (same-core STCDP and cross-core round trip clean on
device; Tier-2 transpose faults) and the real-model handoff classification can be
re-run and audited.
