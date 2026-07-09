# Canonical three-pod FMS benchmark environment

Date: 2026-07-09

This environment removes ambient component selection from PR2 FMS microbenchmark
runs. The same stack was installed and device-validated on:

- `adnan-spyre-dev-pf`
- `adnan-clc-spyre-dev-pf`
- `adnan-cdx-spyre-dev-pf`

DEV and CLC share `/home/adnan`. CDX uses `/home/adnan-cdx`; its container
reports `HOME=/`, so the activation script derives paths from its own location.

## Why the old environments disagreed

The pods selected different components from ambient shell state:

1. DEV/CLC placed stale `sentient` paths before `/opt` in library search paths.
2. The perf-suite parent process and its literal `python` child could use
   different interpreters.
3. The relayout DXP was a source build while `PATH` could select another DXP.
4. Frontend and backend LX fraction settings were hidden in a DXP wrapper.
5. `DEEPTOOLS_PATH` selected different DDL template trees. CDX's `/opt` tree did
   not contain `restickify_lx.ddl`.
6. The first cross-pod DXP copy had a truncated ELF despite a successful tar
   exit. The verifier now checks binary hashes and the DDL template.

## Canonical components

| Component | Identity |
|---|---|
| Python | `<pod>/dt-inductor/.venv/bin/python` |
| PyTorch | `2.11.0+cpu` |
| Torch-spyre base | `6f9b17c7b654533629a6dd9a00516de5ded16401` |
| Torch-spyre `_C.so` SHA256 | `b449a232ec1c07046eb64153d9672447242734005a3f822678f665aabe835c99` |
| perf-suite base | `5640b6859d09273cc814348489f68778dc88d108` |
| DXP SHA256 | `68269b28b10851f7a3e2ba8ad1a98b931128265ef40814345461cdf01a73721a` |
| FMS | `b4f36b5af526b938db506a17dcd32d468a7a91d8` |

The DXP installation contains its executable, dependent libraries, and runtime
`share/` tree. Its executable uses relative RPATHs. `DEEPTOOLS_PATH` points to
the matching installed `share/` tree rather than an ambient `/opt` or sentient
tree.

## Activation and verification

DEV or CLC:

```bash
source /home/adnan/spyre-envs/pr2/activate.sh
python /home/adnan/spyre-envs/pr2/verify_environment.py
```

CDX:

```bash
source /home/adnan-cdx/spyre-envs/pr2/activate.sh
python /home/adnan-cdx/spyre-envs/pr2/verify_environment.py
```

The verifier rejects the wrong interpreter, Torch-spyre source, DXP path, DXP
binary, Torch-spyre extension, or a missing `restickify_lx.ddl`.

## FMS SDPA command

After activation, run:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"

cd "$SPYRE_PERF_SUITE_ROOT"
python run_benchmark.py \
  --op fms_granite_micro.mha_4h_workdiv_h4_lq8 \
  --shape 1 4 512 128 \
  --shape 1 4 4096 128 \
  --shape 1 4 4096 128 \
  --stacks torch-spyre \
  --runs 5 \
  --perf-dir "$PWD/perf" \
  --report "$PWD/report.txt" \
  --spyre_kernel_report "$PWD/spyre_kernel_report.txt" \
  --cpu_kernel_report "$PWD/cpu_kernel_report.txt"
```

`SPYRE_LX_PLANNER_RELAYOUT=1` is the only public feature flag. Torch uses full
frontend LX and passes the backend LX fraction and Deeptools materialization
gate directly to the DXP subprocess. No shell DXP wrapper or perf-suite
environment profile is used.

## Device validation

Shape: Q `[1,4,512,128]`, K/V `[1,4,4096,128]`.

| Pod | Kernel ms | Spyre ms | Transfer ms | Runtime ms |
|---|---:|---:|---:|---:|
| DEV | 0.672 | 1.157 | 0.485 | 1.704 |
| CLC | 0.672 | 1.063 | 0.392 | 1.464 |
| CDX | 0.672 | 1.128 | 0.457 | 1.563 |

Run roots:

- DEV: `/home/adnan/spyre-benchmark-runs/fms-sdpa-canonical-dev-20260709_192525`
- CLC: `/home/adnan/spyre-benchmark-runs/fms-sdpa-canonical-clc-20260709`
- CDX: `/home/adnan-cdx/spyre-benchmark-runs/fms-sdpa-canonical-cdx-20260709-r3`

The report's wall-clock metric includes compilation and is not a steady-state
kernel metric. The matching `0.672 ms` kernel result across all three devices is
the environment equivalence check.

### Matched CLC control

The same CLC environment and shape were run with only the public feature flag
changed:

| Variant | `SPYRE_LX_PLANNER_RELAYOUT` | Kernel ms | Spyre ms | Transfer ms | Runtime ms | Kernel speedup |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0 | 0.826 | 1.368 | 0.542 | 1.932 | 1.000x |
| All-gather + stick relayout | 1 | 0.672 | 1.063 | 0.392 | 1.464 | 1.229x |
| Relayout + experimental aggressive LX eligibility | 1 | 0.301 | 0.752 | 0.451 | 1.149 | 2.744x |

Baseline run root:

- `/home/adnan/spyre-benchmark-runs/fms-sdpa-canonical-clc-baseline-20260709`

Aggressive run root:

- `/home/adnan-cdx/spyre-benchmark-runs/fms-sdpa-aggressive-lx-cdx-20260709`

The aggressive variant changes only
`allow_all_ops_in_lx_planning: False -> True` in an isolated Torch tree. CDX
had already reproduced the narrower relayout result at `0.672 ms`, so its
`0.301 ms` result is comparable to the CLC pair. This setting is useful for
research but is not proposed as the PR2 production default: globally allowing
all intermediate op outputs into LX bypasses the normal safety whitelist.

The isolated research environment is installed on all three pods:

```bash
# DEV / CLC
source /home/adnan/spyre-envs/pr2-aggressive/activate.sh

# CDX
source /home/adnan-cdx/spyre-envs/pr2-aggressive/activate.sh

export SPYRE_LX_PLANNER_RELAYOUT=1
```

### SDSC evidence

Jamie-style first-iteration summaries, raw summarizer logs, and the
communication classification table are under `canonical_environment/sdsc/`:

- `baseline_first_iteration_summarize_sdsc.md`
- `baseline_first_iteration_summarize_sdsc.log`
- `relayout_first_iteration_summarize_sdsc.md`
- `relayout_first_iteration_summarize_sdsc.log`
- `aggressive_first_iteration_summarize_sdsc.md`
- `aggressive_first_iteration_summarize_sdsc.log`
- `edge_classification.md`

## Perf-suite process fix

The canonical perf-suite worktree changes three files:

- `run_benchmark.py`
- `core/suite_driver.py`
- `env_check.py`

Benchmark and environment-check child processes use `sys.executable`. The
environment checker runs in a subprocess and exits without leaving Spyre
runtime threads alive. This prevents parent and child processes from silently
using different Python installations.
