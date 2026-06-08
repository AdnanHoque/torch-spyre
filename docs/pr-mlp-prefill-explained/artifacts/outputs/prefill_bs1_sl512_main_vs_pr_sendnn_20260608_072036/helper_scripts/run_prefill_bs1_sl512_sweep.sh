#!/usr/bin/env bash
set -euo pipefail

RUNROOT="${1:-/tmp/prefill-bs1-sl512-ab-$(date -u +%Y%m%d_%H%M%S)}"
ROOT_REPO=/home/adnan-cdx/codex-worktrees/pr-mlp-fix/torch-spyre
SUITE_SRC=/home/adnan-cdx/spyre-perf-suite
MAIN_REF=upstream/main
FIX_REF=origin/pr-mlp-prefill-explained
COMPAT_REF=824ad4b

mkdir -p "$RUNROOT"

echo "RUNROOT=$RUNROOT"
echo "MAIN_REF=$MAIN_REF"
echo "COMPAT_REF=$COMPAT_REF"
echo "FIX_REF=$FIX_REF"

git -C "$ROOT_REPO" fetch upstream main
git -C "$ROOT_REPO" fetch origin pr-mlp-prefill-explained

git -C "$ROOT_REPO" worktree add --detach "$RUNROOT/torch-spyre-main" "$MAIN_REF"
git -C "$ROOT_REPO" worktree add --detach "$RUNROOT/torch-spyre-fix" "$FIX_REF"
git -C "$RUNROOT/torch-spyre-main" cherry-pick --no-commit "$COMPAT_REF"

cp -a "$SUITE_SRC" "$RUNROOT/suite-main"
cp -a "$SUITE_SRC" "$RUNROOT/suite-fix"

install_sitecustomize() {
  local suite=$1
  cat > "$suite/sitecustomize.py" <<'PY'
import torch_spyre

torch_spyre._autoload()
PY
}

install_sitecustomize "$RUNROOT/suite-main"
install_sitecustomize "$RUNROOT/suite-fix"

write_versions() {
  {
    echo "runroot: $RUNROOT"
    echo "lx_planning: ${LX_PLANNING:-unset}"
    echo "main_ref: $MAIN_REF"
    echo "main_commit: $(git -C "$RUNROOT/torch-spyre-main" rev-parse --short HEAD)"
    echo "main_extra_patch: $COMPAT_REF (benchmark compatibility only: patches.py)"
    echo "fix_ref: $FIX_REF"
    echo "fix_commit: $(git -C "$RUNROOT/torch-spyre-fix" rev-parse --short HEAD)"
    echo "suite_commit: $(git -C "$RUNROOT/suite-fix" rev-parse --short HEAD)"
    echo "flex_commit: $(git -C /home/adnan-cdx/dt-inductor-codex-clean/flex rev-parse --short HEAD 2>/dev/null || true)"
    echo "deeptools_commit: $(git -C /home/adnan-cdx/dt-inductor-codex-clean/deeptools rev-parse --short HEAD 2>/dev/null || true)"
  } > "$RUNROOT/provenance.txt"
}

prepare_env() {
  source /home/adnan-cdx/dt-inductor-codex-clean/env.sh
  source /home/adnan-cdx/dt-inductor-codex-clean/matmul_gap_env.sh
  use_py212_localflex_optdeeptools_spyre_runtime >/dev/null
  export LX_PLANNING=0
}

run_stack() {
  local label=$1
  local src=$2
  local suite=$3
  shift 3
  local stacks=("$@")

  mkdir -p "$suite/logs" "$suite/perf" "$suite/reports"
  (
    prepare_env
    export PYTHONPATH="$src:$suite:${PYTHONPATH:-}"
    export TORCHINDUCTOR_CACHE_DIR="$RUNROOT/cache/$label"
    export DTCOMPILER_EXPORT_DIR="$RUNROOT/export/$label"
    export DEEPRT_EXPORT_DIR="$RUNROOT/export/$label"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$DTCOMPILER_EXPORT_DIR" "$DEEPRT_EXPORT_DIR"
    cd "$suite"
    python - <<'PY'
import torch_spyre
print("torch_spyre_import:", torch_spyre.__file__)
PY

    python run_benchmark.py --op matmul \
      --shape 1 512 4096 --shape 4096 1024 \
      --stacks "${stacks[@]}" --runs 3 \
      --report "$suite/reports/${label}_report.txt" \
      --kernel_report "$suite/reports/${label}_kernel_report.txt" \
      > "$suite/logs/${label}_matmul_kv.log" 2>&1

    python run_benchmark.py --op matmul \
      --shape 1 512 4096 --shape 4096 4096 \
      --stacks "${stacks[@]}" --runs 3 \
      --report "$suite/reports/${label}_report.txt" \
      --kernel_report "$suite/reports/${label}_kernel_report.txt" \
      > "$suite/logs/${label}_matmul_qo.log" 2>&1

    python run_benchmark.py --op matmul \
      --shape 1 512 4096 --shape 4096 12800 \
      --stacks "${stacks[@]}" --runs 3 \
      --report "$suite/reports/${label}_report.txt" \
      --kernel_report "$suite/reports/${label}_kernel_report.txt" \
      > "$suite/logs/${label}_matmul_mlp_proj.log" 2>&1

    python run_benchmark.py --op mlp \
      --shape 1 512 4096 \
      --stacks "${stacks[@]}" --runs 3 \
      --report "$suite/reports/${label}_report.txt" \
      --kernel_report "$suite/reports/${label}_kernel_report.txt" \
      > "$suite/logs/${label}_mlp.log" 2>&1
  )
}

summarize() {
  python - "$RUNROOT" <<'PY'
from pathlib import Path
import re
import sys

runroot = Path(sys.argv[1])

cases = [
    ("matmul KV", "matmul", "1_512_4096__4096_1024_"),
    ("matmul QO", "matmul", "1_512_4096__4096_4096_"),
    ("matmul MLP-proj", "matmul", "1_512_4096__4096_12800_"),
    ("mlp", "mlp", "1_512_4096_"),
]

def metric(path, name):
    text = path.read_text()
    match = re.search(rf"^{re.escape(name)}\\s+[-0-9.]+\\s+([-0-9.]+)\\s+", text, re.M)
    if not match:
        raise RuntimeError(f"missing {name} in {path}")
    return float(match.group(1))

def perf_path(label, stack, op, shape):
    suite = runroot / ("suite-main" if label == "main" else "suite-fix")
    return suite / "perf" / f"{op}_{stack}_shape_{shape}.txt"

rows = []
for name, op, shape in cases:
    main_tsp = perf_path("main", "torch-spyre", op, shape)
    fix_tsp = perf_path("fix", "torch-spyre", op, shape)
    fix_sendnn = perf_path("fix", "sendnn", op, shape)
    main_kernel = metric(main_tsp, "kernel_ms")
    fix_kernel = metric(fix_tsp, "kernel_ms")
    sendnn_kernel = metric(fix_sendnn, "kernel_ms")
    main_pt = metric(main_tsp, "pt_util%")
    fix_pt = metric(fix_tsp, "pt_util%")
    rows.append(
        (
            name,
            main_kernel,
            fix_kernel,
            main_kernel / fix_kernel,
            sendnn_kernel,
            fix_kernel / sendnn_kernel,
            main_pt,
            fix_pt,
        )
    )

lines = []
lines.append("# Prefill bs1 sl512 A/B Sweep")
lines.append("")
lines.append((runroot / "provenance.txt").read_text().strip())
lines.append("")
lines.append("| op | main tsp kernel ms | PR tsp kernel ms | PR speedup vs main | sendnn kernel ms | PR tsp/sendnn | main PT% | PR PT% |")
lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for row in rows:
    lines.append(
        f"| {row[0]} | {row[1]:.3f} | {row[2]:.3f} | {row[3]:.3f}x | "
        f"{row[4]:.3f} | {row[5]:.3f}x | {row[6]:.3f} | {row[7]:.3f} |"
    )

out = runroot / "prefill_bs1_sl512_ab_summary.md"
out.write_text("\\n".join(lines) + "\\n")
print(out)
PY
}

write_versions
run_stack main "$RUNROOT/torch-spyre-main" "$RUNROOT/suite-main" torch-spyre
run_stack fix "$RUNROOT/torch-spyre-fix" "$RUNROOT/suite-fix" torch-spyre sendnn
summarize

echo "DONE=$RUNROOT"
