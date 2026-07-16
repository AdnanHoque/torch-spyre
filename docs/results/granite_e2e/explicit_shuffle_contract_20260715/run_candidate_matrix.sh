#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
usage: run_candidate_matrix.sh [options]

Required:
  --mode hbm|custom|explicit
  --torch-root PATH
  --deeptools-root PATH
  --perf-root PATH
  --output-root PATH
  --dxp-bin PATH       Exact dxp_standalone under test

Optional:
  --image-env PATH     Default: /home/adnan/spyre-envs/image-913f394b4b3f/activate.sh
  --runs N             Default: 6
  --lq N               Select one Lq; repeatable. Default: 512 and 1024
  --mask 0|1           Select one mask state; repeatable. Default: 0 and 1
  --execute            Execute; otherwise print the resolved contract only

The runner intentionally uses one LX partition setting for Torch and Deeptools.
It rejects DXP_BACKEND_LX_FRAC_AVAIL so a split/overcommitted LX experiment
cannot be mistaken for a production-safe result.
EOF
}

mode=
torch_root=
deeptools_root=
perf_root=
output_root=
dxp_bin=
image_env=/home/adnan/spyre-envs/image-913f394b4b3f/activate.sh
runs=6
execute=0
lq_values=()
mask_values=()

while (($#)); do
    case "$1" in
        --mode) mode=$2; shift 2 ;;
        --torch-root) torch_root=$2; shift 2 ;;
        --deeptools-root) deeptools_root=$2; shift 2 ;;
        --perf-root) perf_root=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --dxp-bin) dxp_bin=$2; shift 2 ;;
        --image-env) image_env=$2; shift 2 ;;
        --runs) runs=$2; shift 2 ;;
        --lq) lq_values+=("$2"); shift 2 ;;
        --mask) mask_values+=("$2"); shift 2 ;;
        --execute) execute=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

((${#lq_values[@]})) || lq_values=(512 1024)
((${#mask_values[@]})) || mask_values=(0 1)
for lq in "${lq_values[@]}"; do
    [[ $lq == 512 || $lq == 1024 ]] || {
        echo "--lq must be 512 or 1024" >&2
        exit 2
    }
done
for mask in "${mask_values[@]}"; do
    [[ $mask == 0 || $mask == 1 ]] || {
        echo "--mask must be 0 or 1" >&2
        exit 2
    }
done

case "$mode" in
    hbm|custom|explicit) ;;
    *) echo "--mode must be hbm, custom, or explicit" >&2; exit 2 ;;
esac
for value in torch_root deeptools_root perf_root output_root dxp_bin; do
    if [[ -z ${!value} ]]; then echo "missing --${value//_/-}" >&2; exit 2; fi
done
for path in "$torch_root" "$deeptools_root" "$perf_root" "$image_env"; do
    if [[ ! -e $path ]]; then echo "missing path: $path" >&2; exit 2; fi
done
if [[ -n ${DXP_BACKEND_LX_FRAC_AVAIL:-} ]]; then
    echo "DXP_BACKEND_LX_FRAC_AVAIL must be unset for this production-contract run" >&2
    exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_root="$output_root/${mode}_$stamp"
private_bin="$run_root/bin"
mkdir -p "$private_bin"
[[ -x $dxp_bin ]] || { echo "not executable: $dxp_bin" >&2; exit 2; }
ln -s "$dxp_bin" "$private_bin/dxp_standalone"

cat >"$run_root/contract.txt" <<EOF
mode=$mode
torch_root=$torch_root
deeptools_root=$deeptools_root
perf_root=$perf_root
dxp_bin=$dxp_bin
image_env=$image_env
runs=$runs
LQ_VALUES=${lq_values[*]}
MASK_VALUES=${mask_values[*]}
DXP_LX_FRAC_AVAIL=0.07
DXP_BACKEND_LX_FRAC_AVAIL=UNSET
SPYRE_LX_PLANNER_RELAYOUT=$([[ $mode == hbm ]] && echo UNSET || echo 1)
EOF

if ((execute == 0)); then
    cat "$run_root/contract.txt"
    echo "dry-run output: $run_root"
    exit 0
fi

# shellcheck disable=SC1090
source "$image_env" >/dev/null 2>&1
export TORCH_SPYRE_ROOT="$torch_root"
export SPYRE_PERF_SUITE_ROOT="$perf_root"
export DEEPTOOLS_PATH="$deeptools_root"
export PATH="$private_bin:$PATH"
export PYTHONPATH="$torch_root:$perf_root"
export DXP_LX_FRAC_AVAIL=0.07
unset DXP_BACKEND_LX_FRAC_AVAIL
if [[ $mode == hbm ]]; then
    unset SPYRE_LX_PLANNER_RELAYOUT
else
    export SPYRE_LX_PLANNER_RELAYOUT=1
fi

{
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mode=$mode"
    echo "torch_sha=$(git -C "$torch_root" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "deeptools_sha=$(git -C "$deeptools_root" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "perf_suite_sha=$(git -C "$perf_root" rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum "$dxp_bin"
    env | sort
} >"$run_root/environment.txt"

checker=/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-controls/artifacts/provenance/helpers/check_outputs_pr81_local.py

run_case() {
    local lq=$1
    local mask=$2
    local case_root="$run_root/Lq${lq}_mask${mask}"
    mkdir -p "$case_root/perf" "$case_root/correctness"
    export TORCHINDUCTOR_CACHE_DIR="$case_root/cache"
    local -a shapes=(
        --shape 1 4 "$lq" 128
        --shape 1 4 4096 128
        --shape 1 4 4096 128
    )
    if [[ $mask == 1 ]]; then shapes+=(--shape 1 1 "$lq" 4096); fi

    (cd "$perf_root" && python "$checker" \
        --torch-spyre-only \
        --op experimental.flash_attn_online_softmax \
        "${shapes[@]}" \
        --save-output-dir "$case_root/correctness/outputs") \
        2>&1 | tee "$case_root/correctness/run.log"

    (cd "$perf_root" && python run_benchmark.py \
        --op experimental.flash_attn_online_softmax \
        "${shapes[@]}" \
        --stacks torch-spyre --runs "$runs" --skip-env-check \
        --perf-dir "$case_root/perf" \
        --report "$case_root/report.txt" \
        --spyre_kernel_report "$case_root/spyre_kernel_report.txt" \
        --cpu_kernel_report "$case_root/cpu_kernel_report.txt") \
        2>&1 | tee "$case_root/profile.log"
}

for lq in "${lq_values[@]}"; do
    for mask in "${mask_values[@]}"; do
        run_case "$lq" "$mask"
    done
done

echo "candidate matrix: $run_root"
