#!/usr/bin/env bash

set -uo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <operation> <runner-label>" >&2
    exit 2
fi

op=$1
runner_label=$2
root=/home/adnan/codex-isolated/flash_pr81_relayout_20260714
run_root="$root/runs/pr81_matrix/$op/$runner_label"
activation=/home/adnan/spyre-envs/image-913f394b4b3f/activate.sh
perf_suite="$root/spyre-perf-suite"
regular_torch="$root/torch-spyre"
split_torch="$root/torch-spyre-split-lx-bench"

mkdir -p "$run_root"

run_one() {
    local variant=$1
    local lq=$2
    local masked=$3
    local run="$run_root/lq${lq}_mask${masked}_${variant}"
    local torch_root=$regular_torch

    rm -rf "$run"
    mkdir -p "$run/perf" "$run/cache"

    if [[ $variant == split ]]; then
        torch_root=$split_torch
    fi

    export TORCH_SPYRE_ROOT=$torch_root
    export SPYRE_PERF_SUITE_ROOT=$perf_suite
    # shellcheck disable=SC1090
    source "$activation" >/dev/null 2>&1

    export DEEPTOOLS_PATH="$root/deeptools-clean"
    export PATH="$root/bin:$PATH"
    export TORCHINDUCTOR_CACHE_DIR="$run/cache"
    unset SPYRE_INDUCTOR_LOG SPYRE_INDUCTOR_LOG_LEVEL

    case "$variant" in
        off0p2)
            export SPYRE_LX_PLANNER_RELAYOUT=0
            export DXP_LX_FRAC_AVAIL=0.2
            unset DXP_BACKEND_LX_FRAC_AVAIL
            ;;
        off0p6)
            export SPYRE_LX_PLANNER_RELAYOUT=0
            export DXP_LX_FRAC_AVAIL=0.6
            unset DXP_BACKEND_LX_FRAC_AVAIL
            ;;
        split)
            export SPYRE_LX_PLANNER_RELAYOUT=1
            export DXP_LX_FRAC_AVAIL=0
            export DXP_BACKEND_LX_FRAC_AVAIL=0.6
            ;;
        *)
            echo "unknown variant: $variant" >&2
            return 2
            ;;
    esac

    {
        echo "operation=$op"
        echo "runner=$runner_label"
        echo "lq=$lq"
        echo "masked=$masked"
        echo "variant=$variant"
        echo "torch_sha=$(git -C "$torch_root" rev-parse HEAD)"
        echo "deeptools_sha=$(git -C "$root/deeptools-clean" rev-parse HEAD)"
        echo "perf_suite_sha=$(git -C "$perf_suite" rev-parse HEAD)"
        env | sort
    } > "$run/env.txt"

    local -a shape_args=(
        --shape 1 4 "$lq" 128
        --shape 1 4 4096 128
        --shape 1 4 4096 128
    )
    if [[ $masked == 1 ]]; then
        shape_args+=(--shape 1 1 "$lq" 4096)
    fi

    echo "[$runner_label] $op lq=$lq mask=$masked variant=$variant"
    cd "$perf_suite"
    python run_benchmark.py \
        --op "experimental.$op" \
        "${shape_args[@]}" \
        --stacks torch-spyre \
        --runs 3 \
        --perf-dir "$run/perf" \
        --report "$run/report.txt" \
        --spyre_kernel_report "$run/spyre_kernel_report.txt" \
        --cpu_kernel_report "$run/cpu_kernel_report.txt" \
        > "$run/raw.log" 2>&1
    local rc=$?
    echo "$rc" > "$run/exit_code.txt"
    if [[ $rc -ne 0 ]]; then
        tail -n 80 "$run/raw.log" >&2
    else
        grep -E "wall_clock_ms.mean_ms|kernel_ms.mean_ms|memory_transfer_ms.mean_ms" \
            "$run/report.txt" || true
    fi
    return "$rc"
}

read -r -a lq_values <<< "${LQ_VALUES:-512 1024}"
read -r -a mask_values <<< "${MASK_VALUES:-0 1}"
read -r -a variants <<< "${VARIANTS:-off0p2 off0p6 split}"

failures=0
for lq in "${lq_values[@]}"; do
    for masked in "${mask_values[@]}"; do
        for variant in "${variants[@]}"; do
            run_one "$variant" "$lq" "$masked" || failures=$((failures + 1))
        done
    done
done

echo "[$runner_label] completed $op with $failures failure(s)"
exit "$failures"
