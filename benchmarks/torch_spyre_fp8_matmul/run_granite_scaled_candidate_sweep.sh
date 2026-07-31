#!/usr/bin/env bash

set -uo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/tmp/torch_spyre_fp8_scaled_mm_3a63de5}
RESULT_ROOT=${RESULT_ROOT:?RESULT_ROOT must name an isolated output directory}
PROJECTION=${PROJECTION:?PROJECTION is required}
K=${K:?K is required}
N=${N:?N is required}
WARMUPS=${WARMUPS:-3}
REPS=${REPS:-10}

BENCHMARK=${SOURCE_ROOT}/benchmarks/torch_spyre_fp8_matmul/bench_qo_fp8_poc.py
M_VALUES=(1 2 4 8 16 32 64 128 256 512 1024 2048)

mkdir -p "${RESULT_ROOT}"

run_case() {
    local m=$1
    local label=$2
    shift 2
    local case_root=${RESULT_ROOT}/${PROJECTION}/m${m}/${label}

    mkdir -p "${case_root}/cache" "${case_root}/output"
    if DXP_LX_FRAC_AVAIL=0.2 \
        TORCHINDUCTOR_CACHE_DIR="${case_root}/cache" \
        python "${BENCHMARK}" \
            --m "${m}" \
            --k "${K}" \
            --n "${N}" \
            --warmups "${WARMUPS}" \
            --reps "${REPS}" \
            --output-dir "${case_root}/output" \
            "$@" \
            >"${case_root}/stdout.log" \
            2>"${case_root}/stderr.log"; then
        echo 0 >"${case_root}/exit_code.txt"
    else
        local status=$?
        echo "${status}" >"${case_root}/exit_code.txt"
    fi
}

for m in "${M_VALUES[@]}"; do
    run_case "${m}" fp16 --variant fp16
    run_case "${m}" fp8_scaled --variant fp8_baseline --prepack-weight

    if [[ "${m}" == 512 || "${m}" == 1024 || "${m}" == 2048 ]]; then
        run_case "${m}" fp8_raw_dynamic \
            --variant fp8_raw_baseline --prepack-weight
        run_case "${m}" fp8_raw_prepacked \
            --variant fp8_raw_baseline --prepack-weight --prepack-activation
    fi
done
