#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(shasum -a 256 "${path}" | cut -d " " -f 1)"
  test "${actual}" = "${expected}" || {
    printf "sha256 mismatch: %s\n  expected %s\n  actual   %s\n" \
      "${path}" "${expected}" "${actual}" >&2
    return 1
  }
}

check_gzip_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(gzip -dc "${path}" | shasum -a 256 | cut -d " " -f 1)"
  test "${actual}" = "${expected}" || {
    printf "decompressed sha256 mismatch: %s\n" "${path}" >&2
    return 1
  }
}

check_sha256 \
  implementation/antoni_inference_profile.py \
  12191848dcb0c39f2b44e92c21d9a4dc41ae7b37f7dc021c53baa94742cbb366
check_sha256 \
  implementation/torch_spyre_overlay/setup.py \
  f5eb8acb5ec3a9163d65068bf3946e34d88ee4faca97312b9e061b0e2286a670
check_sha256 \
  implementation/torch_spyre_overlay/execution/kernel_runner.py \
  1a5dc76ed73b75d649d7b6f035b584d5ffb02daff6129a7f4865c5941ed3e6ed
check_sha256 \
  results/2026-07-24/traces/antoni_july2_full_40_layer_2x4.pt.trace.json.gz \
  61daec54cb333b8ddf54d977285f96bd44f4af9129cb9d8cf0b06019fed79f1f
check_sha256 \
  results/2026-07-24/traces/antoni_equivalent_one_layer_20x4.pt.trace.json.gz \
  aba16de1e8eca9b017dc4ad0db93c693b8c719851c53d6f6234e78cd0c76f1fe
check_sha256 \
  results/2026-07-24/traces/repository_validation_one_layer_20x4.pt.trace.json.gz \
  29da9104120fe507a5021e7c2efaf7e404282c0546217687a3265384a5369279
check_gzip_sha256 \
  results/2026-07-24/traces/antoni_july2_full_40_layer_2x4.pt.trace.json.gz \
  cf92955dbab5b2f8e8c0d5ae1286273e44ff0770306586cf953d242f6d47bd14
check_gzip_sha256 \
  results/2026-07-24/traces/antoni_equivalent_one_layer_20x4.pt.trace.json.gz \
  a7edc56cd5411bd776f57cff12d8843bdd99e09ee5cae3b2182c010872692a9d
check_gzip_sha256 \
  results/2026-07-24/traces/repository_validation_one_layer_20x4.pt.trace.json.gz \
  b744c125fccc3063affb7a3a377ca1c4f6e378f7bb34a2f7f71b1603af9c3de3
check_sha256 \
  results/2026-07-25/sendnn/traces/sendnn_one_layer_b1_s512_20x4.pt.trace.json.gz \
  d214c33d18a5e2da7367de9e3d67c799502260450998303e78b294589a30483f
check_gzip_sha256 \
  results/2026-07-25/sendnn/traces/sendnn_one_layer_b1_s512_20x4.pt.trace.json.gz \
  03b44241495422b0a0cd5af014f378d918e1791b5f210de0bba6a34a6e784b6f
check_sha256 \
  results/2026-07-25/sendnn/exports/sendnn_compiler_export.tar.gz \
  838f4bdc76345d7b43c91062c3b9507e9fb5c49cf980aa44b8dce79615bfb3cf
check_sha256 \
  results/2026-07-25/sendnn/logs/run.log.gz \
  0b169a59a02579adbb6817e7c35da18e58cbb993ed9b15f8c7469f21c92c7888
check_sha256 \
  results/2026-07-25/sendnn/logs/old_stack_compiler.log.gz \
  bd57431c555153b13b7f61287cd218deed383bb51c5454c4d9bf9cbea5913921

for script in scripts/*.sh; do
  bash -n "${script}"
done

python3 - <<'PY'
from pathlib import Path

for name in (
    "implementation/antoni_inference_profile.py",
    "implementation/torch_spyre_overlay/setup.py",
    "implementation/torch_spyre_overlay/execution/kernel_runner.py",
    "tools/analyze_granite_trace.py",
    "tools/analyze_sendnn_trace.py",
    "tools/analyze_sendnn_sdsc_lx.py",
):
    path = Path(name)
    compile(path.read_text(), str(path), "exec")
PY

metrics_tmp="$(mktemp)"
sdsc_tmp="$(mktemp -d)"
trap 'rm -f "${metrics_tmp}"; rm -rf "${sdsc_tmp}"' EXIT
python3 tools/analyze_granite_trace.py \
  --reference results/2026-07-24/traces/antoni_july2_full_40_layer_2x4.pt.trace.json.gz \
  --reproduction results/2026-07-24/traces/antoni_equivalent_one_layer_20x4.pt.trace.json.gz \
  --output "${metrics_tmp}"
diff -u results/2026-07-24/metrics.json "${metrics_tmp}"

python3 tools/analyze_granite_trace.py \
  --reference results/2026-07-24/traces/antoni_july2_full_40_layer_2x4.pt.trace.json.gz \
  --reproduction results/2026-07-24/traces/repository_validation_one_layer_20x4.pt.trace.json.gz \
  --output "${metrics_tmp}"
diff -u results/2026-07-24/validation_metrics.json "${metrics_tmp}"

python3 tools/analyze_sendnn_trace.py \
  --trace results/2026-07-25/sendnn/traces/sendnn_one_layer_b1_s512_20x4.pt.trace.json.gz \
  --run-log results/2026-07-25/sendnn/logs/run.log.gz \
  --compiler-log results/2026-07-25/sendnn/logs/old_stack_compiler.log.gz \
  --torch-metrics results/2026-07-24/metrics.json \
  --output "${metrics_tmp}"
diff -u results/2026-07-25/sendnn/metrics.json "${metrics_tmp}"

tar -xzf \
  results/2026-07-25/full_model_comparison/sdsc/sendnn_full_model_post_lxopt_sdsc_20260725.tar.gz \
  -C "${sdsc_tmp}" \
  perfdsc_debug \
  logs/granite/old_stack_compiler.log
python3 tools/analyze_sendnn_sdsc_lx.py \
  "${sdsc_tmp}" \
  --json-out "${sdsc_tmp}/attribution.json" \
  --csv-out "${sdsc_tmp}/attribution.csv"
diff -u \
  results/2026-07-25/full_model_comparison/sdsc/sendnn_sdsc_lx_attribution.json \
  "${sdsc_tmp}/attribution.json"
diff -u \
  results/2026-07-25/full_model_comparison/sdsc/sendnn_sdsc_lx_attribution.csv \
  "${sdsc_tmp}/attribution.csv"

shasum -c results/2026-07-25/sendnn/SHA256SUMS
shasum -c results/2026-07-25/full_model_comparison/SHA256SUMS
gzip -t results/2026-07-25/sendnn/traces/sendnn_one_layer_b1_s512_20x4.pt.trace.json.gz
tar -tzf results/2026-07-25/sendnn/exports/sendnn_compiler_export.tar.gz >/dev/null
tar -tzf results/2026-07-25/full_model_comparison/smc/generated_smc_study_inputs.tar.gz >/dev/null
tar -tzf results/2026-07-25/full_model_comparison/sdsc/sendnn_full_model_post_lxopt_sdsc_20260725.tar.gz >/dev/null

printf 'repository verification passed\n'
