#!/usr/bin/env bash
set -euo pipefail

# Generate the focused torch-spyre SDSC and sendnn DeepRT/DCI artifacts used by
# docs/tsp-sendnn-sdsc/README.md.
#
# Run this from a configured Spyre development environment. The defaults match
# the common pod layout, but every path can be overridden with environment vars:
#
#   TORCH_SPYRE_REPO=/path/to/torch-spyre \
#   SPYRE_PERF_SUITE=/path/to/spyre-perf-suite \
#   RUN_ROOT=/tmp/tsp-sendnn-sdsc-$(date -u +%Y%m%d_%H%M%S) \
#   ./docs/tsp-sendnn-sdsc/generate_tsp_sendnn_sdsc_artifacts.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TORCH_SPYRE_REPO="${TORCH_SPYRE_REPO:-$(git -C "$SCRIPT_DIR/../.." rev-parse --show-toplevel 2>/dev/null || pwd)}"
SPYRE_PERF_SUITE="${SPYRE_PERF_SUITE:-}"
SPYRE_ENV_ROOT="${SPYRE_ENV_ROOT:-/home/adnan-cdx/dt-inductor-codex-clean}"
RUN_ROOT="${RUN_ROOT:-/tmp/tsp-sendnn-sdsc-$(date -u +%Y%m%d_%H%M%S)}"
RUNS="${RUNS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LX_PLANNING="${LX_PLANNING:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

section() {
  printf '\n== %s ==\n' "$*"
}

find_suite() {
  if [[ -n "$SPYRE_PERF_SUITE" ]]; then
    [[ -f "$SPYRE_PERF_SUITE/run_benchmark.py" ]] || die "SPYRE_PERF_SUITE has no run_benchmark.py: $SPYRE_PERF_SUITE"
    return
  fi

  local candidates=(
    "$TORCH_SPYRE_REPO/../spyre-perf-suite"
    "$TORCH_SPYRE_REPO/../spyre-perf-suite-jamie"
    "$SPYRE_ENV_ROOT/spyre-perf-suite"
    "$SPYRE_ENV_ROOT/spyre-perf-suite-jamie"
    "/home/adnan-cdx/spyre-perf-suite"
    "/home/adnan-cdx/dt-inductor-codex-clean/spyre-perf-suite-jamie"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate/run_benchmark.py" ]]; then
      SPYRE_PERF_SUITE="$candidate"
      return
    fi
  done

  die "Could not find spyre-perf-suite. Set SPYRE_PERF_SUITE=/path/to/spyre-perf-suite."
}

maybe_source_spyre_env() {
  if [[ -f "$SPYRE_ENV_ROOT/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SPYRE_ENV_ROOT/env.sh"
  fi
  if [[ -f "$SPYRE_ENV_ROOT/matmul_gap_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SPYRE_ENV_ROOT/matmul_gap_env.sh"
  fi
  if declare -F use_py212_localflex_optdeeptools_spyre_runtime >/dev/null; then
    use_py212_localflex_optdeeptools_spyre_runtime >/dev/null
  fi
}

git_short() {
  local repo="$1"
  if [[ -d "$repo/.git" ]]; then
    git -C "$repo" rev-parse --short HEAD 2>/dev/null || printf 'unknown'
  else
    printf 'not-found'
  fi
}

copy_first_match() {
  local label="$1"
  local root="$2"
  local pattern="$3"
  local dest="$4"
  local match

  match="$(find "$root" -path "$pattern" -type f 2>/dev/null | sort | head -1 || true)"
  if [[ -z "$match" ]]; then
    echo "WARN: did not find $label under $root with pattern $pattern" >&2
    return 1
  fi

  mkdir -p "$(dirname "$dest")"
  cp "$match" "$dest"
  echo "$match -> $dest"
}

copy_all_sdscs() {
  local label="$1"
  local root="$2"
  local dest_dir="$3"
  local found=0

  mkdir -p "$dest_dir"
  while IFS= read -r sdsc; do
    found=1
    local parent
    parent="$(basename "$(dirname "$sdsc")")"
    cp "$sdsc" "$dest_dir/${parent}_$(basename "$sdsc")"
    echo "$sdsc -> $dest_dir/${parent}_$(basename "$sdsc")"
  done < <(find "$root" -name 'sdsc_*.json' -type f 2>/dev/null | sort)

  if [[ "$found" -eq 0 ]]; then
    echo "WARN: did not find any $label SDSCs under $root" >&2
    return 1
  fi
}

run_benchmark_case() {
  local label="$1"
  local op="$2"
  local stack="$3"
  shift 3

  local cache_dir="$RUN_ROOT/cache/$label"
  local export_dir="$RUN_ROOT/export/$label"
  local report="$RUN_ROOT/reports/${label}_report.txt"
  local kernel_report="$RUN_ROOT/reports/${label}_kernel_report.txt"
  local perf_dir="$RUN_ROOT/perf/$label"
  local log="$RUN_ROOT/logs/${label}.log"

  mkdir -p "$cache_dir" "$export_dir" "$perf_dir" "$RUN_ROOT/reports" "$RUN_ROOT/logs"

  section "Running $label"
  echo "op=$op stack=$stack"
  echo "cache:  $cache_dir"
  echo "export: $export_dir"
  echo "log:    $log"

  (
    cd "$SPYRE_PERF_SUITE"
    export PYTHONPATH="$TORCH_SPYRE_REPO:$SPYRE_PERF_SUITE:${PYTHONPATH:-}"
    export PS_TORCH_SPYRE_PATH="$TORCH_SPYRE_REPO"
    export PS_SPYRE_PERF_SUITE_PATH="$SPYRE_PERF_SUITE"
    export TORCHINDUCTOR_CACHE_DIR="$cache_dir"
    export DTCOMPILER_EXPORT_DIR="$export_dir"
    export DEEPRT_EXPORT_DIR="$export_dir"
    export LX_PLANNING
    "$PYTHON_BIN" run_benchmark.py \
      --op "$op" "$@" \
      --stacks "$stack" \
      --runs "$RUNS" \
      --perf-dir "$perf_dir" \
      --report "$report" \
      --kernel_report "$kernel_report"
  ) >"$log" 2>&1
}

write_summary() {
  local summary="$RUN_ROOT/summary.md"

  {
    echo "# TSP/sendnn SDSC Artifact Run"
    echo
    echo "Run root: \`$RUN_ROOT\`"
    echo
    echo "## Versions"
    echo
    echo "| repo | commit | path |"
    echo "| --- | --- | --- |"
    echo "| torch-spyre | $(git_short "$TORCH_SPYRE_REPO") | \`$TORCH_SPYRE_REPO\` |"
    echo "| spyre-perf-suite | $(git_short "$SPYRE_PERF_SUITE") | \`$SPYRE_PERF_SUITE\` |"
    echo "| flex | $(git_short "$SPYRE_ENV_ROOT/flex") | \`$SPYRE_ENV_ROOT/flex\` |"
    echo "| deeptools | $(git_short "$SPYRE_ENV_ROOT/deeptools") | \`$SPYRE_ENV_ROOT/deeptools\` |"
    echo
    echo "LX_PLANNING: \`$LX_PLANNING\`"
    echo "Runs per case: \`$RUNS\`"
    echo
    echo "## Cases"
    echo
    echo "- torch-spyre matmul: \`[[1, 512, 4096], [4096, 12800]]\`"
    echo "- torch-spyre MLP: \`[[1, 512, 4096]]\`"
    echo "- sendnn MLP: \`[[1, 512, 4096]]\`"
    echo
    echo "## Human-readable Artifacts"
    echo
    find "$RUN_ROOT/collected" -type f | sort | sed "s#^$RUN_ROOT#- \`#; s#\$#\`#"
    echo
    echo "## Key torch-spyre SDSC Fields"
    echo
  } >"$summary"

  if command -v jq >/dev/null; then
    {
      echo "Projection SDSC:"
      echo
      echo '```json'
      jq '.[keys[0]] | {
        numWkSlicesPerDim_,
        firstCore: .coreIdToWkSlice_["0"],
        fourthCore: .coreIdToWkSlice_["4"],
        lastCore: .coreIdToWkSlice_["31"],
        N_: .dscs_[0].batchmatmul.N_,
        dataStageParam_: .dscs_[0].batchmatmul.dataStageParam_,
        primaryDsInfo_: .dscs_[0].batchmatmul.primaryDsInfo_
      }' "$RUN_ROOT/collected/tsp/projection_current_sdsc_0.json"
      echo '```'
      echo
      echo "MLP SDSCs:"
      echo
      for sdsc in "$RUN_ROOT"/collected/tsp/mlp/*.json; do
        [[ -f "$sdsc" ]] || continue
        echo "$(basename "$sdsc"):"
        echo
        echo '```json'
        jq '.[keys[0]] | {
          numWkSlicesPerDim_,
          N_: .dscs_[0].batchmatmul.N_?,
          dataStageParam_: .dscs_[0].batchmatmul.dataStageParam_?,
          primaryDsInfo_: .dscs_[0].batchmatmul.primaryDsInfo_?
        }' "$sdsc"
        echo '```'
        echo
      done
      echo "## Key sendnn DCI Fields"
      echo
      for dci in "$RUN_ROOT"/collected/sendnn/*_dci.json; do
        [[ -f "$dci" ]] || continue
        echo "$(basename "$dci"):"
        echo
        echo '```json'
        jq '{dsName_, input_shape_, output_shape_, dcsi_}' "$dci"
        echo '```'
        echo
      done
    } >>"$summary"
  else
    {
      echo
      echo "\`jq\` was not found, so this summary only lists artifact paths."
      echo "Install jq or inspect the JSON files directly."
    } >>"$summary"
  fi

  echo "$summary"
}

find_suite

section "Configuration"
echo "torch-spyre:      $TORCH_SPYRE_REPO"
echo "spyre-perf-suite: $SPYRE_PERF_SUITE"
echo "spyre env root:   $SPYRE_ENV_ROOT"
echo "run root:         $RUN_ROOT"
echo "runs per case:    $RUNS"
echo "LX_PLANNING:      $LX_PLANNING"

mkdir -p "$RUN_ROOT"
maybe_source_spyre_env

if ! command -v "$PYTHON_BIN" >/dev/null; then
  if command -v python3 >/dev/null; then
    PYTHON_BIN=python3
  else
    die "Could not find $PYTHON_BIN or python3. Set PYTHON_BIN=/path/to/python."
  fi
fi

run_benchmark_case tsp_matmul matmul torch-spyre --shape 1 512 4096 --shape 4096 12800
run_benchmark_case tsp_mlp mlp torch-spyre --shape 1 512 4096
run_benchmark_case sendnn_mlp mlp sendnn --shape 1 512 4096

section "Collecting artifacts"
mkdir -p "$RUN_ROOT/collected/tsp/mlp" "$RUN_ROOT/collected/sendnn"

copy_first_match \
  "torch-spyre projection SDSC" \
  "$RUN_ROOT/cache/tsp_matmul" \
  '*/inductor-spyre/*/sdsc_0.json' \
  "$RUN_ROOT/collected/tsp/projection_current_sdsc_0.json" || true

copy_all_sdscs \
  "torch-spyre MLP" \
  "$RUN_ROOT/cache/tsp_mlp" \
  "$RUN_ROOT/collected/tsp/mlp" || true

copy_first_match \
  "sendnn ldsToDciPath" \
  "$RUN_ROOT/export/sendnn_mlp" \
  '*/export_deeprt/ldsToDciPath.json' \
  "$RUN_ROOT/collected/sendnn/ldsToDciPath.json" || true

while IFS= read -r dci; do
  if command -v jq >/dev/null; then
    ds_name="$(jq -r '.dsName_ // "unknown_ds"' "$dci" 2>/dev/null | tr -c 'A-Za-z0-9_.-' '_')"
  else
    ds_name="$(basename "$(dirname "$dci")" | tr -c 'A-Za-z0-9_.-' '_')"
  fi
  cp "$dci" "$RUN_ROOT/collected/sendnn/${ds_name}_dci.json"
  echo "$dci -> $RUN_ROOT/collected/sendnn/${ds_name}_dci.json"
done < <(find "$RUN_ROOT/export/sendnn_mlp" -path '*/HostPrep/dci.json' -type f 2>/dev/null | sort)

section "Writing summary"
summary="$(write_summary)"
echo "Summary: $summary"

section "Done"
echo "Open this first:"
echo "  $summary"
echo
echo "Collected artifacts are under:"
echo "  $RUN_ROOT/collected"
