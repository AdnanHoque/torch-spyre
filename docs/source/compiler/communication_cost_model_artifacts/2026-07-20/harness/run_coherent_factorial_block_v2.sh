#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/adnan/codex-isolated/joint_oracle_factorial_20260720_v1
PY=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
ACTIVATE=/home/adnan/spyre-envs/image-913f394b4b3f/activate.sh
PROBE=$ROOT/scripts/coherent_full_attention_oracle_timing_probe.py
QC=$ROOT/scripts/full_attention_trace_qc.py
BLOCK=${FACTORIAL_BLOCK:?set FACTORIAL_BLOCK}
ORDER_CSV=${FACTORIAL_ORDER:?set FACTORIAL_ORDER}
OUT=${FACTORIAL_OUT:?set FACTORIAL_OUT}
PLACEMENT_ORDER=work_div_inner_first
SEED=$((30 + BLOCK))

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 98; }
mkdir -p "$OUT"
: > "$OUT/status.tsv"

IFS=, read -r -a labels <<< "$ORDER_CSV"
[[ ${#labels[@]} -eq 4 ]] || { echo "expected four labels" >&2; exit 2; }

run_condition() (
  label=$1
  ordinal=$2
  run="$OUT/${ordinal}_${label}"
  mkdir -p "$run"
  source "$ACTIVATE" >/dev/null 2>&1
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export PYTHONPATH="$ROOT/scripts:$ROOT/torch-spyre:$ROOT/spyre-perf-suite"
  export PATH="$ROOT/dxp-wrapper:$PATH"
  export DXP_CLOSURE_ROOT="$ROOT/dxp-closure"
  export DEEPTOOLS_PATH="$ROOT/deeptools"
  export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:/home/adnan/dt-inductor/sentient/deeptools/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/lib64/python3.12/site-packages/torch/lib"
  export DXP_LX_FRAC_AVAIL=0.2 DXP_BACKEND_LX_FRAC_AVAIL=0.2
  export SPYRE_LX_PLANNER_RELAYOUT=1
  export SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=info
  export FULL_ATTN_RUN_DIR="$run" FULL_ATTN_RUNS=30 FULL_ATTN_SEED="$SEED"
  unset SPYRE_TEST_PRESEEDED_LX_RELAYOUT SPYRE_TEST_LX_RELAYOUT_PRESEED_ONLY
  unset DXP_ORACLE_SKIP_LX_RELAYOUT DXP_SPLIT_BUNDLE_AFTER_NODE DXP_DUMP_BUNDLE_NODES
  unset FULL_ATTN_PHYSICAL_CORE_ORDER
  unset JOINT_KEY_PRODUCER_PHYSICAL_CORE_ORDER
  unset FULL_ATTN_QUERY_PRODUCER_PHYSICAL_CORE_ORDER
  unset FULL_ATTN_RELAYOUT_SOURCE_PHYSICAL_CORE_ORDER
  unset JOINT_SCORE_PATH_PHYSICAL_CORE_ORDER
  unset JOINT_SECOND_BMM_PHYSICAL_CORE_ORDER

  case "$label" in
    lx_default) export FULL_ATTN_MODE=lx; placement=default ;;
    oracle_default) export FULL_ATTN_MODE=oracle; placement=default ;;
    lx_coherent) export FULL_ATTN_MODE=lx; placement=joint_coherent ;;
    oracle_coherent) export FULL_ATTN_MODE=oracle; placement=joint_coherent ;;
    *) echo "unknown label: $label" >&2; exit 2 ;;
  esac
  if [[ "$placement" == joint_coherent ]]; then
    export JOINT_KEY_PRODUCER_PHYSICAL_CORE_ORDER=$PLACEMENT_ORDER
    export FULL_ATTN_PHYSICAL_CORE_ORDER=$PLACEMENT_ORDER
    export FULL_ATTN_QUERY_PRODUCER_PHYSICAL_CORE_ORDER=$PLACEMENT_ORDER
    export FULL_ATTN_RELAYOUT_SOURCE_PHYSICAL_CORE_ORDER=$PLACEMENT_ORDER
    export JOINT_SCORE_PATH_PHYSICAL_CORE_ORDER=$PLACEMENT_ORDER
    export JOINT_SECOND_BMM_PHYSICAL_CORE_ORDER=$PLACEMENT_ORDER
  fi

  {
    echo "captured_at=$(date -Is)"
    echo "nodename=$(uname -n)"
    echo "block=$BLOCK"
    echo "ordinal=$ordinal"
    echo "label=$label"
    echo "seed=$SEED"
    echo "placement=$placement"
    echo "process_order=$ORDER_CSV"
    echo "torch_head=$(git -C "$ROOT/torch-spyre" rev-parse HEAD)"
    echo "torch_tree=$(git -C "$ROOT/torch-spyre" rev-parse HEAD^{tree})"
    echo "perf_head=$(git -C "$ROOT/spyre-perf-suite" rev-parse HEAD)"
    echo "deeptools_head=$(git -C "$ROOT/deeptools" rev-parse HEAD)"
    echo "probe_sha256=$(sha256sum "$PROBE" | cut -d' ' -f1)"
    echo "qc_sha256=$(sha256sum "$QC" | cut -d' ' -f1)"
    env | sort
  } > "$run/environment.txt"

  started=$(date -Is)
  set +e
  timeout --kill-after=30s 1200s "$PY" "$PROBE" > "$run/console.log" 2>&1
  probe_rc=$?
  set -e
  qc_rc=99
  if [[ -f "$run/summary.json" ]]; then
    trace=$("$PY" - "$run/summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["trace"])
PY
)
    set +e
    "$PY" "$QC" "$trace" "$run/summary.json" "$run/trace_qc.json" \
      > "$run/qc_console.json"
    qc_rc=$?
    set -e
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ordinal" "$label" "$probe_rc" "$qc_rc" "$started" "$(date -Is)" \
    >> "$OUT/status.tsv"
  [[ $probe_rc -eq 0 && $qc_rc -eq 0 ]]
)

ordinal=0
for label in "${labels[@]}"; do
  ordinal=$((ordinal + 1))
  run_condition "$label" "$ordinal"
done

"$PY" - "$OUT" "$BLOCK" "$ORDER_CSV" <<'PY'
import hashlib
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
status = {
    "gate": "pass",
    "block": int(sys.argv[2]),
    "order": sys.argv[3].split(","),
    "runs": [],
}
for run in sorted(path for path in out.iterdir() if path.is_dir()):
    summary = run / "summary.json"
    qc = run / "trace_qc.json"
    if not summary.exists() or not qc.exists():
        raise SystemExit(2)
    status["runs"].append({
        "label": run.name.split("_", 1)[1],
        "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        "trace_qc_sha256": hashlib.sha256(qc.read_bytes()).hexdigest(),
    })
(out / "BLOCK_SUCCESS.json").write_text(json.dumps(status, indent=2) + "\n")
PY
