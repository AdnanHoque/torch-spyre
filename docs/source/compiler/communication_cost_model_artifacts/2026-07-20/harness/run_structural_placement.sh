#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/adnan/codex-isolated/joint_oracle_factorial_20260720_v1
PY=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
ACTIVATE=/home/adnan/spyre-envs/image-913f394b4b3f/activate.sh
PROBE=$ROOT/scripts/joint_full_attention_prefix_structural_probe.py
PLACEMENT=${1:?usage: run_structural_placement.sh default|joint_all output_dir}
OUT=${2:?usage: run_structural_placement.sh default|joint_all output_dir}
ORDER=work_div_inner_first

case "$PLACEMENT" in
  default|joint_all) ;;
  *) echo "invalid placement: $PLACEMENT" >&2; exit 2 ;;
esac
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 98; }
mkdir -p "$OUT"
: > "$OUT/status.tsv"

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
unset SPYRE_TEST_PRESEEDED_LX_RELAYOUT SPYRE_TEST_LX_RELAYOUT_PRESEED_ONLY
unset DXP_ORACLE_SKIP_LX_RELAYOUT DXP_SPLIT_BUNDLE_AFTER_NODE DXP_DUMP_BUNDLE_NODES
unset FULL_ATTN_PHYSICAL_CORE_ORDER
unset FULL_ATTN_QUERY_PRODUCER_PHYSICAL_CORE_ORDER
unset FULL_ATTN_RELAYOUT_SOURCE_PHYSICAL_CORE_ORDER
unset JOINT_SCORE_PATH_PHYSICAL_CORE_ORDER
unset JOINT_SECOND_BMM_PHYSICAL_CORE_ORDER
if [[ "$PLACEMENT" == joint_all ]]; then
  export FULL_ATTN_PHYSICAL_CORE_ORDER=$ORDER
  export FULL_ATTN_QUERY_PRODUCER_PHYSICAL_CORE_ORDER=$ORDER
  export FULL_ATTN_RELAYOUT_SOURCE_PHYSICAL_CORE_ORDER=$ORDER
  export JOINT_SCORE_PATH_PHYSICAL_CORE_ORDER=$ORDER
  export JOINT_SECOND_BMM_PHYSICAL_CORE_ORDER=$ORDER
fi

{
  echo "captured_at=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "placement=$PLACEMENT"
  echo "torch_head=$(git -C "$ROOT/torch-spyre" rev-parse HEAD)"
  echo "torch_tree=$(git -C "$ROOT/torch-spyre" rev-parse HEAD^{tree})"
  echo "perf_head=$(git -C "$ROOT/spyre-perf-suite" rev-parse HEAD)"
  echo "deeptools_head=$(git -C "$ROOT/deeptools" rev-parse HEAD)"
  echo "probe_sha256=$(sha256sum "$PROBE" | cut -d' ' -f1)"
  echo "target_patch_sha256=$(sha256sum "$ROOT/scripts/joint_oracle_target_patch.py" | cut -d' ' -f1)"
  env | sort
} > "$OUT/environment.txt"

for compile_order in normal_prefix_oracle oracle_prefix_normal; do
  run="$OUT/$compile_order"
  mkdir -p "$run"
  started=$(date -Is)
  set +e
  FULL_PREFIX_RUN_DIR="$run" FULL_PREFIX_COMPILE_ORDER="$compile_order" \
    FULL_PREFIX_SEED=17 timeout --kill-after=30s 1800s "$PY" "$PROBE" \
    > "$run/console.log" 2>&1
  rc=$?
  set -e
  printf '%s\t%s\t%s\t%s\n' "$compile_order" "$rc" "$started" "$(date -Is)" \
    >> "$OUT/status.tsv"
  [[ $rc -eq 0 ]] || { tail -100 "$run/console.log" >&2; exit "$rc"; }
done

"$PY" - "$OUT" "$PLACEMENT" <<'PY'
import hashlib
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
placement = sys.argv[2]
reports = []
for order in ("normal_prefix_oracle", "oracle_prefix_normal"):
    path = out / order / "report.json"
    report = json.loads(path.read_text())
    contract = report.get("factorial_placement_contract", {})
    expected = "joint_all" if placement == "joint_all" else "default"
    if not report.get("all_gates") or contract.get("placement") != expected:
        raise SystemExit(2)
    reports.append({
        "compile_order": order,
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "all_gates": report["all_gates"],
        "gate_count": len(report["gates"]),
    })
status = {"gate": "pass", "placement": placement, "reports": reports}
(out / "STRUCTURAL_SUCCESS.json").write_text(json.dumps(status, indent=2) + "\n")
PY

