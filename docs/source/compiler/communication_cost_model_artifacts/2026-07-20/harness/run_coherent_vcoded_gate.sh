#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/adnan/codex-isolated/joint_oracle_factorial_20260720_v1
PY=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
PROBE=$ROOT/scripts/coherent_vcoded_structural_probe.py
OUT=${1:?usage: run_coherent_vcoded_gate.sh output_dir}
ORDER=work_div_inner_first
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 98; }
mkdir -p "$OUT"

source /home/adnan/spyre-envs/image-913f394b4b3f/activate.sh >/dev/null 2>&1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$ROOT/scripts:$ROOT/torch-spyre:$ROOT/spyre-perf-suite"
export PATH="$ROOT/dxp-wrapper:$PATH"
export DXP_CLOSURE_ROOT="$ROOT/dxp-closure"
export DEEPTOOLS_PATH="$ROOT/deeptools"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:/home/adnan/dt-inductor/sentient/deeptools/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/lib64/python3.12/site-packages/torch/lib"
export DXP_LX_FRAC_AVAIL=0.2 DXP_BACKEND_LX_FRAC_AVAIL=0.2
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=info
export JOINT_KEY_PRODUCER_PHYSICAL_CORE_ORDER=$ORDER
export FULL_ATTN_PHYSICAL_CORE_ORDER=$ORDER
export FULL_ATTN_QUERY_PRODUCER_PHYSICAL_CORE_ORDER=$ORDER
export FULL_ATTN_RELAYOUT_SOURCE_PHYSICAL_CORE_ORDER=$ORDER
export JOINT_SCORE_PATH_PHYSICAL_CORE_ORDER=$ORDER
export JOINT_SECOND_BMM_PHYSICAL_CORE_ORDER=$ORDER
unset SPYRE_TEST_PRESEEDED_LX_RELAYOUT SPYRE_TEST_LX_RELAYOUT_PRESEED_ONLY
unset DXP_ORACLE_SKIP_LX_RELAYOUT DXP_SPLIT_BUNDLE_AFTER_NODE DXP_DUMP_BUNDLE_NODES

{
  echo "captured_at=$(date -Is)"
  echo "nodename=$(uname -n)"
  echo "placement=joint_coherent"
  echo "value_pattern=head-token-channel-coded-V"
  echo "torch_head=$(git -C "$ROOT/torch-spyre" rev-parse HEAD)"
  echo "torch_tree=$(git -C "$ROOT/torch-spyre" rev-parse HEAD^{tree})"
  echo "probe_sha256=$(sha256sum "$PROBE" | cut -d' ' -f1)"
  env | sort
} > "$OUT/environment.txt"

set +e
FULL_PREFIX_RUN_DIR="$OUT" FULL_PREFIX_COMPILE_ORDER=normal_prefix_oracle \
  FULL_PREFIX_SEED=17 timeout --kill-after=30s 1800s "$PY" "$PROBE" \
  > "$OUT/console.log" 2>&1
rc=$?
set -e
echo "$rc" > "$OUT/exit_code.txt"
if [[ -f "$OUT/report.json" ]]; then
  "$PY" - "$OUT/report.json" "$rc" <<'PY'
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
status = {
    "probe_exit_code": int(sys.argv[2]),
    "all_gates": report.get("all_gates"),
    "normal_correct": report.get("gates", {}).get("normal_correct"),
    "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
}
(path.parent / "VCODED_SUCCESS.json").write_text(json.dumps(status, indent=2) + "\n")
PY
fi
exit "$rc"

