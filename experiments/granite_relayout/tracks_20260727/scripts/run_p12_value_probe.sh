#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/adnan/codex-isolated/device_parity_tracks_20260726/p12
RUN_NAME="${1:?run name required}"
PLANNER="${2:?planner flag required}"
RUN="$ROOT/runs/$RUN_NAME"
PYTHON=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
PROBE="$ROOT/torch-spyre/experiments/granite_relayout/tools/p12_residual_add_value_probe.py"

test "$PLANNER" = 0 -o "$PLANNER" = 1
test ! -e "$RUN"
mkdir -p "$RUN/cache" "$RUN/export"
cd "$RUN"

source /home/adnan/spyre-envs/main-e3a79c56-hints-pr2/activate.sh
export PATH="$ROOT/deeptools-build/dxp:$PATH"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/torch-spyre"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache"
export DTCOMPILER_EXPORT_DIR="$RUN/export"
export DEEPRT_EXPORT_DIR="$RUN/export"
export DUMP_SPYRE_CODE=1
export DXP_LX_FRAC_AVAIL=0.2
export SPYRE_LX_PLANNER_RELAYOUT="$PLANNER"
export SPYRE_LX_RELAYOUT_COLLECTIVES=all_to_all
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD_SOURCE=buf0
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD_CONSUMER=buf1
export SPYRE_LX_RELAYOUT_DUMP_PLANS="$RUN/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="$RUN/allocations.jsonl"
export STCDP_DUMP_TRANSFERS=1
unset TORCH_SPYRE_DOWNCAST_WARN

"$PYTHON" "$PROBE" \
  --out "$RUN/result" --label "$RUN_NAME" 2>&1 | tee "$RUN/run.log"
