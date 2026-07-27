#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/adnan/codex-isolated/device_parity_pr2939_20260725
RUN_NAME="${1:?run name required}"
PLANNER="${2:?planner flag required}"
N="${3:-256}"
PATTERN="${4:-hash}"
TRANSPOSE_PRODUCER="${5:-0}"
RUN="$ROOT/runs/$RUN_NAME"
PYTHON=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python

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
export SPYRE_LX_RELAYOUT_COLLECTIVES=all_gather
export SPYRE_LX_RELAYOUT_DUMP_PLANS="$RUN/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="$RUN/allocations.jsonl"
unset TORCH_SPYRE_DOWNCAST_WARN

EXTRA=()
if [[ "$TRANSPOSE_PRODUCER" = 1 ]]; then
  EXTRA+=(--transpose-producer)
fi

"$PYTHON" "$ROOT/p03_value_probe.py" --output "$RUN/output.pt" --n "$N" \
  --pattern "$PATTERN" "${EXTRA[@]}" \
  2>&1 | tee "$RUN/run.log"
