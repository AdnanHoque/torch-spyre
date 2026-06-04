#!/bin/bash
# Serial device probe runner for the MLP study. Waits for the card to be free,
# then runs ONE variant on ONE stack. Env mirrors run_matmul_ab.sh exactly.
#   usage: run_probe.sh <tsp|sendnn> <variant> <M>
set -uo pipefail

STACK="$1"; VARIANT="$2"; M="$3"

TSP_PY=/home/adnan/dt-inductor/.venv/bin/python
SENDNN_PY=/tmp/sendnn210-venv/bin/python

LD_TSP=/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:/home/adnan/dt-inductor/sentient/libaiupti/lib:/home/adnan/dt-inductor/sentient/runtime/lib:/home/adnan/dt-inductor/sentient/deeptools/lib:/opt/ibm/spyre/tvm/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/sentinyexec/lib
LD_SENDNN=/home/adnan/dt-inductor/sentient/libaiupti/lib:/home/adnan/dt-inductor/sentient/runtime/lib:/home/adnan/dt-inductor/sentient/deeptools/lib:/opt/ibm/spyre/tvm/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/sentinyexec/lib

wait_device_free() {
  local waited=0
  while true; do
    local busy
    busy=$(ps -ef | grep -E "gen_sdpa_bundle|dxp_standalone|force_split|hbm_bw|mlp_probe|benchmark.py" | grep -v grep | grep -v "run_probe.sh")
    [[ -z "$busy" ]] && { [[ $waited -gt 0 ]] && echo "[$(date +%T)] device free after ${waited}s"; return 0; }
    echo "[$(date +%T)] device BUSY, waiting..."; sleep 8; waited=$((waited+8))
  done
}

wait_device_free

if [[ "$STACK" == "tsp" ]]; then
  echo "[$(date +%T)] === TSP variant=$VARIANT M=$M ==="
  env PYTHONPATH=/tmp/cost_model_unified_shim \
      DXP_LX_FRAC_AVAIL=1 \
      SENCORES=32 \
      USE_SPYRE_PROFILER=1 \
      "LD_LIBRARY_PATH=$LD_TSP" \
      "$TSP_PY" /tmp/mlp_probe_tsp.py "$VARIANT" "$M"
  exit $?
elif [[ "$STACK" == "sendnn" ]]; then
  echo "[$(date +%T)] === SENDNN variant=$VARIANT M=$M ==="
  env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
      DXP_LX_FRAC_AVAIL=1 \
      SENCORES=32 \
      "LD_LIBRARY_PATH=$LD_SENDNN" \
      "$SENDNN_PY" /tmp/mlp_probe_sendnn.py "$VARIANT" "$M"
  exit $?
else
  echo "unknown stack $STACK"; exit 2
fi
