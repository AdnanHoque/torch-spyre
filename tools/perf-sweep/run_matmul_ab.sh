#!/bin/bash
# Matmul planner OFF-vs-ON A/B over the 6 Priyanka matmul shapes, plus sendnn.
# Same profiler env / same _C.so / same harness for OFF and ON -> the tsp/tsp
# ratio is a clean same-harness improvement factor (no cross-report mixing).
#   OFF = all cost-model planners off  == main heuristic
#   ON  = matmul planner ONLY          == pr-cost-model-matmul behaviour
#   sendnn = torch_sendnn baseline (reference column)
# Serial: one device kernel at a time. Exits early on any failure.
set -uo pipefail
cd /tmp/spyre-perf-suite

PERF=/tmp/spyre-perf-suite/perf.matmul_ab
SPLITLOG=$PERF/splitlogs
mkdir -p "$PERF" "$SPLITLOG"

TSP_PY=/home/adnan/dt-inductor/.venv/bin/python
SENDNN_PY=/tmp/sendnn210-venv/bin/python

LD_TSP=/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:/home/adnan/dt-inductor/sentient/libaiupti/lib:/home/adnan/dt-inductor/sentient/runtime/lib:/home/adnan/dt-inductor/sentient/deeptools/lib:/opt/ibm/spyre/tvm/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/sentinyexec/lib
LD_SENDNN=/home/adnan/dt-inductor/sentient/libaiupti/lib:/home/adnan/dt-inductor/sentient/runtime/lib:/home/adnan/dt-inductor/sentient/deeptools/lib:/opt/ibm/spyre/tvm/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/sentinyexec/lib

TSP_COMMON=(
  PYTHONPATH=/tmp/cost_model_unified_shim
  DXP_LX_FRAC_AVAIL=1
  SENCORES=32
  USE_SPYRE_PROFILER=1
  SPYRE_INDUCTOR_LOG=1
  SPYRE_INDUCTOR_LOG_LEVEL=DEBUG
  "LD_LIBRARY_PATH=$LD_TSP"
)
SENDNN_ENV=(
  TORCH_DEVICE_BACKEND_AUTOLOAD=0
  DXP_LX_FRAC_AVAIL=1
  SENCORES=32
  "LD_LIBRARY_PATH=$LD_SENDNN"
)

wait_device_free() {
  local waited=0
  while true; do
    local busy
    busy=$(ps -ef | grep -E "gen_sdpa_bundle|dxp_standalone|force_split" | grep -v grep | grep -v pylance)
    [[ -z "$busy" ]] && { [[ $waited -gt 0 ]] && echo "[$(date +%T)] device free after ${waited}s"; return 0; }
    echo "[$(date +%T)] device BUSY (foreign job), waiting..."; sleep 10; waited=$((waited+10))
  done
}

run_tsp() {  # mode(off|on) op outname shape_args...
  local mode="$1" op="$2" outname="$3"; shift 3
  local flags
  if [[ "$mode" == "off" ]]; then
    flags=(SPYRE_COST_MODEL_MATMUL_PLANNER=0 SPYRE_COST_MODEL_POINTWISE_PLANNER=0 SPYRE_COST_MODEL_REDUCTION_PLANNER=0)
  else
    flags=(SPYRE_COST_MODEL_MATMUL_PLANNER=1 SPYRE_COST_MODEL_POINTWISE_PLANNER=0 SPYRE_COST_MODEL_REDUCTION_PLANNER=0)
  fi
  local out="$PERF/${op}_torch-spyre_${outname}_${mode}.txt"
  wait_device_free
  echo "[$(date +%T)] === ${op} torch-spyre ${outname} planner=${mode} ==="
  env "${TSP_COMMON[@]}" "${flags[@]}" "SPYRE_LOG_FILE=$SPLITLOG/${op}_${outname}_${mode}.log" \
      "$TSP_PY" benchmark.py --op "$op" --stack torch-spyre --with-profiling "$@" --output "$out"
  local rc=$?
  if [[ $rc -ne 0 ]]; then echo "FAILED rc=$rc tsp ${mode} ${outname}"; return $rc; fi
  return 0
}

run_sendnn() {  # op outname shape_args...
  local op="$1" outname="$2"; shift 2
  local out="$PERF/${op}_sendnn_${outname}.txt"
  wait_device_free
  echo "[$(date +%T)] === ${op} sendnn ${outname} ==="
  env "${SENDNN_ENV[@]}" "$SENDNN_PY" benchmark.py --op "$op" --stack sendnn --with-profiling "$@" --output "$out"
  local rc=$?
  if [[ $rc -ne 0 ]]; then echo "FAILED rc=$rc sendnn ${outname}"; return $rc; fi
  return 0
}

# (outname | --shape args...) -- the 6 Priyanka matmul shapes
shapes=(
  "qo_prefill|--shape|1|512|4096|--shape|4096|4096"
  "qo_bs4|--shape|4|1|4096|--shape|4096|4096"
  "kv_prefill|--shape|1|512|4096|--shape|4096|1024"
  "kv_bs4|--shape|4|1|4096|--shape|4096|1024"
  "mlp_prefill|--shape|1|512|4096|--shape|4096|12800"
  "mlp_bs4|--shape|4|1|4096|--shape|4096|12800"
)

for entry in "${shapes[@]}"; do
  IFS='|' read -ra parts <<< "$entry"
  outname="${parts[0]}"; sargs=("${parts[@]:1}")
  run_tsp off "matmul" "$outname" "${sargs[@]}" || exit 1
  run_tsp on  "matmul" "$outname" "${sargs[@]}" || exit 1
  run_sendnn  "matmul" "$outname" "${sargs[@]}" || exit 1
done

echo "[$(date +%T)] All matmul A/B runs completed."
