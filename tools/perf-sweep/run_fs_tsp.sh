#!/bin/bash
# TSP-arm env wrapper for the force-split harness, copied from
# /tmp/spyre-perf-suite/run_matmul_ab.sh (TSP_COMMON + LD_TSP).
set -uo pipefail
TSP_PY=/home/adnan/dt-inductor/.venv/bin/python
LD_TSP=/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:/home/adnan/dt-inductor/sentient/libaiupti/lib:/home/adnan/dt-inductor/sentient/runtime/lib:/home/adnan/dt-inductor/sentient/deeptools/lib:/opt/ibm/spyre/tvm/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/sentinyexec/lib
env \
  PYTHONPATH=/tmp/cost_model_unified_shim \
  DXP_LX_FRAC_AVAIL=1 \
  SENCORES=32 \
  USE_SPYRE_PROFILER=1 \
  SPYRE_COST_MODEL_MATMUL_PLANNER=1 \
  SPYRE_COST_MODEL_POINTWISE_PLANNER=0 \
  SPYRE_COST_MODEL_REDUCTION_PLANNER=0 \
  "LD_LIBRARY_PATH=$LD_TSP" \
  "$TSP_PY" /tmp/force_split_mnk.py "$@"
