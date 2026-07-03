#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525"
RUN="/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_matmul_operand_only105_fission4_dstoffset_20260703_080553"
timeout 1800 env \
  PYTHONPATH="$ROOT/torch-spyre:$ROOT/torch-spyre/tests/inductor" \
  PATH="$ROOT/tools/dxp-split-wrapper:$ROOT/build-deeptools/dxp:/home/adnan-cdx/dt-inductor-mixed/sentient/runtime/bin:/home/adnan-cdx/dt-inductor-mixed/sentient/deeptools/bin:/home/adnan-cdx/dt-inductor-mixed/.venv/bin:/opt/ibm/spyre/tvm/bin:/opt/ibm/spyre/spyre-comms/bin:/opt/ibm/spyre/runtime/bin:/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/senlib/bin:/opt/ibm/spyre/sentinyexec/bin:/usr/lib64/ccache:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:$ROOT/build-deeptools/dxp:$ROOT/build-deeptools/dsm:/home/adnan-cdx/dt-inductor-mixed/sentient/runtime/lib:/home/adnan-cdx/dt-inductor-mixed/sentient/runtime/lib64:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/home/adnan-cdx/dt-inductor-mixed/.venv/lib/python3.12/site-packages/torch/lib:/home/adnan-cdx/dt-inductor-mixed/sentient/libaiupti/lib" \
  TORCHINDUCTOR_CACHE_DIR="$RUN/cache" \
  DEEPTOOLS_PATH="$ROOT/deeptools" \
  SPYRE_INDUCTOR_LOG=1 \
  SPYRE_INDUCTOR_LOG_LEVEL=DEBUG \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=192 \
  AIU_WORLD_SIZE=1 \
  DXP_LX_FRAC_AVAIL=0 \
  DXP_BACKEND_LX_FRAC_AVAIL=1 \
  DXP_ENABLE_COMPILE_TIME_CORRECTION=1 \
  SPYRE_LX_PLANNER_RELAYOUT=1 \
  LX_BOUNDARY_CLONES=1 \
  SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1 \
  SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1 \
  SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=0 \
  SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1 \
  DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR="$RUN/backend_plans" \
  DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$RUN/backend_plans" \
  DEEPTOOLS_MATMUL_OPERAND_BROADCAST_ONLY_SDSC=105_batchmatmul \
  DEEPTOOLS_MATMUL_OPERAND_BROADCAST_FISSION_ROWS=4 \
  /home/adnan-cdx/dt-inductor-mixed/.venv/bin/python3 "$ROOT/test-spyre-scripts/test_flash.py"
