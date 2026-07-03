#!/usr/bin/env bash
set -o pipefail
ROOT="/home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108"
RUN="/home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108/runs/granite_relayout_s512_kernel_neighbor_after_20260703_234640"
cd "$RUN"
timeout 1800 env \
  PYTHONPATH="$ROOT/torch-spyre:$ROOT/torch-spyre/tests/inductor:$ROOT/foundation-model-stack:$ROOT/spyre-granite-e2e-bench:/home/adnan/dt-inductor/sentient/runtime/lib" \
  PATH="$ROOT/tools/dxp-split-wrapper:$ROOT/build/deeptools/dxp:/opt/ibm/spyre/deeptools/bin:/home/adnan/dt-inductor/sentient/runtime/bin:/home/adnan/dt-inductor/sentient/deeptools/bin:/home/adnan/dt-inductor/.venv/bin:/home/adnan/.local/bin:/home/adnan/bin:/opt/ibm/spyre/tvm/bin:/opt/ibm/spyre/spyre-comms/bin:/opt/ibm/spyre/runtime/bin:/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/senlib/bin:/opt/ibm/spyre/sentinyexec/bin:/usr/lib64/ccache:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  LD_LIBRARY_PATH="/home/adnan/opt-newer/runtime/lib:/home/adnan/opt-newer/spyre-comms/lib:/home/adnan/dt-inductor/sentient/runtime/lib64:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:/home/adnan/dt-inductor/sentient/libaiupti/lib" \
  TORCHINDUCTOR_CACHE_DIR="$RUN/block_prefill/cache" \
  DEEPTOOLS_PATH="$ROOT/deeptools" \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  SPYRE_INDUCTOR_LOG=1 \
  SPYRE_INDUCTOR_LOG_LEVEL=DEBUG \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=192 \
  AIU_WORLD_SIZE=1 \
  SPYRE_LX_PLANNING=1 \
  DXP_LX_FRAC_AVAIL=0 \
  DXP_BACKEND_LX_FRAC_AVAIL=0.2 \
  SPYRE_LX_PLANNER_RELAYOUT=1 \
  LX_BOUNDARY_CLONES=1 \
  SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1 \
  SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1 \
  SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1 \
  SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1 \
  DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1 \
  DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1 \
  DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR="$RUN/backend_plans" \
  DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$RUN/backend_plans" \
  /home/adnan/dt-inductor/.venv/bin/python3 \
  "$ROOT/spyre-granite-e2e-bench/benchmarks/granite_block_layer_probe.py" \
  --fms-root "$ROOT/foundation-model-stack" \
  --run-root "$RUN" \
  --case prefill \
  --seq-len 512 \
  --batch 1 \
  --hidden 4096 \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 1 \
  --warmups 1 \
  --profile \
  --profile-dir "$RUN/block_prefill/profile" \
  --no-profile-memory \
  > "$RUN/stdout.log" 2> "$RUN/stderr.log"
