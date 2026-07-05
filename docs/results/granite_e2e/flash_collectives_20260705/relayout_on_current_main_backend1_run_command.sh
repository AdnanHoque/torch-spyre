#!/usr/bin/env bash
set -u
run_dir="/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_current_main_backend1_20260705_191932"
export TORCHINDUCTOR_CACHE_DIR="$run_dir/cache"
export DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR="$run_dir/backend_plans"
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$run_dir/backend_plans"
export PATCH_MODE="no_h2d,skip_cpu_ref"
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
export SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
export SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
export LX_BOUNDARY_CLONES=1
export TEST_FLASH_SCRIPT=/tmp/test-spyre-scripts/test_flash.py
timeout 900 /home/adnan/dt-inductor/.venv/bin/python3 "$run_dir/bootstrap.py"
