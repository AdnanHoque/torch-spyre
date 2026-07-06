#!/usr/bin/env bash
set -euo pipefail
cd "/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_one_flag_gather_restickify_20260706_175419"
export TEST_FLASH_SCRIPT="/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_one_flag_gather_restickify_20260706_175419/test_flash.py"
export PATCH_MODE="no_h2d,skip_cpu_ref"
export PYTHONPATH="/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/torch-spyre:${PYTHONPATH:-}"
export PATH="/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_one_flag_gather_restickify_20260706_175419/tools:${PATH:-}"
export LD_LIBRARY_PATH="/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools/build-deeptools:/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools/install-deeptools/lib:${LD_LIBRARY_PATH:-}"
export DEEPTOOLS_PATH="/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools"
export DEEPTOOLS_INSTALL_DIR="/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools/install-deeptools"
export TORCHINDUCTOR_CACHE_DIR="/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_one_flag_gather_restickify_20260706_175419/cache"
export SPYRE_INDUCTOR_DEBUG=1
export SPYRE_LX_PLANNER_RELAYOUT=1
unset SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS
unset SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES
unset SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT
unset SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY
export DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR
unset DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
/home/adnan-cdx/dt-inductor/.venv/bin/python3 "/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_one_flag_gather_restickify_20260706_175419/bootstrap.py" >"/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_one_flag_gather_restickify_20260706_175419/stdout.log" 2>"/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_one_flag_gather_restickify_20260706_175419/stderr.log"
