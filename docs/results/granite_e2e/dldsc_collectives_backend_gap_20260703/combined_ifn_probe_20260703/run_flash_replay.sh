#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/adnan/codex-isolated/flash_attention_dldsc_20260703_170000
RUNROOT=${RUNROOT:-$(cd "$(dirname "$0")" && pwd)}
TORCH_ROOT="$ROOT/torch-spyre"
SCRIPT_PATH="$ROOT/test-spyre-scripts/test_flash.py"
PYTHON=/home/adnan/dt-inductor/.venv/bin/python3
TOOLS="$ROOT/tools"
BASE_PATH="$TOOLS:/home/adnan/dt-inductor/.venv/bin:/home/adnan/.local/bin:/home/adnan/bin:/opt/ibm/spyre/runtime/bin:/opt/ibm/spyre/spyre-comms/bin:/opt/ibm/spyre/senlib/bin:/usr/lib64/ccache:/usr/local/bin:/usr/bin:/bin"
BASE_LD="/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:/home/adnan/dt-inductor/sentient/libaiupti/lib:/home/adnan/dt-inductor/sentient/runtime/lib:/home/adnan/dt-inductor/sentient/deeptools/lib:/opt/ibm/spyre/senlib/lib"
BASE_PYTHONPATH="$TORCH_ROOT:$TORCH_ROOT/tests/inductor:/home/adnan/dt-inductor/foundation-model-stack:/home/adnan/dt-inductor/sentient/runtime/lib"

write_bootstrap() {
  local dir=$1
  cat > "$dir/bootstrap.py" <<PY
import os
import runpy
import traceback

script = os.environ["TEST_FLASH_SCRIPT"]
patch_mode = os.environ.get("PATCH_MODE", "")

import torch
import torch_spyre

torch_spyre._autoload()

if "no_h2d" in patch_mode:
    _orig_tensor_to = torch.Tensor.to

    def _compile_probe_to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                device = first
        if device is not None and str(device).startswith("spyre"):
            dtype = kwargs.get("dtype", self.dtype)
            return torch.empty_strided(
                tuple(self.size()), tuple(self.stride()), dtype=dtype, device=device
            )
        return _orig_tensor_to(self, *args, **kwargs)

    torch.Tensor.to = _compile_probe_to
    print("[runtime_patch] no_h2d Tensor.to(device=spyre) compile probe enabled", flush=True)

try:
    torch.spyre.manual_seed_all = lambda seed: None
except Exception:
    pass

try:
    if "skip_cpu_ref" in patch_mode:
        source = open(script, "r", encoding="utf-8").read()
        source = source.replace(
            "    attn_t = flash_cpu(queries_t, keys_t, values_t, mask_t)",
            "    attn_t = torch.empty_like(queries_t)  # runtime compile probe: skip CPU reference",
        )
        source = source.replace(
            "    torch.testing.assert_close(attn_t, attn_t_spyre.cpu(), atol=0.1, rtol=0.1)",
            "    print(\"[runtime_patch] assert_close skipped for compile probe\", flush=True)",
        )
        code = compile(source, script, "exec")
        globs = {"__name__": "__main__", "__file__": script}
        exec(code, globs, globs)
    else:
        runpy.run_path(script, run_name="__main__")
except BaseException:
    traceback.print_exc()
    raise
PY
}

run_mode() {
  local mode=$1
  local dir="$RUNROOT/$mode"
  mkdir -p "$dir/backend_plans" "$dir/cache"
  write_bootstrap "$dir"
  {
    echo "mode=$mode"
    echo "root=$ROOT"
    echo "torch_root=$TORCH_ROOT"
    echo "script=$SCRIPT_PATH"
    echo "python=$PYTHON"
    echo "command=timeout 900 $PYTHON $dir/bootstrap.py"
  } > "$dir/command.txt"

  set +e
  (
    export PATH="$BASE_PATH"
    export LD_LIBRARY_PATH="$BASE_LD"
    export PYTHONPATH="$BASE_PYTHONPATH"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export TORCHINDUCTOR_CACHE_DIR="$dir/cache"
    export TORCHINDUCTOR_FX_GRAPH_CACHE=0
    export TEST_FLASH_SCRIPT="$SCRIPT_PATH"
    export PATCH_MODE="no_h2d,skip_cpu_ref"
    export PYTHONUNBUFFERED=1
    export SPYRE_INDUCTOR_LOG=1
    export SPYRE_INDUCTOR_LOG_LEVEL=DEBUG
    export SPYRE_LX_PLANNING=1
    export LX_PLANNING=1
    export DXP_LX_FRAC_AVAIL=0
    export DXP_BACKEND_LX_FRAC_AVAIL=0.2
    export DEEPTOOLS_PATH=/home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108/deeptools
    export DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR="$dir/backend_plans"
    export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$dir/backend_plans"
    unset DEEPTOOLS_ENABLE_UNSAFE_MATMUL_OPERAND_BROADCAST

    if [[ "$mode" == baseline ]]; then
      export SPYRE_LX_PLANNER_RELAYOUT=0
      export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=0
      export SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=0
      export SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=0
      export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=0
      export LX_BOUNDARY_CLONES=0
    else
      export SPYRE_LX_PLANNER_RELAYOUT=1
      export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
      export SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
      export SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=0
      export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
      export LX_BOUNDARY_CLONES=1
    fi

    env | sort > "$dir/env.txt"
    timeout 900 "$PYTHON" "$dir/bootstrap.py"
  ) > "$dir/stdout.log" 2> "$dir/stderr.log"
  local rc=$?
  echo "$rc" > "$dir/returncode.txt"
  set -e
  return 0
}

run_mode baseline
run_mode metadata
