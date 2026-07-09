#!/usr/bin/env bash

_spyre_activate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_spyre_home_root="$(dirname -- "$(dirname -- "$_spyre_activate_dir")")"

export SPYRE_ENV_ROOT="${SPYRE_ENV_ROOT:-$_spyre_activate_dir}"
export SPYRE_DEV_ROOT="${SPYRE_DEV_ROOT:-$_spyre_home_root/dt-inductor}"
export SPYRE_PERF_PYTHON="$SPYRE_DEV_ROOT/.venv/bin/python"
export SPYRE_PERF_SUITE_ROOT="$SPYRE_ENV_ROOT/spyre-perf-suite"
export TORCH_SPYRE_ROOT="$SPYRE_ENV_ROOT/torch-spyre"
export FMS_ROOT="$SPYRE_DEV_ROOT/foundation-model-stack"
export DEEPTOOLS_PATH="$SPYRE_ENV_ROOT/deeptools-install/share"

export PATH="$SPYRE_DEV_ROOT/.venv/bin:$SPYRE_ENV_ROOT/deeptools-install/bin:/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/runtime/bin:/opt/ibm/spyre/spyre-comms/bin:/opt/ibm/spyre/senlib/bin:/opt/ibm/spyre/sentinyexec/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$TORCH_SPYRE_ROOT:$FMS_ROOT:$SPYRE_PERF_SUITE_ROOT"

_spyre_torch_lib="$SPYRE_DEV_ROOT/.venv/lib/python3.12/site-packages/torch/lib"
if [[ ! -d "$_spyre_torch_lib" ]]; then
  _spyre_torch_lib="$SPYRE_DEV_ROOT/.venv/lib64/python3.12/site-packages/torch/lib"
fi

export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/sentinyexec/lib:$_spyre_torch_lib:$SPYRE_DEV_ROOT/sentient/libaiupti/lib:$SPYRE_DEV_ROOT/sentient/runtime/lib:$SPYRE_DEV_ROOT/sentient/deeptools/lib"
export TORCH_DEVICE_BACKEND_AUTOLOAD=1

unset DXP_LX_FRAC_AVAIL
unset DXP_BACKEND_LX_FRAC_AVAIL
unset DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY
