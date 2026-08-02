#!/usr/bin/env bash
# Run a copied DXP compiler runtime without exposing its private shared
# libraries to the parent Torch-Spyre process.

set -euo pipefail

runtime=${DXP_PRIVATE_RUNTIME:?set DXP_PRIVATE_RUNTIME to the copied DXP runtime}
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset LD_PRELOAD
exec "$runtime/bin/dxp_standalone" "$@"
