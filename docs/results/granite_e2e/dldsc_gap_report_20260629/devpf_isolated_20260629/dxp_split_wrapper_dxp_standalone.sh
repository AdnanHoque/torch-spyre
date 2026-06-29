#!/usr/bin/env bash
set -euo pipefail
BUILD='/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114/deeptools/build-dxp-relayout-isolated'
export LD_LIBRARY_PATH="$BUILD/dxp:$BUILD/dcg:$BUILD/dcg/dcg_fe:$BUILD/dcg/dcg_fe/scheduler:$BUILD/dcg/dcg_be:$BUILD/dip:$BUILD/ddc:$BUILD/dpc:$BUILD/dcc/lib:$BUILD/dsc:$BUILD/sgr:$BUILD/sharedtools:$BUILD/util:$BUILD/common:$BUILD/external/json11:$BUILD/external/g3log:$BUILD/ddc/transformations/automatic_shuffle:$BUILD/ddc/ddl:${LD_LIBRARY_PATH:-}"
if [[ -n "${DXP_BACKEND_LX_FRAC_AVAIL:-}" ]]; then
  export DXP_LX_FRAC_AVAIL="$DXP_BACKEND_LX_FRAC_AVAIL"
fi
exec "$BUILD/dxp/dxp_standalone" "$@"
