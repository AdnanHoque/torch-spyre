#!/usr/bin/env bash

if [[ -n "${DXP_BACKEND_LX_FRAC_AVAIL:-}" ]]; then
    export DXP_LX_FRAC_AVAIL="$DXP_BACKEND_LX_FRAC_AVAIL"
fi

exec /home/adnan/codex-isolated/flash_pr81_relayout_20260714/deeptools-build/dxp/dxp_standalone "$@"
