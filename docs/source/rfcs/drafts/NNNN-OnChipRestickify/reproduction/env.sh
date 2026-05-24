# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Shared environment for the on-chip core-to-core reproduction harness.
#
# Every machine-specific path is an overridable default in `: "${VAR:=...}"`
# form, with the values from the machine where these were developed. Override
# any of them in your shell before running a script, e.g.:
#
#     PYTHON=/path/to/python PATCHED_DXP=/path/to/dxp_standalone \
#         bash devval/devval_roundtrip_fix_512.sh
#
# These are also exported so the Python scripts (which read them via
# os.environ.get) see the same values.

# Directory holding this env.sh (the reproduction/ root), so VAL_BOOT and the
# data dirs resolve to the in-repo copies regardless of cwd.
REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# --- Toolchain --------------------------------------------------------------
# Python interpreter (the venv with torch 2.11 + torch_spyre built in).
: "${PYTHON:=/home/adnan/dt-inductor/.venv/bin/python}"
# Patched dxp standalone (carries the deeptools on-chip foundation patch).
: "${PATCHED_DXP:=/home/adnan/dt-inductor/build/deeptools-onchip/dxp/dxp_standalone}"

# --- On-chip source worktree -----------------------------------------------
# Worktree holding torch_spyre with onchip_bridge.py (the tier0-tier1-onchip
# checkout). val-boot/sitecustomize.py prepends this to sys.path; the splice
# scripts load onchip_bridge.py from it.
: "${ONCHIP_SRC:=/tmp/tier-up}"
: "${ONCHIP_BRIDGE:=${ONCHIP_SRC}/torch_spyre/_inductor/codegen/onchip_bridge.py}"

# --- val-boot import shim ---------------------------------------------------
# In-repo directory whose sitecustomize.py drops torch_spyre from the editable
# finder and prepends ONCHIP_SRC. Goes on PYTHONPATH for device runs.
: "${VAL_BOOT:=${REPRO_ROOT}/val-boot}"

# --- Scratch / work directory ----------------------------------------------
# Root for all spliced-*/*-cache/baseline scratch dirs the scripts create.
: "${WORK_DIR:=/tmp}"

# --- Analysis / baseline input bundles -------------------------------------
# Per-process inductor cache root used by gen_baseline.py.
: "${TORCHINDUCTOR_CACHE_ROOT:=/tmp/torchinductor_adnan}"
# Compiled-bundle code_dirs the edge analysis classifies (real fused kernels).
: "${GRANITE_INDUCTOR:=/tmp/granite_inductor}"
: "${EDGE_GRANITE_RMSNORM:=${GRANITE_INDUCTOR}/inductor-spyre/sdsc_fused_add_linear_mul_rms_norm_6_m56h1rzb}"
: "${EDGE_SDPA:=${TORCHINDUCTOR_CACHE_ROOT}/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_451ht_5h}"
: "${EDGE_ATTN_RMSNORM:=${GRANITE_INDUCTOR}/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_transpose_unsqueeze_view_2_jfvth_by}"

export PYTHON PATCHED_DXP ONCHIP_SRC ONCHIP_BRIDGE VAL_BOOT WORK_DIR
export TORCHINDUCTOR_CACHE_ROOT GRANITE_INDUCTOR
export EDGE_GRANITE_RMSNORM EDGE_SDPA EDGE_ATTN_RMSNORM
