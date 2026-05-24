#!/bin/bash
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
# Per-shape A/B for the MoE dispatch->consumer on-chip splice. SOLO device use.
# For each (E,T,H): compile baseline -> splice the dispatch edge -> recompile with
# patched dxp -> validate on device -> A/B (baseline_HBM vs spliced on-chip).
set +e

PY=/home/adnan/dt-inductor/.venv/bin/python
DIR=/tmp/ab_moe_routing
PDXP_DIR=/home/adnan/dt-inductor/build/deeptools-onchip/dxp
COMMON="PYTHONPATH=/tmp/val-boot TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204'; }

run_shape() {
  local E=$1 T=$2 H=$3
  local CAP=$(( (T + E - 1) / E )); local EC=$(( E * CAP ))
  local SHAPES="MOE_E=$E MOE_T=$T MOE_H=$H"
  local tag="E${E}_T${T}_H${H}"
  local base="$DIR/base_$tag" spl="$DIR/spl_$tag"
  echo "============================================================"
  echo "### SHAPE $tag  (EC=$EC, H=$H)  dispatch buf = EC*H*2 = $(( EC*H*2 )) B"

  # 1. baseline compile -> capture code_dir
  rm -rf /tmp/c_base_$tag "$base"
  eval "env $COMMON $SHAPES ONCHIP_BASELINE=1 ONCHIP_MODE=validate \
    TORCHINDUCTOR_CACHE_DIR=/tmp/c_base_$tag $PY $DIR/devval_moe.py" 2>&1 | clean | \
    grep -E 'DIRECT_VALIDATE_OK|RAS::|BusFence|Error' | tail -1
  local bsrc=$(ls -dt /tmp/c_base_$tag/inductor-spyre/sdsc_fused_mm_*/ 2>/dev/null | head -1)
  if [ -z "$bsrc" ]; then echo "  NO BASELINE BUNDLE -- skip"; return; fi
  cp -r "$bsrc" "$base"

  # 2. splice the dispatch edge (consumer iter: mb=EC, in=H)
  rm -rf "$spl"
  MOE_MB=$EC MOE_IN=$H $PY $DIR/splice_moe_dispatch.py \
    --baseline-dir "$base" --out-dir "$spl" > /tmp/splice_$tag.json 2>&1
  if [ $? -ne 0 ]; then echo "  SPLICE FAILED:"; tail -3 /tmp/splice_$tag.json; return; fi
  local fp=$($PY -c "import json;print(json.load(open('/tmp/splice_$tag.json'))['lx_footprint_bytes'])" 2>/dev/null)
  echo "  spliced: lx_footprint=$fp B (cap 2097152)"

  # 3. recompile spliced with patched dxp
  env LD_LIBRARY_PATH="$PDXP_DIR:$LD_LIBRARY_PATH" \
    "$PDXP_DIR/dxp_standalone" --bundle -d "$spl" > /tmp/dxp_$tag.log 2>&1
  local dxprc=$?
  local sp="$spl/loadprogram_to_device/$(basename $spl)-SenProgSend/init.txt"
  if [ $dxprc -ne 0 ] || [ ! -f "$sp" ]; then
    echo "  PATCHED DXP FAILED rc=$dxprc"; tail -4 /tmp/dxp_$tag.log; return; fi
  echo "  patched dxp OK, senprog $(wc -l < $sp) lines; ring DataDscs=$(grep -c 'Creating PCFG for DataDsc' /tmp/dxp_$tag.log)"

  # 4. POSITIVE validate
  rm -rf /tmp/c_pos_$tag
  eval "env $COMMON $SHAPES ONCHIP_DIR=$spl ONCHIP_MODE=validate \
    TORCHINDUCTOR_CACHE_DIR=/tmp/c_pos_$tag $PY $DIR/devval_moe.py" 2>&1 | clean | \
    grep -E 'DIRECT_VALIDATE_OK|RAS::|BusFence|Error|Mismatch' | tail -1

  # 5. A/B bench
  rm -rf /tmp/c_aA_$tag
  eval "env $COMMON $SHAPES ONCHIP_BASELINE=1 ONCHIP_MODE=bench BENCH_WARMUP=12 BENCH_ITERS=50 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/c_aA_$tag $PY $DIR/devval_moe.py" 2>&1 | clean | \
    grep -E 'BENCH|RAS::|BusFence' | tail -1
  rm -rf /tmp/c_bB_$tag
  eval "env $COMMON $SHAPES ONCHIP_DIR=$spl ONCHIP_MODE=bench BENCH_WARMUP=12 BENCH_ITERS=50 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/c_bB_$tag $PY $DIR/devval_moe.py" 2>&1 | clean | \
    grep -E 'BENCH|RAS::|BusFence' | tail -1
}

for shp in "$@"; do
  IFS=',' read E T H <<< "$shp"
  run_shape "$E" "$T" "$H"
done
echo "### SWEEP DONE"
