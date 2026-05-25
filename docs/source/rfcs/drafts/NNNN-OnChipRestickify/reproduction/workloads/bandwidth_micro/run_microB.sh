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
# Microbench B: effective ring move bandwidth via the proven MoE matmul handoff.
#   graph (perm @ x) @ wexp -> 2-SDSC bundle, handoff buffer [EC,H] @ HBM base 0.
#   BASELINE: handoff through HBM (stock-equivalent fused bundle).
#   ON-CHIP : handoff kept in LX, moved cross-core via STCDP round-trip i->31-i->i
#             (patched dxp). DeltaT = t_base - t_onchip.
# One fresh process per stage (isolates runtime DMA stalls), per-stage timeout.
# SOLO device use. Reuses the /tmp/ab_moe_routing harness (devval + splice).
set +e
PY=/home/adnan/dt-inductor/.venv/bin/python
MOE=/tmp/ab_moe_routing
DIR=/tmp/ab_bw_micro
PDXP_DIR=/home/adnan/dt-inductor/build/deeptools-onchip/dxp
COMMON="PYTHONPATH=/tmp/val-boot TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204|likely passed to kineto|address stability|report this to|collection.cpp'; }
W="${BENCH_WARMUP:-12}"; N="${BENCH_ITERS:-50}"
OUT=$DIR/microB_results.txt
: > "$OUT"

run_shape() {
  local E=$1 T=$2 H=$3
  local CAP=$(( (T + E - 1) / E )); local EC=$(( E * CAP ))
  local SHAPES="MOE_E=$E MOE_T=$T MOE_H=$H"
  local tag="E${E}_T${T}_H${H}"
  local base="$DIR/Bbase_$tag" spl="$DIR/Bspl_$tag"
  local hbuf=$(( EC*H*2 ))
  echo "### B SHAPE $tag EC=$EC H=$H handoff_buf=$hbuf B ($(( hbuf/1048576 )) MB)" | tee -a "$OUT"

  # 1. baseline compile -> capture fused_mm code_dir
  rm -rf /tmp/cB_base_$tag "$base"
  timeout 300 env $COMMON $SHAPES ONCHIP_BASELINE=1 ONCHIP_MODE=validate \
    TORCHINDUCTOR_CACHE_DIR=/tmp/cB_base_$tag stdbuf -oL $PY $MOE/devval_moe.py 2>&1 | clean | \
    grep -E 'DIRECT_VALIDATE_OK|RAS::|BusFence|Error' | tail -1 | tee -a "$OUT"
  local bsrc=$(ls -dt /tmp/cB_base_$tag/inductor-spyre/sdsc_fused_mm_*/ 2>/dev/null | head -1)
  if [ -z "$bsrc" ]; then echo "  NO BASELINE BUNDLE -- skip" | tee -a "$OUT"; return; fi
  cp -r "$bsrc" "$base"

  # 2. splice the dispatch edge (consumer iter: mb=EC, in=H)
  rm -rf "$spl"
  MOE_MB=$EC MOE_IN=$H $PY $MOE/splice_moe_dispatch.py \
    --baseline-dir "$base" --out-dir "$spl" > /tmp/Bsplice_$tag.json 2>&1
  if [ $? -ne 0 ]; then echo "  SPLICE FAILED:" | tee -a "$OUT"; tail -3 /tmp/Bsplice_$tag.json | tee -a "$OUT"; return; fi
  local fp=$($PY -c "import json;print(json.load(open('/tmp/Bsplice_$tag.json'))['lx_footprint_bytes'])" 2>/dev/null)
  echo "  spliced: lx_footprint=$fp B (cap 2097152)" | tee -a "$OUT"

  # 3. recompile spliced with patched dxp; confirm ring signature
  env LD_LIBRARY_PATH="$PDXP_DIR:$LD_LIBRARY_PATH" \
    "$PDXP_DIR/dxp_standalone" --bundle -d "$spl" > /tmp/Bdxp_$tag.log 2>&1
  local dxprc=$?
  local sp="$spl/loadprogram_to_device/$(basename $spl)-SenProgSend/init.txt"
  if [ $dxprc -ne 0 ] || [ ! -f "$sp" ]; then
    echo "  PATCHED DXP FAILED rc=$dxprc" | tee -a "$OUT"; tail -4 /tmp/Bdxp_$tag.log | tee -a "$OUT"; return; fi
  local pcfg=$(grep -c 'Creating PCFG for DataDsc' /tmp/Bdxp_$tag.log)
  local l3=$(grep -c ': L3SU : L3LU' /tmp/Bdxp_$tag.log)
  echo "  patched dxp OK senprog=$(wc -l < $sp) ring_PCFG_DataDsc=$pcfg L3SU_L3LU_lines=$l3" | tee -a "$OUT"

  # 4. POSITIVE validate on device
  rm -rf /tmp/cB_pos_$tag
  timeout 300 env $COMMON $SHAPES ONCHIP_DIR=$spl ONCHIP_MODE=validate \
    TORCHINDUCTOR_CACHE_DIR=/tmp/cB_pos_$tag stdbuf -oL $PY $MOE/devval_moe.py 2>&1 | clean | \
    grep -E 'DIRECT_VALIDATE_OK|RAS::|BusFence|Error|Mismatch' | tail -1 | tee -a "$OUT"

  # 5. A/B bench: baseline_HBM then spliced on-chip
  rm -rf /tmp/cB_aA_$tag
  timeout 300 env $COMMON $SHAPES ONCHIP_BASELINE=1 ONCHIP_MODE=bench BENCH_WARMUP=$W BENCH_ITERS=$N \
    TORCHINDUCTOR_CACHE_DIR=/tmp/cB_aA_$tag stdbuf -oL $PY $MOE/devval_moe.py 2>&1 | clean | \
    grep -E 'BENCH|RAS::|BusFence' | tail -1 | tee -a "$OUT"
  rm -rf /tmp/cB_bB_$tag
  timeout 300 env $COMMON $SHAPES ONCHIP_DIR=$spl ONCHIP_MODE=bench BENCH_WARMUP=$W BENCH_ITERS=$N \
    TORCHINDUCTOR_CACHE_DIR=/tmp/cB_bB_$tag stdbuf -oL $PY $MOE/devval_moe.py 2>&1 | clean | \
    grep -E 'BENCH|RAS::|BusFence' | tail -1 | tee -a "$OUT"
}

for shp in "$@"; do
  IFS=',' read E T H <<< "$shp"
  run_shape "$E" "$T" "$H"
done
echo "### MICROBENCH B DONE" | tee -a "$OUT"
