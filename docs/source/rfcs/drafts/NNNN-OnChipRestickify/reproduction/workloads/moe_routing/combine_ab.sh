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
# Combine (scatter) A/B: (perm_w @ y) @ wout. Same matmul->matmul handoff edge as
# dispatch; splice the perm_w@y output -> consumer-linear input. SOLO device use.
set +e
PY=/home/adnan/dt-inductor/.venv/bin/python
DIR=/tmp/ab_moe_routing
PDXP_DIR=/home/adnan/dt-inductor/build/deeptools-onchip/dxp
COMMON="PYTHONPATH=/tmp/val-boot TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204'; }

E=${1:-8}; T=${2:-512}; H=${3:-2048}
CAP=$(( (T+E-1)/E )); EC=$(( E*CAP ))
SHAPES="MOE_E=$E MOE_T=$T MOE_H=$H"
# combine output is [T, H] -> consumer reads it sharded {mb:T}, in=H. The bridged
# tensor's consumer-input iter: mb=T (rows of combined), in=H.
base="$DIR/cbase"; spl="$DIR/spliced-moe-combine"
echo "### COMBINE shape E=$E T=$T H=$H (EC=$EC); combine out = T*H*2 = $(( T*H*2 )) B"

rm -rf /tmp/cmb_base "$base"
eval "env $COMMON $SHAPES ONCHIP_BASELINE=1 ONCHIP_MODE=validate \
  TORCHINDUCTOR_CACHE_DIR=/tmp/cmb_base $PY $DIR/devval_moe_combine.py" 2>&1 | clean | \
  grep -E 'DIRECT_VALIDATE_OK|RAS::|Error' | tail -1
bsrc=$(ls -dt /tmp/cmb_base/inductor-spyre/sdsc_fused_mm_*/ 2>/dev/null | head -1)
[ -z "$bsrc" ] && { echo "NO BASELINE"; exit 1; }
cp -r "$bsrc" "$base"

rm -rf "$spl"
MOE_MB=$T MOE_IN=$H $PY $DIR/splice_moe_dispatch.py --baseline-dir "$base" --out-dir "$spl" \
  > /tmp/cmb_splice.json 2>&1
[ $? -ne 0 ] && { echo "SPLICE FAILED"; tail -3 /tmp/cmb_splice.json; exit 1; }
echo "  spliced footprint=$($PY -c "import json;print(json.load(open('/tmp/cmb_splice.json'))['lx_footprint_bytes'])") B"

env LD_LIBRARY_PATH="$PDXP_DIR:$LD_LIBRARY_PATH" \
  "$PDXP_DIR/dxp_standalone" --bundle -d "$spl" > /tmp/cmb_dxp.log 2>&1
sp="$spl/loadprogram_to_device/$(basename $spl)-SenProgSend/init.txt"
[ ! -f "$sp" ] && { echo "DXP FAILED"; tail -4 /tmp/cmb_dxp.log; exit 1; }
echo "  patched dxp OK senprog $(wc -l < $sp) lines"

rm -rf /tmp/cmb_pos
eval "env $COMMON $SHAPES ONCHIP_DIR=$spl ONCHIP_MODE=validate \
  TORCHINDUCTOR_CACHE_DIR=/tmp/cmb_pos $PY $DIR/devval_moe_combine.py" 2>&1 | clean | \
  grep -E 'DIRECT_VALIDATE_OK|RAS::|BusFence|Mismatch|Error' | tail -1

# negative control
mv "$sp" "${sp}.bak"; rm -rf /tmp/cmb_neg
eval "env $COMMON $SHAPES ONCHIP_DIR=$spl ONCHIP_MODE=validate \
  TORCHINDUCTOR_CACHE_DIR=/tmp/cmb_neg $PY $DIR/devval_moe_combine.py" 2>&1 | clean | \
  grep -E 'DIRECT_VALIDATE_OK|Failed to open|RuntimeError' | tail -1
mv "${sp}.bak" "$sp"; echo "  (negative control restored)"

rm -rf /tmp/cmb_aA
eval "env $COMMON $SHAPES ONCHIP_BASELINE=1 ONCHIP_MODE=bench BENCH_WARMUP=12 BENCH_ITERS=50 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/cmb_aA $PY $DIR/devval_moe_combine.py" 2>&1 | clean | \
  grep -E 'BENCH|RAS::' | tail -1
rm -rf /tmp/cmb_bB
eval "env $COMMON $SHAPES ONCHIP_DIR=$spl ONCHIP_MODE=bench BENCH_WARMUP=12 BENCH_ITERS=50 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/cmb_bB $PY $DIR/devval_moe_combine.py" 2>&1 | clean | \
  grep -E 'BENCH|RAS::' | tail -1
echo "### COMBINE DONE"
