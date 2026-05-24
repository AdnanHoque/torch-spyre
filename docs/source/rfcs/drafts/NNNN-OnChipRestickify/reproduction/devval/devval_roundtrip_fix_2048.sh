#!/bin/bash
# Device validation: FIXED round-trip bridge, S=2048. Run SOLO (single shared
# accelerator). Positive: redirect runner -> spliced-roundtrip-fix-2048, must be
# value-correct. Negative: remove senprog, must FAIL (proves real load).
set +e
source "$(dirname "$0")/../env.sh"
S=2048
DIR=$WORK_DIR/spliced-roundtrip-fix-$S
SP=$DIR/loadprogram_to_device/spliced-roundtrip-fix-$S-SenProgSend/init.txt
RUN="env ONCHIP_SIZE=$S ONCHIP_DIR=$DIR WORK_DIR=$WORK_DIR PYTHONPATH=$VAL_BOOT TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=$WORK_DIR/rt-fix-$S-cache $PYTHON $(dirname "$0")/devval_roundtrip_fix.py"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204'; }
echo "spliced senprog present: $(wc -l < "$SP" 2>/dev/null) lines"
echo "### POSITIVE: must load spliced + be VALUE-CORRECT (no Compute-CB)"
rm -rf "$WORK_DIR/rt-fix-$S-cache"
eval "$RUN" 2>&1 | clean | grep -E 'REDIRECT|DIRECT_VALIDATE_OK|max_err|Error|assert|Mismatch|Traceback|Exception|No such|Compute CB|ComputeHardware|RAS::' | tail -8
echo ""
echo "### NEGATIVE CONTROL: remove senprog -> must FAIL"
mv "$SP" "${SP}.bak"; rm -rf "$WORK_DIR/rt-fix-$S-cache"
eval "$RUN" 2>&1 | clean | grep -E 'DIRECT_VALIDATE_OK|Error|No such|RuntimeError' | tail -6
mv "${SP}.bak" "$SP"; echo "### restored. DONE"
