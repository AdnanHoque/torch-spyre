#!/bin/bash
# Device validation: 2048 cross-core round trip. Run SOLO (single shared
# accelerator). Positive: redirect runner -> spliced-roundtrip, must load spliced
# + be value-correct (no Compute-CB). Negative: remove senprog, must FAIL.
set +e
source "$(dirname "$0")/../env.sh"
SP=$WORK_DIR/spliced-roundtrip/loadprogram_to_device/spliced-roundtrip-SenProgSend/init.txt
RUN="env WORK_DIR=$WORK_DIR PYTHONPATH=$VAL_BOOT TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=$WORK_DIR/rt-direct-cache $PYTHON $(dirname "$0")/devval_roundtrip.py"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204'; }
echo "spliced senprog present: $(wc -l < "$SP" 2>/dev/null) lines"
echo "### POSITIVE: redirect runner -> spliced-roundtrip (must load spliced + be VALUE-CORRECT, no Compute-CB)"
rm -rf "$WORK_DIR/rt-direct-cache"
eval "$RUN" 2>&1 | clean | grep -E 'REDIRECT|DIRECT_VALIDATE_OK|max_err|Error|assert|Mismatch|Traceback|Exception|No such|Compute CB|ComputeHardware|RAS::' | tail -8
echo ""
echo "### NEGATIVE CONTROL: remove spliced senprog -> must FAIL (proves device loads from there)"
mv "$SP" "${SP}.bak"
rm -rf "$WORK_DIR/rt-direct-cache"
eval "$RUN" 2>&1 | clean | grep -E 'REDIRECT|DIRECT_VALIDATE_OK|max_err|Error|assert|Traceback|Exception|No such|RuntimeError' | tail -6
mv "${SP}.bak" "$SP"
echo "### restored. DONE"
