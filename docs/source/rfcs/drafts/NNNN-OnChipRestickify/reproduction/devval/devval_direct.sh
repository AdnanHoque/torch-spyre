#!/bin/bash
# Device validation: 2048 Tier-2 transpose proof (the splice_2048_bmm bundle,
# which FAULTS with the Compute-CB hardware error). Run SOLO (single shared
# accelerator). Positive: redirect runner -> spliced-2048, must load spliced.
# Negative: remove senprog, must FAIL (proves it loads from there).
set +e
source "$(dirname "$0")/../env.sh"
SP=$WORK_DIR/spliced-2048/loadprogram_to_device/spliced-2048-SenProgSend/init.txt
RUN="env WORK_DIR=$WORK_DIR PYTHONPATH=$VAL_BOOT TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=$WORK_DIR/direct-cache $PYTHON $(dirname "$0")/devval_direct.py"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204'; }
echo "spliced senprog present: $(wc -l < "$SP" 2>/dev/null) lines"
echo "### POSITIVE: redirect runner -> spliced-2048 (must load spliced + be correct)"
rm -rf "$WORK_DIR/direct-cache"
eval "$RUN" 2>&1 | clean | grep -E 'REDIRECT|DIRECT_VALIDATE_OK|Error|assert|Mismatch|Traceback|Exception|No such' | tail -6
echo ""
echo "### NEGATIVE CONTROL: remove spliced senprog -> must FAIL (proves it loads from there)"
mv "$SP" "${SP}.bak"
rm -rf "$WORK_DIR/direct-cache"
eval "$RUN" 2>&1 | clean | grep -E 'REDIRECT|DIRECT_VALIDATE_OK|Error|assert|Traceback|Exception|No such|RuntimeError' | tail -6
mv "${SP}.bak" "$SP"
echo "### restored. DONE"
