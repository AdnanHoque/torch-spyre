#!/bin/bash
# 2048 baseline vs same-core vs round-trip latency, two reps. Run SOLO (single
# shared accelerator). Writes BENCH lines to $WORK_DIR/bench_onchip_results.txt.
set +e
source "$(dirname "$0")/../env.sh"
RES=$WORK_DIR/bench_onchip_results.txt; : > "$RES"
run() {  # $1=label  $2=spliced_dir(or empty)  $3=cachedir
  env PYTHONPATH=$VAL_BOOT WORK_DIR=$WORK_DIR TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
      TORCHINDUCTOR_CACHE_DIR="$3" SPLICED_DIR="$2" BENCH_SIZE=2048 \
      "$PYTHON" "$(dirname "$0")/bench_onchip.py" 2>/dev/null | grep '^BENCH'
}
for rep in 1 2; do
  echo "## rep $rep" | tee -a "$RES"
  rm -rf "$WORK_DIR/bo-base-$rep" "$WORK_DIR/bo-sc-$rep" "$WORK_DIR/bo-rt-$rep"
  run "baseline"  ""                          "$WORK_DIR/bo-base-$rep" | tee -a "$RES"
  run "samecore"  "$WORK_DIR/spliced-stcdp"      "$WORK_DIR/bo-sc-$rep"   | tee -a "$RES"
  run "roundtrip" "$WORK_DIR/spliced-roundtrip"  "$WORK_DIR/bo-rt-$rep"   | tee -a "$RES"
done
echo "DONE" | tee -a "$RES"
