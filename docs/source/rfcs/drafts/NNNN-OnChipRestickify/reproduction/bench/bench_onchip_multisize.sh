#!/bin/bash
# baseline vs same-core vs round-trip latency across sizes 512/1024/2048/4096.
# Run SOLO (single shared accelerator). Writes BENCH lines to
# $WORK_DIR/bench_onchip_multisize.txt.
set +e
source "$(dirname "$0")/../env.sh"
RES=$WORK_DIR/bench_onchip_multisize.txt; : > "$RES"
run() {  # $1=size $2=spliced_dir(or empty) $3=cachedir
  env PYTHONPATH=$VAL_BOOT WORK_DIR=$WORK_DIR TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
      TORCHINDUCTOR_CACHE_DIR="$3" SPLICED_DIR="$2" BENCH_SIZE="$1" BENCH_WARMUP=15 BENCH_ITERS=60 \
      "$PYTHON" "$(dirname "$0")/bench_onchip.py" 2>/dev/null | grep '^BENCH'
}
for S in 512 1024 2048 4096; do
  if [ "$S" = "2048" ]; then SC=$WORK_DIR/spliced-stcdp; RT=$WORK_DIR/spliced-roundtrip;
  else SC=$WORK_DIR/spliced-stcdp-$S; RT=$WORK_DIR/spliced-roundtrip-$S; fi
  echo "## size $S" | tee -a "$RES"
  rm -rf "$WORK_DIR/ms-base-$S" "$WORK_DIR/ms-sc-$S" "$WORK_DIR/ms-rt-$S"
  run "$S" ""    "$WORK_DIR/ms-base-$S" | tee -a "$RES"
  run "$S" "$SC" "$WORK_DIR/ms-sc-$S"   | tee -a "$RES"
  run "$S" "$RT" "$WORK_DIR/ms-rt-$S"   | tee -a "$RES"
done
echo "DONE" | tee -a "$RES"
