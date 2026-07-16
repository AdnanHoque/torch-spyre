#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/adnan-cdx/codex-isolated/explicit-shuffle-contract-20260715-fix
REPO=$ROOT/deeptools
RUN=$ROOT/replay/variant-a-bounded-v2/final-exact-backend-20260716

cmake --build "$ROOT/build" --target dxp_standalone -j 16 \
  >"$ROOT/logs/build-final-bounded-shuffle.log" 2>&1

rm -rf "$RUN"
mkdir -p "$RUN/fixture"
cp "$ROOT/replay/variant-a-bounded-v2/fixture/sdsc_0.json" "$RUN/fixture/"
cp "$ROOT/replay/variant-a-bounded-v2/manual-debug-20260716T055917Z/fixture/bundle.mlir" \
  "$RUN/fixture/"

DXP_DEBUG=1 DXP_VERBOSE=1 "$ROOT/build/dxp/dxp_standalone" \
  -d "$RUN/fixture" -b sentient \
  >"$RUN/replay.stdout.log" 2>"$RUN/replay.stderr.log"
echo "$?" >"$RUN/replay.rc"

POST=$RUN/fixture/debug/sdsc_0/sdsc_0.out.out.out.json
python3 "$ROOT/tools/verify_variant_a_bounded.py" \
  --sdsc "$POST" \
  --json-out "$RUN/verification.json" \
  --markdown-out "$RUN/verification.md"

python3 "$ROOT/tools/verify_variant_a_order.py" \
  --integration-dir "$ROOT/replay/variant-a-bounded-v2/integration-order-fixture" \
  --authoritative-shuffle "$ROOT/replay/variant-a-bounded-v2/fixture/sdsc_0.json" \
  --lowered-sdsc "$POST" \
  --json-out "$RUN/ordering-verification.json" \
  --markdown-out "$RUN/ordering-verification.md"

git -C "$REPO" diff --check
