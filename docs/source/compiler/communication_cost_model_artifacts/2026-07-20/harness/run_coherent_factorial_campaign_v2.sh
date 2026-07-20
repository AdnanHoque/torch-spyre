#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/adnan/codex-isolated/joint_oracle_factorial_20260720_v1
PY=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
RUN_BLOCK=$ROOT/scripts/run_coherent_factorial_block_v2.sh
ANALYZER=$ROOT/scripts/analyze_coherent_oracle_factorial_v2.py
VERIFY=$ROOT/scripts/verify_dxp_closure.py
PREREG=$ROOT/PREREGISTRATION_V2.txt
OUT=$ROOT/results/coherent_factorial_20260720_v2
STRUCTURAL_DEFAULT=$ROOT/results/factorial_20260720_v1/structural_default
STRUCTURAL_COHERENT=$ROOT/results/coherent_structural_20260720_v1
VCODED=$ROOT/results/coherent_vcoded_20260720_v1/report.json

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 98; }
[[ -f "$PREREG" ]] || { echo "missing V2 preregistration" >&2; exit 2; }
[[ -f "$STRUCTURAL_DEFAULT/STRUCTURAL_SUCCESS.json" ]] || exit 2
[[ -f "$STRUCTURAL_COHERENT/STRUCTURAL_SUCCESS.json" ]] || exit 2
[[ -f "$ROOT/results/coherent_vcoded_20260720_v1/VCODED_SUCCESS.json" ]] || exit 2
[[ -z $(git -C "$ROOT/torch-spyre" status --porcelain) ]] || {
  echo "combined Torch tree is dirty" >&2; exit 2;
}
git -C "$ROOT/torch-spyre" merge-base --is-ancestor \
  2a20cf3b7ac8aadf629314e40e5059ad82471911 HEAD
[[ $(git -C "$ROOT/deeptools" rev-parse HEAD) == \
  19280fd7c6bbd91000c63c2a6719a0253e513f4a ]] || exit 2

mkdir -p "$OUT"
cp "$PREREG" "$OUT/preregistration.txt"
"$PY" "$VERIFY" "$ROOT/dxp-closure" --out "$OUT/closure_before.json" \
  > "$OUT/closure_before_console.json"

finalize() {
  original_rc=$?
  trap - EXIT
  set +e
  "$PY" "$VERIFY" "$ROOT/dxp-closure" --out "$OUT/closure_after.json" \
    > "$OUT/closure_after_console.json"
  closure_rc=$?
  compare_rc=99
  if [[ $closure_rc -eq 0 ]]; then
    "$PY" - "$OUT/closure_before.json" "$OUT/closure_after.json" <<'PY'
import json, sys
a, b = (json.load(open(path)) for path in sys.argv[1:])
for key in ("pass", "manifest_sha256", "files", "loader_returncode", "errors"):
    if a[key] != b[key]:
        raise SystemExit(f"closure mismatch: {key}")
PY
    compare_rc=$?
  fi
  effective_rc=$original_rc
  if [[ $effective_rc -eq 0 && $closure_rc -ne 0 ]]; then effective_rc=$closure_rc; fi
  if [[ $effective_rc -eq 0 && $compare_rc -ne 0 ]]; then effective_rc=$compare_rc; fi
  "$PY" - "$OUT" "$original_rc" "$closure_rc" "$compare_rc" "$effective_rc" <<'PY'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
report = out / "factorial_report.json"
status = {
    "gate": "pass" if int(sys.argv[5]) == 0 else "fail",
    "campaign_body_exit_code": int(sys.argv[2]),
    "closure_after_exit_code": int(sys.argv[3]),
    "closure_compare_exit_code": int(sys.argv[4]),
    "effective_exit_code": int(sys.argv[5]),
    "factorial_report_sha256": (
        hashlib.sha256(report.read_bytes()).hexdigest() if report.exists() else None
    ),
}
(out / "TERMINAL_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
PY
  exit "$effective_rc"
}
trap finalize EXIT

run_block() {
  block=$1
  order=$2
  FACTORIAL_BLOCK=$block FACTORIAL_ORDER=$order \
    FACTORIAL_OUT="$OUT/block_$block" bash "$RUN_BLOCK"
}

run_block 1 lx_default,lx_coherent,oracle_coherent,oracle_default
run_block 2 lx_coherent,oracle_default,lx_default,oracle_coherent
run_block 3 oracle_default,oracle_coherent,lx_coherent,lx_default
run_block 4 oracle_coherent,lx_default,oracle_default,lx_coherent
run_block 5 lx_default,lx_coherent,oracle_default,oracle_coherent

"$PY" "$ANALYZER" "$OUT" "$STRUCTURAL_DEFAULT" \
  "$STRUCTURAL_COHERENT" "$VCODED"

