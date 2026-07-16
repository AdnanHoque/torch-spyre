#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ROOT VARIANT RUN_LABEL" >&2
  exit 2
fi

root=$1
variant=$2
run_label=$3
run_dir="$root/replay/$run_label/$variant"
fixture_dir="$root/replay/fixtures/$variant"

rm -rf "$run_dir"
mkdir -p "$run_dir"
cp "$fixture_dir/sdsc_0.json" "$run_dir/"
cp "$fixture_dir/bundle.mlir" "$run_dir/"
printf '%q ' env -u FLEX_COMPUTE DXP_DEBUG=1 DUMP_SPYRE_CODE=1 \
  "DEEPTOOLS_PATH=$root/source/deeptools" \
  "$root/build/deeptools/dxp/dxp_standalone" \
  -d . -b sentient --dump-bundle-module > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"

cd "$run_dir"
set +e
env -u FLEX_COMPUTE DXP_DEBUG=1 DUMP_SPYRE_CODE=1 \
  "DEEPTOOLS_PATH=$root/source/deeptools" \
  "$root/build/deeptools/dxp/dxp_standalone" \
  -d . -b sentient --dump-bundle-module >stdout.log 2>stderr.log
rc=$?
set -e
printf '%s\n' "$rc" > return_code.txt
find . -maxdepth 4 -type f -printf '%P\n' | sort > file_inventory.txt
exit 0
