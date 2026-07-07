# Comms Collectives Replay Payloads - 2026-07-07

These payloads make the latest all-gather/restickify checkpoint reproducible
without depending on the original pod cache path.

## Payloads

Directory:

`docs/results/granite_e2e/comms_collectives_20260707/replay_payloads/artifact_payload_20260707_overnight`

Files:

- `flash_saved_superdsc_bundle_20260707.tgz`: saved flash SuperDSC bundle used for the full DXP replay.
- `full_flash_dxp_replay_default_chunk_policy_20260707.tgz`: successful replay output, including full `dxp.log` and all backend plan artifacts.
- `broadcast_multicast_bounded_experiment_20260707.tgz`: failed bounded broadcast/multicast experiment and diff.
- `SHA256SUMS.txt`: checksums for the three tarballs.

## Expected Code State

- Torch branch: `gather-restickify`
- Torch SHA: `bced14b49acf4fae92ef4df07d2f5229806c672b`
- Deeptools branch: `ah/comms-collectives`
- Deeptools SHA: `262b28c05`

## Replay Command

From a Deeptools checkout at the SHA above:

```bash
WORK=/tmp/lx_relayout_replay_20260707
mkdir -p "$WORK"
tar -C "$WORK" -xzf flash_saved_superdsc_bundle_20260707.tgz

BUNDLE="$WORK/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_haegy4lc"
RUN="$WORK/replay"
mkdir -p "$RUN/backend_plans" "$RUN/post_sdsc"

export DEEPTOOLS_PATH=/path/to/deeptools
export DXP_LX_FRAC_AVAIL=1
export SPYRE_LX_PLANNER_RELAYOUT=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$RUN/backend_plans"
export DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR="$RUN/backend_plans"
export DEEPTOOLS_DEBUG_RELAYOUT_SDSC_DIR="$RUN/post_sdsc"

/path/to/deeptools/build-deeptools/dxp/dxp_standalone -d "$BUNDLE" \
  > "$RUN/dxp.log" 2>&1

echo "rc=$?"
find "$RUN/backend_plans" -type f | wc -l
```

Expected:

- return code `0`
- backend plan count `64`
- no `Max IBUFF`, `wrong locale`, or `ERROR` lines in `dxp.log`

## Unit Gates

```bash
export DXP_LX_FRAC_AVAIL=1

build-deeptools/dxp/dxp_unit_test \
  --gtest_filter="*CoreWorkDivIncomptLxRelayout*:*MatmulOperandBroadcast*:*PartialViewGather*"

build-deeptools/util/util_unit_test \
  --gtest_filter="LayoutAllgatherRestickify.*"
```

Expected:

- DXP focused tests: `8/8` pass
- util all-gather/restickify tests: `32/32` pass

## Broadcast/Multicast Experiment

`broadcast_multicast_bounded_experiment_20260707.tgz` records a deliberately
small experiment that tried to route `broadcast` and `multicast` through the
same staged `STCDPOpLx + ReStickifyOpLx` carrier used by all-gather.

Result:

- The code change is small and included as an experimental diff.
- Broadcast/multicast classification already exists.
- The current synthetic fixture is not a valid proof fixture for these classes:
  it rewrites the communication pattern while leaving source allocation and
  target tensor coordinates shaped like the all-gather case.
- The failing signal was an invalid source-address/buffer-offset path, not a
  parser/classifier failure.

Interpretation:

Broadcast/multicast remain in the gap list until we add a fixture whose source
tensor is actually resident for the fanout coordinates being requested, or until
Torch emits a real Granite/flash broadcast edge with the correct redundant
tensor distribution metadata.
