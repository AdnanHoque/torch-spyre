# Full Flash Saved-Bundle Replay Attempt At Deeptools 3a4349e62

Date: 2026-07-07

This note records a compiler-only DXP replay attempt of the archived full flash
SuperDSC bundle against the latest Deeptools communication-collectives branch.

## Code State

- Torch source bundle: archived
  `flash_saved_superdsc_bundle_20260707.tgz`
- Deeptools branch: `Adnan-Hoque1/deeptools:ah/comms-collectives`
- Deeptools commit: `3a4349e62baff978faa21b8cbad376a524658398`

## Command Shape

```bash
export DXP_LX_FRAC_AVAIL=1
export SPYRE_LX_PLANNER_RELAYOUT=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$RUN/backend_plans"
export DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR="$RUN/backend_plans"
export DEEPTOOLS_DEBUG_RELAYOUT_SDSC_DIR="$RUN/post_sdsc"

build-deeptools/dxp/dxp_standalone -d "$BUNDLE" > "$RUN/dxp.log" 2>&1
```

CDX paths during the attempt:

```text
/tmp/lx_relayout_replay_latest_20260707_172841
/tmp/lx_relayout_replay_latest_20260707_172841/replay_3a4349e62
```

## Observed Signal

Before interruption:

- DXP process was active at about `99%` CPU for several minutes.
- `backend_plans/` contained `64` plan artifacts.
- `post_sdsc/` contained `0` files.
- `dxp.log` was still `0` bytes.
- No early `Max IBUFF`, `wrong locale`, or parser/classifier error was emitted.

The process was manually interrupted with Ctrl-C after the long silent run:

```text
interrupted_rc = 130
```

## Interpretation

This is not evidence that the communication classifier regressed. The replay
had already emitted the expected `64` backend plan artifacts, matching the
previous saved-bundle all-gather run.

The open question is whether latest DXP/DCC now takes substantially longer, or
whether the replay was stuck after plan emission. Because the run was
interrupted and the pod auth expired before the run directory could be copied
back into the artifact branch, treat this as an incomplete validation attempt.

## Next Step

Re-run the same saved-bundle DXP replay when pod auth is refreshed. Capture:

- return code;
- wall time;
- `backend_plans` count;
- `dxp.log`;
- any post-relayout SDSCs;
- the complete run directory tarball.

The bounded backend-unit status remains green at `3a4349e62`:

- DXP focused tests: `10/10` passed.
- `LayoutAllgatherRestickify.*`: `32/32` passed.
