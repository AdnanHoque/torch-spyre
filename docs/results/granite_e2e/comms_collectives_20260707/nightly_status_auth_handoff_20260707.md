# Nightly Status / Auth Handoff - 2026-07-07

## Goal Status

The active goal is still the DLDSC LX communication substrate for Granite:

- classify the in-scope copy communication edges;
- express them through DLDSC tensor/compute distribution metadata;
- prove Deeptools can realize bounded LX-resident tiles;
- keep oversized/full-streaming cases as WSR-owned rather than rebuilding WSR here.

This branch has made real progress toward that goal, but it is not complete.

## Remote Branch Map

As of this handoff:

| Repo | Branch | SHA | Role |
| --- | --- | --- | --- |
| `AdnanHoque/torch-spyre` | `ah/comms-collectives` | current branch head; at least `3532cf0426b7d32a353a0a8e65d7eaa6a53a17df` | Artifact branch with docs, patches, run payloads, and handoffs. |
| `AdnanHoque/torch-spyre` | `gather-restickify` | `7a18839f83d74d2c576f4c85585e11638d30c20b` | Torch prototype branch for flash/gather-restickify experiments. Adds planner-level tests for generic gather, broadcast, multicast, and all-gather relayout metadata. |
| `AdnanHoque/torch-spyre` | `pr-lx-relayout-scatter` | `ba365fe6234527e17558520ab41e21d8c6c696e2` | PR1 scatter/permutation Torch branch. |
| `AdnanHoque/torch-spyre` | `pr-lx-relayout-dldsc` | `a9a3bb505b966a3716d48854d1ecc22e46624476` | Older DLDSC/scatter exploration branch. |
| `Adnan-Hoque1/deeptools` | `ah/comms-collectives` | `3a4349e62baff978faa21b8cbad376a524658398` | Current Deeptools communication-collectives branch. |
| `Adnan-Hoque1/deeptools` | `gather-restickify` | `57c6f040b02ff592bc6cb207d9783375d2043d78` | Clean gather/restickify split branch. |
| `Adnan-Hoque1/deeptools` | `pr-lx-relayout-dldsc-scatter` | `b8c09743c46505b4cac46b434b9eb3243ae0b685` | PR1 scatter Deeptools branch. |

## Feature Flag State

The intended user-facing gate for the prototype is one flag:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

That flag is documented in:

```text
docs/results/granite_e2e/comms_collectives_20260706/antoni_minimal_flags_20260706.md
docs/results/granite_e2e/comms_collectives_20260706/portable_onegate_patches_20260707/README.md
```

For normal Torch-launched runs with the portable one-gate patches, do not ask
users to set the older per-feature flags. They remain debug/compatibility knobs.

For manual DXP replay that bypasses Torch, `DXP_LX_FRAC_AVAIL=1` is still needed
because Torch is not launching the DXP subprocess and cannot pass the backend
workspace setting.

For full Granite/full-LX benchmarking, the historical split setting is:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

Treat that as a capacity/runtime setup for reproducing large Granite runs, not
as a second feature flag.

## What Is Solid

Scatter/permutation is the production PR1 baseline.

Bounded all-gather/replicate is working through the staged
`gather_then_restickify` path. The saved full-flash DXP replay passed at
Deeptools `23010446e` with:

- return code `0`;
- `64` backend plan artifacts;
- no `wrong locale` failure;
- no IBUFF failure after the larger default chunk policy;
- all-gather/replicate plans lowered through `gather_then_restickify`.

The artifact is:

```text
docs/results/granite_e2e/comms_collectives_20260707/replay_payloads/artifact_payload_20260707_overnight/full_flash_dxp_replay_default_chunk_policy_20260707.tgz
```

Bounded broadcast and bounded multicast now have focused Deeptools unit
coverage at latest Deeptools head `3a4349e62`:

- focused DXP tests: `10/10` passed;
- `LayoutAllgatherRestickify.*`: `32/32` passed;
- new bounded broadcast and multicast fixtures are archived under:

```text
docs/results/granite_e2e/comms_collectives_20260707/bounded_broadcast_gather_restickify_20260707
docs/results/granite_e2e/comms_collectives_20260707/bounded_multicast_gather_restickify_20260707
```

## What Is Still Unverified

The latest Deeptools head is:

```text
3a4349e62baff978faa21b8cbad376a524658398
```

At that head, a saved full-flash DXP replay was started on CDX. It emitted the
expected `64` backend plan artifacts, but the process stayed silent in DXP/DCC
for several minutes:

- `backend_plans/`: `64` files;
- `post_sdsc/`: `0` files;
- `dxp.log`: `0` bytes;
- no early parser/classifier error;
- no `wrong locale`;
- no IBUFF message before interruption.

The run was interrupted manually and the run directory was not copied back
because local kube auth expired immediately afterward. Treat latest-head
full-flash replay as incomplete, not failed.

The note for that attempt is:

```text
docs/results/granite_e2e/comms_collectives_20260707/full_flash_replay_latest_3a4349e62_attempt_20260707.md
```

## Current Operational Blocker

Local Kubernetes/OpenShift auth expired:

```text
kubectl get pods -n a6-quantization
error: You must be logged in to the server (Unauthorized)
```

An `oc login --web` flow was started, but Chrome is currently parked at the IBM
w3id passkey page. The alternate-sign-in link did not move the page forward.
The waiting local `oc login` shell was stopped after the browser tabs were left
open.

Two Chrome tabs were left open as handoff tabs:

1. w3id passkey login page for the active `oc login --web` attempt;
2. OpenShift OAuth token page, which also needs a fresh login.

Once the user completes the passkey/login, start a fresh `oc login --web` or use
the OpenShift token page, then verify:

```bash
kubectl get pods -n a6-quantization
```

## Resume Command For Latest Full-Flash Replay

On CDX, the relevant checkouts were:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/torch-spyre
```

The archived bundle was copied to:

```text
/tmp/flash_saved_superdsc_bundle_20260707.tgz
```

After auth is refreshed, rerun with a timeout and archive the whole run:

```bash
cd /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools
git checkout ah/comms-collectives
git rev-parse HEAD

RUN=/tmp/lx_relayout_replay_latest_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN/bundle" "$RUN/backend_plans" "$RUN/post_sdsc"
tar -xzf /tmp/flash_saved_superdsc_bundle_20260707.tgz -C "$RUN/bundle"

export DXP_LX_FRAC_AVAIL=1
export SPYRE_LX_PLANNER_RELAYOUT=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$RUN/backend_plans"
export DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR="$RUN/backend_plans"
export DEEPTOOLS_DEBUG_RELAYOUT_SDSC_DIR="$RUN/post_sdsc"

timeout 900s build-deeptools/dxp/dxp_standalone -d "$RUN/bundle" \
  > "$RUN/dxp.log" 2>&1
echo $? > "$RUN/rc"
find "$RUN/backend_plans" -type f | wc -l > "$RUN/backend_plan_count"
find "$RUN/post_sdsc" -type f | wc -l > "$RUN/post_sdsc_count"
tar -czf "${RUN}.tgz" -C "$(dirname "$RUN")" "$(basename "$RUN")"
```

Expected baseline for comparison:

- at `23010446e`, this saved replay passed with return code `0`;
- at `3a4349e62`, it should ideally also pass, because the latest multicast
  commit is test-only and the bounded broadcast production change should not
  alter the all-gather/replicate carrier.

If the latest replay hangs or times out, compare `23010446e`, `071e293cf`, and
`3a4349e62` on the same saved bundle to isolate whether the slowdown comes from
the bounded broadcast production change or unrelated build/runtime drift.

## Local Source Sanity Check

After pod auth expired, the exact remote branches were shallow-cloned locally
and inspected.

Torch `gather-restickify` at `7a18839f` matches the one-gate artifact:

- `SPYRE_LX_PLANNER_RELAYOUT=1` enables the experimental subfeatures;
- the older `SPYRE_LX_PLANNER_RELAYOUT_*` flags are compatibility/debug
  overrides, not required user-facing gates;
- Torch defaults frontend `DXP_LX_FRAC_AVAIL` to `0` when the relayout flag is
  on;
- Torch defaults backend `DXP_BACKEND_LX_FRAC_AVAIL` to `1` for the DXP
  subprocess when the relayout flag is on.

Deeptools `ah/comms-collectives` at `3a4349e62` matches the latest artifact
claims:

- `SPYRE_LX_PLANNER_RELAYOUT=1` enables the staged gather/restickify path;
- `shouldUseMatmulOperandGatherRestickify(...)` selects
  `gather_then_restickify` before the loop-scoped/kernel-neighbor path;
- `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR` is not implicitly enabled
  by `SPYRE_LX_PLANNER_RELAYOUT`;
- latest commit `3a4349e62` is test-only;
- the production changes after the last green full-flash replay are
  `071e293` and `2301044`.

Recent Deeptools commits:

```text
3a4349e [DXP] test bounded multicast gather restickify
071e293 [DXP] enable bounded broadcast gather restickify
2301044 [DXP] honor tensor split contract for relayout cells
262b28c [DXP] tune matmul operand relayout chunking
cd30c2a [DXP] add bounded gather restickify relayout path
```

This makes the latest saved full-flash replay failure mode more likely to be a
run-completion/runtime issue than a classifier regression, because plan emission
had already produced the expected `64` artifacts and the unsafe loop-scoped path
is no longer selected by the public flag.

## Torch Planner-Level Test Update

After the source sanity check, the Torch `gather-restickify` branch was updated
to:

```text
7a18839f83d74d2c576f4c85585e11638d30c20b
```

The commit adds focused planner-level tests for generic copy collectives:

- `test_planner_records_generic_gather_relayout`
- `test_planner_records_generic_broadcast_relayout`
- `test_planner_records_generic_multicast_relayout`
- `test_planner_records_generic_all_gather_relayout`

These tests call `plan_lx_relayouts(...)` directly on fake computed-buffer
producer/consumer edges and assert that the recorded relayout payload contains
the expected communication class, pattern, fan-in/fan-out counts, transfer
count, and producer/consumer coordinate maps.

Local Mac checks:

```bash
python3 -m py_compile tests/inductor/test_lx_relayout_dldsc.py
python3 -m ruff check tests/inductor/test_lx_relayout_dldsc.py
git diff --check
```

all passed. A local pytest run still cannot execute on the Mac because this
shell does not have `torch` installed:

```text
ModuleNotFoundError: No module named 'torch'
```

Once pod auth is refreshed, run:

```bash
cd <torch-spyre-gather-restickify-checkout>
git fetch origin gather-restickify
git checkout gather-restickify
git reset --hard origin/gather-restickify
python -m pytest tests/inductor/test_lx_relayout_dldsc.py -q
```

## Remaining Goal Gaps

This work has not yet removed all non-weight Granite HBM spills.

The communication substrate covers or is close to covering:

- scatter/permutation;
- bounded all-gather/replicate with layout restickify;
- bounded broadcast/multicast fixtures.

Remaining gaps:

- full Granite runs still need fresh latest-head AIU/profile validation;
- any activation that is too large to hold as a resident full tensor should be
  WSR/tile-scoping work, not a private streaming relayout in this branch;
- reduce/all-reduce remain future arithmetic collectives, not copy relayout;
- flash numeric correctness is not a valid oracle because the baseline flash
  kernel currently has an independent broadcast/zero-stride correctness issue.
