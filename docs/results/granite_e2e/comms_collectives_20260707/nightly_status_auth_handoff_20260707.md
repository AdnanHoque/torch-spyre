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
| `AdnanHoque/torch-spyre` | `ah/comms-collectives` | `092e14d82e7ea4aedce67d209152eb5b4d835039` | Artifact branch with docs, patches, run payloads, and handoffs. |
| `AdnanHoque/torch-spyre` | `gather-restickify` | `bced14b49acf4fae92ef4df07d2f5229806c672b` | Torch prototype branch for flash/gather-restickify experiments. |
| `AdnanHoque/torch-spyre` | `pr-lx-relayout-scatter` | `ba365fe6234527e17558520ab41e21d8c6c696e2` | PR1 scatter/permutation Torch branch. |
| `AdnanHoque/torch-spyre` | `pr-lx-relayout-dldsc` | `a9a3bb505b966a3716d48854d1ecc22e46624476` | Older DLDSC/scatter exploration branch. |
| `Adnan-Hoque1/deeptools` | `ah/comms-collectives` | `3a4349e62baff978faa21b8cbad376a524658398` | Current Deeptools communication-collectives branch. |
| `Adnan-Hoque1/deeptools` | `gather-restickify` | `57c6f040b02ff592bc6cb207d9783375d2043d78` | Clean gather/restickify split branch. |
| `Adnan-Hoque1/deeptools` | `pr-lx-relayout-dldsc-scatter` | `b8c09743c46505b4cac46b434b9eb3243ae0b685` | PR1 scatter Deeptools branch. |

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

Two Chrome tabs were left open as handoff tabs:

1. w3id passkey login page for the active `oc login --web` attempt;
2. OpenShift OAuth token page, which also needs a fresh login.

Once the user completes the passkey/login, poll the waiting `oc login` shell if
it is still alive, then verify:

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
