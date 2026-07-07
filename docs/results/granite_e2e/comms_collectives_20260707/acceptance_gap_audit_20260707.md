# DLDSC Comms Acceptance Gap Audit - 2026-07-07

This audit maps the active goal to current evidence. It is intentionally strict:
partial structural evidence is not counted as full completion when the goal asks
for a useful Granite/flash path.

## Current Branches

| Component | Branch | SHA |
| --- | --- | --- |
| Torch artifacts | `AdnanHoque/torch-spyre:ah/comms-collectives` | current artifact branch |
| Torch prototype | `AdnanHoque/torch-spyre:gather-restickify` | `bced14b49acf4fae92ef4df07d2f5229806c672b` |
| Torch PR1 scatter | `AdnanHoque/torch-spyre:pr-lx-relayout-scatter` | `ba365fe6234527e17558520ab41e21d8c6c696e2` |
| Deeptools collectives | `Adnan-Hoque1/deeptools:ah/comms-collectives` | `3a4349e62baff978faa21b8cbad376a524658398` |
| Deeptools PR1 scatter | `Adnan-Hoque1/deeptools:pr-lx-relayout-dldsc-scatter` | `b8c09743c46505b4cac46b434b9eb3243ae0b685` |

## Acceptance Matrix

| Requirement | Current Evidence | Status |
| --- | --- | --- |
| Classify scatter/permutation | PR1 scatter branches and earlier scatter artifacts. | Complete for PR1 class. |
| Classify all-gather/replicate | `flash_allgather_failclosed_checkpoint_20260707.md` records `all_gather_replicate` plans; Granite S512 checkpoint records two attention RHS all-gather/replicate handoffs. | Complete structurally; latest-head full replay still needs re-run. |
| Classify broadcast/multicast | Latest Deeptools fixtures generate `broadcast` and `multicast` plans under `bounded_broadcast_gather_restickify_20260707` and `bounded_multicast_gather_restickify_20260707`. | Complete for bounded synthetic fixtures; not yet proven on a real Granite/flash edge. |
| Classify gather | Partial-view gather artifacts and offset guards exist under `partial_view_gather_*` and `fanout_physical_fixture_probe_20260707`. | Partial. Needs clearer useful-workload edge or a focused bounded value test. |
| Torch emits DLDSC metadata for scatter | PR1 branch emits tensor-vs-compute distribution metadata consumed by Deeptools scatter relayout. | Complete for scatter. |
| Torch emits DLDSC metadata for all-gather/replicate | Torch `gather-restickify` emits `matmul_operand_broadcast`/`all_gather_replicate` metadata for flash and Granite attention RHS handoffs. | Complete for current attention/flash class. |
| Torch emits DLDSC metadata for broadcast/multicast | The current evidence is mostly backend synthetic fixtures, not Torch-emitted useful workload SDSCs. | Incomplete. Need a Torch-side bounded real/synthetic producer of broadcast/multicast metadata. |
| Deeptools realizes bounded scatter | PR1 Deeptools branch and unit/device evidence. | Complete for bounded scatter. |
| Deeptools realizes bounded all-gather/replicate | Saved full-flash DXP replay passed at `23010446e`; bounded M16 all-gather replay passed; chunk policy fixed IBUFF. | Complete at `23010446e`; latest head `3a4349e62` needs replay completion. |
| Deeptools realizes bounded broadcast | Focused DXP/util tests pass at `071e293cf`; bounded plan artifact archived. | Complete for bounded fixture. |
| Deeptools realizes bounded multicast | Focused DXP/util tests pass at `3a4349e62`; bounded plan artifact archived. | Complete for bounded fixture. |
| Unsupported or oversized cases fail closed/fallback | Fail-closed docs record direct broadcast/multicast wrong-locale avoidance, IBUFF boundary, and chunk/cap behavior. | Mostly complete; needs a single current-head negative test summary after latest broadcast/multicast changes. |
| Artifacts identify removed Granite HBM spills | Granite S512 checkpoint records disabled/enabled SDSC counts and maps `sdsc_7` and `sdsc_15 -> sdsc_16` activation handoffs to on-chip movement. | Complete for the profiled `backend2162` run; needs refresh at latest branches. |
| Artifacts identify remaining Granite HBM spills | Granite S512 checkpoint classifies remaining explicit `ReStickifyOpHBM` rows as weight-format rows. | Complete for that run. Keep weight restickifies out of comms scope. |
| Flash attention used as validation | Flash structural runs show zero `ReStickifyOpHBM` and 32/64 backend plans depending on fixture. | Complete structurally. Flash value correctness remains out of scope because baseline is independently value-wrong. |
| Reduce/all-reduce scoped separately | Docs consistently classify reduce/all-reduce as future arithmetic collectives. | Complete as a scope decision, not implementation. |

## What Is Actually Closed

1. **Scatter/permutation PR1**: the production-shaped direct relayout lane exists.
2. **All-gather/replicate with layout restickify**: bounded and saved full-flash
   DXP evidence exists, with the old IBUFF failure addressed by chunk policy.
3. **Bounded broadcast/multicast backend fixtures**: latest Deeptools branch has
   green focused tests and archived plan artifacts.
4. **One public feature gate**: `SPYRE_LX_PLANNER_RELAYOUT=1` is the intended
   user-facing flag. Capacity knobs are runtime setup, not feature gates.

## What Is Not Yet Closed

1. **Latest-head full-flash replay**:
   - Last green full replay is at Deeptools `23010446e`.
   - Latest Deeptools head `3a4349e62` emitted the expected `64` plans but was
     interrupted before DXP completion.
   - This needs one timed replay after pod auth is refreshed.

2. **Torch-emitted broadcast/multicast useful workload**:
   - Deeptools bounded fixtures are green.
   - We still need a Torch-emitted SDSC edge that naturally classifies as
     bounded broadcast or bounded multicast, or an explicit note that the
     current Granite/flash in-scope edges are all-gather/replicate rather than
     broadcast/multicast.

3. **Gather useful workload/value boundary**:
   - Partial-view gather metadata and guards exist, but the evidence is less
     strong than all-gather/replicate.
   - Add a bounded patterned value test or point to a Granite edge where a
     gather-like copy is actually realized.

4. **Fresh Granite run at latest heads**:
   - The strongest Granite speed/SDSc proof is the `backend2162` S512 run.
   - Current comms head has moved since then. Once auth is back, rerun Granite
     S512 with one-gate setup and compare:
     - explicit `ReStickifyOpHBM`;
     - `ReStickifyOpLx`;
     - backend plan count and plan classes;
     - kernel time and wall time.

5. **WSR boundary**:
   - Oversized full-tensor activation materialization should fail closed or fall
     back to HBM.
   - The comms branch should not build private streaming. It should identify
     the communication class and prove bounded resident tiles.

## Next Device Validation Order

When OpenShift auth is restored, use this order:

1. **Saved full-flash DXP replay at latest Deeptools head**
   - No AIU needed, fastest compiler check.
   - Expected: `rc=0`, `64` backend plans.
   - If it times out, bisect `23010446e -> 071e293cf -> 3a4349e62`.

2. **Flash Python compile probe**
   - Structural validation only.
   - Expected: compile/smoke success, `ReStickifyOpHBM=0`, `ReStickifyOpLx>0`,
     backend plans present.
   - Do not use numeric mismatch as a failure of comms unless baseline is fixed.

3. **Granite S512 one-layer causal prefill**
   - Use empty/fake weights and one public relayout flag.
   - Expected: non-weight attention handoffs on LX; remaining explicit HBM rows
     should be weight-format rows or WSR-scoped oversize cases.

4. **Broadcast/multicast Torch fixture**
   - If no real Granite edge appears, add a small Torch-generated bounded
     synthetic fixture so the frontend metadata path is proven, not only the
     Deeptools backend path.

## Completion Bar

Do not mark the goal complete until:

- latest-head all-gather/replicate replay is green or its failure is classified
  as an explicit fail-closed capacity/WSR boundary;
- bounded broadcast/multicast have both backend and Torch-emitted metadata
  evidence, or are explicitly downgraded out of required Granite scope;
- gather has either a useful-workload proof or a bounded value-oriented proof;
- latest Granite artifacts identify every remaining non-weight HBM spill as
  either removed, comm-substrate gap, or WSR/tile-scoping gap.

