# DLDSC Comms Acceptance Gap Audit - 2026-07-07

This audit maps the active goal to current evidence. It is intentionally strict:
partial structural evidence is not counted as full completion when the goal asks
for a useful Granite/flash path.

## Current Branches

| Component | Branch | SHA |
| --- | --- | --- |
| Torch artifacts | `AdnanHoque/torch-spyre:ah/comms-collectives` | current artifact branch |
| Torch prototype | `AdnanHoque/torch-spyre:gather-restickify` | `102520820da890d6a62f781e86573f38dcc6f244` |
| Torch PR1 scatter | `AdnanHoque/torch-spyre:pr-lx-relayout-scatter` | `ba365fe6234527e17558520ab41e21d8c6c696e2` |
| Deeptools collectives | `Adnan-Hoque1/deeptools:ah/comms-collectives` | `a5ff55eee627c5c2bd4b7b0518bb0cbaad385952` |
| Deeptools PR1 scatter | `Adnan-Hoque1/deeptools:pr-lx-relayout-dldsc-scatter` | `b8c09743c46505b4cac46b434b9eb3243ae0b685` |

## Current Test Evidence

Latest acceptance validation at the current pushed heads is archived here:

```text
docs/results/granite_e2e/comms_collectives_20260707/latest_acceptance_validation_20260707/
```

Results:

```text
Torch tests/inductor/test_lx_relayout_dldsc.py: 38 passed in 6.70s
Deeptools LayoutAllgatherRestickify.*: 32 passed
Deeptools DXP focused comms tests: 10 passed
```

The focused DXP set includes bounded broadcast, bounded multicast,
partial-view gather, malformed partial-view gather fail-closed guards,
oversized matmul-operand relayout fail-closed guards, and core-work-division
LX relayout.

The full Torch relayout test file now passes on the CDX pod at the current
`gather-restickify` head:

```bash
python3 -m pytest -q tests/inductor/test_lx_relayout_dldsc.py
```

Result:

```text
38 passed in 3.82s
```

Archived output:

```text
docs/results/granite_e2e/comms_collectives_20260707/latest_head_validation_20260707/torch_full_lx_relayout_test_20260707.txt
```

Older DEV-pod validation still documents static/import-light checks and the
local DEV `_C.so` / `libspyre_comms.so.1` ABI mismatch:

```text
docs/results/granite_e2e/comms_collectives_20260707/latest_head_validation_20260707/torch_gather_restickify_validation_summary_devpf.md
```

## Acceptance Matrix

| Requirement | Current Evidence | Status |
| --- | --- | --- |
| Classify scatter/permutation | PR1 scatter branches and earlier scatter artifacts. | Complete for PR1 class. |
| Classify all-gather/replicate | `flash_allgather_failclosed_checkpoint_20260707.md` records `all_gather_replicate` plans; Granite S512 checkpoint records two attention RHS all-gather/replicate handoffs; latest-head-equivalent saved full-flash DXP replay emits 64 all-gather/replicate plans. | Complete structurally for the current flash/attention class. |
| Classify broadcast/multicast | Latest Deeptools fixtures generate bounded broadcast/multicast test coverage; Torch full relayout test file now passes planner-level and bundle/codegen DLDSC tests for both classes; regenerated bounded broadcast/multicast artifacts show gather-then-restickify lowering. | Complete for bounded fixtures and generic graph/planner/codegen metadata. No current Granite/flash useful-workload edge naturally classifies as bounded broadcast/multicast; current useful workload evidence is all-gather/replicate. |
| Classify gather | `tests/inductor/test_lx_relayout_dldsc.py` now passes full-file validation including topology classification, generic DLDSC emission, and partial-view gather bundle enrichment; latest Deeptools focused DXP tests compile bounded partial-view gather and fail closed for missing/invalid offsets. | Complete for bounded metadata/codegen and bounded DXP compile. Useful-workload value proof remains separate from this substrate checkpoint. |
| Torch emits DLDSC metadata for scatter | PR1 branch emits tensor-vs-compute distribution metadata consumed by Deeptools scatter relayout. | Complete for scatter. |
| Torch emits DLDSC metadata for all-gather/replicate | Torch `gather-restickify` emits `matmul_operand_broadcast`/`all_gather_replicate` metadata for flash and Granite attention RHS handoffs. | Complete for current attention/flash class. |
| Torch emits DLDSC metadata for broadcast/multicast | `tests/inductor/test_lx_relayout_dldsc.py` passes planner-level fake-graph tests and generic broadcast/multicast DLDSC emission tests through `compile_op_spec`. | Complete for generic planner/codegen metadata emission. |
| Deeptools realizes bounded scatter | PR1 Deeptools branch and unit/device evidence. | Complete for bounded scatter. |
| Deeptools realizes bounded all-gather/replicate | Saved full-flash DXP replay passes at current Deeptools head `a5ff55eee` with `rc=0`, 64 backend plans, all `all_gather_replicate -> gather_then_restickify`. | Complete for bounded saved flash replay. |
| Deeptools realizes bounded broadcast | Focused DXP/util tests pass, Deeptools `2ccd5ce` fixes one diagnostic cause of stale `stages`, `320630da` adds a guard test, and regenerated plan artifact cleanly shows `broadcast -> gather_then_restickify`. | Complete for bounded fixture. |
| Deeptools realizes bounded multicast | Focused DXP/util tests pass at `3a4349e62`; bounded plan artifact archived. | Complete for bounded fixture. |
| Deeptools realizes bounded gather | Latest focused DXP tests pass `PartialViewGatherBoundedOffsetRelayoutCompiles` and the missing/invalid source-offset fail-closed guards. | Complete for bounded partial-view gather. |
| Unsupported or oversized cases fail closed/fallback | Fail-closed docs record direct broadcast/multicast wrong-locale avoidance, IBUFF boundary, chunk/cap behavior, and the Granite full-activation 1 MiB Torch fallback policy. Focused Deeptools negative tests pass in `MatmulOperandBroadcast*FailsClosed`. | Complete for current bounded-substrate scope. |
| Artifacts identify removed Granite HBM spills | Granite S512 checkpoint records disabled/enabled SDSC counts and maps `sdsc_7` and `sdsc_15 -> sdsc_16` activation handoffs to on-chip movement. | Complete for the profiled `backend2162` run; needs refresh at latest branches. |
| Artifacts identify remaining Granite HBM spills | Granite S512 checkpoint classifies remaining explicit `ReStickifyOpHBM` rows as weight-format rows. | Complete for that run. Keep weight restickifies out of comms scope. |
| Flash attention used as validation | Flash structural runs show zero `ReStickifyOpHBM` and 32/64 backend plans depending on fixture. | Complete structurally. Flash value correctness remains out of scope because baseline is independently value-wrong. |
| Reduce/all-reduce scoped separately | Docs consistently classify reduce/all-reduce as future arithmetic collectives. | Complete as a scope decision, not implementation. |
| Latest current-head Granite spill classification | Current-head Granite S512 was rerun on CDX. Clean FMS blocked before SDSC on element-arrangement mismatch. Pinned FMS emitted SDSCs, then stopped at runtime on `convert_address not yet implemented`; no backend relayout plans were emitted. The generated SDSC table has five explicit `ReStickifyOpHBM` ops: four weight/kernel prelayout rows and one non-weight attention value-side activation handoff. | Partially complete. Bounded comms substrate is green, but current-head full Granite still has one non-weight activation HBM handoff, classified as WSR/loop-scoped collectives boundary. |

## What Is Actually Closed

1. **Scatter/permutation PR1**: the production-shaped direct relayout lane exists.
2. **All-gather/replicate with layout restickify**: bounded and saved full-flash
   DXP evidence exists, with the old IBUFF failure addressed by chunk policy.
3. **Bounded broadcast/multicast backend fixtures**: latest Deeptools branch has
   green focused tests. The multicast artifact is clean, and the regenerated
   broadcast artifact at `320630da`-equivalent head now cleanly shows
   `broadcast -> gather_then_restickify`.
4. **Torch generic collectives metadata/codegen**: current `gather-restickify`
   passes the full `tests/inductor/test_lx_relayout_dldsc.py` file, including
   scatter, gather, broadcast, multicast, all-gather, matmul operand contracts,
   and partial-view gather bundle enrichment.
5. **Bounded partial-view gather**: latest Deeptools DXP focused tests compile
   the bounded partial-view gather fixture and verify missing/invalid
   source-offset metadata fails closed.
6. **One public feature gate**: `SPYRE_LX_PLANNER_RELAYOUT=1` is the intended
   user-facing flag. Capacity knobs are runtime setup, not feature gates.
7. **Plan artifact stage consistency**: Deeptools `2ccd5ce` refreshes emitted
   `matmul_operand_broadcast` artifact stages after selecting the physical
   carrier, and `320630da` adds a focused test guard. This is observability
   cleanup, not new lowering functionality.
8. **Latest bounded-substrate unit evidence**: current pushed heads pass the
   full Torch relayout test file and the focused Deeptools comms fixtures:
   `38/38` Torch tests, `32/32` util all-gather/restickify tests, and `10/10`
   focused DXP comms tests.

## What Is Not Yet Closed

1. **Useful-workload broadcast/multicast**:
   - Generic Torch planner/codegen tests cover broadcast and multicast.
   - Deeptools bounded fixtures are green.
   - Current useful Granite/flash in-scope edges observed so far classify as
     all-gather/replicate or oversized all-gather/replicate, not bounded
     broadcast/multicast. If a later Granite/WSR tile exposes natural bounded
     broadcast/multicast, reuse this substrate and add that workload artifact.

2. **Fresh Granite run at latest heads**:
   - Current-head Granite S512 has now been rerun structurally on CDX.
   - With the clean Codex FMS checkout, both disabled/enabled variants fail
     before SDSC generation on an element-arrangement mismatch in `mul`.
   - With the older known-good pinned FMS checkout, SDSCs are generated, then
     execution stops on `convert_address not yet implemented`.
   - The pinned-FMS SDSC evidence shows no backend plans at the current bounded
     head and one remaining non-weight activation `ReStickifyOpHBM`:
     attention value-side activation handoff into the value-side BMM operand.
   - The other explicit `ReStickifyOpHBM` ops in that run are weight/kernel
     prelayout rows and remain out of scope.
   - The strongest Granite speed proof remains the older `backend2162` S512
     run. It should not be presented as current-head completion evidence.

3. **WSR boundary**:
   - Oversized full-tensor activation materialization should fail closed or fall
     back to HBM.
   - The comms branch should not build private streaming. It should identify
     the communication class and prove bounded resident tiles.
   - The latest Granite structural run confirms this boundary: bounded
     collectives are green, but full Granite still needs WSR or loop-scoped
     tile execution to remove the remaining large attention activation handoff
     at current heads.

## Next Device Validation Order

Use this order for the next device validation pass:

1. **Flash Python compile probe**
   - Structural validation only.
   - Expected: compile/smoke success, `ReStickifyOpHBM=0`, `ReStickifyOpLx>0`,
     backend plans present.
   - Do not use numeric mismatch as a failure of comms unless baseline is fixed.

2. **Granite S512 one-layer causal prefill**
   - Use empty/fake weights and one public relayout flag.
   - Expected: non-weight attention handoffs on LX; remaining explicit HBM rows
     should be weight-format rows or WSR-scoped oversize cases.

3. **Broadcast/multicast graph fixture**
   - If no real Granite edge appears, add a small full-graph bounded synthetic
     fixture so planner-produced SDSC metadata is proven, not only hand-built
     `OpSpec` metadata and backend fixtures.
   - The bounded backend broadcast artifact is now regenerated and clean; this
     remaining item is about a graph-produced workload edge.

## Completion Bar

Do not mark the goal complete until latest Granite artifacts identify every
remaining non-weight HBM spill as either removed, a communication-substrate gap,
or a WSR/tile-scoping gap.

Broadcast/multicast are no longer the strict blocker for this checkpoint:
generic Torch planner/codegen coverage is green, Deeptools bounded fixtures are
green. Gather is also no longer a strict blocker for the bounded substrate:
Torch generic/partial-view metadata is green, and Deeptools has a bounded
partial-view gather DXP compile proof plus fail-closed guards. Current
Granite/flash useful edges observed so far are all-gather/replicate or
oversized all-gather/replicate rather than bounded broadcast/multicast.

Latest Granite status: current-head bounded substrate is not enough by itself to
remove all Granite non-weight HBM spills. The pinned-FMS current-head SDSCs leave
one non-weight attention activation handoff in HBM. That handoff is classified
as a WSR/loop-scoped all-gather/restickify gap, not a missing bounded
scatter/broadcast/gather primitive.
