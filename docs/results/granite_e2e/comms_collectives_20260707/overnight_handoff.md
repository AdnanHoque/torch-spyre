# Granite LX communication collectives overnight handoff

Date: 2026-07-07

## Latest checkpoint

See `overnight_status_20260707_late.md`,
`fanout_physical_fixture_probe_20260707/README.md`, and
`bounded_broadcast_gather_restickify_20260707/README.md`, and
`bounded_multicast_gather_restickify_20260707/README.md` for the latest state.
The latest full saved-flash replay attempt is recorded in
`full_flash_replay_latest_3a4349e62_attempt_20260707.md`.

The current Deeptools branch is green at `3a4349e62` for the focused bounded
cases:

- `DxpTestFixture.CoreWorkDivIncomptLxRelayout*`
- `DxpTestFixture.MatmulOperandBroadcastChunkCapFailsClosed`
- `DxpTestFixture.MatmulOperandBroadcastPattern*FailsClosed`
- `DxpTestFixture.MatmulOperandBroadcastPatternBroadcastGatherRestickifyCompiles`
- `DxpTestFixture.MatmulOperandBroadcastPatternMulticastGatherRestickifyCompiles`
- `DxpTestFixture.PartialViewGather*`
- `LayoutAllgatherRestickify.*`

The important clarification from the late probe is:

- generic DLDSC core-work-div relayout already covers bounded copy-cardinality
  changes such as full producer to sliced consumers and sliced producers to one
  full consumer;
- flash/attention all-gather plus layout conversion is green through the staged
  `STCDPOpLx + ReStickifyOpLx` path;
- bounded broadcast and multicast plus layout conversion before BMM now have
  positive Deeptools tests using the same staged carrier;

This note records the current state of the Granite non-weight HBM spill removal work. The goal is to build the DLDSC LX communication substrate for Granite without duplicating WSR: classify communication edges in Torch, emit a compact DLDSC coordinate contract, and let Deeptools realize bounded LX-resident movement. Large full-tensor streaming remains WSR-owned.

## Current progress

- **Scatter / permutation**: PR1 class is effectively done. Torch emits the tensor-vs-compute coordinate contract and Deeptools can realize bounded scatter/permutation relayout.
- **Flash structural all-gather/gather-restickify**: the larger CDX prototype reaches the desired structural result on the flash probe: `ReStickifyOpHBM: 0`, `ReStickifyOpLx: 64`, `matmul_operand_broadcast: 32`, backend plans all `gather_then_restickify`.
- **Bounded correctness**: old green bounded M16 probe previously passed with `ALLCLOSE True`, `MAX_DIFF 0.001953125`, `MISMATCH 0 / 4096`.
- **Current blocker**: real generated Torch/Granite SDSC validation is still
  needed for the new fanout fixtures. Full Granite spill removal still needs
  WSR/tile-scoping for oversized
  live activations, and reduce/all-reduce remain separate arithmetic
  communication classes.

## Historical A/B Evidence

Earlier in this investigation, the exact same Torch-generated SDSC bundle was
replayed through two backends:

| Backend | Result | Meaning |
|---|---:|---|
| Earlier slim/diagnostic backend | DXP rc `134`, `wrong locale for dst operand` | that backend lowering stack was incomplete |
| Old green backend wrapper | DXP rc `0` | the SDSC/Inductor contract is sufficient for this bounded case |

Current failing DXP replay:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/dxp_replay_after_lxsu_consumer_20260707_130244
```

Old green passing DXP replay of the same bundle:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/dxp_replay_oldgreen_backend_same_bundle_20260707_130308
```

The shared SDSC bundle used for the A/B was:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/current_restored_sdsc_relayout_M16_N256_20260707_125927/cache/inductor-spyre/sdsc_fused_mm_mul_0_17j5_fux
```

## What This Means

The remaining gap is not "does Torch know which communication class is needed?"
For the bounded matmul operand broadcast case, Torch emits enough metadata for
Deeptools to plan the movement.

At `071e293cf`, the Deeptools branch now has a bounded broadcast positive test
for the staged `STCDPOpLx + ReStickifyOpLx` carrier. The old `wrong locale`
failure is still useful history, but it is no longer the latest status for the
bounded broadcast fixture.

At `3a4349e62`, the same branch adds a bounded multicast positive test for the
same carrier. That commit is test-only relative to the broadcast production
gate.

The old green backend patch stack contains two families of changes:

1. **Materialized gather/restickify path**: emit `STCDPOpLx` gather into a temporary LX tensor, then `ReStickifyOpLx` into the consumer KERNEL layout before compute.
2. **DL transfer-node/local-stage path**: represent ring transfers and local restickify stages directly in the schedule tree so DCC can lower L3 ring movement plus local LXLU/LXSU copies.

The first path has prior bounded value correctness and now has a bounded
broadcast and multicast compiler-positive tests. The second path is a better
long-term architecture, but still needs careful slimming and validation.

## Runs And Logs Archived Here

- `current_backend_m16_n256_wrong_locale.log`: current backend full probe, reaches DCC and aborts.
- `current_backend_dxp_replay_wrong_locale.log`: DXP-only replay with current backend, same abort.
- `oldgreen_backend_same_bundle_dxp_replay_pass.log`: DXP-only replay with old green backend, passes.
- `current_torch_oldgreen_backend_runtime_hardware_error.log`: current Torch + old green backend reached runtime, then CDX AIU threw a hardware scheduler error. Treat this as device noise, not value evidence.
- `oldgreen_backend_full_dirty_patch_20260707.patch`: complete old green backend dirty patch stack.
- `current_backend_working_tree_patch_20260707.patch`: current diagnostic backend working tree patch stack.
- `bounded_broadcast_gather_restickify_20260707/`: latest positive bounded
  broadcast checkpoint at Deeptools `071e293cf`.
- `bounded_multicast_gather_restickify_20260707/`: latest positive bounded
  multicast checkpoint at Deeptools `3a4349e62`.
- `full_flash_replay_latest_3a4349e62_attempt_20260707.md`: latest saved full
  flash replay attempt. It emitted `64` backend plan artifacts but was
  interrupted after a long silent DXP/DCC run, so full replay completion remains
  unverified at `3a4349e62`.

## Next Steps

1. Keep the PR1 scatter branch separate and clean.
2. For `ah/comms-collectives`, keep hardening the production path for
   gather/restickify:
   - short-term: continue validating the materialized `STCDPOpLx +
     ReStickifyOpLx` path because it has bounded correctness evidence;
   - next: validate the fanout classes against real Torch/Granite-generated
     SDSC bundles;
   - long-term: move toward the DL transfer-node/local-stage path so backend
     owns physical lowering without a deprecated data-op surface.
3. Reproduce bounded M16/M32 value correctness after each backend change using the same `stable_matmul_operand_broadcast.py` probe.
4. For flash, continue using structural criteria until baseline flash value correctness is fixed elsewhere. Our pass should not introduce additional correctness bugs, but the existing flash kernel is not a clean value-correct oracle today.
5. Do not implement full streaming in this branch. If a tensor is too large for bounded resident movement, fail closed and leave HBM fallback; WSR should tile the region later.

## Reset Note

On CDX, `aiu_dd2_hot_reset -t chip -d 0000:b0:00.0` reaches the device but aborts with `RISCV config not found`; `-t linux` requires elevated privileges. If the AIU throws RAS runtime errors, restart the pod rather than treating the run as a compiler failure.
