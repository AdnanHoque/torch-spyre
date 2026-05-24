# On-chip handoff realization (design + first cut)

A *same-layout cross-core handoff* keeps a producer to consumer activation edge
resident in LX instead of round-tripping through HBM at the SDSC boundary. The
Tier-1 planner (`onchip_handoff.py`) already DETECTS eligible edges; this
document specifies how the **realize** path turns a detected plan into the
mixed DL + data-op SuperDSC proven on device, and what the offline first cut
implements versus stubs.

The mechanism, schema, LX bases, and device proof are in
`../rfcs/drafts/NNNN-OnChipRestickify/CoreToCoreDataMovementRecipe.md` (the SPEC;
§6b documents the compiler-driven realization this doc designs). The manual
ground truth is the two splice scripts under the recipe's `reproduction/splice/`:
`splice_2048_stcdp.py` (same-core) and `splice_2048_roundtrip.py` (cross-core).
Realization reproduces the splice *in-compiler* rather than post-hoc.

## 1. Hook points

The on-chip unit is the SDSC, not the bundle (LX does not persist across
`sdsc_execute`). A bundle is a list of SDSCs: `codegen/bundle.py
generate_bundle` calls `compile_op_spec(idx, ks)` (`codegen/superdsc.py:612`)
per `OpSpec`, then writes one `sdsc_{name}.json` per SDSC plus `bundle.mlir`.
`generate_sdsc` (`codegen/compute_ops.py:208`) builds each SDSC dict: the DL op
in `dscs_`, a degenerate `coreIdToDscSchedule` of `[[-1,0,0,0]]`, and `numWkSlicesPerDim_`
(the consumer sharding). The realize fold is the **only** place mixed SDSCs can
be assembled.

- Producer of realization output: `onchip_handoff.py` (planner). Today it
  returns fail-closed `OnChipHandoffPlan`s. Realize adds a pure function
  `realize_plan(...) -> OnChipRealization | None` returning the LX bases,
  synthesized `datadscs_`/schedule/`opFuncsUsed_`, and producer/consumer
  LX-flip descriptors.
- Consumer of realization output: `codegen/bundle.py generate_bundle`,
  immediately after the `compile_op_spec` loop (line 32). The consumer SDSC dict
  gets `datadscs_`/`coreIdToDscSchedule`/`opFuncsUsed_` installed and the
  producer SDSC's output `labeledDs` flipped to LX. `bundle.mlir` is unchanged
  for same-core/cross-core (all SDSCs kept); a standalone in-graph restickify is
  dropped only for the layout-changing Tier-2 path (out of scope).

## 2. Transform, manual splice -> in-compiler

| # | Manual splice (post-hoc) | In-compiler (realize) |
|---|---|---|
| 1 | trace HBM addr to find prod/cons | planner already pairs prod/cons by edge |
| 2 | fixed bases 16384/8192 | `allocate_lx_bases` from `per_core_slice_bytes` |
| 3 | `_flip_tensor_to_lx` on prod out + cons in | descriptors -> bundle applies |
| 4 | `build_same_layout_bridge` | `onchip_bridge` same fn (sharding-matched) |
| 5 | install scaffolding, keep mlir | bundle fold; keep mlir (same-core) |

## 3. The binding (the hard part)

The bridge dataOUT must feed the SPECIFIC consumer input. Memory/§11c: torch-
spyre cannot bind via `DscSenGraph` (edge `index_` is a graph port, not an
internal `labeledDs_` idx). The splice binds by **LX address coincidence**:
producer-out and bridge dataIN at one base, bridge dataOUT and consumer-in at
another; the consumer reads its own input from a known LX base. The realize
pass binds the same way — by LX-base agreement, NOT a graph API. The consumer
ldsIdx is `inputLabeledDs[0]`; producer ldsIdx is `outputLabeledDs[0]`.

Pure-inductor reach: descriptor synthesis + base agreement + JSON fold. NOT
pure-inductor: the dxp import gate / mixed-codegen dispatch (§5 of the SPEC).
Conclusion: binding is pure-inductor (base coincidence), execution needs the
deeptools Foundation gate.

## 4. LX liveness within 2 MB/core

`per_core_slice_bytes` = rows x stick-padded chunk; `allocate_lx_bases` packs
non-overlapping stick-aligned regions with `region0` headroom for the DL op's
own LX. Same-core = 2 regions; real cross-core = 2; round-trip = 3 (NOFIT at
4096). Realize asserts in-capacity, else fail-closed.

## 5. Scope

Pure inductor: detect, bases, synth, flip-descriptors, fold (default-off, fail-
closed). Needs deeptools gate: mixed import + dispatch + senprog. First cut:
same-core same-shard offline plan; fold seam is `TODO(onchip-realize)`.
