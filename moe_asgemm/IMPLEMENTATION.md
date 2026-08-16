# Implementation

## Algorithm

The implemented operation is:

```text
Y[t,h] = sum over e of alpha[e,t,0] * Wd[e](
    gelu(X[t,:] * Wg[e]) * (X[t,:] * Wu[e]))
```

Physical inputs are:

```text
X       [T,H]
Wg      [H,E,F]
Wu      [H,E,F]
Wd      [F,E,H]
alpha   [E,T,1]
Y       [T,H]
```

The explicit singleton in `alpha[E,T,1]` is intentional. It preserves the
broadcast-compatible device layout used by the post-down multiply without
restickifying the full down-projection result.

## Emitted program

The full program is one bundle with one static `E=128` loop:

```text
HBM X -> distributed X_LX
initialize Y_LX

for expert in 0..127:
    gate   = X_LX @ Wg[expert]
    gate   = gelu_tanh(gate)
    up     = X_LX @ Wu[expert]
    hidden = gate * up
    down   = hidden @ Wd[expert]
    weight = alpha[expert,:,0]
    contrib = down * weight
    Y_LX += contrib

Y_LX -> HBM output
```

The representative bundle contains twelve SDSCs:

```text
sdsc_0   X HBM-to-LX preheader
sdsc_1   fixed accumulator initialization
sdsc_2   gate projection
sdsc_3   GELU
sdsc_4   up projection
sdsc_5   gate/up multiply
sdsc_6   down projection
sdsc_7   runtime alpha HBM-to-LX copy
sdsc_8   post-down alpha multiply
sdsc_9   collapsed unit expert contribution
sdsc_10  loop-carried accumulator add
sdsc_11  final LX-to-HBM drain
```

## Torch-Spyre implementation pieces

`torch_spyre/_inductor/customops.py`

- Defines shared-LHS and prepacked expert-matmul contracts.
- Avoids constructing a physical `[E,T,H]` copy of `X`.

`torch_spyre/_inductor/lowering.py`

- Lowers shared-LHS gate/up projections and prepacked down projections.
- Retains expert metadata needed by coarse tiling and layout propagation.

`torch_spyre/_inductor/wsr/coarse_tile.py`

- Constructs the flat temporal expert loop.
- Compacts invariant reads to their active dependency dimensions.
- Hoists the single `X` load before the expert loop.
- Preserves full expert-slab HBM advances after a unit expert dimension is
  tiled and squeezed.
- Builds a fixed accumulator plus one post-loop drain.
- Replaces a stale local unit-size sum with an identity contribution while the
  loop-carried add performs the real expert reduction.

`torch_spyre/_inductor/loop_info.py`

- Carries owner and lifetime metadata for invariant preheaders and fixed loop
  state.

`torch_spyre/_inductor/spyre_kernel.py`

- Retains loop symbols that appear only in tensor argument advances.
- Produces effective affine expert addresses after the expert dimension is
  squeezed from the local tensor shape.

`torch_spyre/_inductor/scratchpad/allocator.py`

- Admits narrowly marked fixed reduction state into LX.
- Extends the invariant activation lifetime through the owning loop without
  counting a synthetic read.
- Preserves ordinary mutation, capacity, ownership, and alias checks.

`torch_spyre/_inductor/scratchpad/lx_relayout.py`

- Aligns a marked invariant activation copy with the exact common work
  division and physical core map of its gate/up consumers.
- Aligns the activation-stationary loop chain to the transport-free M32 row
  ownership used by the accepted full-shape program.
- Fails closed for incompatible consumers, partial ownership, or mappings that
  cannot be represented exactly.

`torch_spyre/_inductor/codegen/superdsc.py`

- Preserves the selected core-contiguous mapping through SuperDSC emission.

`torch_spyre/_inductor/constants.py`, `ir.py`, `pass_utils.py`, `passes.py`,
`propagate_layouts.py`, `scratchpad/plan_solver.py`, and
`scratchpad/utils.py`

- Carry fail-closed metadata, ordering, placement, and validation contracts.

`torch_spyre/execution/async_compile.py`

- Adds an opt-in compiler timeout override for the large counted-expert bundle.

## Tests

The branch extends:

- `tests/inductor/test_coarse_tiling.py`
- `tests/inductor/test_core_mapping.py`
- `tests/inductor/test_scratchpad_solver.py`
- `tests/inductor/test_work_division_hint.py`

Coverage includes compact invariant copies, loop ownership, expert-stride
preservation, fixed accumulator placement, unit-sum collapse, affine bundle
addresses, exact 32-core maps, rollback, alias rejection, and ordinary-path
negative controls.

## What this is not

- It is not Antoni and Swagath's PR293 chunked program.
- It is not a clone-based `[E,T,H]` dense proxy.
- It is not one fused custom DDL.
- It is not a grouped GEMM.
- It is not an end-to-end MoE implementation.
