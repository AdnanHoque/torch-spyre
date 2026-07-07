# Current-head Python value-probe attempts - 2026-07-07

This archive records attempts to get a current-head Python/AIU value probe for the bounded DLDSC matmul-operand communication path.

## Code heads

- Torch branch: `gather-restickify`
- Torch SHA: `bced14b4 inductor: enrich partial-view gather contracts`
- Deeptools branch: `ah/comms-collectives`
- Deeptools SHA: `9cd9c79c3 [DXP] test partial-view gather offset validation`

## What changed from the earlier blocked probe

The earlier current-head value probe failed before SDSC emission in PyTorch/Spyre fake-tensor setup. The key workaround found here is:

```python
import torch._inductor.config as inductor_config
inductor_config.use_joint_graph_passes = False
```

That bypasses the Inductor joint-graph SDPA/SFDP pattern initialization path that trips Spyre copy/fake-tensor behavior. With that setting, the probe reaches DXP.

## Attempts

### `jointoff_original/`

Original attention-shaped probe with K transpose and `v * 1.0`, plus joint graph passes disabled.

Result: fails in DXP before relayout plan emission.

Failure:

```text
Command [dxp_standalone, -d, .../sdsc_fused_mul_transpose_0_...] died with SIGSEGV
```

### `directkt_mulproducer/`

Pre-transposed K input to avoid the fused transpose prelude. Producer is still `v * 1.0`.

Result: fails in DXP before relayout plan emission.

Failure:

```text
DtException: op->inpSP_.at(inpSPIdx).dimToSize_.at(dimNameOuter) >= stickDim
... sdsc_fused_mul_0 ... died with SIGABRT
```

### `directkt_addproducer/`

Pre-transposed K input and producer changed to `v + 0.001` to avoid fused mul.

Result: fails in DXP before relayout plan emission.

Failure:

```text
DtException: op->inpSP_.at(inpSPIdx).dimToSize_.at(dimNameOuter) >= stickDim
... sdsc_fused_add_0 ... died with SIGABRT
```

## Interpretation

These runs do not disprove the DLDSC communication substrate. They all fail before a backend relayout plan is emitted, either in an upstream DXP pointwise/transpose prelude or earlier compiler setup.

The current verified evidence for the bounded communication substrate remains:

- DXP focused relayout insertion: 8/8 passed
- `LayoutAllgatherRestickify.*`: 32/32 passed
- older structural flash run: backend emitted and realized 32 `gather_then_restickify` plans with 8192 logical transfers and zero structural `ReStickifyOpHBM`

## Next useful validation step

A value probe should avoid unsupported pointwise producer SDSCs. The cleanest next route is a purpose-built lower-level fixture or a Torch graph that starts from an already LX-resident producer without requiring a separate pointwise SDSC before the target matmul operand relayout.
