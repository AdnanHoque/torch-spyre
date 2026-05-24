# Stage 006: MoE Static Routing Splice A/B

Date: 2026-05-24

## Purpose

Stage 005 tested the executable `overwrite_f` scatter-like primitive and found no
on-chip realization.  The `on-chip-restickify` reproduction tree suggested a
better immediate experiment: express MoE dispatch/combine as static one-hot
permutation matmuls, then splice the intermediate `batchmatmul -> batchmatmul`
handoff with the proven same-stick `STCDPOpLx` roundtrip bridge.

This is not dynamic MoE routing.  It is a static routing proxy where the
permutation matrix is known at compile time.  That makes the handoff statically
addressable, unlike true top-k routing where token-to-slot placement is runtime
index data.

## Source Inspiration

The experiment was adapted from the `on-chip-restickify` reproduction directory:

```text
docs/source/rfcs/drafts/NNNN-OnChipRestickify/reproduction/workloads/moe_routing/
  eligibility.md
  projection.md
  devval_moe.py
  devval_moe_combine.py
  derive_moe_placement.py
  splice_moe_dispatch.py
```

The important design shift from Stage 005 is this formulation:

```python
# dispatch / gather proxy
out = (perm @ x) @ wexp

# combine / scatter-add proxy
out = (perm_w @ y) @ wout
```

Both compile into a two-SDSC bundle.  The first `batchmatmul` materializes the
routed activation buffer and the second `batchmatmul` consumes it.  In the stock
bundle that intermediate buffer is shared through HBM; the splice flips the
producer output and consumer input to LX and folds two `STCDPOpLx` data ops into
the consumer SDSC.

## Method

Environment:

```sh
export HOME=/home/adnan-cdx
export DTI_PROJECT_ROOT=$HOME/dt-inductor-mixed
source "$DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh"
export PYTHON="$DTI_PROJECT_ROOT/.venv/bin/python3"
export PATCHED_DXP="$DTI_PROJECT_ROOT/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone"
export PATH="$(dirname "$PATCHED_DXP"):$PATH"
```

The reproduction scripts were copied to `/tmp/ab_moe_routing` from
`origin/on-chip-restickify` and run against this branch's
`torch_spyre/_inductor/codegen/onchip_bridge.py`.  DXP was run with
`DXP_DEBUG=1` so `senprog.txt` was regenerated for the spliced bundles.

For each shape:

1. Compile and validate the stock HBM bundle.
2. Copy the generated `sdsc_fused_mm_*` code directory.
3. Run `splice_moe_dispatch.py` with `MOE_MB`/`MOE_IN` set to the bridged tensor
   extents.
4. Re-run clean DXP on the spliced directory.
5. Redirect the runtime kernel runner to the fresh spliced code directory.
6. Validate value correctness and benchmark stock HBM vs spliced on-chip.

The tested shape family was `E=8, T=512`, with `H=1024` and `H=2048`.

## Edge Classification

For both dispatch and combine, `derive_moe_placement.py` reported:

```text
shared_hbm_base: true
producer output layout: [mb, out], stick=out, split_dim=mb, n_cores=32
consumer input layout: [mb, in], stick=in, split_dim=mb, n_cores=32
same_stick: true
same_shard: true
stick_is_split: false
```

The stick names differ because producer output uses `out` and consumer matmul
input uses `in`, but both refer to the hidden dimension.  The token/slot dimension
is split across cores; hidden remains the stick dimension and is not split.

Splice footprints:

```text
H=1024: per_core_slice_bytes=32768, lx_footprint=98304
H=2048: per_core_slice_bytes=65536, lx_footprint=196608
```

Both are comfortably below the 2 MB/core LX capacity.

## Results

All timings are wall-clock medians for the compiled call with
`torch.accelerator.synchronize()` inside the benchmark loop.  The profiler
`PrivateUse1` fields reported `0.0000` in this environment, so wall-clock is the
usable measurement.

| Workload | E | T | H | Variant | Iters | Median ms | Min ms | Max error |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| dispatch `(perm @ x) @ wexp` | 8 | 512 | 1024 | HBM | 25 | 0.1536 | 0.1451 | 0.000854 |
| dispatch `(perm @ x) @ wexp` | 8 | 512 | 1024 | spliced on-chip | 25 | 0.1204 | 0.1176 | 0.001221 |
| combine `(perm_w @ y) @ wout` | 8 | 512 | 1024 | HBM | 25 | 0.1511 | 0.1450 | 0.000671 |
| combine `(perm_w @ y) @ wout` | 8 | 512 | 1024 | spliced on-chip | 25 | 0.1212 | 0.1196 | 0.001038 |
| dispatch `(perm @ x) @ wexp` | 8 | 512 | 2048 | HBM | 15 | 0.3308 | 0.3240 | 0.001465 |
| dispatch `(perm @ x) @ wexp` | 8 | 512 | 2048 | spliced on-chip | 15 | 0.2648 | 0.2574 | 0.001709 |
| combine `(perm_w @ y) @ wout` | 8 | 512 | 2048 | HBM | 15 | 0.3251 | 0.3198 | 0.001221 |
| combine `(perm_w @ y) @ wout` | 8 | 512 | 2048 | spliced on-chip | 15 | 0.2628 | 0.2599 | 0.001236 |

Median speedups:

| Workload | H | Speedup | Median delta ms |
|---|---:|---:|---:|
| dispatch | 1024 | 1.276x | 0.0332 |
| combine | 1024 | 1.247x | 0.0299 |
| dispatch | 2048 | 1.249x | 0.0660 |
| combine | 2048 | 1.237x | 0.0623 |

## Descriptor And Senprog Evidence

Each spliced consumer was `sdsc_1_batchmatmul.json` with:

```text
opFuncsUsed_=["STCDPOpLx", "STCDPOpLx"]
num_dataops=2
```

DXP debug evidence for both dispatch and combine at both hidden sizes:

```text
sdsc_1_batchmatmul/senprog.txt: HBM=0 L3_LDU=64 L3_STU=64
```

Spliced directories:

```text
/tmp/ab_moe_routing/spl_dispatch_E8_T512_H1024
/tmp/ab_moe_routing/spl_combine_E8_T512_H1024
/tmp/ab_moe_routing/spl_dispatch_E8_T512_H2048
/tmp/ab_moe_routing/spl_combine_E8_T512_H2048
```

## Interpretation

This experiment changes the MoE conclusion from Stage 005 in an important way:
the current core-to-core primitive can help MoE-shaped routing if the routed
activation appears as a static, same-stick `batchmatmul -> batchmatmul` handoff.
The static permutation-matmul proxy gets a consistent `~1.24x-1.28x` speedup and
preserves value correctness.

The result does not solve dynamic top-k MoE routing.  True `gather`/`scatter` is
still blocked in the compiler path, and the true router mapping is runtime data,
not static `PieceInfo.memId`.  The right next production step is either:

1. Add a production realizer for same-stick `batchmatmul -> batchmatmul` handoffs
   so this static proxy no longer needs an offline splice.
2. Investigate an index-driven same-stick STCDP/data-op path for real dynamic
   routing.

The first item is immediately actionable in our current architecture because the
splice already proves the descriptor shape and device behavior.  The second item
is the real MoE routing frontier.
