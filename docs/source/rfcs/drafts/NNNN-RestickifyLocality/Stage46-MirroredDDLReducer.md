# Stage 46: Mirrored DDL Contract Reducer

## Summary

Stage 45 left one clean blocker: the graph-input DDL bridge was unsafe and is
now gated off, while the high-signal in-graph restickify is the mirrored
direction:

```text
input layout/stick:   [out, mb], stick=mb
output layout/stick:  [mb, out], stick=out
split:                mb:32
```

Stage 46 adds a reducer for that contract:

```text
tools/restickify_mirrored_ddl_reducer.py
```

The reducer synthesizes `ReStickifyOpHBM` SDSCs through the real
`SDSCSpec -> generate_sdsc -> generate_restickify_ddl_bridge_sdsc` path, then
sweeps sizes, loop orders, and LX address modes through DDC, DCC, and optional
DXP-with-pre-DDC-bypass.

No Deeptools changes were made.

## What The Reducer Tests

The reducer has two important direction modes:

- `forward`: output stick dimension equals the split dimension. This is the
  Stage 42 direction that Deeptools can compile to an HBM-free LX/SFP/PT
  program.
- `mirrored`: input stick dimension equals the split dimension, while output
  stick dimension does not. This is the in-graph direction Stage 3B wants.

It also has address modes:

- `generated`: use the compact start addresses produced by synthetic
  Torch-Spyre SDSC generation.
- `input-strided`: force the input LX allocation to use production-like
  per-core starts: `core * lxSize`.
- `stage44-like`: `input-strided` plus the large output base seen in the real
  Stage 44 failing SDSC.

The address-mode knob turned out to be the key.

## Commands

Size/address sweep:

```sh
python tools/restickify_mirrored_ddl_reducer.py \
  --output-dir /tmp/stage46-mirrored-ddl-reducer \
  --size 64 --size 128 --size 256 --size 512 --size 1024 --size 1536 --size 2048 \
  --direction forward --direction mirrored \
  --split-dim mb \
  --loop-order input-reversed \
  --address-mode generated \
  --address-mode input-strided \
  --run-deeptools --run-dxp-preload
```

Loop-order sweep at 2048:

```sh
python tools/restickify_mirrored_ddl_reducer.py \
  --output-dir /tmp/stage46-loop-order \
  --size 2048 \
  --direction mirrored \
  --split-dim mb \
  --loop-order input-reversed \
  --loop-order input \
  --loop-order output-reversed \
  --loop-order output \
  --address-mode generated \
  --address-mode input-strided \
  --run-deeptools --run-dxp-preload
```

Address-mode isolation:

```sh
python tools/restickify_mirrored_ddl_reducer.py \
  --output-dir /tmp/stage46-address-modes-v2 \
  --size 2048 \
  --direction mirrored \
  --split-dim mb \
  --loop-order input-reversed \
  --address-mode generated \
  --address-mode input-strided \
  --address-mode stage44-like \
  --run-deeptools --run-dxp-preload
```

Local artifacts:

```text
artifacts/stage46_mirrored_ddl_reducer/
```

## Results

### Size Sweep

Failure counts:

```text
ok:               6
ddc-fail:         14
dcc-lrf-boundary: 1
```

Key rows:

| Contract | Sizes | Result |
|---|---:|---|
| forward, generated addresses | 64, 128 | DDC fails: `Cannot find a valid minimum value for data stage 3` |
| forward, generated addresses | 256, 512, 1024, 1536, 2048 | DDC/DCC/DXP pass |
| mirrored, generated addresses | 64 through 1536 | DDC fails: `Cannot find a valid minimum value for data stage 4` |
| mirrored, generated addresses | 2048 | DDC/DCC/DXP pass |
| mirrored, input-strided addresses | 2048 | DDC passes, DCC/DXP fail with LXLU LRF boundary |

The successful generated-address DXP rows emit no HBM tokens in `senprog.txt`:

```text
HBM=0, L3LU=0, L3SU=0, LXLU=32, LXSU=32, SFP=896, PT=8928
```

So the DDL template can produce an HBM-free restickify-style program for the
compact-address contract.

### Address-Mode Isolation

At mirrored 2048:

| Address mode | Result | Boundary |
|---|---|---|
| generated | passes DDC/DCC/DXP | none |
| input-strided | DDC passes, DCC/DXP fail | `lxlu0 : LRF0 : 2359168` |
| stage44-like | DDC passes, DCC/DXP fail | `lxlu0 : LRF0 : 2359168` |

The failing rows reproduce the Stage 44 pattern. DCC reports failures from
core 9 through core 31, ending at:

```text
core 31: lxlu0 : LRF0 : 8126336
```

That value corresponds to the production-like input LX start address:

```text
31 * 262144 - 128 = 8126336
```

The large output base in `stage44-like` is not required to trigger the failure.
Input per-core LX striding alone is enough.

### Loop-Order Sweep

At mirrored 2048:

| Loop order | generated addresses | input-strided addresses |
|---|---|---|
| input-reversed | pass | LXLU LRF boundary |
| input | pass | LXLU LRF boundary |
| output-reversed | pass | LXLU LRF boundary |
| output | pass | LXLU LRF boundary |

Loop order does not explain the blocker. The failure follows the input LX
addressing mode.

## Interpretation

The mirrored direction is not impossible in Deeptools. It compiles all the way
through DXP when the LX allocations use compact starts.

The blocker is more specific:

```text
mirrored DDL restickify + production-like per-core LX input starts
```

DCC appears to treat the strided input LX starts as LRF initialization
addresses and rejects high-core values once they exceed the register-file
addressing boundary. This matches the real Stage 44 failure, where the failing
core-31 value was also `8126336`.

That means the next implementation problem is not "make the mirrored DDL
algorithm exist." It is:

```text
How do we represent an in-graph producer's per-core LX-resident tensor to the
DDL restickify template without feeding global/strided LX addresses into an
LXLU-local register-file address path?
```

## Consequence For The E2E Bridge

The current Stage 45 skip remains correct:

```text
output-stick-is-not-split-dim
```

If we simply allowed the mirrored in-graph bridge through, real production-like
addresses can hit the same LXLU boundary. But the reducer also shows a possible
way forward: the DDL path can compile if the bridge presents compact per-core
local LX addresses.

That is not automatically safe. If the producer wrote the tensor at a real
per-core LX address, blindly normalizing the consumer side to zero could read
the wrong place. The bridge needs a real contract for producer-consumer LX
aliasing, not just a smaller number.

## Next Step

The next stage should inspect the DDC/DDL allocation contract rather than tune
Stage 3B mapping further:

1. Compare generated-address and input-strided post-DDC SDSCs for allocation
   fields around `allocate_Tensor0_lx_internalInput`, `allocate_Tensor0_lx`,
   `lxlu_input`, and `sfp_input`.
2. Inspect Deeptools handling of LX `startAddressCoreCorelet_` for DDL
   `get_external_data_transfer_allocation(... memory="lx")`.
3. Prototype a consumer-side compact-local address contract that still aliases
   the producer's actual per-core LX allocation.
4. Only after that compile path works, re-enable the mirrored in-graph DDL
   bridge and rerun correctness on `adds_then_matmul 2048`.

## Validation

Local:

```text
python3 -m py_compile tools/restickify_mirrored_ddl_reducer.py
```

Pod:

```text
python -m py_compile tools/restickify_mirrored_ddl_reducer.py
```

The reducer sweeps completed and artifacts were copied back under:

```text
artifacts/stage46_mirrored_ddl_reducer/
```
