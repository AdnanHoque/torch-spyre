# Stage 007: Production Static MoE Routing Handoff

Date: 2026-05-25

## Purpose

Stage 006 proved that a static MoE routing proxy can benefit from the Tier 1
same-stick bridge when spliced offline.  This stage moves that splice into the
compiler path: a gated production realizer for static
`batchmatmul -> batchmatmul` handoffs.

This still does not solve dynamic top-k MoE routing.  The supported target is a
static, compiler-visible routed activation, such as:

```python
# dispatch / gather proxy
out = (perm @ x) @ wexp

# combine / scatter-add proxy
out = (perm_w @ y) @ wout
```

The intermediate `perm @ x` or `perm_w @ y` tensor is a same-stick activation
handoff from one `batchmatmul` SDSC to the next.

## Implementation

The realization is gated by:

```sh
SPYRE_ONCHIP_HANDOFF_REALIZE=1
SPYRE_ONCHIP_STATIC_MATMUL_HANDOFF=1
SPYRE_ONCHIP_HANDOFF_MIN_BYTES=1048576
```

Code changes:

- `torch_spyre/_inductor/config.py`
  - Added `SPYRE_ONCHIP_STATIC_MATMUL_HANDOFF`.
- `torch_spyre/_inductor/codegen/bundle.py`
  - Passes the static-matmul flag into `realize_onchip_handoff`.
- `torch_spyre/_inductor/onchip_realize.py`
  - Added `detect_static_matmul_handoff`.
  - Added `realize_static_matmul_handoff`.
  - The detector requires:
    - producer op is `batchmatmul`;
    - exactly one future consumer of the producer HBM address;
    - consumer op is `batchmatmul`;
    - producer output and consumer input preserve the same physical stick;
    - producer and consumer split the same dim by the same 32-core factor;
    - the handoff is above the configured byte threshold;
    - three LX regions fit in the 2 MB/core LX capacity.
  - The realizer flips producer output and consumer input to LX, then folds a
    two-`STCDPOpLx` roundtrip bridge into the consumer SDSC.
- `tests/_inductor/test_onchip_realize_logic.py`
  - Added static matmul detection, realization, min-size, layout-change, and
    fanout tests.

The physical-stick check allows producer output to name the hidden axis `out_`
and consumer input to name it `in_`, but only when the stick position and paired
extents match and all non-stick dims match by name.

## Validation

Logic tests:

```text
tests/_inductor/test_onchip_handoff_logic.py      3/3 passed
tests/_inductor/test_onchip_realize_logic.py     13/13 passed
tests/_inductor/test_onchip_streaming_logic.py    9/9 passed
```

Device runs used the clean foundation DXP binary:

```sh
export HOME=/home/adnan-cdx
export DTI_PROJECT_ROOT=$HOME/dt-inductor-mixed
source "$DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh"
export PYTHON="$DTI_PROJECT_ROOT/.venv/bin/python3"
export PATCHED_DXP="$DTI_PROJECT_ROOT/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone"
export PATH="$(dirname "$PATCHED_DXP"):$PATH"
export PYTHONPATH="$DTI_PROJECT_ROOT/torch-spyre-core-to-core-primitive:$PYTHONPATH"
```

The `PYTHONPATH` setting is important when running reproduction scripts from
`/tmp`; otherwise Python imports the installed Torch-Spyre tree instead of this
worktree.

## Production A/B Results

All runs used `E=8, T=512`.  Timings are wall-clock medians for the compiled call
with device synchronization inside the loop.  The reproduction scripts still
print `baseline_HBM` in their label because `ONCHIP_BASELINE=1` disables their
old redirect shim; the artifact mode below is the authoritative variant.

| Workload | H | Variant | Iters | Median ms | Min ms | Max error | Mixed SDSC |
|---|---:|---|---:|---:|---:|---:|---|
| dispatch `(perm @ x) @ wexp` | 1024 | HBM | 25 | 0.1465 | 0.1426 | 0.000854 | none |
| dispatch `(perm @ x) @ wexp` | 1024 | production on-chip | 25 | 0.1357 | 0.1320 | 0.001221 | `sdsc_1_batchmatmul` |
| combine `(perm_w @ y) @ wout` | 1024 | HBM | 25 | 0.1447 | 0.1356 | 0.000671 | none |
| combine `(perm_w @ y) @ wout` | 1024 | production on-chip | 25 | 0.1235 | 0.1207 | 0.001038 | `sdsc_1_batchmatmul` |
| dispatch `(perm @ x) @ wexp` | 2048 | HBM | 15 | 0.3302 | 0.3264 | 0.001465 | none |
| dispatch `(perm @ x) @ wexp` | 2048 | production on-chip | 15 | 0.2619 | 0.2589 | 0.001709 | `sdsc_1_batchmatmul` |
| combine `(perm_w @ y) @ wout` | 2048 | HBM | 15 | 0.3320 | 0.3280 | 0.001221 | none |
| combine `(perm_w @ y) @ wout` | 2048 | production on-chip | 15 | 0.2658 | 0.2576 | 0.001236 | `sdsc_1_batchmatmul` |

Median speedups:

| Workload | H | Speedup | Median delta ms |
|---|---:|---:|---:|
| dispatch | 1024 | 1.080x | 0.0108 |
| combine | 1024 | 1.172x | 0.0212 |
| dispatch | 2048 | 1.261x | 0.0683 |
| combine | 2048 | 1.249x | 0.0662 |

## Descriptor Evidence

For every production on-chip run:

```text
sdsc_1_batchmatmul.json opFuncsUsed_=["STCDPOpLx", "STCDPOpLx"] datadscs_=2
consumer input ldsIdx_=0 hbmSize_=0 lxSize_=2147483647
```

DXP debug evidence:

```text
sdsc_1_batchmatmul/senprog.txt: HBM=0 L3_LDU=64 L3_STU=64
```

The stock HBM runs emitted no mixed SDSCs and no L3 traffic in
`sdsc_1_batchmatmul`.

Caches:

```text
dispatch H=1024 HBM      /tmp/moe-prod-dispatch-hbm-H1024-483332-19633
dispatch H=1024 on-chip  /tmp/moe-prod-dispatch-onchip-H1024-483332-10930
combine  H=1024 HBM      /tmp/moe-prod-combine-hbm-H1024-483332-5023
combine  H=1024 on-chip  /tmp/moe-prod-combine-onchip-H1024-483332-6266
dispatch H=2048 HBM      /tmp/moe-prod-dispatch-hbm-H2048-483332-20343
dispatch H=2048 on-chip  /tmp/moe-prod-dispatch-onchip-H2048-483332-12465
combine  H=2048 HBM      /tmp/moe-prod-combine-hbm-H2048-483332-25360
combine  H=2048 on-chip  /tmp/moe-prod-combine-onchip-H2048-483332-151
```

## Conclusion

The offline static MoE splice is now a compiler/codegen feature behind
`SPYRE_ONCHIP_STATIC_MATMUL_HANDOFF=1`.  The production path emits the same
mixed consumer shape, runs value-correct on device, and shows a clear benefit at
the larger hidden size (`~1.25x` for both dispatch and combine).

The next MoE frontier remains dynamic routing.  True `gather`/`scatter` still
does not compile through this path, and true top-k routing needs runtime
index-driven placement rather than static `PieceInfo.memId`.
