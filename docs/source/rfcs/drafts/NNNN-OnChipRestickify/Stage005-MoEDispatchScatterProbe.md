# Stage 005: MoE Dispatch Scatter Probe

Date: 2026-05-24

## Purpose

This stage checks whether the current Tier 1 on-chip handoff realization gives a
latency gain for an MoE-like scatter/gather workload.  The goal was not to claim
full MoE support; it was to find the closest executable token-routing pattern in
the current Spyre Inductor stack and compare flag-off against flag-on.

## Operator Support Probe

General tensor-indexed routing operators are not currently executable through
this branch's Spyre Inductor path.  Small probes for these ops failed before
DXP/codegen with `TypeError: Cannot convert symbols to int`:

```text
torch.index_select(x, 0, idx)
x[idx]
torch.gather(x, 1, idx)
out.scatter(0, idx, src)
out.scatter_add(0, idx, src)
```

The executable scatter-like primitive today is `torch.ops.spyre.overwrite_f`,
which is already used by the paged-KV-cache scatter tests.  I therefore used a
static token-dispatch microbenchmark:

```python
def dispatch_fn(dispatch, tokens):
    for i, slot in enumerate(slots):
        dispatch = torch.ops.spyre.overwrite_f(
            tokens[i : i + 1], dispatch, [0], [slot]
        )
    return dispatch
```

This models the dispatch side of MoE routing: contiguous token rows are placed
into expert-capacity slots.  It does not model dynamic gather or combine because
those paths do not compile yet as real tensor-indexed ops.

One additional probe tried to add a pointwise expert epilogue after dispatch:
`dispatch -> mul -> add`.  That shape did trigger the generic pointwise on-chip
bridge (`sdsc_16_add.json opFuncsUsed_=["STCDPOpLx"]`), but it was not
value-correct (`max_err=5.296875`), so it is excluded from the benchmark claim.

## Method

Runs used the same clean foundation DXP binary as the SDPA proof:

```sh
export HOME=/home/adnan-cdx
export DTI_PROJECT_ROOT=$HOME/dt-inductor-mixed
source "$DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh"
export PYTHON="$DTI_PROJECT_ROOT/.venv/bin/python3"
export PATCHED_DXP="$DTI_PROJECT_ROOT/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone"
export PATH="$(dirname "$PATCHED_DXP"):$PATH"
```

Each pair used fresh `TORCHINDUCTOR_CACHE_DIR` values and measured the compiled
Spyre call after warmup with `torch.spyre.synchronize()` before stopping the
timer.

Flag-off baseline:

```sh
SPYRE_ONCHIP_HANDOFF_REALIZE=0
SPYRE_ONCHIP_ATTENTION_SCORE_HANDOFF=0
```

Flag-on probe:

```sh
SPYRE_ONCHIP_HANDOFF_REALIZE=1
SPYRE_ONCHIP_ATTENTION_SCORE_HANDOFF=0
SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
```

## Results

| Routes | Capacity | Hidden | Variant | Warmup / iters | Median ms | Mean ms | Min ms | Max ms | Max error | Mixed SDSCs |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 16 | 32 | 256 | off-chip | 10 / 30 | 0.183324 | 0.183758 | 0.174785 | 0.196256 | 0.001953 | none |
| 16 | 32 | 256 | on-chip flag | 10 / 30 | 0.190184 | 0.189604 | 0.178548 | 0.205064 | 0.001953 | none |
| 32 | 64 | 512 | off-chip | 5 / 20 | 0.240256 | 0.239567 | 0.229051 | 0.253135 | 0.001953 | none |
| 32 | 64 | 512 | on-chip flag | 5 / 20 | 0.255544 | 0.255253 | 0.247218 | 0.266491 | 0.001953 | none |
| 64 | 128 | 512 | off-chip | 3 / 10 | 0.410156 | 0.405195 | 0.361053 | 0.444822 | 0.001953 | none |
| 64 | 128 | 512 | on-chip flag | 3 / 10 | 0.411476 | 0.411288 | 0.383371 | 0.434548 | 0.001953 | none |

Median deltas:

| Routes | Median speedup | Median delta ms | Interpretation |
|---:|---:|---:|---|
| 16 | 0.964x | -0.006860 | flag noise; no on-chip realization |
| 32 | 0.940x | -0.015288 | flag noise; no on-chip realization |
| 64 | 0.997x | -0.001320 | effectively equal; no on-chip realization |

Caches:

```text
R=16 offchip=/tmp/moe-dispatch-offchip-R16-C32-H256-477771-10615
R=16 onchip=/tmp/moe-dispatch-onchipflag-R16-C32-H256-477771-3779
R=32 offchip=/tmp/moe-dispatch-offchip-R32-C64-H512-477771-4516
R=32 onchip=/tmp/moe-dispatch-onchipflag-R32-C64-H512-477771-428
R=64 offchip=/tmp/moe-dispatch-offchip-R64-C128-H512-477771-18606
R=64 onchip=/tmp/moe-dispatch-onchipflag-R64-C128-H512-477771-28696
```

The on-chip-flag caches were checked for mixed data-op emission:

```text
/tmp/moe-dispatch-onchipflag-R16-C32-H256-477771-3779  STCDPOpLx_lines=0 datadscs_lines=0
/tmp/moe-dispatch-onchipflag-R32-C64-H512-477771-428   STCDPOpLx_lines=0 datadscs_lines=0
/tmp/moe-dispatch-onchipflag-R64-C128-H512-477771-28696 STCDPOpLx_lines=0 datadscs_lines=0
```

## Conclusion

There is no measured on-chip gain for MoE dispatch routing yet because the
current realization does not target the executable scatter primitive.  The
flag-on runs stay value-correct and fail closed, but they emit no mixed SDSCs,
no `STCDPOpLx`, and therefore no HBM-eliminating handoff.

Strategically, this confirms the next MoE step: add a realizer for
`overwrite_f`/token-routing edges, or first make real tensor-indexed
`gather`/`scatter` compile through Spyre Inductor.  Until one of those exists,
the current Tier 1 primitive helps SDPA score handoff and pointwise same-stick
edges, but not MoE routing itself.
