# Stage 002: SDPA Score Handoff Realization

Date: 2026-05-24

## Purpose

This stage records the move from a blockwise SDPA decomposition experiment to a
stock-SDPA score-matrix handoff realization.

Stage 001 proved that an Inductor-visible online-softmax decomposition can be
value-correct, but the typical prefill benchmark expanded into many SDSCs and
did not emit the intended mixed `STCDPOpLx` attention edge.  The better next
target is the existing stock SDPA graph: the `QK^T` `batchmatmul` score tensor
is written to HBM and immediately consumed by softmax `max` and `sub`.

## Method

The new path is default-off and requires both flags:

```sh
SPYRE_ONCHIP_HANDOFF_REALIZE=1
SPYRE_ONCHIP_ATTENTION_SCORE_HANDOFF=1
```

The size gate is:

```sh
SPYRE_ONCHIP_HANDOFF_MIN_BYTES=1048576
```

The realizer detects stock SDPA score fanout by matching the producer output HBM
allocation address to future `max` and `sub` consumer input allocation addresses
in the generated SDSCs.  This uses `scheduleTree_` allocation addresses because
cached SDSCs do not carry resolved `hbmStartAddress_` fields on DL `labeledDs_`.

The implementation deliberately bridges both consumers.  Bridging only `sub`
would be wrong: flipping the `QK^T` producer output to LX would leave `max`
reading the old HBM address.  Both softmax consumers therefore become mixed
SuperDSCs with `datadscs_`, `coreIdToDscSchedule`, and `opFuncsUsed_`.

## Design Choice

The first production compiler path uses a same-stick `STCDPOpLx` handoff on the
consumer score geometry.  The producer and consumer use different internal dim
labels for the same logical score matrix (`batchmatmul` labels the query axis as
`mb`, while softmax labels it as `x`), so the bridge uses the softmax layout and
split as the canonical byte geometry.  This preserves the same-stick contract and
avoids the `ReStickifyOpWithPTLx` path that still faults Compute CB.

This is still a Tier 1 handoff: no stick-changing path is enabled, and sub-MB
score rows fail closed to HBM.

## Evidence To Record

For each validation run, record:

- cache directory and exact flags;
- mixed SDSC filenames;
- `opFuncsUsed_` values;
- value-correctness max error;
- `senprog.txt` HBM/L3 evidence;
- benchmark median/mean/min/max against stock HBM SDPA;
- any runtime scheduler or Compute CB failures.

## Initial Evidence

Focused logic tests passed:

```text
tests/_inductor/test_onchip_handoff_logic.py
tests/_inductor/test_onchip_realize_logic.py
tests/_inductor/test_onchip_streaming_logic.py

20 passed in 0.14s
```

Compiler smoke with:

```sh
SPYRE_ONCHIP_HANDOFF_REALIZE=1
SPYRE_ONCHIP_ATTENTION_SCORE_HANDOFF=1
SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
TORCHINDUCTOR_CACHE_DIR=/tmp/spyre-attn-score-smoke-onchip-20260524-lxsize2m
```

and shape `(B=1, H=2, L=128, D=64)` emitted the intended mixed SDSCs:

```text
sdsc_3_batchmatmul.json  opFuncsUsed_=None
sdsc_4_max.json          opFuncsUsed_=['STCDPOpLx']
sdsc_5_sub.json          opFuncsUsed_=['STCDPOpLx']
```

The producer score output was LX-only and both score consumers were LX-fed, so
the compiler-side fanout transformation is present.

Device validation did not complete.  The run reached generated runtime artifacts
and then hung at the D2H barrier; it had to be terminated.  A follow-up HBM
control run in the same pod also hung after the terminated mixed runs, so the pod
runtime/device state was no longer clean enough to produce a reliable benchmark
in this iteration.  The feature remains default-off.

The most likely remaining technical risk is the lifetime/scheduling contract for
one producer LX value feeding two later mixed SDSCs.  The compiler can express
the fanout, but the device run has not proven that this multi-consumer LX
lifetime is legal.
