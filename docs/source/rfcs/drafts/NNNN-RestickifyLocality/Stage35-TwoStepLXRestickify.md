# Stage 35: Two-Step LX Restickify Prototype

## Summary

This stage tested whether a restickify-shaped transformation can be represented as
two LX-resident data operations instead of a single `ReStickifyOpHBM` compute-op:

1. `ReStickifyOpLx`: local per-core stick-layout conversion.
2. `STCDPOpLx`: same-stick cross-core LX-to-LX ownership movement.

The prototype is diagnostic only and remains behind
`SPYRE_RESTICKIFY_LX_DATAOP=1`.

## Implementation

Added a small SuperDsc combiner for standalone data-op payloads:

- `torch_spyre/_inductor/codegen/restickify_lx_dataop.py`
  - `combine_dataop_sdscs(...)`

Extended the diagnostic probe:

- `tools/restickify_lx_dataop_probe.py`
  - `--two-step-lx-restickify`

For size `2048`, the probe emits:

- baseline: local `ReStickifyOpLx`, then `STCDPOpLx` from `out_:32` ownership to
  `mb_:32` ownership.
- stage3b: local `ReStickifyOpLx`, then `STCDPOpLx` preserving `out_:32`
  ownership.

## Validation Commands

```sh
export HOME=/home/adnan-cdx
export DTI_PROJECT_ROOT=$HOME/dt-inductor
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/runtime/bin:$PATH
export PYTHONPATH=/tmp/torch-spyre-lx-dataop:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /tmp/torch-spyre-lx-dataop

python -m pytest tests/inductor/test_restickify_lx_dataop.py -q

SPYRE_RESTICKIFY_LX_DATAOP=1 \
python tools/restickify_lx_dataop_probe.py \
  --two-step-lx-restickify \
  --size 2048 \
  --num-cores 32 \
  --output-dir /tmp/restickify-lx-two-step \
  --run-dcg
```

## Results

The focused unit test passed:

```text
5 passed
```

`dcg_standalone -initSdsc` accepted both composed artifacts:

```text
baseline TwoStepReStickifyLxStcdp: dcg_rc=0
stage3b TwoStepReStickifyLxStcdp: dcg_rc=0
```

The DCG logs show the expected transfer-function difference.

Baseline:

```text
Computing Re-StickifyOp transfer function..
Creating pcfg for coreID:* : LX : PE0 ...
Computing transfer function metaData..
0 --> [ 0 1 2 ... 31 ]
...
31 --> [ 0 1 2 ... 31 ]
maxConsumers: 32
Creating pcfg for coreID:* : L3SU : L3LU : LX : PE0 ...
```

Stage 3B:

```text
Computing Re-StickifyOp transfer function..
Creating pcfg for coreID:* : LX : PE0 ...
Computing transfer function metaData..
0 --> [ 0 ]
...
31 --> [ 31 ]
maxConsumers: 1
Creating pcfg for coreID:* : LX : PE0 ...
```

The exported DCG SDSCs contain two data-ops and no compute DSCs:

| mode | dataops | compute dscs |
| --- | ---: | ---: |
| baseline | 2 | 0 |
| stage3b | 2 | 0 |

Both modes keep all data-op pieces in LX and report `hbmSize_ = 0`:

| data-op | input stick | output stick | placements | hbmSize_ |
| --- | --- | --- | --- | ---: |
| `ReStickifyOpLx` | `out_` | `mb_` | `lx:32 -> lx:32` | 0 |
| `STCDPOpLx` | `mb_` | `mb_` | `lx:32 -> lx:32` | 0 |

## Interpretation

This is a backend-contract proof, not a hardware-counter proof.

What it supports:

- `ReStickifyOpLx` can express a local stick-layout conversion with LX-resident
  input and output pieces.
- `STCDPOpLx` can express same-stick LX-to-LX movement.
- A composed `ReStickifyOpLx -> STCDPOpLx` SuperDsc is accepted by DCG.
- Stage 3B-style ownership continuity changes the second step from all-to-all
  consumer fanout to one-to-one local ownership.

What it does not yet prove:

- No physical HBM traffic occurred on hardware.
- A production TorchInductor lowering can safely replace `ReStickifyOpHBM` with
  this sequence.
- The existing standalone IR dumper can emit MLIR for this composed data-op
  sequence.

## Tool Limitation

`DataOpStandalone` can run DCG for the composed artifact, but its
PCFG-to-DataflowIR path currently asserts that a SuperDsc contains exactly one
data-op:

```text
DtException: sdsc_.dataOpdscs_.size() == 1
file /project_src/deeptools/dcc/src/Conversion/PCFGToDataflowIR/PCFGToDFManager.cpp line 89
```

This means we can inspect DCG transfer metadata for the composed sequence, but
we cannot currently dump one combined dataflow MLIR through this standalone
tool. Single-dataop MLIR inspection still works and was covered in Stage 34.

## Next Step

There are two reasonable follow-ups:

1. Integrate this only as a diagnostic path in TorchInductor and measure whether
   the generated executable runs correctly.
2. Patch Deeptools' standalone data-op IR dump path so it can handle
   `dataOpdscs_.size() > 1`, then use it to inspect the composed MLIR directly.

The current evidence is enough to say that LX-resident restickify is representable
as a Deeptools data-op sequence. It is not enough to claim measured zero-HBM
execution without AIUPTI, aiu-monitor, or lower-level fabric/HBM counters.
