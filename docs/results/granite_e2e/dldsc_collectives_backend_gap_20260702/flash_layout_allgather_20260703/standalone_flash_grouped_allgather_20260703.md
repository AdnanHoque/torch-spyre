# Standalone Flash Grouped All-Gather DataOp Probe, 2026-07-03

## Summary

This probe isolates one flash-attention relayout edge from the full `test_flash.py` bundle and lowers it as a standalone `STCDPOpLx` data-op. The goal was to answer one narrow question:

```text
Can Deeptools/DCG lower the exact grouped all-gather descriptor shape that the flash bundle wants?
```

Answer: yes, for the isolated descriptor. `DataOpStandalone` exits `0`, emits PCFG/MLIR, and computes a transfer table with the expected 1-to-8 fanout.

This does not prove full flash correctness. The full flash graph is currently value-wrong even when all relayout flags are off in this experimental checkout, so the full graph is not a clean correctness oracle yet.

## Run

Pod:

```text
adnan-cdx-spyre-dev-pf
```

Root:

```text
/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525
```

Command:

```bash
SENARCH=sen1p5 ./dcc/bin/DataOpStandalone \
  --dataDscSample FlashGroupedAllgather \
  --ddsc-out-dir /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/dataop_flash_grouped_allgather_20260703_032921
```

Exit:

```text
0
```

## Descriptor Shape

The standalone data-op models the sampled flash edge:

```text
producer LX tensor -> consumer batchmatmul KERNEL LX operand
```

The logical data shape is:

```text
out = 4096
in  = 128
x   = 4
```

The source side has 32 producer subpieces:

```text
each producer subpiece: out=512, in=128, x=1
source LX base: 131072
```

The destination side has 256 consumer subpieces:

```text
each consumer subpiece: out=512, in=128, x=1
each destination core receives all 8 out chunks for one x value
destination LX base: 0
```

The transfer table has:

```text
pSubPiece rows: 32
cSubPiece rows: 256
dtTable rows: 32
fanout per producer row: 8
maxConsumers: 8
```

Example stdout:

```text
0 --> [ 0 4 8 12 16 20 24 28 ]
1 --> [ 1 5 9 13 17 21 25 29 ]
2 --> [ 2 6 10 14 18 22 26 30 ]
3 --> [ 3 7 11 15 19 23 27 31 ]
...
maxConsumers: 8
```

## Comparison With Full Flash Debug Artifact

The standalone descriptor and the full flash relayout debug descriptor agree on the important generated fields:

| Field | Standalone data-op | Full flash debug relayout |
| --- | ---: | ---: |
| `pSubPiece` count | 32 | 32 |
| `cSubPiece` count | 256 | 256 |
| `dtTable_` count | 32 | 32 |
| Fanout per `dtTable_` row | 8 | 8 |
| Example source LX base | `131072` | `131072` |
| Example destination LX base | `0` | `0` |

The first generated transfer row is also structurally the same:

```text
pIDX=0
cIDXs=[0, 32, 64, 104, 136, 168, 208, 240]
cMemIDs=[0, 12, 16, 20, 24, 28, 4, 8]
trVolume=65536
numTransactions_=128
selectedMCMode=1
```

This strongly suggests the direct `STCDPOpLx` carrier can represent this grouped all-gather class once the source/destination coordinates are explicit.

## What This Proves

- `STCDPOpLx` is not limited to single-destination scatter for this shape.
- DCG can compute a legal multicast/grouped-transfer table for the flash-like 1-to-8 LX movement.
- The generated standalone descriptor matches the flash relayout debug descriptor at the `pSubPiece`, `cSubPiece`, and `dtTable_` level.

## What This Does Not Prove

- It does not prove full `test_flash.py` correctness.
- It does not prove runtime value correctness after the movement.
- It does not prove that the full mixed compute/data schedule binds the consumer batchmatmul to the post-relayout LX location correctly.

The full flash control runs are currently value-wrong even with relayout disabled:

```text
all relayout flags off: ~75.1% mismatch, 0 backend plans
collectives disabled:  ~75.1% mismatch, 0 backend plans
one relayout edge on:  ~90.7% mismatch, 1 backend plan
```

So the full graph cannot yet be used as the only oracle for this communication class.

## Current Best Read

The backend transfer primitive can lower the grouped all-gather descriptor. The remaining risk has moved to integration:

- consumer operand rebinding to the relayout output allocation;
- schedule placement relative to producer/consumer compute;
- source/destination address lifetime overlap in the full bundle;
- or a broader value issue in the current flash test environment.

The next useful test is a small patterned runtime harness that writes known values into the producer LX shards, runs exactly this grouped all-gather, and checks the destination LX bytes before any downstream attention math.

## Captured Files

- [standalone_flash_grouped_allgather/sdsc.json](standalone_flash_grouped_allgather/sdsc.json)
- [standalone_flash_grouped_allgather/sdsc_pre.json](standalone_flash_grouped_allgather/sdsc_pre.json)
- [standalone_flash_grouped_allgather/dataop_stdout.txt](standalone_flash_grouped_allgather/dataop_stdout.txt)
- [standalone_flash_grouped_allgather/dataop_stderr.txt](standalone_flash_grouped_allgather/dataop_stderr.txt)
- [standalone_flash_grouped_allgather/dataop_exit.txt](standalone_flash_grouped_allgather/dataop_exit.txt)
- [standalone_flash_grouped_allgather/deeptools_flash_grouped_allgather_experiment.patch](standalone_flash_grouped_allgather/deeptools_flash_grouped_allgather_experiment.patch)
- [standalone_flash_grouped_allgather/deeptools_flash_grouped_allgather_experiment_diff_stat.txt](standalone_flash_grouped_allgather/deeptools_flash_grouped_allgather_experiment_diff_stat.txt)
