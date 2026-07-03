# Grouped LX All-Gather Standalone Sweep, 2026-07-03

## Summary

After the full flash-sized grouped all-gather failed `senulator -v store`, we reduced the descriptor while preserving the same communication class:

```text
many producer LX chunks -> multiple consumer cores needing a grouped/all-gathered LX operand
```

The result is sharper than the full flash smoke:

```text
small grouped all-gather works
full topology with small payload works
full topology fails once the generated transfer row reaches 32+ transactions
```

This points away from a pure coordinate-contract issue and toward a backend executable-lowering issue for larger grouped `STCDPOpLx` transfers.

## Results

| Sample | Cores | Producer Pieces | Consumer Pieces | Fanout | `trVolume` | `numTransactions_` | Store Check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `FlashGroupedAllgatherSmall` | 4 | 4 | 8 | 2 | 8192 | 16 | passed |
| `FlashGroupedAllgatherFanout4` | 8 | 8 | 32 | 4 | 8192 | 16 | passed |
| `FlashGroupedAllgatherFanout8X1` | 8 | 14 | 64 | 8 | 8192 | 16 | passed |
| `FlashGroupedAllgatherMid` | 16 | 16 | 64 | 4 | 8192 | 16 | passed |
| `FlashGroupedAllgatherFullIn16` | 32 | 32 | 256 | 8 | 8192 | 16 | passed |
| `FlashGroupedAllgatherFullIn32` | 32 | 32 | 256 | 8 | 16384 | 32 | failed |
| `FlashGroupedAllgatherFullIn64` | 32 | 32 | 256 | 8 | 32768 | 64 | failed |
| `FlashGroupedAllgather` | 32 | 32 | 256 | 8 | 65536 | 128 | failed |

## Read

The pass/fail boundary is currently not fanout alone:

- fanout 8 with 8 cores passes;
- fanout 8 with 32 cores and small payload passes;
- full flash topology with 32 cores and fanout 8 fails when transfer volume grows.

The cleanest backend question is now:

```text
Why does grouped STCDPOpLx execution diverge from PCFG reference once each producer row reaches 32 or more transactions?
```

That is likely much easier for Deeptools/DCC owners to reason about than the original full flash attention failure.

## DCC Flag Sweep

I also tried a small DCC flag sweep on the first failing case, `FlashGroupedAllgatherFullIn32`:

| Variant | `senpcfg` | `dcc-opt` | `senulator` | Store Check |
| --- | ---: | ---: | ---: | --- |
| baseline | 0 | 0 | 1 | failed |
| `--dcc-multicast-canonicalization-disable` | 0 | 134 | skipped | unknown |
| `--dcc-burst-splitting-disable` | 0 | 0 | 1 | failed |
| `--dcc-sync-send-recv-fusion-disable` | 0 | 0 | 1 | failed |
| `--dcc-store-and-forward-fusion-disable` | 0 | 0 | 1 | failed |
| `--dcc-mutable-addr-splitting-disable` | 0 | 0 | 1 | failed |

None of the obvious DCC toggles fixed the mismatch. Disabling multicast canonicalization aborts in DCC, which is also a useful clue: this path appears to rely on multicast canonicalization, but the resulting executable program is still value-wrong for larger transfer rows.

## Chunking Experiment

I also tried splitting the full flash-sized payload into smaller logical `in=16` chunks while preserving the full `32-core, fanout-8` grouped all-gather topology.

| Sample | Producer Pieces | Consumer Pieces | Fanout | `trVolume` | `numTransactions_` | Store Check |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `FlashGroupedAllgatherChunkedIn16` | 256 | 2048 | 8 | 8192 | 16 | failed |

This rules out a purely per-row transaction-count explanation. The failure may also involve the number of grouped rows, repeated multicast groups targeting the same destination allocation, or a DCC lowering state issue across many `STCDPOpLx` grouped transfer rows.

## Implication For Granite/Flash Collectives

This confirms that `broadcast` / `multicast` / `layout_allgather` are plausible through the DLDSC coordinate contract, but the production path still needs a backend fix before we can use this class to remove the flash attention HBM round trips.

Scatter PR1 remains separate: it covers the smaller one-to-one class and is not invalidated by this result.

## Captured Files

- [grouped_allgather_sweep/metadata_summary.csv](grouped_allgather_sweep/metadata_summary.csv)
- [grouped_allgather_sweep/topology_sweep_summary.csv](grouped_allgather_sweep/topology_sweep_summary.csv)
- [grouped_allgather_sweep/payload_sweep_summary.csv](grouped_allgather_sweep/payload_sweep_summary.csv)
- [grouped_allgather_sweep/in32_dcc_flag_sweep_summary.csv](grouped_allgather_sweep/in32_dcc_flag_sweep_summary.csv)
- [grouped_allgather_sweep/chunked_in16_summary.csv](grouped_allgather_sweep/chunked_in16_summary.csv)
- [grouped_allgather_sweep/chunked_in16_lowering_and_senulator.txt](grouped_allgather_sweep/chunked_in16_lowering_and_senulator.txt)
- [grouped_allgather_sweep/deeptools_grouped_allgather_sweep_experiment.patch](grouped_allgather_sweep/deeptools_grouped_allgather_sweep_experiment.patch)
- [grouped_allgather_sweep/deeptools_grouped_allgather_chunked_experiment.patch](grouped_allgather_sweep/deeptools_grouped_allgather_chunked_experiment.patch)
- [grouped_allgather_sweep/deeptools_grouped_allgather_sweep_experiment_diff_stat.txt](grouped_allgather_sweep/deeptools_grouped_allgather_sweep_experiment_diff_stat.txt)
