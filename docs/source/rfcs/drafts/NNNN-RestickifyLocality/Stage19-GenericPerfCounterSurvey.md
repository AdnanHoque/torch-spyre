# Stage 19: Generic Perf Counter Survey

This note follows up on Stage 18 by asking a narrower question: is there a no-people path to program a generic hardware counter or raw metric selector so we can directly observe RIU data-ring traffic for restickify?

## Short Answer

Not with the currently visible installed software stack.

We found strong topology and performance-model evidence that the RIU data BiRing is the path for cross-core LX-LX traffic, but we did not find a shipped `aiu-smi`, senlib, libaiupti, or monitoring-script interface that exposes RIU traffic as a named metric or selector.

## Sources Checked

| Source | Path | Finding |
|---|---|---|
| DD2 sysconfig | `/tmp/ai-chip-toolchain-deeptools/dsc/HardwareArchMapping/sysConfigs2.0/sentient_dd2_sysconfig.json` | Defines `RIU` and `RIURequest` BiRings plus SFP UniRings and on-core FIFO links. |
| Deeptools RIU estimator | `/tmp/ai-chip-toolchain-deeptools/dsm/perfEstimator/IntraEntityScheduler/RCUIntraEntityScheduler.cpp` | Models HBM-LX and LX-LX transfer paths through core RIU/MNI nodes and RIU links. |
| libaiupti metric writer | `/tmp/ai-chip-toolchain-libaiupti/src/aiupti/aiupti_metric.cpp` | Old-format writer maps HMI, RMI/RMO, DMI/DMO, power, and temperature terms; no RIU term. New-format writer exists but is not implemented. |
| Flex telemetry | `/tmp/ai-chip-toolchain-flex/flex/src/telemetry/telemetry_utils.cpp`, `/tmp/ai-chip-toolchain-flex/flex/src/telemetry/aiupti_profiler.cpp` | `QueryMetrics` returns metric term arrays, but the visible call does not request a specific fabric selector. |
| senlib monitoring configs | `/opt/ibm/spyre/senlib/etc/senlib_config_monitoring*.json` | Configures DMA, HMI, MCI, RMI, RMO, SLU, MSG, power, and temperature monitoring. No `SNT_RIU`. |
| aiu-smi config | `/home/adnan-cdx/dt-inductor-profiler/.venv-py212/etc/senlib_config_aiusmi.json` | Same pattern as senlib monitoring: HMI/RMI/RMO/DMA-style counters, no RIU section. |
| senlib monitoring parser | `/opt/ibm/spyre/senlib/etc/scripts/monitoring/counter.py` | Has counter tables and readers for DMI, DMO, HMI, MCI, SLU, MSG, RMI, and RMO. No RIU reader/table. |
| installed `ibm-aiu-monitor` | `/home/adnan-cdx/dt-inductor-profiler/.venv-py212/lib/dcr_parser.py`, `libaiusmi*.so` | User-facing metric groups are `rdmem`, `wrmem`, PCIe/RDMA rates, request rates, power, temperature, busy, and memory usage. No RIU-facing field. |
| Spyre Knowledgebase | `wiki/foundations/hardware/observability.md`, `wiki/artifacts/rfcs/0601-spyre-profiling-toolkit.md` | Describes the intended profiling stack and inter-core communication efficiency as a goal, but does not identify a currently surfaced RIU counter. |

## Topology Confirmation

The DD2 sysconfig confirms the hardware fabric facts we have been using:

| Fabric | Sysconfig object | Frequency | Bandwidth | Role |
|---|---|---:|---:|---|
| RIU data BiRing | `RIU_Grp_All` | 1.3 GHz | `RIU-To-RIU-Link = 128 B/cyc` | HBM/core data and cross-core LX-LX data via HBM and MNI agents. |
| RIU request BiRing | `RIURequest_Grp_All` | 1.3 GHz | `RIURequest-To-RIURequest-Link = 1 B/cyc` | HBM read/write request headers. |
| SFP UniRing, corelet 0 | `SFPDataIU_Corelet0` | 1.1 GHz | `32 B/cyc` | SFP/PSUM traffic for corelet 0. |
| SFP UniRing, corelet 1 | `SFPDataIU_Corelet1` | 1.1 GHz | `32 B/cyc` | SFP/PSUM traffic for corelet 1. |
| On-core FIFO links | `FIFO-Links` | 1.1 GHz | `128 B/cyc` | Per-core LX/MNI/PT/PE/SFP component links. |

This is why our Stage 3B byte-hop model is pointed at the RIU data ring rather than the SFP rings. ReStickify movement between two different physical cores' LX ownership regions is an MNI/RIU problem in the deeptools model.

## What aiu-smi Actually Exposes

The installed `aiu-smi` package exposes these user-facing data-rate metrics:

- `rdmem`
- `wrmem`
- `rxpci`
- `txpci`
- `rdrdma`
- `wrrdma`

It also exposes request-rate variants such as `n_rdmem` and `n_wrmem`, plus power, temperature, busy, and memory reservation/usage fields.

The parser and its binary extension do not expose fields named `riu`, `ring`, `lx_lx`, `sfp`, or similar. Strings in `libaiusmi*.so` show counter classes for DMA, RDMA, memory, power, and temperature. They do not show an RIU counter class.

That means `aiu-smi` is useful for HMI/LPDDR-facing validation and coarse memory-traffic timing, but not for direct RIU fabric attribution.

## Raw Metric Attempt

We also tried a raw metric smoke around the known restickify workload using:

- `SENLIB_DEVEL_CONFIG_FILE=/home/adnan-cdx/dt-inductor-profiler/.venv-py212/etc/senlib_config_aiusmi.json`
- `AIUPTI_ENABLE_METRICS=1`
- `AIUPTI_METRIC_PATH=/tmp/restickify-raw-term-smoke2/metrics.%BUSID`
- `DTLOG_LEVEL=TRACE`

The run completed, and senlib produced a metric file at:

```text
/tmp/metrics.0000:aa:00.0
```

However, the workload log did not contain `QueryMetrics`, `oldFormatWriter`, `Skip TERM_ID`, `AIUptiMetric`, or raw term IDs. In other words, this did not reveal hidden RIU term IDs through the libaiupti path.

Reading the produced metric file through the installed parser shows the expected `aiu-smi` CSV schema, but the columns are still the visible HMI/host/RDMA-style fields:

```text
rdmem, wrmem, rxpci, txpci, rdrdma, wrrdma, n_rdmem, n_wrmem, ...
```

No RIU/fabric-specific columns appear.

## Why New AIUPTI Metric Format Does Not Solve It Yet

`aiupti_metric.cpp` contains an `AIUPTI_USE_NEW_FORMAT` path, but the new-format writer begins with an assertion and is explicitly not implemented. So there is no currently usable "dump every term in a new format" escape hatch in the libaiupti code we inspected.

The old-format writer is intentionally narrow. It maps a small set of old metric offsets:

- DMI/DMO transfer events
- HMI SOC read/write data beats
- RMI/RMO LPDDR transfer terms
- MCI terms
- power and temperature

Unknown term IDs would be logged as skipped, but our raw smoke did not show unknown returned terms in the workload log.

## Interpretation For Restickify

This strengthens, but also narrows, the claim we can make:

1. The hardware topology and deeptools performance model support the premise that in-graph restickify can create RIU LX-LX traffic when ownership moves between cores.
2. Stage 3B reduces the compiler-modeled RIU byte-hop cost by aligning producer and restickify ownership.
3. `aiu-smi` can show whether the kernel causes LPDDR/HMI-facing traffic, but it cannot prove the RIU portion directly.
4. The installed counter stack does not appear to expose the specific fabric counter we want.

So Stage 3B remains a compiler locality optimization backed by static topology and runtime timing, not by direct RIU hardware counters.

## Practical Consequence

For restickify experiments, the best available measurement stack remains:

| Evidence | What it proves | What it does not prove |
|---|---|---|
| Stage 3B telemetry | Modeled core-to-core byte-hops from producer ownership to restickify ownership. | Actual fabric counter increments. |
| SDSC/opfunc names and kernel timings | Which generated kernel ran and how long it took. | Whether the payload used RIU, HBM, or both internally. |
| `aiu-smi` `rdmem`/`wrmem` | LPDDR/HMI-facing traffic and bandwidth. | Cross-core RIU LX-LX traffic. |
| Deeptools sysconfig/perf estimator | The intended fabric path and bandwidth model. | Runtime counter evidence for one kernel execution. |

## Recommended Next Step

The no-people path is close to exhausted for direct RIU counters. The next useful work should be measurement design rather than more blind source searching:

1. Keep using `aiu-smi` to separate HMI-heavy and HMI-light restickify cases.
2. Use marker-separated kernel timing to isolate `ReStickifyOpHBM` runtime.
3. Use compiler telemetry to classify the same kernels as graph-input/weight, in-graph nonlocal, or Stage 3B-local.
4. Build a differential table:
   - graph-input/weight restickify: expect HMI traffic, Stage 3B cannot help;
   - in-graph baseline restickify: expect modeled RIU byte-hops plus any HMI materialization;
   - in-graph Stage 3B restickify: expect modeled RIU byte-hops near zero, HMI traffic unchanged if materialization is still present.

If that table lines up, it gives us a defensible story even without an RIU counter: Stage 3B removes the compiler-modeled cross-core locality loss, while separate HMI counters explain why the end-to-end latency gain is bounded.
