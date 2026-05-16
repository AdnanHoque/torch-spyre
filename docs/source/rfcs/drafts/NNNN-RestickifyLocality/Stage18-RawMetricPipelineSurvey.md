# Stage 18: Raw Metric Pipeline Survey

This note records what we found after inspecting the currently available toolchain sources for fabric-specific observability. The short version is: the hardware topology explicitly models RIU LX-LX movement, and deeptools has a software performance model for LX-to-LX relayouts, but the visible `QueryMetrics`/AIUPTI/`aiu-smi` metric surface we can inspect does not currently expose a named RIU data-ring traffic counter.

## Sources Checked

| Source | Path | Finding |
|---|---|---|
| Deeptools sysconfig | `/tmp/ai-chip-toolchain-deeptools/dsc/HardwareArchMapping/sysConfigs2.0/sentient_dd2_sysconfig.json` | Canonical fabric topology and bandwidths are present in `systemDef.connections`. |
| Deeptools RIU perf estimator | `/tmp/ai-chip-toolchain-deeptools/dsm/perfEstimator/IntraEntityScheduler/RCUIntraEntityScheduler.cpp` | Models `HBM_to_LX`, `LX_to_HBM`, and `LX_to_LX` data-transfer entities using RIU/MNI links and sysconfig bandwidths. |
| Flex telemetry | `/tmp/ai-chip-toolchain-flex/flex/src/telemetry/telemetry_utils.cpp` and `aiupti_profiler.cpp` | Sends `QueryMetrics`, receives `AiuMetrics`, and packages returned term IDs into AIUPTI metric activities. |
| libaiupti metric writer | `/tmp/ai-chip-toolchain-libaiupti/src/aiupti/aiupti_metric.cpp` | The old-format metric mapping includes HMI, RMI/RMO, MCI, DMI/DMO, power, and temperature terms; no RIU data-ring term was found. |
| Installed senlib metric IDs | pod path `/opt/ibm/spyre/senlib/include/hal/aiu_metrics.hpp` | Exposes generic HW perf counters plus HMI/MCI/RMI/RMO/DMI/DMO terms, but no obvious `RIU`, `RING`, `DATARING`, or `SFPDataIU` value term. |
| aiu-toolbox | `/tmp/ai-chip-toolchain-aiu-toolbox` | Contains inter-device P2P/RDMA/topology benchmarks and helpers, not an on-chip RIU LX-LX counter path. |
| Spyre KB | `wiki/foundations/hardware/observability.md`, `wiki/artifacts/rfcs/0601-spyre-profiling-toolkit.md` | Confirms the intended profiling stack shape, but does not add a currently surfaced RIU counter beyond the toolchain paths above. |

## Fabric Topology From Sysconfig

The DD2 sysconfig directly matches the hardware notes we have been using:

| Fabric | Sysconfig source | Nodes | Frequency | Link bandwidth | Agents / role |
|---|---:|---:|---:|---:|---|
| RIU data BiRing | `RIU_Grp_All` | 33 | 1.3 GHz | `RIU-To-RIU-Link = 128 B/cyc` | HBM and MNI; carries HBM-core data and cross-core LX-LX data. |
| RIU request BiRing | `RIURequest_Grp_All` | 33 | 1.3 GHz | `RIURequest-To-RIURequest-Link = 1 B/cyc` | HBM and MNI request/control traffic. |
| SFP data UniRing clockwise | `SFPDataIU_Corelet0` | 32 | 1.1 GHz | `32 B/cyc` | SFP/PSUM traffic for corelet 0. |
| SFP data UniRing counter-clockwise | `SFPDataIU_Corelet1` | 32 | 1.1 GHz | `32 B/cyc` | SFP/PSUM traffic for corelet 1. |
| On-core FIFO links | `FIFO-Links` | per core | 1.1 GHz | `128 B/cyc` | LX, MNI, PT, PE, and SFP component-to-component links. |

This matters for ReStickify because the relevant path for an in-graph computed relayout is not SFP traffic. The sysconfig names the RIU data BiRing as the fabric connecting HBM and per-core MNIs, which is the path deeptools uses for core-to-core LX-LX data transfer modeling.

## QueryMetrics / AIUPTI Path

The visible metric pipeline looks like this:

1. `libaiupti` starts a sampler thread.
2. Flex telemetry sends a `QueryMetrics` message.
3. Firmware/runtime writes an `AiuMetrics` payload into a pinned host buffer.
4. Flex iterates over returned `AiuMetricsTermHdr` records.
5. AIUPTI records the returned metric term IDs and values.
6. `aiu-smi`/old metric files map a small subset of known term IDs to user-facing names.

The important limitation is that `QueryMetrics` itself is not a rich selector API in the code we inspected. The visible call passes a device buffer address and length; it does not request "RIU data ring bytes" or "SFP ring bytes" as a named metric. The returned term IDs are therefore only as useful as the firmware/runtime-selected metric set and the exposed `AiuMetricsTermId` catalog.

## Exposed Metric Terms We Found

The installed `aiu_metrics.hpp` exposes:

- Generic hardware perf counter controls and slots:
  - `TERM_ID_VALUE_HW_PERF_COUNTER_CTRL`
  - `TERM_ID_VALUE_HW_PERF_COUNTER_0..7`
  - `TERM_ID_VALUE_HW_SOC_PERF_COUNTER_CTRL`
  - `TERM_ID_VALUE_HW_SOC_PERF_COUNTER_0..3`
- HMI / HMI SOC memory counters:
  - `TERM_ID_VALUE_HMI_SOC_PERF_WR_DATA_BEATS`
  - `TERM_ID_VALUE_HMI_SOC_PERF_RD_DATA_BEATS`
- MCI request counters:
  - `TERM_ID_VALUE_MCI_REQ_READ_*`
  - `TERM_ID_VALUE_MCI_REQ_WRITE_*`
- RMI/RMO LPDDR and transfer counters:
  - `TERM_ID_VALUE_RMI_HCI2RMI_WRITE_LPDDR`
  - `TERM_ID_VALUE_RMO_HCI2RMO_READ_LPDDR`
  - related `XFER*`, `CRC`, `WDONE`, and control terms
- DMI/DMO event, duration, and stall counters
- power, temperature, and other device-level terms

We did not find a named term containing `RIU`, `RING`, `DATARING`, or `SFPDataIU` in the exposed metric IDs or the libaiupti old-format writer.

## Deeptools LX-LX Model

Deeptools does know about the LX-LX relayout path as a software/performance-estimation concept:

- It classifies data-transfer entities as `HBM_to_LX`, `LX_to_HBM`, or `LX_to_LX`.
- For `HBM_to_LX`, it forms a path like:
  - `HBM -> HBM_RIU -> CoreRIU -> CoreMNI`
- For `LX_to_HBM`, it forms the reverse:
  - `CoreMNI -> Core_RIU -> HBM_RIU -> HBM`
- For `LX_to_LX`, it:
  - compares producer and consumer logical regions,
  - builds a transfer table for overlapping regions where producer core and consumer core differ,
  - maps that traffic onto RIU links,
  - identifies a critical RIU link by aggregate volume,
  - estimates cycles from `volume / sysconfig_bandwidth`.

That is a strong independent confirmation that the compiler/backend model treats core-to-core relayout as an RIU LX-LX problem, not as an SFP ring problem.

## How This Connects To ReStickifyOpHBM

`ReStickifyOpHBM` in generated SDSC names does not by itself prove a full round trip through HBM for an in-graph restickify. The name says the op materializes/restickifies an HBM-backed tensor object in the generated program interface. Deeptools still has separate data-transfer modeling paths for HBM-LX and LX-LX movement.

Our observed `aiu-smi` balanced read/write traffic during isolated `ReStickifyOpHBM` runs is still real evidence of device-memory traffic, but it is HMI/LPDDR-facing evidence. It cannot, by itself, distinguish:

- pure HBM round-trip,
- LX-LX relayout plus HBM-backed materialization,
- or a mixture of HBM traffic and on-chip RIU traffic inside the kernel.

The Stage 3B compiler telemetry remains a software model of the eligible in-graph LX-LX component. It estimates producer-core to restickify-core movement using logical region ownership and physical core distance. The `aiu-smi` counters validate device-memory traffic around the kernel, not the exact RIU byte-hop term.

## Current Conclusion

The best defensible conclusion is:

1. The sysconfig and deeptools model confirm that RIU data-ring LX-LX movement is a valid path for relayout/restickify-like data movement.
2. The exposed `QueryMetrics`/AIUPTI/`aiu-smi` metric surface we can inspect today does not provide a named, user-facing RIU data-ring traffic counter.
3. Current `rdmem`/`wrmem` measurements should be interpreted as HMI/LPDDR-facing memory traffic, not as direct RIU traffic.
4. Stage 3B byte-hop telemetry is therefore a compiler-side locality model, not yet a hardware-counter measurement.
5. To directly prove RIU traffic, we need either a hidden selector for the generic HW perf counter slots, firmware/runtime support that returns RIU terms through `AiuMetrics`, or another tool that already configures those counters.

## Next No-People Step

The most useful next step that does not involve asking other teams is to chase the generic hardware perf counter controls:

1. Search deeptools, flex, senlib headers, and installed configs for selector names or config keys that program `TERM_ID_VALUE_HW_PERF_COUNTER_CTRL` and `TERM_ID_VALUE_HW_SOC_PERF_COUNTER_CTRL`.
2. Dump raw `AiuMetrics` term IDs during an isolated restickify run and confirm whether generic counter slots change in a useful way.
3. If selector programming is discoverable, run a differential experiment:
   - HBM-only copy or graph-input restickify,
   - in-graph baseline restickify with nonzero Stage 3B byte-hops,
   - Stage 3B-aligned restickify with zero modeled byte-hops.
4. Only treat a counter as RIU evidence if it changes with the in-graph byte-hop delta and does not track the HMI read/write counters.

Until then, the strongest story is: Stage 3B is certified by the compiler model and supported by the deeptools architecture/perf model, while hardware validation is currently limited to coarse HMI memory counters and kernel timing.
