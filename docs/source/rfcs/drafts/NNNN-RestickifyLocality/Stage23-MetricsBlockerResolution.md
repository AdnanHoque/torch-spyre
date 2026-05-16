# Stage 23: Metrics Blocker Resolution

## Summary

The next blocker was the lower metrics writer:

```text
/tmp/metrics.<bus>
```

We wanted to patch below torch-spyre/libaiupti, in the senlib/Flex path that
actually produces that file, so that hardware counter samples could be surfaced
more directly into our profiling workflow.

The result:

- the active writer is in senlib, not in the libaiupti checkout we patched
- the source file is identified by the installed binary as
  `/project_src/senlib/senlib/1p0/monitoring.cpp`
- the normal torch-spyre developer checkout does not clone senlib source
- the accessible `spyre-runtime` repo is a packaging/image repo, not the senlib
  source repo
- the installed `aiu-monitor` wheel already contains a supported parser path
  (`aiu-smi`, `dcr_parser.py`, `libaiusmi`)

Therefore the practical solution is two-track:

1. Use an `aiu-smi`/`libaiusmi` sidecar for current measurements.
2. Only attempt the lower senlib patch if we can fetch the senlib source RPM or
   source repo.

## Evidence

The runtime log reports:

```text
[monitoring.cpp: 61] Opening Metrics File: /tmp/metrics.0000:aa:00.0
```

`strings` on the installed libraries points at:

```text
/project_src/senlib/senlib/1p0/monitoring.cpp
Opening Metrics File:
monitoring.cpp
```

The installed RPMs identify the exact source package names:

```text
ibm-senlib-core-2.0.0-0.main.1+109.22167bb_0.el10.src.rpm
ibm-senlib-dd2-2.0.0-0.main.1+109.22167bb_0.el10.src.rpm
```

Installed binary RPMs:

```text
ibm-senlib-core 2.0.0-0.main.1+109.22167bb_0.el10
ibm-senlib-dd2  2.0.0-0.main.1+109.22167bb_0.el10
```

The packaging repo is reachable:

```text
git@github.ibm.com:ai-foundation/spyre-runtime.git
```

but it contains only image/RPM packaging, not senlib source. Its scripts install:

```text
ibm-senlib-core
ibm-senlib-dd2
ibm-flex
ibm-libaiupti
```

The usual source checkout script also does not clone senlib. It has a build flag
for `--senlib`, but only if a sibling `senlib/` directory already exists.

## Immediate Solution: Supported Sidecar

Use `aiu-smi` as the stable counter sidecar around the exact timed region.

This is already implemented for restickify probes:

```sh
python tools/restickify_aiusmi_marker_probe.py \
  --case adds_then_matmul_x \
  --size 2048 \
  --mode baseline \
  --mode stage3b \
  --warmup 5 \
  --iters 200 \
  --sample-interval 0.05 \
  --output-dir /tmp/restickify-aiusmi-marker-stage3b
```

The tool:

- starts `aiu-smi` only around the timed loop
- records all exposed metric groups with `aiu-smi -s -g A`
- archives `/tmp/metrics.0000:aa:00.0`
- writes CSV, JSONL, SVG, and interactive HTML
- preserves compiler restickify telemetry for comparison

The installed `aiu-monitor` wheel exposes:

```text
aiu-smi
dcr_parser.py
metric_state_helper.py
libaiusmi.cpython-312-x86_64-linux-gnu.so
```

`libaiusmi` exposes these counter families:

```text
cardMem
dma
dmaDsc
dmaIn
dmaOut
mem
power
rdma
rdmaIn
rdmaOut
thermal
```

No RIU/ring counter is exposed in this installed parser API.

## Lower Patch Solution, If Source Is Available

If we can obtain the source RPM or source repo, the lower patch should target:

```text
senlib/1p0/monitoring.cpp
```

Suggested workflow:

```sh
mkdir -p /home/adnan-cdx/dt-inductor-profiler/senlib-src
cd /home/adnan-cdx/dt-inductor-profiler/senlib-src

# Fetch one of these from Artifactory/source RPM storage:
#   ibm-senlib-core-2.0.0-0.main.1+109.22167bb_0.el10.src.rpm
#   ibm-senlib-dd2-2.0.0-0.main.1+109.22167bb_0.el10.src.rpm

rpm2cpio ibm-senlib-core-*.src.rpm | cpio -idmv
```

Then inspect:

```sh
rg -n "Opening Metrics File|/tmp/metrics|monitoring.cpp|Metric" .
```

The useful patch would add a debug/env-gated secondary emitter next to the
existing metrics write path, for example:

```text
SPYRE_SENLIB_METRIC_TAP_JSONL=/tmp/senlib-metric-tap.jsonl
```

The tap should emit sample timestamp, bus id, counter group, raw value fields,
and the source metric file path. It should not replace the existing
`/tmp/metrics.<bus>` file, because `aiu-smi` depends on that file today.

## Why The Sidecar Is The Right Current Path

The sidecar gives us the best available observability without modifying runtime
libraries:

- torch-profiler/acelyzer gives per-kernel AIU timing
- restickify telemetry gives modeled byte movement and byte-hops
- `aiu-smi` gives HMI/LPDDR-facing memory rates and request rates
- the interactive HTML gives a single report for comparing these layers

It does not prove RIU traffic directly. But our searches show that the exposed
installed tools also do not name a direct RIU counter. Until a senlib source
patch, source RPM, or internal counter selector appears, the honest claim is:

```text
Stage 3B removes modeled RIU byte-hops; observed HMI/LPDDR traffic stays
roughly unchanged; direct RIU counter evidence is not currently exposed.
```

## Next Actions

1. Keep using the sidecar flow for restickify experiments.
2. Add acelyzer trace parsing to the sidecar report so kernel timings and
   memory counters are shown in one HTML.
3. Fetch the senlib source RPM from Artifactory if we need to patch
   `monitoring.cpp`.
4. If source RPM access is unavailable, treat direct RIU proof as blocked and
   use synthetic P2P/ring benchmarks from `aiu-toolbox` only as calibration.
