# Stage 21: Profiler Device Properties Fix

## Summary

This stage fixed the immediate `acelyzer` ingestion blocker for
torch-profiler PrivateUse1 traces. The exported Chrome trace now contains a
non-empty AIU `deviceProperties` object instead of `deviceProperties: []`.

The fix is on the separate profiler branch:

```text
AdnanHoque/profiler-metric-visibility
d975d57 debug: emit AIU device properties
```

No PR was created.

## Knowledgebase Grounding

The Spyre knowledgebase does not currently document Chrome trace
`deviceProperties` directly. The useful clue came from
`wiki/interfaces/pytorch-to-torch-spyre.md`, which says the Spyre device
interface already reports device availability, a 32-core/multi-processor count,
and compute capability to PyTorch/Dynamo.

The live PR-1856 profiler bridge already implemented `getDeviceInfo()`, which
creates the visible trace process row labeled `AIU 0`. However,
`IActivityProfilerSession::getDeviceProperties()` defaulted to an empty string,
and torch-spyre did not override it. That matched the observed trace:

```text
deviceProperties: []
```

## Prototype Fix

The profiler branch now overrides:

```c++
std::string AiuptiActivityProfilerSession::getDeviceProperties();
```

and returns a minimal AIU metadata object:

```json
{
  "id": 0,
  "name": "AIU 0",
  "type": "AIU",
  "multiProcessorCount": 32,
  "computeCapability": "dd2",
  "coreCount": 32
}
```

This is intentionally minimal. The values mirror the current Spyre device
interface and are enough for downstream trace ingestion. A production version
should eventually query runtime/device properties rather than hardcoding DD2
and 32 cores.

## Validation

Smoke workload:

```sh
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 128 \
  --skip-correctness \
  --torch-profiler \
  --warmup 0 \
  --iters 1 \
  --output-dir /tmp/restickify-profiler-deviceprops-smoke \
  --fail-on-error
```

Trace check:

```text
deviceProperties= [{'id': 0, 'name': 'AIU 0', 'type': 'AIU',
                    'multiProcessorCount': 32,
                    'computeCapability': 'dd2',
                    'coreCount': 32}]
```

`acelyzer` now ingests the unmodified trace:

```text
ACELYZER_RC=0
Exported events: 36
```

The exported trace still contains:

```text
ERROR: gpuGetDeviceCount failed with code 35
```

but this is a PyTorch CUDA-side device-properties warning and no longer blocks
the AIU trace analyzer.

## What This Improves

- Before: torch-profiler traces had PrivateUse1 kernel timing but could not be
  ingested cleanly by `acelyzer` because `deviceProperties` was empty.
- After: torch-profiler traces have PrivateUse1 kernel timing **and** a usable
  AIU device metadata record.

This does not expose RIU traffic counters or AIUPTI metric records. It does
unlock the next useful profiler step: run `acelyzer` on marker-separated
restickify traces and compare its derived timeline output with torch-profiler
kernel timings and `aiu-smi`/senlib memory counters.
