# Stage 16: Visual Counter Reports

## Summary

This sidequest adds automatic visual summaries for restickify counter runs. The
goal is to make each run produce a compact, Nsight-style report instead of only
raw JSONL/CSV rows.

The visual report combines:

- AIU memory/fabric context from the Spyre Knowledgebase: 32 cores on a
  bidirectional RIU data ring, per-core LX scratchpad ownership, and off-chip
  device-memory traffic using ring-facing data paths.
- Compiler locality telemetry: total byte-hops, average/max hops, bytes moved,
  and restickify source attribution.
- Runtime timing: median, p10, and p90 for the measured loop.
- `aiu-smi` counters: average and peak device read/write memory bandwidth.

## Tool Change

Added:

```text
tools/restickify_profile_viz.py
```

It is dependency-free and uses only Python's standard library. This matters
because the Spyre pod environments are intentionally lean, and plotting packages
are not always installed.

The marker-separated counter probe now calls the visualizer by default:

```text
tools/restickify_aiusmi_marker_probe.py
```

Each run writes:

```text
aiusmi_marker_summary.svg
aiusmi_marker_summary.html
```

Use `--no-figure` to disable report generation for automated runs that only
want raw data.

## Example

For the Stage 15 `adds_then_matmul_x` run at size `2048`, the generated report
is available locally at:

```text
artifacts/restickify_aiusmi_marker/fused-2048-v2/aiusmi_marker_summary.svg
artifacts/restickify_aiusmi_marker/fused-2048-v2/aiusmi_marker_summary.html
```

The figure highlights the key interpretation:

- Stage 3B removes the modeled in-graph byte-hops.
- Restickify bytes and restickify count stay unchanged.
- `aiu-smi` read/write memory traffic remains present in both modes because the
  fused workload still contains graph-input/weight traffic, matmul, pointwise,
  input/output, and ordinary runtime/device-memory activity.

## Why This Shape

The report deliberately separates compiler-modeled locality from hardware
counters. That avoids the common mistake of treating `byte_hops == 0` as proof
that the whole fused kernel stopped touching device memory. The visualization
makes that distinction visible:

- compiler byte-hop bars answer "did the compiler remove the modeled ownership
  mismatch?";
- latency bars answer "did the measured loop get faster?";
- `aiu-smi` bars answer "is the full workload still producing device read/write
  traffic?";
- source bars answer "which restickifies were eligible in-graph edges versus
  graph-input/weight boundaries?".

## Next Improvements

The current report is intentionally static and portable. Good follow-ups are:

1. add a timeline view once per-op AIUPTI/PrivateUse1 events include stable
   restickify kernel names;
2. add RIU/HBM counter lanes if `aiu-smi` or AIUPTI exposes fabric-specific
   counters separately from aggregate read/write memory traffic;
3. annotate generated SDSC opfunc names such as `ReStickifyOpHBM` when the
   probe exports the bundle directory;
4. embed repeated-run distributions so we can distinguish noise from stable
   speedups.

