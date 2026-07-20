# Coherent Flash placement factorial artifacts (2026-07-20)

This package contains the compact, reviewer-facing evidence for the corrected
Flash placement-by-handoff experiment. The campaign passed its preregistered
correctness, structural, trace-inventory, determinism, closure, and factorial
analysis gates.

## Result

At `B1 H4 Lq512 Lk4096 D128`, group size 8, coherent placement reduced the
mean LX process-median kernel time from 210.7515 us to 199.0549 us, a mean
1.05877x speedup across five fresh-process blocks. The default explicit-shuffle
residual relative to the graph oracle was 29.9292 us; the coherent residual was
17.1684 us. The paired interaction was 12.7608 us (95% Student-t interval
11.9086 to 13.6130 us), closing 42.62% (95% interval 41.12% to 44.12%) of the
default residual.

The coherent LX arm reached 91.37% of its placement-matched oracle rate. The
remaining oracle headroom is therefore about 1.0944x for this shape. This is an
end-to-end Flash kernel result, not a standalone LX bandwidth measurement.

## What made the placement value-correct

The corrected contract assigns `work_div_inner_first` to the actual scaled-K
producer, the synthetic relayout source, the query producer, the first BMM, the
stable-softmax path, and the second BMM. The semantic rule is:

> Every unshuffled local-LX producer-to-consumer edge must retain one physical
> mapping. Only an explicit shuffle may bridge mappings.

The first prototype violated this rule by changing the synthetic relayout-source
view while leaving the actual scaled-K producer under its default mapping. The
high-contrast correctness gate found 241,384 mismatches out of 262,144 elements
at `rtol=atol=1e-2`. That arm was stopped before timing. The source-only negative
control reproduced the failure; source-closed and destination-closed controls
passed. The fully coherent contract passed in both compile orders and with an
adversarial V-coded value pattern.

## Oracle and statistical method

The oracle is allocation-matched and graph-local: an untimed prefix executes
through the shuffle and seeds its destination immediately before each timed
oracle graph; the timed graph differs by omitting only the explicit shuffle.
Wrong-preseed, no-reseed, restore, graph-identity, and materialization checks
make sure that the oracle cannot silently read a hidden input or stale value.

The factorial contains `lx_default`, `oracle_default`, `lx_coherent`, and
`oracle_coherent`. Each of five counterbalanced fresh-process blocks collected
30 accepted device events per condition. Oracle setup events were classified
separately and excluded. Inference uses the process median in each cell and a
paired two-sided Student-t interval across blocks. These are descriptive
single-device process-block intervals, not inference across five independent
devices. Performance was not an execution gate.

## Route-proxy warning

`remote_relations`, `total_hop_units`, and `max_directed_link_units` in this
package are **software shortest-path proxies** derived from compiler mappings.
They are not hardware-counter measurements, measured bytes, achieved bandwidth,
or physical link occupancy. The proxy changed from 224/2048/40 under default
placement to 224/672/16 under coherent placement. This supports the route-shape
explanation but does not by itself establish percent-of-peak ring utilization.

## Package map

- `SUMMARY.json`: compact headline values, intervals, structural outcomes, and
  limitations.
- `PROVENANCE.json`: allowlisted revision and toolchain provenance; hardware and
  host identifiers are deliberately omitted.
- `factorial/PREREGISTRATION.txt`: the corrected preregistration captured before
  inferential timing.
- `factorial/factorial_report.json`: canonical analyzer output.
- `factorial/RUN_QC_SUMMARY.json`: path-free per-run gate, checksum, and timing
  summary. Raw traces and generated summaries are intentionally excluded.
- `factorial/*SUCCESS.json` and `TERMINAL_STATUS.json`: campaign and block status.
- `structural/`: compact negative and positive controls. Each compact report
  records the SHA-256 of its original large report; those large reports are
  omitted because they embed generated artifacts and absolute paths.
- `harness/`: the exact experiment scripts used for structural gating, timing,
  and analysis. `HARNESS_SHA256SUMS` records their package hashes.
- `SHA256SUMS`: package integrity manifest, excluding the manifest itself.

Excluded on purpose: environment dumps, hardware/host identifiers, traces,
tensor outputs, compiler caches, console logs, credentials, and external links.
