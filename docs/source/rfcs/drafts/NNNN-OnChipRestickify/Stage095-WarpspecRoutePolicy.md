# Stage095 - Warpspec Route Policy

## Question

The decoupled loader-specialized FlashAttention path is correctness-certified
on an eight-row gate island, but it is not uniformly faster than
`onchip_master`. What is the next production-facing interface between the gate
and a future dispatcher?

## Change

Add:

```text
tools/onchip_sdpa_route_policy.py
tests/_inductor/test_onchip_sdpa_route_policy_logic.py
```

The tool consumes the JSON produced by `tools/onchip_sdpa_perf_compare.py` and
emits a shape-selective route table for one baseline comparison.

Typical use:

```text
tools/onchip_sdpa_perf_compare.py \
  --gate onchip_warpspec_decoupled \
  --cases all \
  --baseline-variants flash_hbm,onchip_master \
  --warmup 2 \
  --iters 7 \
  --seed 42865 \
  --output-json /tmp/sdpa-warpspec-perf.json

tools/onchip_sdpa_route_policy.py \
  --input-json /tmp/sdpa-warpspec-perf.json \
  --baseline-variant onchip_master \
  --min-speedup 1.0 \
  --output-json /tmp/sdpa-warpspec-route-policy.json
```

The policy is intentionally conservative:

```text
select target route only if:
  baseline_status == ok
  target_status == ok
  baseline_median_ms / target_median_ms >= min_speedup

otherwise:
  select the fallback route
```

By default, the target route is the perf JSON's `target_variant`, and the
fallback route is the selected baseline variant.

## Why This Is Needed

The promotion gate proves that the target row emitted the required artifact:

```text
current-prefetch sidecar
loader core 31
loader fanout
full-tile fanout pieces
serialized loader-core prefetch
STCDPOpHBM
pointwise handoff
value correctness within the gate bound
```

That is necessary for production, but not sufficient for route selection. The
dispatcher also needs to know whether the certified target is preferred over
the strongest available baseline for that shape.

Stage234 showed:

```text
PERF_SUMMARY baseline=flash_hbm ok_pairs=8/8 geomean_speedup=1.1518x
PERF_SUMMARY baseline=onchip_master ok_pairs=8/8 geomean_speedup=0.9929x
```

So the decoupled target is clearly better than `flash_hbm`, but still
shape-selective relative to `onchip_master`.

## Stage234 Route Table At `min_speedup=1.0`

Using the Stage234 medians, the initial performance-preferred subset is:

```text
+---------------------+------+---------------+--------------+----------+--------------------+
| Shape               | L    | onchip_master | decoupled ms | Speedup  | Route              |
+---------------------+------+---------------+--------------+----------+--------------------+
| B1 H4 D64 block64   | 768  | 1.626099      | 1.571266     | 1.0349x  | decoupled warpspec |
| B1 H4 D64 block64   | 1024 | 2.194906      | 2.173174     | 1.0100x  | decoupled warpspec |
| B1 H8 D64 block64   | 384  | 0.954747      | 0.968222     | 0.9861x  | onchip_master      |
| B1 H8 D64 block64   | 512  | 1.267971      | 1.275051     | 0.9944x  | onchip_master      |
| B2 H4 D128 block64  | 384  | 1.102760      | 1.151739     | 0.9575x  | onchip_master      |
| B2 H4 D128 block64  | 512  | 1.486903      | 1.555549     | 0.9559x  | onchip_master      |
| B2 H4 D128 block64  | 768  | 3.115857      | 3.109740     | 1.0020x  | decoupled warpspec |
| B2 H4 D128 block64  | 1024 | 4.821906      | 4.796391     | 1.0053x  | decoupled warpspec |
+---------------------+------+---------------+--------------+----------+--------------------+
```

The B2/H4/D128 long-row wins are small. They are useful evidence for an initial
policy table, but they should be repeated before being treated as a durable
default in a runtime dispatcher.

## Output Contract

`tools/onchip_sdpa_route_policy.py` prints a concise text summary:

```text
ROUTE_POLICY_SUMMARY gate=onchip_warpspec_decoupled baseline=onchip_master \
target=onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled \
selected=4/8 min_speedup=1.0000

ROUTE_ROW case=... B=1 H=4 D=64 L=768 block=64 \
route=onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled \
reason=speedup_met_threshold speedup=1.0349x ...
```

The JSON output contains:

```text
gate
cases
baseline_variant
target_variant
target_route
fallback_route
min_speedup
routes[]
summary
```

Each route row records shape fields, chosen route, reason, statuses, medians,
and speedup.

## Interpretation

This stage does not make warpspec the production default. It creates the missing
selection artifact that a production dispatcher can consume later.

The current route-policy implication is:

```text
correctness-certified island:
  all 8 decoupled gate rows

performance-preferred island at min_speedup=1.0:
  B1 H4 D64  block64 L768,L1024
  B2 H4 D128 block64 L768,L1024

fallback to onchip_master:
  B1 H8 D64  block64 L384,L512
  B2 H4 D128 block64 L384,L512
```

This gives us a clean next boundary:

1. Repeat the route-policy perf input with stronger benchmark settings.
2. Decide the minimum speedup margin for production routing.
3. Wire the resulting table into the actual compile-time/runtime selection path
   only after the table is stable.
