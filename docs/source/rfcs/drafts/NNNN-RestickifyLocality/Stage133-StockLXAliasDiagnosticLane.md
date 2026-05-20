# Stage 133: Stock LX Alias Diagnostic Lane

## Goal

Keep the stock-restickify-by-LX-aliasing path, but explicitly treat it as a
diagnostic probe rather than a production candidate.

The hypothesis for this lane is narrow:

```text
Reuse stock restickification by LX aliasing:
  useful as a probe
  not production-worthy yet
  likely brittle
  correctness failure is a fundamental metadata/contract mismatch
```

Stage 132 showed why the existing alias can launch but produce wrong values:
HBM is a global exchange point, while LX addresses are local to each core. If a
restickify core reads local LX without matching the producer's physical
ownership, it reads the wrong logical region.

## Implementation

`tools/restickify_scenario_probe.py` now adds explicit ownership diagnostics to
the stock LX alias path.

When `SPYRE_RESTICKIFY_STOCK_LX_ALIAS=1`, the probe records:

- producer work splits
- restickify work splits
- producer/restickify core chunk sizes
- producer/restickify logical dimension totals
- whether core maps are equal
- whether the restickify core map is valid for its own split factors
- per-core local overlap between:
  - the producer-owned region in that core's LX
  - the restickify-needed region read from that core's LX
- `direct_lx_alias_safe`
- `requires_remote_lx_fetch`

The new safety switch:

```text
SPYRE_RESTICKIFY_STOCK_LX_ALIAS_REQUIRE_SAFE=1
```

turns the path into a pure diagnostic. If direct local aliasing is not
ownership-safe, the probe logs the mismatch and leaves the original stock HBM
path untouched.

This is useful because it lets us test whether a candidate graph is even
eligible for direct aliasing before we let the unsafe alias produce wrong
values.

## Validation Run

Command run in the pod:

```sh
SPYRE_RESTICKIFY_STOCK_LX_ALIAS=1 \
SPYRE_RESTICKIFY_STOCK_LX_ALIAS_REQUIRE_SAFE=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 \
  --ring-telemetry \
  --kernel-launch-log \
  --copy-kernel-code \
  --lx-boundary-stitch-prototype \
  --output-dir /tmp/stage133-stock-lx-alias-diagnostic \
  --fail-on-error
```

Result:

```text
status: ok
restickifies: 1
bytes: 524288
byte_hops: 4194304
```

The run stayed value-correct because `REQUIRE_SAFE=1` rejected the unsafe alias
and preserved the HBM path.

## Ownership Diagnostic

Before copying the producer core map:

| Field | Value |
|---|---:|
| producer split | `mb:32,out:1` |
| restickify split | `mb:4,out:8` |
| core maps equal | `false` |
| restickify core map valid | `true` |
| invalid restickify map entries | `0` |
| min local overlap | `0.0` |
| avg local overlap | `0.03125` |
| max local overlap | `0.125` |
| direct LX alias safe | `false` |
| requires remote LX fetch | `true` |

After copying the producer core map onto the restickify:

| Field | Value |
|---|---:|
| producer split | `mb:32,out:1` |
| restickify split | `mb:4,out:8` |
| core maps equal | `true` |
| restickify core map valid | `false` |
| invalid restickify map entries | `28` |
| min local overlap | `0.0` |
| avg local overlap | `0.00390625` |
| max local overlap | `0.125` |
| direct LX alias safe | `false` |
| requires remote LX fetch | `true` |

Core 0 sample before the copy:

```text
producer owns:      mb [0,16),   out [0,512)
restickify needs:   mb [0,128),  out [0,64)
overlap:            0.125
```

Core 1 sample before the copy:

```text
producer owns:      mb [16,32),   out [0,512)
restickify needs:   mb [128,256), out [0,64)
overlap:            0.0
```

## Interpretation

Copying the producer core map is not enough. It can make the maps look equal,
but the copied indices may be invalid for the restickify op's own split
contract. In this run, copying `mb:0..31,out:0` onto a restickify split of
`mb:4,out:8` creates 28 invalid `mb` slice indices.

So this path confirms the fundamental blocker:

```text
direct LX aliasing is only sound when each physical core already owns exactly
the logical region that the restickify op will read from that core's LX.
```

For the normal stock restickify split, that condition is false. The missing
piece is not just metadata polish; it is a real ownership transfer/fetch
contract.

## Next Use

This diagnostic lane remains useful for:

1. finding rare cases where direct aliasing is actually safe;
2. proving why a candidate graph is not safe;
3. validating Stage 3B-style locality certificates;
4. preventing accidental "works because it launches" conclusions.

It should not be presented as the production solution. For a general LX-to-LX
restickify, we still need either:

- a real stock `ReStickifyOpLx` contract with valid producer/restickify/consumer
  ownership, or
- an explicit remote-LX fetch/data-op bridge.
