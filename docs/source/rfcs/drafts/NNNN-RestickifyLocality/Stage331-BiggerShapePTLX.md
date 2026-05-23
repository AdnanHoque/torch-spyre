# Stage 331: Bigger-Shape PT-LX Boundary Check

## Summary

Stage 331 checked whether the allocator-backed mixed PT-LX restickify prototype
extends past the proven 2048 square case.

The result is negative for the current default-off full-tensor prototype:

- 2048 still patches and is the only allocator-backed success in this sweep.
- 2176, 2304, 2432, 2560, 3072, and 4096 compile in the probe, but the PT-LX
  replacement is skipped.
- The skip reason is `producer-endpoint-not-allocator-backed:prototype-default`.
- `DXP_LX_FRAC_AVAIL=0` does not fix it; it makes the upstream add fail
  Deeptools scheduling for 2176 and larger.

So larger shapes need either better producer endpoint planning or the streaming
PT-LX bridge.  The current full-tensor endpoint path should stay scoped to the
allocator-backed 2048 case.

## Default Compile-Only Sweep

Common settings:

```sh
LX_PLANNING=1
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS unset
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E unset
```

Command shape:

```sh
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --size 2176 \
  --size 2304 \
  --size 2432 \
  --size 2560 \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code \
  --output-dir /tmp/stage331-cutoff-default \
  --fail-on-error
```

Probe rows reported successful compile for all sizes:

| Size | Probe status | Restickifies | Bytes |
|---:|---|---:|---:|
| 2048 | ok | 1 | 8,388,608 |
| 2176 | ok | 1 | 9,469,952 |
| 2304 | ok | 1 | 10,616,832 |
| 2432 | ok | 1 | 11,829,248 |
| 2560 | ok | 1 | 13,107,200 |

But the bridge audit shows only 2048 used the mixed PT-LX replacement:

| Size | Audit status | Result |
|---:|---|---|
| 2048 | patched | `1_MixedReStickifyOpWithPTLxConsumer` |
| 2176 | skipped | `producer-endpoint-not-allocator-backed:prototype-default` |
| 2304 | skipped | `producer-endpoint-not-allocator-backed:prototype-default` |
| 2432 | skipped | `producer-endpoint-not-allocator-backed:prototype-default` |
| 2560 | skipped | `producer-endpoint-not-allocator-backed:prototype-default` |

A separate compile-only sweep for 2560, 3072, and 4096 had the same skip reason
for all three sizes.

## DXP Scratchpad Headroom Check

Setting `DXP_LX_FRAC_AVAIL=0` was tested as a diagnostic.  It did not make
larger shapes allocator-backed.  Instead, Deeptools rejected the upstream add:

```text
DtException: Unable to map graph within architecture constraints:
The initial chunk parameters must fit in LX for SuperDSC: 0_add
```

This means the larger-shape blocker is not only our PT-LX replacement gate.
The producer side of the graph is no longer being planned in the same
allocator-backed LX endpoint form.

## Forced Endpoint Diagnostic

Forced environment endpoints were also tested compile-only:

```sh
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=1048576
```

Results:

| Size | Forced result | Meaning |
|---:|---|---|
| 2176 | patched, then DXP cardinality error | bridge can be emitted, but generated contract is invalid |
| 2304 | patched, then DXP cardinality error | bridge can be emitted, but generated contract is invalid |
| 2560 | skipped, `missing-intermediate-lx-space` | full-tensor intermediate no longer fits with chosen endpoints |
| 3072 | skipped, `missing-intermediate-lx-space` | full-tensor intermediate no longer fits with chosen endpoints |

For 2560 and 3072 the audit produced a valid streaming candidate:

| Size | Source cores | Dest cores | Tile workspace | Transfer bytes | Byte-hops |
|---:|---:|---:|---:|---:|---:|
| 2560 | 32 | 20 | 24,576 | 26,214,400 | 104,857,600 |
| 3072 | 32 | 24 | 24,576 | 37,748,736 | 150,994,944 |

This is not zero-hop locality.  It is an LX-to-LX streaming candidate that
would avoid HBM but still use RIU data-ring movement because the producer and
destination work distributions do not match.

## Interpretation

The full-tensor mixed PT-LX prototype is a narrow success:

```text
2048 square: allocator-backed, zero-hop, value-correct on hardware.
>2048 square: not allocator-backed by default; full-tensor bridge is not enough.
```

The larger-shape path should not try to force the 2048 mechanism.  It needs a
production-shaped streaming/tiled bridge:

- consume the producer's real LX output;
- allocate a bounded tile workspace instead of a full-tensor intermediate;
- materialize the restickified consumer view tile-by-tile;
- preserve the consumer-side metadata contract;
- tolerate source/destination core-count mismatch;
- model RIU cost explicitly because larger shapes are not necessarily
  zero-hop.

## Next Step

The next implementation step is the streaming PT-LX bridge, not broader
enablement of the full-tensor path.  Acceptance should be:

- 2048 keeps the allocator-backed zero-hop fast path;
- 2560 and 3072 compile through a streaming LX-to-LX path instead of
  `ReStickifyOpHBM`;
- hardware correctness passes before timing;
- the audit reports transfer bytes and byte-hops because the larger-shape path
  is locality-aware but not necessarily core-local.
