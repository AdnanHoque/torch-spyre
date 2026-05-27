# Stage 055: Layout-Xform Hoisted-Future Builder

Date: 2026-05-27

## Purpose

Stage055 adds the next fail-closed probe toward a warp-specialized prefill
attention analogue on Spyre:

```text
row 0: hoist an independent future ReStickifyOpHBM producer
row 1: compute the current batchmatmul while prefetching the future input in LX
```

This differs from Stage054 lookahead.  Lookahead requires the future producer to
already be before the current consumer in bundle order.  Hoist targets the real
block64 graph shape where a future layout-transform producer can be independent
but appears after the current compute tile.

## Change

A new default-off tile gate was added:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_HOIST_TILE=-2
```

The sweep harness exposes:

```text
layout_xform_hoist_auto
```

When legal, the hoist builder emits:

```text
*_current_consumer
  compute 0: future ReStickifyOpHBM producer, output LX-pinned
  datadsc 0: STCDPOpLx layout-transform prefetch into the future input buffer
  compute 1: current batchmatmul
  schedule: [[-1, 0, 0, 1], [0, 1, 1, 0]]

*_future_consumer
  original future batchmatmul, selected input LX-pinned to the prefetched buffer
```

The original future `ReStickifyOpHBM` is omitted from `bundle.mlir`; otherwise
the hoisted producer would execute once inside the current sidecar and again at
its original bundle position.  The original JSON is still written for debugging.

The builder rejects unless:

- the future producer is exactly `ReStickifyOpHBM`;
- the selected edge is still a strict layout-transform pair edge;
- the future producer appears after the current consumer and before the future
  consumer;
- the future producer inputs are HBM-backed and already available before the
  current consumer;
- replacement and omission names do not conflict.

The layout-transform pair-edge helper now accepts nonzero consumer inputs for
this hoist scan.  For non-input0 operands, it maps the producer split dimension
through the layout transform when the consumer op's split dimension is not part
of the selected operand layout.

## Local Validation

```text
python3 -m py_compile torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_config_logic.py \
  tests/_inductor/test_onchip_realize_logic.py \
  tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_config_logic.py
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_realize_logic.py
python3 tests/_inductor/test_onchip_flash_pipeline_logic.py
git diff --check
```

Results:

```text
test_config_logic.py: 12/12 passed
test_onchip_sdpa_sweep_logic.py: 14/14 passed
test_onchip_realize_logic.py: 59/59 passed
test_onchip_flash_pipeline_logic.py: 11/11 passed
git diff --check: clean
```

The new tests cover:

- concrete hoist gate parsing;
- the `layout_xform_hoist_auto` sweep variant;
- current-sidecar schedule shape;
- future-consumer LX pinning for `input1`;
- bundle omission of the original future `ReStickifyOpHBM`;
- rejection of dependent future producers;
- rejection of non-`ReStickifyOpHBM` producers.

## Pod Result

The touched files were synced into:

```text
/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
```

The same torch-free pod tests passed:

```text
test_config_logic.py: 12/12 passed
test_onchip_sdpa_sweep_logic.py: 14/14 passed
test_onchip_realize_logic.py: 59/59 passed
test_onchip_flash_pipeline_logic.py: 11/11 passed
```

The block64 L128 probe used:

```text
variant=layout_xform_hoist_auto
cache=/tmp/sdpa-stage055-layout-xform-hoist-localdxp-layout_xform_hoist_auto-B1-H2-L128-D64-C0-638830-57807
```

It did not select a hoist sidecar.  The real generated graph showed:

```text
3_ReStickifyOpHBM -> 4_batchmatmul input1
```

as the independent future K/V layout-transform candidate, but that edge is not
yet representable by the current LX prefetch builder:

```text
future_tile1:input1:invalid_split:mb_
```

The producer is a 2-core `ReStickifyOpHBM` whose producer split maps to the
future consumer's `x_` operand dimension.  The future `batchmatmul` is a 32-core
consumer split over `mb_`, and `mb_` is not present in the K/V `input1` layout.
The current `STCDPOpLx` descriptor builder can model producer-LX to consumer-LX
layout transforms when the destination operand is sharded compatibly with the
consumer cores.  It cannot yet model the broadcast/repack needed to make a
2-core K/V LX buffer available to a 32-core matmul split over query rows.

The later Stage053-style edge was also visible:

```text
14_ReStickifyOpHBM -> 15_batchmatmul input0
```

That edge is a valid layout-transform pair edge, but it is not a legal hoist
candidate: the producer depends on the intervening softmax/pointwise chain, so
moving it before the current tile would be a true data-dependency violation.

The L256 block64 probe showed the same family of blockers: independent K/V
input1 candidates either had multiple consumers, an unsupported operand split,
or a low-core-count producer feeding a 32-core future matmul.  The block32 probe
did not compile because layout propagation rejected the generated matmul layout.

## Current Status

Stage055 makes the hoist path explicit and bundle-safe, but the current real
SDPA graphs still do not expose a device-exercisable, value-correct hoisted LX
prefetch.  The next required primitive is a K/V operand repack or broadcast
contract that can take a low-core-count restickify output and materialize the
LX input expected by a 32-core future batchmatmul, or a scheduler contract that
can overlap that low-core producer without forcing the K/V operand into the
current same-shard STCDP model.
