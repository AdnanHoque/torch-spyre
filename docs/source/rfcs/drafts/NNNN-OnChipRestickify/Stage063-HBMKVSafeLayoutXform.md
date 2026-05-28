# Stage 063: HBM-KV-Safe Layout Xform

Date: 2026-05-27

## Purpose

Stage061 and Stage062 ruled out the narrow descriptor fixes for consuming K/V
input1 directly from an LX-only sidecar.  The direct HBM-to-consumer-LX bytes
are value-clean when copied back to HBM, but the generated `batchmatmul`
consumer still needs Foundation's normal HBM-backed input1 staging path.

Stage063 pivots to the production-aligned overlap boundary that is already
passing: keep K/V HBM-backed, and overlap/promote the layout-transform path on
the query-side `batchmatmul` input0 adjacency.

## New Sweep Row

Stage063 adds an explicit sweep variant:

```text
onchip_hbm_kv_layout_xform
```

It is intentionally shaped like the passing master layout-transform path:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM=1
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE=<unset>
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PLAN_ARTIFACT=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_TILE=-1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_TILE=-1
SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
```

Unsetting the low-level layout-transform tile lets the config adjunct choose
its default auto tile.  The K/V-repack gates are explicitly disabled so parent
probe env cannot accidentally turn this HBM-KV-safe row into an LX-K/V row.

## Device Result

Run:

```text
B=1, H=8, L=256, D=64, block=64, causal=0, seed=0
warmup=1, iters=2
```

Baseline Stage063 controls:

```text
variant                     status  median_ms  max_abs_error  mixed_sdscs
flash_hbm                   pass    0.619872   0.00488281     0
onchip_layout_xform         pass    0.569912   0.00488281     19
onchip_master_layout_xform  pass    0.574272   0.00488281     19
```

Named HBM-KV-safe row:

```text
variant                    status  median_ms  max_abs_error  mixed_sdscs
onchip_hbm_kv_layout_xform pass    0.565406   0.00488281     19
```

The timing is only a two-iteration diagnostic sample, not a benchmark.  The
important result is value correctness with mixed sidecars generated while K/V
remains on the HBM-backed path.

## Descriptor Evidence

The named row cache was:

```text
/tmp/sdpa-stage063-hbmkv-variant-onchip_hbm_kv_layout_xform-B1-H8-L256-D64-C0-680111-97920
```

No generated path in that cache contained `kv`.  The layout-transform consumer
sidecar was:

```text
inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_h6mrvhfw/sdsc_mixed_flash_layout_xform_pair_tile_2_consumer.json
```

Its `flashAttentionPipeline_` metadata showed:

```text
layout_xform_pair_role = consumer
layout_xform_attached_input_idx = 0
layout_xform_predecessor_sdsc = 14_ReStickifyOpHBM
layout_xform_predecessor_output_idx = 1
layout_xform_consumer_sdsc = 15_batchmatmul
split_dim = mb_
iter_sizes = {x_: 8, mb_: 256, in_: 64}
slice_bytes = 262144
layout_xform_predecessor_lx_base = 16384
layout_xform_input_lx_base = 8192
```

The consumer `batchmatmul` LDS summary was:

```text
lds0 Tensor0 INPUT  memOrg = lx       allocate-Tensor0_lx
lds1 Tensor1 KERNEL memOrg = hbm + lx
lds2 Tensor2 OUTPUT memOrg = hbm + lx
```

This is the boundary we want: the layout-transform sidecar attaches only to
input0, while K/V input1 remains a normal HBM-backed `KERNEL` operand.

## Interpretation

The working near-term path is not to force K/V input1 into LX.  The validated
route is to keep the K/V operand under Foundation's HBM input staging contract
and continue moving the safe query-side layout-transform and pointwise/score
handoff work into mixed sidecars.

This gives the warp-specialized attention effort a stable production lane:

- `onchip_hbm_kv_layout_xform` is the explicit correctness row for HBM-backed
  K/V plus on-chip layout-transform overlap.
- K/V repack variants remain diagnostics until there is a backend-supported way
  to reproduce the HBM-style input1 staging path from LX.
- The next implementation work should promote this HBM-KV-safe matrix and then
  layer additional already-safe sidecars around it, rather than reopening raw
  LX-only K/V fanout knobs.

## Verification

Local:

```text
python3 -m py_compile tools/onchip_sdpa_sweep.py tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py  # 55/55 pass
```

Pod:

```text
python3 -m py_compile tools/onchip_sdpa_sweep.py tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py  # 55/55 pass
```

Device:

```text
tools/onchip_sdpa_sweep.py \
  --lengths 256 \
  --variants onchip_hbm_kv_layout_xform \
  --batch 1 --heads 8 --dim 64 --block-size 64 \
  --seed -256 --warmup 1 --iters 2 --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage063-hbmkv-variant \
  --output-json /tmp/sdpa-stage063-hbmkv-variant.json
```
