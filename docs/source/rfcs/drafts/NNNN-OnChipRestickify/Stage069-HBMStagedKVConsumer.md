# Stage 069: HBM-Staged K/V Consumer

Date: 2026-05-27

## Purpose

Stage061 and Stage062 showed that direct HBM-to-consumer-LX K/V movement was
value-clean as data movement, but value-wrong once batchmatmul consumed that
operand from an LX-only Tensor1 descriptor.  The lower-stack trace explained the
failure: DXP only synthesizes the normal HBM-to-LX staging transfer when the
labeledDs is HBM-pinned.  The LX-only sidecar instead takes the local
`no_component -> LX` path.

Stage069 adds a K/V consumer mode that keeps the original HBM K/V producer and
leaves the consumer Tensor1 descriptor HBM-pinned.  This is not overlap yet; it
is the contract bridge needed before we can safely schedule future K/V staging
next to compute.

## Implementation

New gate:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_STAGED=1
```

The K/V pair builder now accepts `hbm_staged=True`.  In that mode it:

- keeps the original HBM K/V producer;
- does not `apply_lx_flip` to the consumer input1;
- does not add the `no_component -> LX` input-fetch marker;
- emits only a `nop` barrier dataop before the consumer compute row;
- records `kv_repack_hbm_staged: true` in consumer metadata.

The sweep variant is:

```text
kv_repack_pair_hbm_staged_auto
```

## Device Result

Run:

```text
tools/onchip_sdpa_sweep.py \
  --variants kv_repack_pair_hbm_staged_auto \
  --batch 1 --heads 8 --lengths 256 --dim 64 --block-size 64 \
  --warmup 1 --iters 3 --timeout-s 240 --dxp-debug \
  --cache-prefix /tmp/sdpa-stage069-hbm-staged-b64 \
  --output-json /tmp/sdpa-stage069-hbm-staged-b64.json
```

Result:

```text
status = ok
median = 0.610877 ms
max_abs_error = 0.00439453
mixed_sdscs = 9
consumer sidecar = sdsc_mixed_flash_kv_repack_broadcast_pair_3_input1_consumer
```

Control run with the previous direct-load canonical-name variant on the same
shape still emitted the K/V consumer sidecar and failed values:

```text
variant = kv_repack_pair_hbm_direct_load_canonical_name_auto
status = failed
mismatches = 2663 / 131072
max_abs_error = 0.3720703125
```

## Descriptor Evidence

Pre-DXP consumer sidecar:

```text
Tensor1 dsType = KERNEL
Tensor1 memOrg = hbm + lx
Tensor1 allocate = allocate-Tensor1_hbm
input_fetch_transfer = false
opFuncsUsed = [nop]
kv_repack_hbm_staged = true
```

Final DXP JSON:

```text
Tensor1 memOrg = hbm + lx + ptxrf
allocate_lds1_lx prev = loop_ds0_ds1_in
transfer_lds1_src:hbm_dst:lx src = {unit: l3lu, storage: hbm}
transfer_lds1_src:lxlu_dst:ptrow0..7 src = {unit: lxlu, storage: lx}
```

This proves DXP selected the HBM-pinned staging branch instead of the previous
LX-local `no_component -> LX` branch.

## Interpretation

The new mode closes the immediate correctness gap for the K/V consumer
descriptor.  We now have a passing K/V pair sidecar for the same block-64 H8/L256
family where the LX-only direct-load variant failed values.

The next step is to build real overlap on top of this contract: replace the
current `nop` barrier with a scheduled future K/V stage when the hoist scanner
finds a producer-consumer pair that requires 8-core to 32-core K/V staging.

## Verification

Local:

```text
python3 -m py_compile \
  torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_config_logic.py \
  tests/_inductor/test_onchip_realize_logic.py \
  tests/_inductor/test_onchip_sdpa_sweep_logic.py

manual no-arg test harness:
  test_config_logic: 38 loaded
  test_onchip_sdpa_sweep_logic: 59 loaded
  test_onchip_realize_logic: 113 loaded
  ran 210 tests

git diff --check
```

Pod:

```text
/home/adnan-cdx/dt-inductor-mixed/.venv/bin/python -m py_compile \
  torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py
```
