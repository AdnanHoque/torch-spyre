# Stage 062: K/V Input1 LX Allocator Shape

Date: 2026-05-27

## Purpose

Stage061 ruled out the simple `dsType_=INPUT` fix for the LX-backed K/V
`batchmatmul` input1 contract.  Stage062 tests the next narrow descriptor
hypothesis: maybe the failure is not the K/V bytes or `dsType_`, but the LX
allocation shape created by `apply_lx_flip`.

The passing HBM-backed input1 path ends after DXP with:

```text
Tensor1 memOrg = hbm allocate-Tensor1_hbm
              + lx allocate_lds1_lx
              + ptxrf allocate_lds1_ptxrf
```

The failing executable pair ended with:

```text
Tensor1 memOrg = lx allocate-Tensor1_lx
              + ptxrf allocate_lds1_ptxrf
```

The new probe asks whether using Foundation's canonical `allocate_lds1_lx`
name for the prefilled LX endpoint changes the K/V input1 result.

## New Gate

Stage062 adds:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_CONSUMER_LX_ALLOC_STYLE=
```

Values:

```text
""              preserve apply_lx_flip's default allocate-TensorN_lx name
canonical_name  rename the consumer LX endpoint to allocate_ldsN_lx
canonical_loop  try to attach that allocation to loop_ds0_ds1_in when present
```

The sweep rows are:

```text
kv_repack_pair_hbm_direct_load_canonical_name_auto
kv_repack_pair_hbm_direct_load_canonical_loop_auto
```

Both keep the original HBM-backed K/V producer, direct-load the K/V source
from HBM into every consumer core's LX slot, disable the optional
input-fetch-neighbor transfer marker, and leave the K/V input as
`dsType_=KERNEL`.

## Device Result

Run:

```text
B=1, H=8, L=256, D=64, block=64, causal=0, seed=0
cache prefix: /tmp/sdpa-stage062-kv-lxalloc
```

Results:

```text
kv_repack_pair_hbm_direct_load_canonical_name_auto:
  status = failed
  mismatches = 3162 / 131072
  max abs diff = 0.56396484375 at (0, 1, 178, 13)

kv_repack_pair_hbm_direct_load_canonical_loop_auto:
  status = failed
  mismatches = 3162 / 131072
  max abs diff = 0.56396484375 at (0, 1, 178, 13)
```

This is identical to the prior direct-load pair signature.

The canonical-name pre-DXP sidecar had the intended mutation:

```text
kv_repack_consumer_lx_alloc_style = canonical_name
Tensor1 dsType_ = KERNEL
Tensor1 memOrg = lx allocate_lds1_lx
allocate_lds1_lx first address = 278528
```

After DXP, the descriptor still had the canonical LX endpoint and PTXRF:

```text
Tensor1 memOrg = lx allocate_lds1_lx
              + ptxrf allocate_lds1_ptxrf
allocate_lds1_lx first address = 278528
```

The real pre-DXP `batchmatmul` descriptor does not yet contain
`loop_ds0_ds1_in`; Foundation creates that staging loop during DXP.  Therefore
`canonical_loop` cannot attach the allocation to that loop before DXP in this
path, and on the real device row it effectively collapses to the
canonical-name case.  DXP's final loop still did not include an input1
`allocate_lds1_lx` or `transfer_lds1_src:hbm_dst:lx` staging step.

## Interpretation

The K/V input1 mismatch is not caused by the non-canonical
`allocate-Tensor1_lx` name alone.  Renaming the prefilled LX endpoint to the
same `allocate_lds1_lx` name used by Foundation's HBM-backed path compiles,
survives into the final descriptor, and still produces the exact same value
error.

The remaining delta is the Foundation-generated HBM input staging path itself:

```text
allocate_lds1_lx
transfer_lds1_src:hbm_dst:lx
...
transfer_lds1_src:lxlu_dst:ptrow*
```

The executable pair only pre-populates an LX allocation before compute.  It
does not trigger the same DXP-generated HBM-to-LX staging transfer for input1.
The K/V `batchmatmul` contract is therefore more specific than "Tensor1 is
present in an LX allocation with the right logical bytes."

The next production-aligned route should stop spending turns on raw LX input1
fanout knobs unless a backend-supported way to create that HBM-style staging
path from LX is identified.  Keeping K/V HBM-backed while overlapping the
other flash pipeline work is now the more credible implementation path.

## Verification

Local:

```text
python3 -m py_compile ...                                         # pass
python3 tests/_inductor/test_config_logic.py                      # 37/37 pass
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py           # 54/54 pass
python3 tests/_inductor/test_onchip_realize_logic.py              # 111/111 pass
```

Pod:

```text
python3 -m py_compile ...                                         # pass
python3 tests/_inductor/test_config_logic.py                      # 37/37 pass
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py           # 54/54 pass
python3 tests/_inductor/test_onchip_realize_logic.py              # 111/111 pass
```

Device:

```text
tools/onchip_sdpa_sweep.py \
  --lengths 256 \
  --variants kv_repack_pair_hbm_direct_load_canonical_name_auto,kv_repack_pair_hbm_direct_load_canonical_loop_auto \
  --batch 1 --heads 8 --dim 64 --block-size 64 \
  --seed -256 --warmup 1 --iters 2 --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage062-kv-lxalloc \
  --output-json /tmp/sdpa-stage062-kv-lxalloc.json
```
