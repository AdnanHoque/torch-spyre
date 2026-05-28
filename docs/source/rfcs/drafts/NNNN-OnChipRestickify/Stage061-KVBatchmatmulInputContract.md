# Stage 061: K/V Batchmatmul Input Contract

Date: 2026-05-27

## Purpose

Stage061 narrows the K/V repack investigation to the descriptor contract for
the flash prefill `batchmatmul` K/V operand.  Stage060 showed that direct
HBM-to-consumer-LX movement is value-clean when copied back to the original HBM
input, while the executable pair remains value-wrong when the same LX region is
consumed directly by `batchmatmul`.

The remaining question is whether the bad executable pair is caused by a
simple logical descriptor mismatch on the K/V operand, especially `dsType_`, or
by a deeper Foundation/DL input1 layout contract.

## New Gate

Stage061 adds a focused diagnostic override:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_CONSUMER_DS_TYPE=
```

The default empty value preserves existing behavior.  A non-empty value is
written onto the LX-flipped consumer input LDS after `apply_lx_flip`.

The first sweep row using this gate is:

```text
kv_repack_pair_hbm_direct_load_dsinput_auto
```

It sets:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_TILE=-2
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_IFN_TRANSFER=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_DIRECT_LOAD=1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_CONSUMER_CORE_STATE_INIT=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_CONSUMER_DS_TYPE=INPUT
```

This variant keeps the original HBM-backed K/V producer, directly loads K/V
from HBM into every future consumer core's LX input slot, disables the optional
input-fetch-neighbor marker, omits the explicit `coreStateInit_` values, and
tries to make the LX-backed K/V operand look like an `INPUT` operand rather than
a `KERNEL` operand.

## Descriptor Evidence

The failing direct-load executable pair with default `dsType_=KERNEL` used:

```text
cache: /tmp/sdpa-stage060-direct-load-csi-rerun-kv_repack_pair_hbm_direct_load_auto-B1-H8-L256-D64-C0-674974-529552
sidecar: sdsc_mixed_flash_kv_repack_broadcast_pair_3_input1_consumer.json
```

The K/V consumer input was:

```text
ldsIdx_              = 1
dsName_              = Tensor1
dsType_              = KERNEL
memOrg_              = lx allocate-Tensor1_lx
hbmStartAddress_     = -1
hbmSize_             = 0
consumer_lx_base     = 278528
source_lx_base       = 16384
source_piece_count   = 8
destination_pieces   = 256
consumer layout      = [in_, x_, out_]
stick dim            = out_
mapped split         = x_
consumer split       = mb_
```

DXP kept that operand as an LX-only `KERNEL` operand and the value run failed
with:

```text
3162 / 131072 mismatches
max abs diff 0.56396484375 at (0, 1, 178, 13)
```

The direct-load copyback controls used the same HBM-to-consumer-LX load pattern
but copied a selected consumer replica back to the original HBM input before
the unchanged HBM-backed consumer ran.  These passed across core selections:

```text
kv_repack_copyback_hbm_direct_load_auto:   PASS, max err 0.0048828125
kv_repack_copyback_hbm_direct_load_core0:  PASS, max err 0.0048828125
kv_repack_copyback_hbm_direct_load_core1:  PASS, max err 0.0048828125
kv_repack_copyback_hbm_direct_load_core8:  PASS, max err 0.0048828125
kv_repack_copyback_hbm_direct_load_core16: PASS, max err 0.0048828125
kv_repack_copyback_hbm_direct_load_core31: PASS, max err 0.0048828125
```

This separates data movement from consumption: the K/V bytes placed in
consumer LX are good enough to round-trip through HBM and feed the normal HBM
consumer, but not good enough for the generated LX-backed K/V `batchmatmul`
operand.

The unchanged HBM-backed `batchmatmul` path has a different post-DXP shape.
Its input1 remains HBM-backed, and Foundation also creates an internal LX
allocation plus PTXRF plumbing.  That suggests the normal K/V path is not a
plain "load these HBM-layout bytes into an arbitrary LX address" contract;
Foundation's HBM load path appears to restage or format the operand for
`batchmatmul` input1.

Stage039 also provides a useful contrast.  The successful layout-transform pair
flipped `batchmatmul` input0 to LX as `dsType_=INPUT`, and DXP accepted it.
That path uses a different operand position and descriptor class from K/V
input1.  It should not be assumed to prove that nonzero K/V input1 can be
treated the same way.

## Device Result

The `dsType_=INPUT` A/B row was run on the neutral K/V candidate shape:

```text
B=1, H=8, L=256, D=64, block=64, causal=0, seed=0
variant: kv_repack_pair_hbm_direct_load_dsinput_auto
cache: /tmp/sdpa-stage061-kv-dsinput-kv_repack_pair_hbm_direct_load_dsinput_auto-B1-H8-L256-D64-C0-677192-938926
```

The generated sidecar had the intended input override:

```text
ldsIdx_          = 1
dsName_          = Tensor1
dsType_          = INPUT
memOrg_          = lx allocate-Tensor1_lx
hbmStartAddress_ = -1
hbmSize_         = 0
coreStateInit_   = []
```

The metadata also recorded:

```text
kv_repack_hbm_direct_load = true
kv_repack_consumer_core_state_init = false
kv_repack_consumer_ds_type = INPUT
```

DXP did not reach device execution.  It aborted in DDL conversion:

```text
terminate called after throwing an instance of 'DtException'
what(): DtException: Could not find any suitable dimension mapping,
file /home/adnan-cdx/dt-inductor-mixed/deeptools-onchip-foundation-clean/ddc/ddl/ddl_conversion.cpp line 2493
```

The printed dimension properties showed invalid/drop dimensions for the early
refs and only `x` candidates for the later refs, so the nonzero K/V input1
`INPUT` descriptor is rejected before value execution.

## Interpretation

The `CONSUMER_DS_TYPE=INPUT` result rules out the simplest descriptor fix.  A
nonzero K/V `batchmatmul` input1 cannot be made into the same kind of LX
`INPUT` operand that worked for the Stage039 input0 layout-transform path.

Together with the Stage060 copyback passes, the current contract is:

- HBM-to-consumer-LX direct-load data movement is correct.
- The explicit `coreStateInit_` field is not the only cause; removing it and
  changing `dsType_` to `INPUT` fails earlier in DXP.
- LX-only `KERNEL` input1 compiles, but consumes the K/V operand incorrectly.
- LX-only `INPUT` input1 does not compile for this K/V edge.
- The normal HBM-backed input1 path likely relies on Foundation-generated
  restaging/formatting that is not represented by the current sidecar LX fill.

The next useful route is therefore not more STCDP fanout tuning.  It is either
to identify the backend-supported LX format/dataop for `batchmatmul` K/V
input1, or to keep K/V HBM-backed in the production overlap design and use the
validated HBM-source/copyback controls only as diagnostics.

## Verification

Local:

```text
python3 -m py_compile torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_config_logic.py \
  tests/_inductor/test_onchip_sdpa_sweep_logic.py \
  tests/_inductor/test_onchip_realize_logic.py
python3 tests/_inductor/test_config_logic.py                      # 36/36 pass
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py           # 52/52 pass
python3 tests/_inductor/test_onchip_realize_logic.py              # 109/109 pass
git diff --check
```

Pod:

```text
python3 -m py_compile ...                                         # pass
python3 tests/_inductor/test_config_logic.py                      # 36/36 pass
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py           # 52/52 pass
python3 tests/_inductor/test_onchip_realize_logic.py              # 109/109 pass
```

Device:

```text
tools/onchip_sdpa_sweep.py \
  --lengths 256 \
  --variants kv_repack_pair_hbm_direct_load_dsinput_auto \
  --batch 1 --heads 8 --dim 64 --block-size 64 \
  --seed -256 --warmup 1 --iters 2 --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage061-kv-dsinput \
  --output-json /tmp/sdpa-stage061-kv-dsinput.json

status = failed
returncode = 1
failure = DXP SIGABRT from DtException: Could not find any suitable dimension mapping
```
