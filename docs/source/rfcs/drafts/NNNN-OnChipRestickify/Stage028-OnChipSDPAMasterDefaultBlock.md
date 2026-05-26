# Stage 028: On-Chip SDPA Master Default Block Size

Date: 2026-05-26

## Purpose

Stage 027 added the production-candidate gate:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
```

That gate initially inherited the older flash-prefill default block size of
128.  Stage 022 had already suggested that larger blocks reduce launch and graph
overhead for longer sequence lengths.  This stage reruns the sweep using the
new `onchip_master` variant, compares it to vanilla and flash-HBM, then updates
the master gate default to the better production policy.

## Sweep

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Command for the default-before-change block-128 sweep:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128,256,512,1024 \
  --variants vanilla,flash_hbm,onchip_master \
  --batch 1 --heads 2 --dim 64 --block-size 128 \
  --warmup 2 --iters 5 \
  --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage028-bs128 \
  --output-json /tmp/sdpa-stage028-bs128.json
```

Results:

| L | Variant | Block | Median ms | Max error | Mixed SDSCs |
|---:|---|---:|---:|---:|---:|
| 128 | vanilla | 128 | 0.121117 | 0.00537109 | 0 |
| 128 | flash_hbm | 128 | 0.249775 | 0.00341797 | 0 |
| 128 | onchip_master | 128 | 0.246158 | 0.00341797 | 1 |
| 256 | vanilla | 128 | 0.172945 | 0.00268555 | 0 |
| 256 | flash_hbm | 128 | 0.343077 | 0.00341797 | 0 |
| 256 | onchip_master | 128 | 0.336099 | 0.00341797 | 4 |
| 512 | vanilla | 128 | 0.230368 | 0.00231934 | 0 |
| 512 | flash_hbm | 128 | 0.609910 | 0.00195312 | 0 |
| 512 | onchip_master | 128 | 0.554964 | 0.00195312 | 10 |
| 1024 | vanilla | 128 | 0.531044 | 0.00292969 | 0 |
| 1024 | flash_hbm | 128 | 1.418527 | 0.00134277 | 0 |
| 1024 | onchip_master | 128 | 1.301631 | 0.00134277 | 21 |

On-chip speedup over flash-HBM at block 128:

| L | Speedup |
|---:|---:|
| 128 | 1.015x |
| 256 | 1.021x |
| 512 | 1.099x |
| 1024 | 1.090x |

Block-512 follow-up:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 512,1024 \
  --variants flash_hbm,onchip_master \
  --batch 1 --heads 2 --dim 64 --block-size 512 \
  --warmup 2 --iters 5 \
  --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage028-bs512 \
  --output-json /tmp/sdpa-stage028-bs512.json
```

Results:

| L | Variant | Block | Median ms | Max error | Mixed SDSCs |
|---:|---|---:|---:|---:|---:|
| 512 | flash_hbm | 512 | 0.539735 | 0.00195312 | 0 |
| 512 | onchip_master | 512 | 0.548683 | 0.00195312 | 0 |
| 1024 | flash_hbm | 512 | 1.172071 | 0.00134277 | 0 |
| 1024 | onchip_master | 512 | 1.160441 | 0.00134277 | 2 |

The block-512 `L=1024` on-chip row is faster than the block-128 `L=1024`
on-chip row:

```text
block 128 onchip_master: 1.301631 ms
block 512 onchip_master: 1.160441 ms
speedup from block policy: 1.122x
```

It is still slower than stock Spyre SDPA:

```text
vanilla L=1024:          0.531044 ms
onchip_master block 512: 1.160441 ms
relative to vanilla:     0.458x
```

## Implementation

Updated:

```text
torch_spyre/_inductor/config.py
```

When the production-candidate master gate is enabled and the user has not set
`SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE`, the default block size is now 512:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=0 -> default block size 128
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1 -> default block size 512
```

Explicit user overrides still win:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE=128
```

keeps block size 128.

Updated:

```text
tests/_inductor/test_config_logic.py
```

Coverage now checks the master default is 512 and that an explicit block-size
override is preserved.

Updated:

```text
tools/onchip_sdpa_sweep.py
```

The harness now records the effective Torch-Spyre config block size instead of
only echoing the env var.  It also treats:

```text
--block-size 0
```

as "leave `SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE` unset."  This lets the
harness validate default config policy in a fresh child process.

## Default-Policy Device Proof

Command:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 1024 \
  --variants onchip_master \
  --batch 1 --heads 2 --dim 64 --block-size 0 \
  --warmup 2 --iters 5 \
  --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage028-master-default \
  --output-json /tmp/sdpa-stage028-master-default.json
```

Result:

```text
L=1024 onchip_master status=ok median=1.239145ms mean=2.631605ms
max_err=0.00134277 mixed=2
cache=/tmp/sdpa-stage028-master-default-onchip_master-B1-H2-L1024-D64-518337-80770
```

The mean includes one timing outlier, but the correctness and policy evidence
are clear:

```text
block_size=512
block_size_env=""
mixed SDSCs:
  sdsc_11_mul opFuncsUsed=["STCDPOpLx"] datadscs=1
  sdsc_17_add opFuncsUsed=["STCDPOpLx"] datadscs=1
```

Representative senprog evidence for the mixed SDSCs:

```text
sdsc_11_mul: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=64
sdsc_17_add: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=64
```

## Validation

Pod validation after the implementation:

```text
tests/_inductor/test_config_logic.py                  4/4 passed
tests/_inductor/test_onchip_realize_logic.py         30/30 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py  10/10 passed
combined focused run:                                44/44 passed
py_compile(config.py, test_config_logic.py, onchip_sdpa_sweep.py) passed
git diff --check passed
```

## Interpretation

The master gate is now a better production-shaped policy:

- It still enables only value-correct, fail-closed serial on-chip handoffs.
- It now defaults to the lower-overhead flash-prefill block size observed in the
  sweep.
- It keeps user and probe overrides explicit.

This does not solve the bigger performance gap to stock SDPA.  The useful
finding is sharper: serial on-chip handoffs improve the flash-prefill path, but
the flash-prefill decomposition still loses to stock Spyre SDPA because of graph
and launch overhead.  More same-layout serial handoffs will not close that gap
by themselves.

## Next Step

The next implementation effort should target launch/fusion/overlap rather than
additional standalone handoff edges.  Two concrete paths remain:

```text
1. Foundation/DXP support for flash-shaped InputFetchNeighbor descriptors
   using mb/x/in/out coordinates instead of the current i/j-only path.

2. A compiler-produced fused flash tile SDSC that keeps score-scale and the
   online-softmax pointwise update in one mixed artifact, reducing standalone
   SDSC launches without relying on the blocked overlap-prefix path.
```
