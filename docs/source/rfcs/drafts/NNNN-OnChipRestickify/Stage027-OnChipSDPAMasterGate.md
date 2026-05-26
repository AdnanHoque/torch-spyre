# Stage 027: On-Chip SDPA Master Gate

Date: 2026-05-26

## Purpose

Stage 026 showed that the planned overlap-prefix path is still blocked by the
current Foundation `InputFetchNeighbor` contract for generated flash
`batchmatmul` descriptors.  The useful production move is therefore not to hide
that path behind a broad "turn everything on" flag.  This stage adds a narrower
production-candidate switch for the parts that are already value-correct and
fail-closed:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
```

The switch enables the generated flash-prefill decomposition plus the certified
same-stick on-chip handoffs inside that graph.  It intentionally does not enable
mixed sidecar artifacts, tile replacement probes, value-flow tile replacement,
or overlap-prefix scheduling.

## Implementation

Added:

```text
torch_spyre/_inductor/config.py
  flash_attention_onchip_sdpa
```

The master gate derives these existing flags:

```text
flash_attention_mixed_pipeline = flash_attention_onchip_sdpa || legacy env
flash_attention_pointwise_handoff = flash_attention_onchip_sdpa || legacy env
flash_attention_score_scale_handoff = flash_attention_onchip_sdpa || legacy env
```

It leaves these gates independently controlled:

```text
flash_attention_prefill
flash_attention_mixed_pipeline_overlap
flash_attention_mixed_pipeline_artifact
flash_attention_mixed_pipeline_execute_tile
flash_attention_mixed_pipeline_value_flow_tile
```

The `flash_attention_prefill` flag remains separate because
`flash_attention_mixed_pipeline` already selects the flash-prefill decomposition
through the decomposition predicate.

Added config coverage:

```text
tests/_inductor/test_config_logic.py
```

The test reads config values in fresh Python subprocesses so import-time env
state is deterministic.  It disables Torch backend auto-loading for those pure
config imports:

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

Updated the sweep harness:

```text
tools/onchip_sdpa_sweep.py
```

The harness now clears all flash-attention probe flags in a shared base variant
environment and adds:

```text
onchip_master:
  SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
```

This prevents inherited shell flags from polluting `vanilla`, `flash_hbm`, and
legacy `onchip` rows, and gives future sweeps a direct production-candidate
variant.

## Pod Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Static and logic tests:

```text
tests/_inductor/test_config_logic.py             3/3 passed
tests/_inductor/test_onchip_realize_logic.py    30/30 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py 10/10 passed
py_compile(config.py, test_config_logic.py, onchip_sdpa_sweep.py) passed
git diff --check passed
```

Device smoke with only the new master gate selecting the on-chip behavior:

```sh
export SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
unset SPYRE_FLASH_ATTENTION_MIXED_PIPELINE
unset SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF
unset SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF
unset SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP
unset SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_ARTIFACT
unset SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE
unset SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE
export SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE=64
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-stage027-onchip-sdpa-master-1779828812
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 7 deselected in 19.18s
```

Mixed SDSC evidence from the device cache:

```text
cache=/tmp/sdpa-stage027-onchip-sdpa-master-1779828812
mixed_sdsc_count=4

sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_me8qcpe9/4_mul:
  datadscs=1 opfuncs=["STCDPOpLx"] HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=128

sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_c6fthoon/11_mul:
  datadscs=1 opfuncs=["STCDPOpLx"] HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=64

sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_c6fthoon/17_add:
  datadscs=1 opfuncs=["STCDPOpLx"] HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=64

sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_c6fthoon/5_mul:
  datadscs=1 opfuncs=["STCDPOpLx"] HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=128
```

Tiny harness smoke for the new variant:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128 \
  --variants onchip_master \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --warmup 1 --iters 2 \
  --timeout-s 180 \
  --cache-prefix /tmp/sdpa-stage027-master-sweep \
  --output-json /tmp/sdpa-stage027-master-sweep.json
```

Result:

```text
L=128 onchip_master status=ok median=0.263357ms mean=0.263357ms
max_err=0.00341797 mixed=4
cache=/tmp/sdpa-stage027-master-sweep-onchip_master-B1-H2-L128-D64-514796-406550
```

The timing is a smoke measurement only because it used two iterations.  The
important acceptance facts for this stage are value correctness, four emitted
mixed `STCDPOpLx` SDSCs, and `HBM=0` in their senprog dumps.

## Interpretation

This stage gives the on-chip SDPA prototype a single production-shaped entry
point while keeping the riskier research paths explicit.  The enabled pieces are:

```text
score batchmatmul output -> score-scale scalar mul
selected SFP pointwise edges in the online-softmax/update chain
```

The overlap design is still separate.  Stage 026 found that current
`InputFetchNeighbor` lowering assumes `i/j` coordinates and does not accept the
`mb/x/in/out` geometry used by generated flash `batchmatmul` tiles.  Until that
Foundation contract changes, `SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1` should mean
"value-correct serial on-chip handoffs inside flash prefill", not "warp
specialized load/compute overlap."

## Next Step

Use the new `onchip_master` harness variant for larger sweeps:

```text
--variants vanilla,flash_hbm,onchip_master
```

Then resume the overlap work only after the DXP/Foundation side supports
flash-shaped input-neighbor descriptors or a different mixed-SDSC schedule shape
that does not rely on the current `i/j`-only path.
