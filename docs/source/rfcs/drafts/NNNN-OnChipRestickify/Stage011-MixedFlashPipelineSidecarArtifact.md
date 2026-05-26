# Stage 011: Mixed Flash Pipeline Sidecar Artifact

Date: 2026-05-26

## Purpose

Stage 010 identified the real flash-prefill SDSC graph emitted by
`SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1`.  Stage 011 makes the compiler produce a
real mixed-SDSC proof artifact from that generated graph.

The artifact is intentionally not executed yet.  It is emitted next to the
normal SDSCs, while `bundle.mlir` continues to run the stock flash-prefill graph.
That gives us a correctness-preserving control run plus an inspectable mixed
descriptor for DXP/senprog validation.

## Implementation

New gate:

```sh
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_ARTIFACT=1
```

Optional descriptor-only overlap schedule:

```sh
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1
```

Code changes:

- `torch_spyre/_inductor/config.py`
  - Added `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_ARTIFACT`.
- `torch_spyre/_inductor/onchip_realize.py`
  - Added `build_flash_attention_pipeline_artifact`.
  - The helper finds generated `batchmatmul` flash-prefill tile compute SDSCs in
    one bundle and wraps them with the Stage 009 double-buffered `STCDPOpLx`
    data-op schedule.
  - The helper records metadata: source, layout, split dim, stick dim, row dim,
    iter sizes, tile bytes, data-op count, tile count, and whether overlap rows
    are present.
- `torch_spyre/_inductor/codegen/bundle.py`
  - Emits `sdsc_mixed_flash_pipeline_artifact.json` when both
    `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1` and
    `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_ARTIFACT=1` are set.
  - Does not add the sidecar file to `bundle.mlir`.
- `tests/_inductor/test_onchip_realize_logic.py`
  - Added sidecar artifact tests for serial schedule, overlap candidate rows,
    and no-batchmatmul fail-closed behavior.

## Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Standalone tests:

```text
tests/_inductor/test_onchip_realize_logic.py         16/16 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   9/9 passed
tests/_inductor/test_onchip_streaming_logic.py        9/9 passed
tests/_inductor/test_onchip_handoff_logic.py          3/3 passed
```

Device/compiler proof, serial artifact:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_ARTIFACT=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-mixed-artifact-1779818247
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q
```

Result:

```text
1 passed, 6 deselected in 13.35s
```

Artifacts emitted:

```text
.../sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_.../sdsc_mixed_flash_pipeline_artifact.json
.../sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_.../sdsc_mixed_flash_pipeline_artifact.json
```

The sidecar name does not appear in either generated `bundle.mlir`, so it is not
executed in the correctness run.

Serial artifact metadata:

```text
bundle 0: datadscs=2 dscs=1 opFuncs=["STCDPOpLx", "STCDPOpLx"]
          tile_count=1 dataop_count=2 overlap_candidate=false tile_bytes=1024

bundle 1: datadscs=6 dscs=3 opFuncs=["STCDPOpLx" x 6]
          tile_count=3 dataop_count=6 overlap_candidate=false tile_bytes=384
          schedule0=[
            [0,-1,0,1], [1,-1,1,1], [-1,0,1,1],
            [2,-1,1,1], [3,-1,1,1], [-1,1,1,1],
            [4,-1,1,1], [5,-1,1,1], [-1,2,1,0]
          ]
```

Descriptor-only overlap artifact:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-mixed-artifact-overlap-1779818291
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q
```

Result:

```text
1 passed, 6 deselected in 15.84s
```

The three-tile bundle marks `overlap_candidate=true` and emits overlap rows:

```text
[[0,-1,0,1], [1,-1,1,1],
 [2,0,1,1], [3,-1,1,1],
 [4,1,1,1], [5,-1,1,1],
 [-1,2,1,0]]
```

## Conclusion

The compiler now emits a real mixed flash-pipeline SDSC artifact derived from
the generated flash-prefill graph.  The artifact is not executed yet, which keeps
the SDPA value-correctness run unchanged.  The next step is to feed this sidecar
through DXP directly, inspect `senprog.txt`, and then decide whether the first
executed realizer should replace one serial tile-chain bundle or start with a
single bundled score/PV subchain.
