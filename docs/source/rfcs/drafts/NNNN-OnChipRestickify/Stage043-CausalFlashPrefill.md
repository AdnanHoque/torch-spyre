# Stage 043: Causal Flash Prefill

Date: 2026-05-27

## Purpose

Stage042 left causal and masked SDPA outside the on-chip layout-transform gate.
This stage brings square causal prefill into the flash-prefill path and adds a
small causal promotion-gate case for the production-shaped layout-transform
variant.

This is still narrower than arbitrary attention masking.  `attn_bias is not
None` remains fail-closed and routes through the existing non-flash fallback.

## Implementation Summary

Updated:

```text
torch_spyre/_inductor/decompositions.py
tools/onchip_sdpa_sweep.py
tools/onchip_sdpa_promotion_gate.py
tests/inductor/test_building_blocks.py
tests/_inductor/test_onchip_sdpa_sweep_logic.py
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py
```

The flash-prefill predicate no longer rejects every `is_causal=True` call.
Instead, it allows the causal path only for square prefill, where
`q_len == kv_len`:

```text
query.size(2) == key.size(2)
```

Non-square causal decode/prefill remains outside this decomposition for now.

The causal mask is represented as a triangular additive bias in query/key
coordinates.  The bias is the upper triangle of a `-inf` score matrix, and each
key block adds:

```text
causal_bias[:, start:end]
```

before the score tensor is transposed into the running softmax layout.  This
keeps the existing blockwise accumulation and layout-transform sidecar flow
unchanged.

The sweep harness now accepts `--is-causal`, passes it to
`F.scaled_dot_product_attention`, records `is_causal` in the result JSON, and
includes the causal bit in the compile-cache directory key.  The promotion gate
validates that bit so a stale noncausal sweep cannot satisfy the causal case.

The expected causal promotion-gate case is `b1h2d64_block64_causal`.  With that
case included, the default `onchip_layout_xform` gate has eight cases and twenty
rows.  The new row group is:

| Shape | Block | Lengths | Causal | Mixed floor |
| --- | --- | --- | --- | --- |
| B1 H2 D64 | 64 | 128, 256 | yes | 8, 16 |

## Pod Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
```

Unit command:

```sh
"$PYTHON" -m unittest \
  tests.inductor.test_building_blocks.TestBuildingBlocks.test_sdpa_flash_attention_mixed_pipeline_causal_prefill
```

Result:

```text
FallbackWarning: aten.triu.default is falling back to cpu
.
Ran 1 test in 13.129s
OK
```

Causal sweep command:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128,256 \
  --variants onchip_master_layout_xform \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --is-causal \
  --warmup 1 --iters 2 --timeout-s 480 \
  --cache-prefix /tmp/sdpa-stage043-causal-smoke \
  --output-json /tmp/sdpa-stage043-causal-smoke.json
```

Result:

```text
L=128 onchip_master_layout_xform status=ok median=0.461393ms mean=0.461393ms max_err=0.00585938 mixed=8
L=256 onchip_master_layout_xform status=ok median=0.669666ms mean=0.669666ms max_err=0.00390625 mixed=16
```

The generated rows included a layout-transform consumer sidecar for both
lengths.

## Local Validation

```text
tests/_inductor/test_onchip_sdpa_sweep_logic.py           4/4 passed
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py  7/7 passed
py_compile(decompositions.py, sweep, promotion gate, tests) passed
git diff --check passed
```

## Interpretation

Square causal prefill is now value-correct through the mixed flash-prefill path
and exercises the same layout-transform sidecar path as the noncausal rows.  The
change is intentionally scoped: arbitrary `attn_bias` masks and non-square causal
cases still fail closed.

The remaining implementation caveat is mask construction.  The triangular bias
currently uses `aten.triu.default`, and the pod unit test reported a CPU fallback
for that op.  That is acceptable for this proof and gate coverage, but it should
be replaced with a device-native mask construction before broad/default causal
enablement.

## Documentation Check

```text
git diff --check --no-index /dev/null docs/source/rfcs/drafts/NNNN-OnChipRestickify/Stage043-CausalFlashPrefill.md passed
python3 -m sphinx -b html -W --keep-going docs/source /tmp/torch-spyre-docs-stage043-check unavailable: No module named sphinx
```

## Next

- run the updated twenty-row promotion gate on the pod;
- replace the `aten.triu` mask construction with a device-native equivalent; and
- decide whether masked `attn_bias` support should be implemented separately or
  explicitly excluded from the defaultable on-chip SDPA variant.
