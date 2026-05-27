# Stage 048: Causal Bias Custom-Op Scaffold

Date: 2026-05-27

## Purpose

Stage047 identified the boundary for removing the causal `aten.triu.default`
fallback: a score-layout-anchored primitive must produce the additive causal
bias from output coordinates and the current key-block start.

This stage adds the Torch-Spyre side of that primitive without routing SDPA
through it yet.  The remaining missing piece is DeepTools/SDSC support for the
new coordinate-aware compute op.

## Implementation

Added:

```text
spyre::causal_score_bias_like(scores: Tensor, key_start: int) -> Tensor
```

Files:

```text
torch_spyre/_inductor/customops.py
torch_spyre/_inductor/lowering.py
torch_spyre/_inductor/spyre_kernel.py
torch_spyre/_inductor/constants.py
tests/inductor/test_building_blocks.py
```

The custom op now has:

- a fake implementation returning `scores.new_empty(scores.size())`;
- a CPU kernel for direct semantic tests;
- a Spyre lowering that reads `scores` as the layout anchor;
- a `SpyreOpFuncs` entry that emits one `causal_score_bias_like` pointwise op;
- `keyStart` carried in `op_info["constants"]`.

The CPU semantics are:

```text
bias[..., q, k_block] = -inf if key_start + k_block > q else 0
```

## Why SDPA Is Not Wired Yet

The generated OpSpec is structurally the right shape: one score-shaped input,
one score-shaped output, and `keyStart` in constants.  But the backend still
needs an implementation of the `causal_score_bias_like` compute op that reads
the output coordinates and applies the triangular predicate.

Until that exists, replacing the current `triu` fallback in
`_flash_attention_prefill` would only move the failure later, from PyTorch CPU
fallback readiness to DeepTools compile/runtime.

## Validation Target

When backend support lands, the SDPA call site can switch from:

```text
scores = scores + causal_bias[:, start:end]
```

to:

```text
scores = scores + torch.ops.spyre.causal_score_bias_like(scores, start)
```

and the proof remains:

```sh
"$PYTHON" tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --cases b1h2d64_block64_causal \
  --forbid-fallbacks
```

Expected result:

```text
PROMOTION_GATE_PASSED gate=onchip_layout_xform cases=1 rows=2
```

## Pod Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
```

Commands:

```sh
"$PYTHON" -m py_compile \
  torch_spyre/_inductor/customops.py \
  torch_spyre/_inductor/lowering.py \
  torch_spyre/_inductor/spyre_kernel.py \
  torch_spyre/_inductor/constants.py \
  tests/inductor/test_building_blocks.py

"$PYTHON" -m unittest \
  tests.inductor.test_building_blocks.TestBuildingBlocks.test_causal_score_bias_like_cpu \
  tests.inductor.test_building_blocks.TestBuildingBlocks.test_flash_attention_prefill_causal_tail_block
```

Results:

```text
spyre.causal_score_bias_like registered
packet_registered True
overload_registered True
spyre_opfunc True
Ran 2 tests in 0.021s
OK
```

The existing causal prefill helper still uses the `triu` fallback.  This stage
only proves the new primitive's Torch-Spyre registration and CPU semantics.

## Compile Probe

A direct compile probe:

```python
def f(x):
    return torch.ops.spyre.causal_score_bias_like(x, 2)

x = torch.empty(1, 2, 4, 64, dtype=torch.float16).to("spyre")
torch.compile(f, backend="inductor")(x)
```

now reaches the generated SDSC:

```text
sdsc_fused_causal_score_bias_like_0
sdsc_0_causal_score_bias_like.json
constantInfo_: name_="keyStart"
computeOp_: opFuncName="causal_score_bias_like"
```

and fails at DeepTools with the expected backend-missing signature:

```text
DtException: Unrecognized opFunc: causal_score_bias_like
```

This confirms that the remaining blocker is the backend compute op, not custom
op registration or Spyre lowering discovery.

## Local Validation

```text
py_compile(customops.py, lowering.py, spyre_kernel.py, constants.py, test_building_blocks.py) passed
tests/_inductor/test_onchip_sdpa_sweep_logic.py          6/6 passed
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py 9/9 passed
git diff --check passed
```

## Next

- add DeepTools/SDSC support for `causal_score_bias_like`; and
- wire `_flash_attention_prefill` to the custom op once the backend accepts it.
