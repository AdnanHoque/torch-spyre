# Flash frontend blocker report

Context inspected:
- Torch checkout: `/home/adnan/codex-isolated/dldsc_runtime_path_20260702_074814/torch-spyre`, commit `75040ee6d9f48518d0c194b72d1075035bb37b7b` (`ah/comms-collectives-dldsc-agent`).
- Latest repo flash: `/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/repos/test-spyre-scripts/test_flash.py`, commit `afda166e58b23519d0b4ca871350b011b56d91a3`, SHA256 `622622e262e9829868d82cbb5632522625073bde9d49945d7a1385370e0a7818`.
- Pod-local all-gather flash clone: `/home/adnan/codex-isolated/flash_allgather_runtime_20260701_abi_matched/test_flash.py`, SHA256 `5b87bc624e1401d567760c75e01b507e2484a4180646c429db034fe6c8bbef72`.

## 1. `buf10 (Pointwise): no mechanism to resolve stick incompatibility`

Observed in:
- `/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/current_baseline_aiu1_noh2d_20260702_080924/stderr.log`
- `/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/current_optimized_allgather_restickify_noh2d_20260702_081118/stderr.log`

User code producing the failing node:
- `/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/repos/test-spyre-scripts/test_flash.py:114`, `running_max = torch.maximum(real_max, block_max)`.
- Inputs are `buf4` from `test_flash.py:95` (`real_max = real_max.amax(dim=-1)`) and `buf9` from `test_flash.py:113` (`block_max = torch.amax(scores, dim=-1)`).

Torch path:
- `torch_spyre/_inductor/patches.py:120` calls pre-scheduling passes from patched `_update_scheduler`.
- `torch_spyre/_inductor/passes.py:317-343` defines pass order; this fails during `optimize_restickify_locations`, before `finalize_layouts`, `insert_restickify`, work division, scratchpad planning, or SDSC emission.
- `torch_spyre/_inductor/propagate_layouts.py:581-690` handles multi-arg pointwise joins. For `buf10`, it emits only non-zero-stick output candidates from surviving input stick expressions (`:663-668`) and sets `AllSameNode` (`:689`). It does not add a zero-stick output candidate when zero-stick input layouts are present and non-zero candidates also exist.
- `torch_spyre/_inductor/optimize_restickify.py:85-93` maps infeasible `compute_restickify_needed(...)->(True, None)` to infinite cost.
- `torch_spyre/_inductor/pass_utils.py:949-994` computes restickify feasibility. For zero-stick to non-zero-stick, `compute_restickify_target_layout()` cannot find a matching old stick host dim and returns `None` (`:887-894`).
- `torch_spyre/_inductor/optimize_restickify.py:472-485` exhausts all candidate states and raises.
- `torch_spyre/_inductor/optimize_restickify.py:255-299` constructs the `NotImplementedError`; `_stick_incompatibility_reason()` at `:227-238` supplies `No mechanism to gather elements from multiple sticks into single stick`.

Classification:
- Frontend stick-layout limitation, not SDSC/DLDSC and not a probe artifact.
- If a pointwise join is forced from zero-stick input to non-zero-stick output, that would require an unsupported gather/scatter class. In this graph it is avoidable: `buf9` has only a zero-stick layout and `buf4` has zero-stick candidates, but `_multi_arg_pointwise_layouts()` excludes zero-stick output once any non-zero input stick exists. The diagnostic text reports `buf4` first because `first_blocking_edge()` scans all possible input STLs on the first edge, not only the actual dead-end state; `buf9` is also a zero-stick constraint.

Minimal Torch-side change for latest repo flash:
- Add a zero-stick output candidate in `_multi_arg_pointwise_layouts()` when any non-index input candidate has stick expression `0`, even if non-zero stick expressions are also present. Concretely, around `torch_spyre/_inductor/propagate_layouts.py:663-668`, call `_try_stick_dim(-1)` before/after iterating `offset_free_stick_exprs` when zero-stick is present. This lets the optimizer choose the all-zero-stick `maximum(real_max, block_max)` layout and avoids requiring the missing gather communication class at `buf10`.

## 2. `Unexpected stick expression 4*(Mod(d4, 16))`

Observed in:
- `/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/custom_allgather_baseline_empty_20260702_081844/stderr.log`
- `/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/custom_allgather_optimized_empty_20260702_082029/stderr.log`
- Diagnostic wrapper: `/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/diagnose_unexpected_stick_20260702_083142`.

User code producing the failing node:
- `/home/adnan/codex-isolated/flash_allgather_runtime_20260701_abi_matched/test_flash.py:94`, `torch.matmul(exp_scores.transpose(-1, -2), values)` inside `output.copy_(...)`.
- This is `op16`, a `batchmatmul` reading restickified `buf26` (from `exp_scores`, `test_flash.py:90`) and `arg2_1` values.

Torch path:
- Restickify optimizer succeeds for this clone, then `insert_restickify` injects two restickifies before failure:
  - `buf16` input `buf15`: old stride map `[65536, 256, 64, 2097152, 1]` to target `[2097152, 1, 16384, 65536, 256]`.
  - `buf17` input `buf16`: old stride map `[32768, 128, 64, 1048576, 1]` to target `[32768, 1, 8192, 1048576, 128]`.
- `torch_spyre/_inductor/passes.py:317-343` then runs `propagate_named_dims`, `assign_dim_hints`, `_maybe_coarse_tile`, and `span_reduction`.
- `torch_spyre/_inductor/work_division.py:1331-1341` visits reduction ops; `divide_reduction_op()` calls `span_reduction_pass()` at `:1296`; `span_reduction_pass()` calls `collect_tensor_deps()` at `:764`; `collect_tensor_deps()` constructs `TensorDep` at `:615`; `TensorDep.__post_init__()` calls `device_coordinates()` at `:72-75`.
- `torch_spyre/_inductor/pass_utils.py:676-705` computes device coordinates and calls `_check_stick_expr_supported()`.
- `torch_spyre/_inductor/pass_utils.py:664-673` rejects anything other than `Mod(var, 64)`, bare variable, `0`, or constant-offset variants. It raises on `4*(Mod(d4, 16))`.

Concrete failing dep from diagnostic wrapper:
- dep: `buf26`
- dep index: `2097152*d0 + 65536*d1 + d2 + 256*d4`
- dep ranges after coarse tiling: `{d0: 2, d1: 4, d2: 64, d3: 128, d4: 256}`
- STL device size: `[4, 64, 4, 2, 64]`
- STL stride map: `[16384, 1, 4096, 65536, 64]`
- elems per stick: `64`
- rejected stick coordinate: `4*(Mod(d4, 16))`

Classification:
- Frontend stick-coordinate canonicalization/support limitation, not a true missing communication class and not SDSC/DLDSC.
- It is not caused by the empty-tensor probe itself; the probe compiles the same all-gather flash graph. The expression comes from the combination of coarse tiling/restickify layout and the batchmatmul read of `buf26`. The empty probe only avoids full data execution/CPU reference.
- To reach SDSC emission for this all-gather clone, Torch must either avoid producing this sub-stick expression for the restickified `buf26` layout, or teach `pass_utils._check_stick_expr_supported()` and downstream stick helpers to accept/canonicalize scaled modulo forms equivalent to sub-stick packing, such as `k*Mod(var, N)` where `k*N == elems_per_stick`.

## DLDSC relayout metadata connection

These blockers are before SDSC generation and before the DLDSC relayout emission path can matter.
- `passes.py:317-343` shows failure 1 occurs at `optimize_restickify_locations` and failure 2 occurs at `span_reduction`; both precede scratchpad/LX planning completion and SDSC lowering.
- DLDSC/LX relayout metadata clearing lives in `lx_relayout.clear_lx_relayout_metadata()` (`lx_relayout.py:199-207`) and scratchpad retry fallback (`scratchpad/allocator.py:415-427`). SDSC-side restickify LX selection is later in `codegen/superdsc.py:641-648` and `:935-940`.
- Therefore the current flash frontend blockers are not caused by DLDSC relayout metadata being cleared before SDSC. The earlier replay evidence about cleared metadata remains a separate later-stage issue; these runs never reach that stage.
