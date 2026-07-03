# Sidecar Granite Prefill Sanity - DLDSC Collectives

## Scope

- Pod: `adnan-spyre-dev-pf`
- Worktree root: `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155`
- Workload: one-layer FMS Granite block, causal prefill, shape `[1, 512, 4096]`, empty Spyre weights.
- Harness: `/home/adnan/codex-isolated/comms_collectives_20260629/spyre-granite-e2e-bench/benchmarks/granite_block_layer_probe.py`
- Requested split env: Torch `DXP_LX_FRAC_AVAIL=0`, DXP subprocess `DXP_BACKEND_LX_FRAC_AVAIL=0.2` via per-run `tools/dxp_standalone` wrapper.

## Checkouts

### Torch

```text
torch_branch=ah/comms-collectives
torch_sha=a39388e6d93fb5317b198aee30b30f7aa5ddded9
?? docs/results/granite_e2e/devpf_granite_prefill_profile_current_dldsc_20260702.md
```

### Deeptools

```text
deeptools_branch=ah/comms-collectives
deeptools_sha=3d54e87eb404b54c0ba74b98d6caa83945b2ef5b
```

### Granite harness

```text
bench_branch=main
bench_sha=76cd51426ba1de6e99dd8fbf613cb0f32b71e87f
```

## Runs

### Relayout disabled control

- Run dir: `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/sidecar_granite_prefill_disabled_20260703_144448`
- Command: `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/sidecar_granite_prefill_disabled_20260703_144448/command.sh`
- Result: pass
- `returncode`: `0`
- wall median ms: `26.6721248626709`
- trace `kernel_ms_per_iter`: `14.671158`
- trace `memory_ms_per_iter`: `0.21924100000000002`
- trace: `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/sidecar_granite_prefill_disabled_20260703_144448/block_prefill/profile/adnan-spyre-dev-pf_622424.1783090003027326602.pt.trace.json`

Key env:

```text
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=0.2
SPYRE_LX_PLANNER_RELAYOUT=0
LX_BOUNDARY_CLONES=0
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=0
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=0
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=0
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=0
```

### Relayout enabled collectives

- Run dir: `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/sidecar_granite_prefill_enabled_20260703_144732`
- Command: `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/sidecar_granite_prefill_enabled_20260703_144732/command.sh`
- Result: fail during DXP compile, before profiled runtime iteration
- `returncode`: `1`
- wall/kernel timing: unavailable
- backend plan dir: `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/sidecar_granite_prefill_enabled_20260703_144732/backend_plans`

Key env:

```text
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=0.2
SPYRE_LX_PLANNER_RELAYOUT=1
LX_BOUNDARY_CLONES=1
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
```

Backend plan files emitted:

```text
/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/sidecar_granite_prefill_enabled_20260703_144732/backend_plans/10_batchmatmul_Tensor1_0_layout_allgather_restickify_plan.json
/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/sidecar_granite_prefill_enabled_20260703_144732/backend_plans/18_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
```

Failure signature:

```text
DtException: [buildFoldFromAllocation] Can not propagate coordinates for coreletSplit dimensionmb from allocateNode allocate-Tensor0_lx with custom coreIdToWkSlice., file /home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/deeptools/ddc/ddc_fold.cpp line 3010
```

Result error tail:

```text
lf._compile_to_module_lines(wrapper_code)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/adnan/dt-inductor/.venv/lib64/python3.12/site-packages/torch/_inductor/graph.py", line 2584, in _compile_to_module_lines
    mod = PyCodeCache.load_by_key_path(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/adnan/dt-inductor/.venv/lib64/python3.12/site-packages/torch/_inductor/codecache.py", line 3764, in load_by_key_path
    mod = _reload_python_module(key, path, set_sys_modules=in_toplevel)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/adnan/dt-inductor/.venv/lib64/python3.12/site-packages/torch/_inductor/runtime/compile_tasks.py", line 35, in _reload_python_module
    exec(code, mod.__dict__, mod.__dict__)
  File "/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/sidecar_granite_prefill_enabled_20260703_144732/block_prefill/cache/qd/cqdnrwib2cofwj3wj5buuemao2omhfwstykmyztugpjyexzdj7ri.py", line 942, in <module>
    sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2 = async_compile.sdsc('sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2',
                                                                                                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/torch-spyre/torch_spyre/execution/async_compile.py", line 63, in sdsc
    subprocess.run(["dxp_standalone", "--bundle", "-d", output_dir], check=True)
  File "/usr/lib64/python3.12/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
torch._inductor.exc.InductorError: CalledProcessError: Command '['dxp_standalone', '--bundle', '-d', '/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/sidecar_granite_prefill_enabled_20260703_144732/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_uqg3wh28']' died with <Signals.SIGABRT: 6>.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


```

## Interpretation

- The pod/env can run the Granite causal prefill control with the requested split LX settings.
- Current full collectives relayout path is not e2e-ready for Granite on this checkout: it fails deterministically in Deeptools DDC/fold handling for an LX allocation with custom `coreIdToWkSlice`.
- This is different from the previous backend-fraction-1 attempt, which reached runtime and bus-fenced; with backend fraction `0.2`, the blocker is now compile-time and easier to debug.
- Since enabled does not reach runtime, there is no enabled kernel or wall timing to compare.
