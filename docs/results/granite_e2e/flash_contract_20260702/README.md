# Flash DLDSC layout-allgather contract run - 2026-07-02

## Runs

Before contract run: /home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/latest_after_zero_stick_optimized_20260702_084153

After contract run: /home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/latest_after_zero_stick_split_lx_layout_allgather_20260702_085442

## Summary

| Metric | Before | After |
| --- | ---: | ---: |
| return code | 0 | 1 |
| SDSC count | 550 | 549 |
| ReStickifyOpHBM rows | 32 | 0 |
| ReStickifyOpLx rows | 0 | 32 |
| layout_allgather classifications | 0 | 32 |
| LX residency coordinate entries | 0 | 32 |

## Interpretation

The zero-stick pointwise fix lets latest flash pass the earlier frontend stick incompatibility and emit SDSCs. With Torch full-LX planning, the layout-allgather contract flag, and a split DXP wrapper, the flash activation edge changes shape in SDSC:

- Before: 32 ReStickifyOpHBM rows and no relayout classifications.
- After: 32 ReStickifyOpLx rows and 32 layout_allgather_restickify classifications, each with transfer_count=256, max_fanout=8, and max_fanin=8.

This proves the frontend can now express the flash activation layout/restickify handoff as an on-chip DLDSC contract. The run is not e2e correct yet: DXP aborts with scheduler failure for ReStickifyOpLx, which is the remaining backend physical-lowering gap.

## After Failure

DXP failure tail:

                          ^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/adnan/dt-inductor/.venv/lib64/python3.12/site-packages/torch/_inductor/graph.py", line 2499, in compile_to_module
        return self._compile_to_module()
               ^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/adnan/dt-inductor/.venv/lib64/python3.12/site-packages/torch/_inductor/graph.py", line 2509, in _compile_to_module
        mod = self._compile_to_module_lines(wrapper_code)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/adnan/dt-inductor/.venv/lib64/python3.12/site-packages/torch/_inductor/graph.py", line 2584, in _compile_to_module_lines
        mod = PyCodeCache.load_by_key_path(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/adnan/dt-inductor/.venv/lib64/python3.12/site-packages/torch/_inductor/codecache.py", line 3764, in load_by_key_path
        mod = _reload_python_module(key, path, set_sys_modules=in_toplevel)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/adnan/dt-inductor/.venv/lib64/python3.12/site-packages/torch/_inductor/runtime/compile_tasks.py", line 35, in _reload_python_module
        exec(code, mod.__dict__, mod.__dict__)
      File "/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/latest_after_zero_stick_split_lx_layout_allgather_20260702_085442/cache/v2/cv2eoax5gxga6hrxmo7kdrkrd6hojrlee45kzrrh2c5obyutjnt6.py", line 237, in <module>
        sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1 = async_compile.sdsc('sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1',
                                                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/adnan/codex-isolated/dldsc_runtime_path_20260702_074814/torch-spyre/torch_spyre/execution/async_compile.py", line 63, in sdsc
        subprocess.run(["dxp_standalone", "--bundle", "-d", output_dir], check=True)
      File "/usr/lib64/python3.12/subprocess.py", line 571, in run
        raise CalledProcessError(retcode, process.args,
    torch._inductor.exc.InductorError: CalledProcessError: Command '['dxp_standalone', '--bundle', '-d', '/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/latest_after_zero_stick_split_lx_layout_allgather_20260702_085442/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_ihw4zzb8']' died with <Signals.SIGABRT: 6>.
    
    Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
