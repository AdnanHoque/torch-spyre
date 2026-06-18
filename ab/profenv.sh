# Locked profiler stack for kernel-time runs (source before run_ab.py).
# harvest stable-device libs (opt-newer) + the USE_SPYRE_PROFILER build on the
# latest-main tree (editable into .venv) + .venv torch 2.11. self_device_time_total
# only appears with this exact pairing; mismatch -> 0.0us profiler / import fail /
# device-busy. See CORE_TO_CORE_SWIGLU_BASELINE.md "Perf" for the rationale.
export LD_LIBRARY_PATH="/home/adnan/opt-newer/runtime/lib:/home/adnan/opt-newer/spyre-comms/lib:/home/adnan/opt-newer/deeptools/lib:/home/adnan/opt-newer/senlib/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib"
export PATH="/home/adnan/opt-newer/deeptools/bin:$PATH"
export PYTHONPATH="/home/adnan/dt-inductor/foundation-model-stack"
export TORCH_SPYRE_DOWNCAST_WARN=0
# Interpreter: /home/adnan/dt-inductor/.venv/bin/python (torch 2.11, the ABI the
# profiler _C.so was built against). torch_spyre resolves via the editable install
# to /home/adnan/dt-inductor/torch-spyre (the profiler build; verify with
#   nm -D .../torch_spyre/_C*.so | grep SpyreActivityProfiler).
