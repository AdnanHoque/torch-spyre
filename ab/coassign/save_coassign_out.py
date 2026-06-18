# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lean device value-capture for co-assignment.

Compile the co-assigned SwiGLU on spyre and run **one** forward, then save the
output for an offline max_err diff (``check_maxerr.py``). Leaner than
``check_outputs.py`` worker mode (which warms up + measures = two device calls);
here a single call both compiles (dxp-accept) and produces the output tensor.

Co-assign is installed via the in-process patch (compile_threads=1,
fx_graph_cache=False — same as run_ab.py) before any compile so the work-division
flips fire. Everything prints with flush so a tail of the log shows progress.

    python save_coassign_out.py <op> <seed> <out.pt> <D0> <D1> <D2> ...
"""

import sys

sys.path.insert(0, "/tmp/core-to-core-wt")
sys.path.insert(0, "/tmp/core-to-core-wt/ab")
sys.path.insert(0, "/home/adnan/dt-inductor/spyre-perf-suite-aisw")


def main():
    op = sys.argv[1]
    seed = int(sys.argv[2])
    out_path = sys.argv[3]
    shape = tuple(int(x) for x in sys.argv[4:])
    print(f"[SAVE] op={op} seed={seed} shape={shape} -> {out_path}", flush=True)

    import torch
    import torch._inductor.config as ic

    ic.compile_threads = 1
    ic.fx_graph_cache = False
    import torch_spyre  # noqa: F401

    from ab.coassign.coassign import apply_coassign

    apply_coassign()
    print("[SAVE] co-assign installed", flush=True)

    import check_outputs as co

    co._seed_cpu_generator(seed)
    custom_module, resolved_op = co.resolve_custom_op(op, None)
    tensors = co.create_tensors(torch, [shape], resolved_op, "torch-spyre", custom_module)
    target = co.get_operation_target(
        resolved_op, torch, "torch-spyre", [shape], custom_module
    )
    device = torch.device("spyre")
    target = co.prepare_module_target(target, tensors, device=device)
    compiled = torch.compile(target)
    run_tensors = tuple(t.to(device) for t in tensors if isinstance(t, torch.Tensor))
    print("[SAVE] compiling + single forward ...", flush=True)

    with torch.no_grad():
        result = compiled(*run_tensors)
    result = co._normalize_output(result, resolved_op, "torch-spyre")
    print("[SAVE] forward done; saving", flush=True)

    import torch as _t

    _t.save(result, out_path)
    print(f"[SAVE] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
