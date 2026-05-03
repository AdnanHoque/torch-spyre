# Copyright 2025 The Torch-Spyre Authors.
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

"""Phase 2 exploration — what does V.graph know about nn.Parameters?

Compile a tiny nn.Linear model on spyre and dump every signal V.graph
exposes that could distinguish a weight from an activation:
  - V.graph.named_parameters
  - V.graph.constants
  - V.graph.graph_input_names
  - V.graph.graph_inputs (per-name InputBuffer / TensorBox)
  - V.graph.allocated_constant_name
  - placeholder metadata on the orig_gm
  - the orig_gm's graph signature, if any

We instrument SpyreKernel.create_tensor_arg so we can see, at the
moment a TensorArg is created, whether that buffer name traces back
to a parameter or an activation.

Goal: find the cleanest API to use in Phase 3 to set
`is_static=True` only on tensors that came from `nn.Parameter`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_SPYRE_DOWNCAST_WARN", "0")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

import torch_spyre  # noqa: E402, F401

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402
from torch._inductor.virtualized import V  # noqa: E402


# ---- monkey-patch create_tensor_arg to capture context ------------------

from torch_spyre._inductor import spyre_kernel as _sk  # noqa: E402

_orig_create_tensor_arg = _sk.SpyreKernel.create_tensor_arg
_observations: list[dict] = []


def _patched_create_tensor_arg(self, is_input, name, tensor):
    arg = _orig_create_tensor_arg(self, is_input, name, tensor)

    g = V.graph
    obs = {
        "name": name,
        "is_input": is_input,
        "in_named_parameters": name in getattr(g, "named_parameters", {}),
        "in_graph_input_names": name in getattr(g, "graph_input_names", []),
        "in_constants": name in getattr(g, "constants", {}),
        "in_allocated_constant_name": name
        in getattr(g, "allocated_constant_name", {}),
        "buffer_input_node_target": None,
        "graph_input_target": None,
        "in_static_input_indices": None,
    }
    # Many Inductor versions stash this on V.graph as a list of input idx
    sii = getattr(g, "static_input_indices", None)
    if sii is not None and name in getattr(g, "graph_input_names", []):
        idx = list(g.graph_input_names).index(name)
        obs["in_static_input_indices"] = idx in set(sii)
        obs["static_input_indices_value"] = list(sii)

    # Look at the graph_input itself — if `name` IS a graph input, what's
    # behind it?
    gi = getattr(g, "graph_inputs", {}).get(name)
    if gi is not None:
        obs["graph_input_target"] = type(gi).__name__
        if hasattr(gi, "data") and hasattr(gi.data, "data"):
            obs["graph_input_inner"] = type(gi.data.data).__name__

    # Walk the orig_gm placeholders to find one whose target matches name
    if hasattr(g, "orig_gm") and g.orig_gm is not None:
        for node in g.orig_gm.graph.nodes:
            if node.op == "placeholder" and node.name == name:
                obs["placeholder_meta_keys"] = sorted(node.meta.keys())
                desc = node.meta.get("desc")
                if desc is not None:
                    obs["placeholder_desc"] = repr(desc)[:300]
                    obs["placeholder_desc_type"] = type(desc).__name__
                tm = node.meta.get("tensor_meta")
                if tm is not None:
                    obs["placeholder_tensor_meta"] = repr(tm)[:300]
                src = node.meta.get("source_fn_stack")
                if src:
                    obs["source_fn_stack"] = str(src)[:200]
                break

    _observations.append(obs)
    return arg


_sk.SpyreKernel.create_tensor_arg = _patched_create_tensor_arg


# ---- Inductor knobs needed for the patch to actually fire ---------------

import torch._inductor.config as _icfg  # noqa: E402

_icfg.compile_threads = 1
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False


# ---- the model under test ----------------------------------------------


class TinyMM(torch.nn.Module):
    def __init__(self, M, N, K):
        super().__init__()
        # nn.Linear(K, N) — weight shape is (N, K), bias shape (N,)
        self.lin = torch.nn.Linear(K, N, bias=False, dtype=torch.float16)

    def forward(self, x):
        return self.lin(x)


def main() -> int:
    M, N, K = 128, 4096, 4096

    model = TinyMM(M, N, K).to("spyre")
    x = torch.randn(M, K, dtype=torch.float16, device="spyre")

    torch._dynamo.reset()
    compiled = torch.compile(model, dynamic=False)

    out = compiled(x)
    _ts.synchronize()
    _ = out.shape

    print("=" * 70)
    print(f"Captured {len(_observations)} create_tensor_arg() calls")
    print("=" * 70)
    for obs in _observations:
        print()
        for k, v in obs.items():
            print(f"  {k}: {v}")

    print()
    print("=" * 70)
    print("V.graph snapshot keys (final compile)")
    print("=" * 70)
    g = V.graph
    print(f"named_parameters:     {sorted(getattr(g, 'named_parameters', {}).keys())}")
    print(f"graph_input_names:    {getattr(g, 'graph_input_names', [])}")
    print(f"constants:            {sorted(getattr(g, 'constants', {}).keys())}")
    print(
        f"allocated_constant_name: "
        f"{dict(getattr(g, 'allocated_constant_name', {}))}"
    )
    if hasattr(g, "orig_gm") and g.orig_gm is not None:
        placeholders = [
            (n.name, sorted(n.meta.keys()))
            for n in g.orig_gm.graph.nodes
            if n.op == "placeholder"
        ]
        print(f"orig_gm placeholders: {placeholders}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
