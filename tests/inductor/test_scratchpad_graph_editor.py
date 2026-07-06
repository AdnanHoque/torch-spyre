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

import importlib.util
import pathlib
import types
import sys
import unittest

import torch
from torch._inductor.ir import FixedLayout, InputBuffer, ReinterpretView, StorageBox


def _load_graph_editor():
    """Load GraphEditor without importing extension-backed helper modules."""
    module_path = (
        pathlib.Path(__file__).parents[2]
        / "torch_spyre"
        / "_inductor"
        / "scratchpad"
        / "graph_editor.py"
    )
    stub_names = (
        "torch_spyre._inductor.pass_utils",
        "torch_spyre._inductor.ir",
    )
    old_modules = {name: sys.modules.get(name) for name in stub_names}
    parent_attrs = {}
    for name in stub_names:
        parent_name, attr = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            parent_attrs[name] = (parent, hasattr(parent, attr), getattr(parent, attr, None))

    pass_utils = types.ModuleType("torch_spyre._inductor.pass_utils")
    pass_utils.copy_op_metadata = lambda *, src, dst: None
    spyre_ir = types.ModuleType("torch_spyre._inductor.ir")
    spyre_ir.FixedTiledLayout = type("FixedTiledLayout", (), {})

    try:
        sys.modules["torch_spyre._inductor.pass_utils"] = pass_utils
        sys.modules["torch_spyre._inductor.ir"] = spyre_ir
        spec = importlib.util.spec_from_file_location(
            "_test_scratchpad_graph_editor", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.GraphEditor
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
        for name, (parent, had_attr, value) in parent_attrs.items():
            _, attr = name.rsplit(".", 1)
            if had_attr:
                setattr(parent, attr, value)
            elif hasattr(parent, attr):
                delattr(parent, attr)


class TestGraphEditor(unittest.TestCase):
    def test_change_graph_output_preserves_reinterpret_view(self):
        GraphEditor = _load_graph_editor()
        storage_layout = FixedLayout(torch.device("cpu"), torch.float16, [2, 3], [3, 1])
        view_layout = FixedLayout(torch.device("cpu"), torch.float16, [3, 2], [1, 3])
        old_buffer = InputBuffer(name="buf11", layout=storage_layout)
        new_buffer = InputBuffer(name="buf11_clone", layout=storage_layout)

        graph_output = ReinterpretView(
            data=StorageBox(old_buffer),
            layout=view_layout,
        )
        editor = GraphEditor.__new__(GraphEditor)
        editor.lowering = types.SimpleNamespace(graph_outputs=[graph_output])

        editor.change_graph_output(old_buffer, new_buffer)  # type: ignore[arg-type]

        replacement = editor.lowering.graph_outputs[0]
        self.assertIsInstance(replacement, ReinterpretView)
        self.assertIs(replacement.layout, view_layout)
        self.assertIsInstance(replacement.data, StorageBox)
        self.assertIs(replacement.data.data, new_buffer)


if __name__ == "__main__":
    unittest.main()
