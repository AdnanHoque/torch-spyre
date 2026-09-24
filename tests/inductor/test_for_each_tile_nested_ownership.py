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

"""Where a nested for_each_tile's pre-loop carry copy runs.

When an inner loop's initial value is not private to that loop, splicing the
inner loop inserts one copy before it (#4838). The inner loop sits inside the
outer loop's body, so the copy must run on every trip of every enclosing loop,
and never inside the inner loop itself. CPU only; no Spyre device needed.
"""

import unittest

from torch._inductor.virtualized import V

import test_for_each_tile_lowering as lowering_tests
from for_each_tile_fixtures import (
    matmul_inputs,
    nested_two_inner_loops_shared_init_fn,
)


class TestNestedPreLoopCopyGroup(unittest.TestCase):
    # The lowering tests' own capture-and-lower helper, reused rather than
    # copied.
    _run_graph = lowering_tests.TestSpliceWhileLoops._run_graph

    def test_inner_pre_loop_copy_belongs_to_enclosing_loop_only(self):
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            splice_while_loops,
        )

        (X, Y), _ref = matmul_inputs()
        graph = self._run_graph(nested_two_inner_loops_shared_init_fn, (X, Y))
        with V.set_graph_handler(graph):
            splice_while_loops(graph)

        copies = [
            op for op in graph.operations if "while_loop_carry_copy_" in op.get_name()
        ]
        # The first inner loop spliced copies the shared fill. The second may
        # then own the fill outright: a copy of a pure fill inlines the fill
        # rather than reading it, so the copy is not a second reader.
        self.assertTrue(
            [op for op in copies if len(op.get_size()) == 2],
            "the shared fill must be copied before an inner loop",
        )
        for op in copies:
            info = getattr(op, "loop_info", None)
            self.assertIsNotNone(
                info, f"{op.get_name()} is not a member of the enclosing loop"
            )
            self.assertEqual(
                info.loop_group_id,
                (0,),
                f"{op.get_name()} must run once per outer trip, "
                "not inside the inner loop",
            )


if __name__ == "__main__":
    unittest.main()
