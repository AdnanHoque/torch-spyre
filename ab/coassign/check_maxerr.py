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
"""Compare a saved co-assign device output against the CPU eager reference.

Pure-CPU: loads the ``--_worker-output`` .pt produced by ``check_outputs.py``
(the co-assigned device run) and diffs it against ``_run_eager_cpu`` for the
same op/shape/seed. No device touched. Confirms co-assignment value-correctness
(expect fp16-noise ~1e-2, the same level as the unfused baseline 0.011 — NOT
the ~0 corruption of the data-op reshard).

    python check_maxerr.py <saved_out.pt> <op> <seed> <D0> <D1> <D2> ...
"""

import sys

import torch

sys.path.insert(0, "/home/adnan/dt-inductor/spyre-perf-suite-aisw")
import check_outputs  # noqa: E402


def main():
    out_path = sys.argv[1]
    op = sys.argv[2]
    seed = int(sys.argv[3])
    shape = tuple(int(x) for x in sys.argv[4:])

    actual = torch.load(out_path, map_location="cpu", weights_only=False)
    expected = check_outputs._run_eager_cpu(op, [shape], None, seed)

    if isinstance(actual, (list, tuple)):
        actual = actual[0]
    if isinstance(expected, (list, tuple)):
        expected = expected[0]
    actual = actual.detach().to(torch.float32).cpu()
    expected = expected.detach().to(torch.float32).cpu()

    diff = (actual - expected).abs()
    print(f"op={op} shape={shape} seed={seed}")
    print(f"  actual   mean={actual.mean().item():.6g}  std={actual.std().item():.6g}")
    print(f"  expected mean={expected.mean().item():.6g}  std={expected.std().item():.6g}")
    print(f"  max_abs_diff={diff.max().item():.6g}")
    print(f"  mean_abs_diff={diff.mean().item():.6g}")
    print(f"  allclose(atol=1e-2,rtol=1e-2)={torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)}")


if __name__ == "__main__":
    main()
