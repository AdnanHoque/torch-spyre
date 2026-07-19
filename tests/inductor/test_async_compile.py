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

import os
from unittest import TestCase
from unittest.mock import patch

from torch_spyre._inductor import config
from torch_spyre.execution.async_compile import SpyreAsyncCompile


class TestSpyreAsyncCompile(TestCase):
    def test_forwards_planned_lx_fraction_to_dxp(self):
        with (
            config.patch({"dxp_lx_frac_avail": 0.375}),
            patch.dict(
                os.environ,
                {"DXP_LX_FRAC_AVAIL": "0.9", "SPYRE_TEST_ENV": "preserved"},
            ),
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/test_bundle",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run") as run,
            patch("torch_spyre.execution.async_compile.SpyreSDSCKernelRunner"),
        ):
            SpyreAsyncCompile().sdsc("test_kernel", [])

        run.assert_called_once()
        backend_env = run.call_args.kwargs["env"]
        self.assertEqual(backend_env["DXP_LX_FRAC_AVAIL"], "0.375")
        self.assertEqual(backend_env["SPYRE_TEST_ENV"], "preserved")
        self.assertIsNot(backend_env, os.environ)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
