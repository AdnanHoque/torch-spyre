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

import tempfile
from typing import Any
import os
import subprocess
import shutil

from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import OpSpec, UnimplementedOp
from torch_spyre._inductor.codegen.bundle import generate_bundle
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")


_RESTICKIFY_DDL_PREDDC_SHIM_SRC = r'''
#include <iostream>
class SuperDsc;
class Dsm {
 public:
  static void doCoreletSplitSdsc(SuperDsc* sdsc);
};
class L3DlOpsScheduler {
 public:
  void run(SuperDsc& sdsc);
};
void Dsm::doCoreletSplitSdsc(SuperDsc*) {
  std::cerr << "[torch-spyre] skipped Dsm::doCoreletSplitSdsc for restickify DDL bridge\n";
}
void L3DlOpsScheduler::run(SuperDsc&) {
  std::cerr << "[torch-spyre] skipped L3DlOpsScheduler::run for restickify DDL bridge\n";
}
'''


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

    def sdsc(self, kernel_name: str, specs: list[OpSpec | UnimplementedOp]):
        unimp = [s for s in specs if isinstance(s, UnimplementedOp)]
        if len(unimp) != 0:
            logger.warning(
                f"WARNING: Compiling unimplemented {unimp[0].op} to runtime exception"
            )
            return SpyreUnimplementedRunner(kernel_name, unimp[0].op)

        # Generate SDSC Bundle from OpSpecs
        output_dir = get_output_dir(kernel_name)
        op_specs = [s for s in specs if isinstance(s, OpSpec)]
        generate_bundle(kernel_name, output_dir, op_specs)

        # Invoke backend compiler of SDSC Bundle
        env = _dxp_env_for_bundle(output_dir)
        subprocess.run(
            ["dxp_standalone", "--bundle", "-d", output_dir],
            check=True,
            env=env,
        )

        return SpyreSDSCKernelRunner(kernel_name, output_dir)

    def wait(self, scope: dict[str, Any]) -> None:
        pass


def _dxp_env_for_bundle(output_dir: str) -> dict[str, str] | None:
    if not _spyre_config.restickify_ddl_bridge_e2e:
        return None
    if not _spyre_config.restickify_ddl_bridge_preddc_shim:
        return None
    if not _bundle_contains_only_restickify_ddl_bridge(output_dir):
        return None

    shim = _compile_restickify_ddl_preddc_shim(output_dir)
    env = os.environ.copy()
    old_preload = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = shim if not old_preload else f"{shim}:{old_preload}"
    return env


def _bundle_contains_only_restickify_ddl_bridge(output_dir: str) -> bool:
    files = [
        name
        for name in os.listdir(output_dir)
        if name.startswith("sdsc_") and name.endswith(".json")
    ]
    return bool(files) and all("_ddl_bridge" in name for name in files)


def _compile_restickify_ddl_preddc_shim(output_dir: str) -> str:
    src = os.path.join(output_dir, "restickify_ddl_preddc_shim.cpp")
    lib = os.path.join(output_dir, "librestickify_ddl_preddc_shim.so")
    if os.path.exists(lib):
        return lib
    with open(src, "w", encoding="utf-8") as handle:
        handle.write(_RESTICKIFY_DDL_PREDDC_SHIM_SRC)
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if cxx is None:
        raise RuntimeError("restickify DDL bridge shim needs g++, c++, or clang++")
    subprocess.run(
        [cxx, "-shared", "-fPIC", "-std=c++17", src, "-o", lib],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return lib
