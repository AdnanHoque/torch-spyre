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
#include <dlfcn.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include <dsc/superdsc.h>
#include <dsc/designSpaceConfig.h>
#include <dsc/dscdefn.h>

class Dsm {
 public:
  static void doCoreletSplitSdsc(SuperDsc* sdsc);
};

class L3DlOpsScheduler {
 public:
  void run(SuperDsc& sdsc);
};

namespace ddc {
class Ddc {
 public:
  bool run_v1(SuperDsc& sdsc);
  void coordinateCapture();
  bool buildFoldForTransfer(dsc2::TransferNode* transferNode,
                            dsc2::CoordPropInfoType& coordPropInfo);
};
}  // namespace ddc

static bool skip_bridge_ddc_after_l3 = false;
static bool skip_coordinate_capture_after_bridge_l3 = false;

static bool has_bridge_name(const SuperDsc* sdsc) {
  if (!sdsc) {
    return false;
  }
  if (sdsc->name_.find("_ddl_bridge") != std::string::npos) {
    return true;
  }
  for (const auto& dsc : sdsc->dscs_) {
    if (dsc.name_.find("_ddl_bridge") != std::string::npos) {
      return true;
    }
  }
  return false;
}

static bool is_restickify_ddl_bridge(const SuperDsc* sdsc) {
  if (!has_bridge_name(sdsc) || sdsc->dscs_.empty()) {
    return false;
  }
  for (const auto& dsc : sdsc->dscs_) {
    if (dsc.computeOp_.size() != 1) {
      return false;
    }
    const auto op = dsc.computeOp_.front().opFuncName;
    if (op != OpFuncs::ReStickifyOpHBM && op != OpFuncs::ReStickifyOpLx) {
      return false;
    }
    if (dsc.dataStageParam_.size() < 2) {
      return false;
    }
  }
  return true;
}

static bool name_matches_any(const SuperDsc* sdsc, const char* raw_names) {
  if (!sdsc || raw_names == nullptr || raw_names[0] == '\0') {
    return false;
  }
  const std::string names(raw_names);
  const auto has_match = [&names](const std::string& candidate) {
    std::size_t start = 0;
    while (start <= names.size()) {
      auto end = names.find(',', start);
      auto token = names.substr(start, end == std::string::npos ? end : end - start);
      if (!token.empty() && candidate.find(token) != std::string::npos) {
        return true;
      }
      if (end == std::string::npos) {
        break;
      }
      start = end + 1;
    }
    return false;
  };
  if (has_match(sdsc->name_)) {
    return true;
  }
  for (const auto& dsc : sdsc->dscs_) {
    if (has_match(dsc.name_)) {
      return true;
    }
  }
  return false;
}

static void rename_prefilled_interslice_nodes_before_l3(SuperDsc& sdsc) {
  if (!name_matches_any(&sdsc, "interslicetranspose_fp16_ddl_bridge")) {
    skip_bridge_ddc_after_l3 = false;
    skip_coordinate_capture_after_bridge_l3 = false;
    return;
  }
  skip_bridge_ddc_after_l3 = true;
  skip_coordinate_capture_after_bridge_l3 = true;
  for (auto& dsc : sdsc.dscs_) {
    for (auto* node : dsc.scheduleTree_.traverseTreeDFSMutable()) {
      if (node->name_ == "transfer_lds0_src:no_component_dst:lx_lx_local") {
        node->name_ = "prefill_transfer_lds0_src:no_component_dst:lx_lx_local";
      } else if (node->name_ == "transfer_lds1_src:lx_dst:no_component_lx_local") {
        node->name_ = "prefill_transfer_lds1_src:lx_dst:no_component_lx_local";
      } else if (node->name_ == "loop_ds0_ds1_y") {
        node->name_ = "prefill_loop_ds0_ds1_y";
      } else if (node->name_ == "loop_ds0_ds1_out") {
        node->name_ = "prefill_loop_ds0_ds1_out";
      } else if (node->name_ == "loop_ds0_ds1_mb") {
        node->name_ = "prefill_loop_ds0_ds1_mb";
      } else if (node->name_ == "lx_below_schedule") {
        node->name_ = "prefill_lx_below_schedule";
      }
    }
  }
}

template <typename Fn>
static Fn required_next_symbol(const char* name) {
  dlerror();
  auto* sym = dlsym(RTLD_NEXT, name);
  const char* err = dlerror();
  if (err != nullptr || sym == nullptr) {
    std::cerr << "[torch-spyre] restickify DDL bridge shim could not find "
              << name << ": " << (err == nullptr ? "missing symbol" : err)
              << "\n";
    std::abort();
  }
  return reinterpret_cast<Fn>(sym);
}

void Dsm::doCoreletSplitSdsc(SuperDsc* sdsc) {
  if (is_restickify_ddl_bridge(sdsc)) {
    const char* run_bridge_corelet =
        std::getenv("SPYRE_RESTICKIFY_DDL_SHIM_RUN_CORELET_FOR_BRIDGE");
    if (run_bridge_corelet != nullptr && std::string(run_bridge_corelet) == "1") {
      using Fn = void (*)(SuperDsc*);
      static Fn next = required_next_symbol<Fn>("_ZN3Dsm18doCoreletSplitSdscEP8SuperDsc");
      next(sdsc);
      return;
    }
    std::cerr << "[torch-spyre] skipped Dsm::doCoreletSplitSdsc for "
              << sdsc->name_ << "\n";
    return;
  }
  const char* skip_names =
      std::getenv("SPYRE_RESTICKIFY_DDL_SHIM_SKIP_CORELET_NAMES");
  if (name_matches_any(sdsc, skip_names)) {
    std::cerr << "[torch-spyre] skipped Dsm::doCoreletSplitSdsc by name for "
              << sdsc->name_ << "\n";
    return;
  }
  using Fn = void (*)(SuperDsc*);
  static Fn next = required_next_symbol<Fn>("_ZN3Dsm18doCoreletSplitSdscEP8SuperDsc");
  next(sdsc);
}

void L3DlOpsScheduler::run(SuperDsc& sdsc) {
  if (is_restickify_ddl_bridge(&sdsc)) {
    const char* run_bridge_l3 =
        std::getenv("SPYRE_RESTICKIFY_DDL_SHIM_RUN_L3_FOR_BRIDGE");
    if (run_bridge_l3 != nullptr && std::string(run_bridge_l3) == "1") {
      using Fn = void (*)(L3DlOpsScheduler*, SuperDsc&);
      static Fn next = required_next_symbol<Fn>("_ZN16L3DlOpsScheduler3runER8SuperDsc");
      next(this, sdsc);
      return;
    }
    std::cerr << "[torch-spyre] skipped L3DlOpsScheduler::run for "
              << sdsc.name_ << "\n";
    return;
  }
  const char* skip_names =
      std::getenv("SPYRE_RESTICKIFY_DDL_SHIM_SKIP_L3_NAMES");
  if (name_matches_any(&sdsc, skip_names)) {
    std::cerr << "[torch-spyre] skipped L3DlOpsScheduler::run by name for "
              << sdsc.name_ << "\n";
    return;
  }
  rename_prefilled_interslice_nodes_before_l3(sdsc);
  using Fn = void (*)(L3DlOpsScheduler*, SuperDsc&);
  static Fn next = required_next_symbol<Fn>("_ZN16L3DlOpsScheduler3runER8SuperDsc");
  const char* swallow_l3_errors =
      std::getenv("SPYRE_RESTICKIFY_DDL_SHIM_SWALLOW_L3_ERRORS");
  if (swallow_l3_errors != nullptr &&
      std::string(swallow_l3_errors) == "1" && has_bridge_name(&sdsc)) {
    try {
      next(this, sdsc);
    } catch (const std::exception& ex) {
      std::cerr << "[torch-spyre] swallowed L3DlOpsScheduler::run exception for "
                << sdsc.name_ << ": " << ex.what() << "\n";
    }
    return;
  }
  next(this, sdsc);
}

bool ddc::Ddc::run_v1(SuperDsc& sdsc) {
  using Fn = bool (*)(ddc::Ddc*, SuperDsc&);
  static Fn next = required_next_symbol<Fn>("_ZN3ddc3Ddc6run_v1ER8SuperDsc");
  const char* skip_bridge_ddc =
      std::getenv("SPYRE_RESTICKIFY_DDL_SHIM_SKIP_BRIDGE_DDC_AFTER_L3");
  if (skip_bridge_ddc != nullptr && std::string(skip_bridge_ddc) == "1" &&
      skip_bridge_ddc_after_l3 && has_bridge_name(&sdsc)) {
    std::cerr << "[torch-spyre] skipped DDC run_v1 after interslice bridge L3 "
              << "scheduling for " << sdsc.name_ << "\n";
    skip_bridge_ddc_after_l3 = false;
    skip_coordinate_capture_after_bridge_l3 = false;
    return true;
  }
  skip_bridge_ddc_after_l3 = false;
  return next(this, sdsc);
}

void ddc::Ddc::coordinateCapture() {
  using Fn = void (*)(ddc::Ddc*);
  static Fn next = required_next_symbol<Fn>("_ZN3ddc3Ddc17coordinateCaptureEv");
  const char* skip_bridge_coordinate_capture =
      std::getenv("SPYRE_RESTICKIFY_DDL_SHIM_SKIP_BRIDGE_COORDINATE_CAPTURE");
  if (skip_bridge_coordinate_capture != nullptr &&
      std::string(skip_bridge_coordinate_capture) == "1" &&
      skip_coordinate_capture_after_bridge_l3) {
    std::cerr << "[torch-spyre] skipped DDC coordinateCapture after "
              << "interslice bridge L3 scheduling\n";
    skip_coordinate_capture_after_bridge_l3 = false;
    return;
  }
  skip_coordinate_capture_after_bridge_l3 = false;
  next(this);
}

bool ddc::Ddc::buildFoldForTransfer(dsc2::TransferNode* transferNode,
                                    dsc2::CoordPropInfoType& coordPropInfo) {
  using Fn = bool (*)(ddc::Ddc*, dsc2::TransferNode*, dsc2::CoordPropInfoType&);
  static Fn next = required_next_symbol<Fn>(
      "_ZN3ddc3Ddc20buildFoldForTransferEPN4dsc212TransferNodeERNS1_17CoordPropInfoTypeE");
  const char* log_transfers =
      std::getenv("SPYRE_RESTICKIFY_DDL_SHIM_LOG_DDC_TRANSFERS");
  const bool should_log =
      log_transfers != nullptr && std::string(log_transfers) == "1";
  if (should_log && transferNode != nullptr) {
    std::cerr << "[torch-spyre] DDC buildFoldForTransfer "
              << transferNode->name_ << "\n";
    coordPropInfo.print(std::cerr);
    std::cerr << "\n";
  }
  try {
    return next(this, transferNode, coordPropInfo);
  } catch (const std::exception& ex) {
    std::cerr << "[torch-spyre] DDC buildFoldForTransfer failed";
    if (transferNode != nullptr) {
      std::cerr << " for " << transferNode->name_;
    }
    std::cerr << ": " << ex.what() << "\n";
    coordPropInfo.print(std::cerr);
    std::cerr << "\n";
    const char* swallow_transfer_errors =
        std::getenv("SPYRE_RESTICKIFY_DDL_SHIM_SWALLOW_DDC_TRANSFER_ERRORS");
    if (swallow_transfer_errors != nullptr &&
        std::string(swallow_transfer_errors) == "1") {
      std::cerr << "[torch-spyre] swallowed DDC buildFoldForTransfer failure";
      if (transferNode != nullptr) {
        std::cerr << " for " << transferNode->name_;
      }
      std::cerr << "\n";
      return false;
    }
    throw;
  }
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
    if not _bundle_contains_restickify_ddl_bridge(output_dir):
        return None

    shim = _compile_restickify_ddl_preddc_shim(output_dir)
    env = os.environ.copy()
    old_preload = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = shim if not old_preload else f"{shim}:{old_preload}"
    return env


def _bundle_contains_restickify_ddl_bridge(output_dir: str) -> bool:
    files = [
        name
        for name in os.listdir(output_dir)
        if name.startswith("sdsc_") and name.endswith(".json")
    ]
    return any("_ddl_bridge" in name for name in files)


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
    deeptools_include = os.path.join(
        os.environ.get("DEEPTOOLS_INSTALL_DIR", "/opt/ibm/spyre/deeptools"),
        "include",
    )
    subprocess.run(
        [
            cxx,
            "-shared",
            "-fPIC",
            "-std=c++17",
            f"-I{deeptools_include}",
            src,
            "-ldl",
            "-o",
            lib,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return lib
