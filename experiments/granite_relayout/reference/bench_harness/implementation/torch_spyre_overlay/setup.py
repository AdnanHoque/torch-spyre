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

import os
import shutil
from pathlib import Path
from typing import cast

os.environ.setdefault(
    "TORCH_DEVICE_BACKEND_AUTOLOAD", "0"
)  # must be before torch import
os.environ.setdefault(
    "SEN_COMMON_HEADERS", str(Path(__file__).resolve().parent.parent / "flex")
)


import glob

from setuptools import Command, setup

PATH_NAME = "torch_spyre"
PACKAGE_NAME = "torch_spyre"
DISTRIBUTED_PACKAGE_NAME = "spyre_ccl"


def get_torch_spyre_version() -> str:
    version_ns: dict[str, object] = {}
    with open(f"{PATH_NAME}/version.py") as f:
        exec(f.read(), version_ns)
        version = cast(str, version_ns["__version__"])
    return version


version = get_torch_spyre_version()


def check_libflex():
    ld_library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    for path in ld_library_paths:
        if glob.glob(os.path.join(path, "libflex.so")):
            return True
    return False


ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / PATH_NAME / "csrc"
DISTRIBUTED_SRC_DIR = CSRC_DIR / "distributed"
PROFILER_SRC_DIR = CSRC_DIR / "profiler"


# Automatically download json.hpp if not present
def maybe_download_nlohmann_json():
    """return path to header files"""
    import urllib.request

    NLOHMANN_URL = "https://raw.githubusercontent.com/nlohmann/json/v3.11.2/single_include/nlohmann/json.hpp"
    SHARED_PATH = Path(
        os.environ.get("SHARED_DEPS_DIR", ROOT_DIR / PATH_NAME / "csrc" / "external")
    )
    NLOHMANN_INC_DIR = SHARED_PATH / "nlohmann" / "include"
    NLOHMANN_DIR = NLOHMANN_INC_DIR / "nlohmann"

    NLOHMANN_HEADER = os.path.join(NLOHMANN_DIR, "json.hpp")
    if not os.path.exists(NLOHMANN_HEADER):
        os.makedirs(NLOHMANN_DIR, exist_ok=True)
        print("Downloading nlohmann/json.hpp...")
        urllib.request.urlretrieve(NLOHMANN_URL, NLOHMANN_HEADER)
    return NLOHMANN_INC_DIR


INCLUDE_DIRS = [
    CSRC_DIR,
    # "tracy/public"
]
LIBRARY_DIRS = []


INCLUDE_DIRS += [maybe_download_nlohmann_json()]

cmake_include_path = os.environ.get("CMAKE_INCLUDE_PATH", "")
extra_include_dirs = cmake_include_path.split(":") if cmake_include_path else []
INCLUDE_DIRS += [Path(p) for p in extra_include_dirs if p]

cmake_library_path = os.environ.get("CMAKE_LIBRARY_PATH", "")
extra_library_dirs = cmake_library_path.split(":") if cmake_library_path else []
LIBRARY_DIRS += [Path(p) for p in extra_library_dirs if p]

if "RUNTIME_INSTALL_DIR" in os.environ:
    # take lower precedence than CMAKE_LIBRARY_PATH and CMAKE_INCLUDE_PATH
    RUNTIME_DIR = Path(os.environ["RUNTIME_INSTALL_DIR"])
    SENLIB_DIR = Path(os.environ["SENLIB_INSTALL_DIR"])
    DEEPTOOLS_DIR = Path(os.environ["DEEPTOOLS_INSTALL_DIR"])
    INCLUDE_DIRS += [
        RUNTIME_DIR / "include",
    ]
    INCLUDE_DIRS += [
        RUNTIME_DIR / "include" / "concurrentqueue" / "moodycamel",
    ]
    INCLUDE_DIRS += [
        SENLIB_DIR / "include",
    ]
    INCLUDE_DIRS += [
        DEEPTOOLS_DIR / "include",
    ]
    LIBRARY_DIRS += [RUNTIME_DIR / "lib"]

# The USE_SPYRE_CCL environment variable can be used to build torch-spyre
# without support for Multi-Spyre. This is for developers only.
# If set to '0' then Multi-Spyre support is disabled.
# Otherwise (default) Multi-Spyre support is enabled.
use_spyre_ccl = os.environ.get("USE_SPYRE_CCL", "1") != "0"

if not use_spyre_ccl:
    print("=" * 80)
    print("WARNING: Multi-Spyre support has been disabled")
    print("=" * 80)
else:
    if "SPYRE_COMMS_INSTALL_DIR" in os.environ:
        SPYRE_COMMS_DIR = Path(os.environ["SPYRE_COMMS_INSTALL_DIR"])
        if not SPYRE_COMMS_DIR.exists():
            raise RuntimeError(
                f"SPYRE_COMMS_INSTALL_DIR directory does not exist: {SPYRE_COMMS_DIR}"
            )
        SPYRE_COMMS_INCLUDE_DIR = SPYRE_COMMS_DIR / "include"
        if not SPYRE_COMMS_INCLUDE_DIR.exists():
            raise RuntimeError(
                f"SPYRE_COMMS_INSTALL_DIR include directory does not exist: {SPYRE_COMMS_INCLUDE_DIR}"
            )
        SPYRE_COMMS_LIB_DIR = SPYRE_COMMS_DIR / "lib"
        if not SPYRE_COMMS_LIB_DIR.exists():
            raise RuntimeError(
                f"SPYRE_COMMS_INSTALL_DIR lib directory does not exist: {SPYRE_COMMS_LIB_DIR}"
            )
        INCLUDE_DIRS += [
            SPYRE_COMMS_INCLUDE_DIR,
        ]
        LIBRARY_DIRS += [SPYRE_COMMS_LIB_DIR]
    else:
        raise RuntimeError(
            "SPYRE_COMMS_INSTALL_DIR not set. "
            "Set USE_SPYRE_CCL=0 to build without Multi-Spyre support, "
            "or set the SPYRE_COMMS_INSTALL_DIR to the Spyre Comms install directory."
        )

INCLUDE_DIRS += [os.environ["SEN_COMMON_HEADERS"]]

use_new_system = os.environ.get("NEW_SYSTEM_SETUP", "0") == "1"
compile_aiupti = os.environ.get("USE_SPYRE_PROFILER", "0") == "1"

if compile_aiupti:
    if "LIBAIUPTI_INSTALL_DIR" not in os.environ:
        raise RuntimeError(
            "LIBAIUPTI_INSTALL_DIR must be set when USE_SPYRE_PROFILER=1"
        )
    LIBAIUPTI_DIR = Path(os.environ["LIBAIUPTI_INSTALL_DIR"])
    LIBAIUPTI_INCLUDE_DIR = LIBAIUPTI_DIR / "include"
    LIBAIUPTI_LIB_DIR = LIBAIUPTI_DIR / "lib"
    if not LIBAIUPTI_INCLUDE_DIR.exists():
        raise RuntimeError(
            f"libaiupti include directory not found: {LIBAIUPTI_INCLUDE_DIR}"
        )
    if not (LIBAIUPTI_LIB_DIR / "libaiupti.so").exists():
        raise RuntimeError(f"libaiupti shared library not found: {LIBAIUPTI_LIB_DIR}")
    INCLUDE_DIRS += [
        LIBAIUPTI_INCLUDE_DIR,
        LIBAIUPTI_INCLUDE_DIR / "libaiupti",
    ]
    LIBRARY_DIRS += [LIBAIUPTI_LIB_DIR]

if use_new_system:
    LIBRARIES = ["flex"]
else:
    LIBRARIES = ["sendnn", "sendnn_interface", "flex"]
if compile_aiupti:
    LIBRARIES.append("aiupti")
if use_spyre_ccl:
    LIBRARIES.append("spyre_comms")

# FIXME: added no-deprecated as this fails in sentensor_shape.hpp
# - we need to fix there
# Note that we always compile with debug info
# EXTRA_CXX_FLAGS = ["-g", "-Wall", "-Werror", "-Wno-deprecated"]
# Set TORCH_SPYRE_DEBUG=1 to build with -O0 for easier debugging
NO_OPT_BUILD = os.environ.get("TORCH_SPYRE_DEBUG", "0") == "1"

EXTRA_CXX_FLAGS = ["-g", "-Wall", "-Wno-deprecated", "-std=c++20"]
if NO_OPT_BUILD:
    EXTRA_CXX_FLAGS += ["-O0"]


class clean(Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove torch_spyre extension
        for path in (ROOT_DIR / PATH_NAME).glob("**/*.so"):
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


if __name__ == "__main__":
    import sys

    is_meta = any(
        cmd in sys.argv for cmd in ["dist_info", "egg_info", "install_egg_info"]
    )

    if is_meta:
        setup(
            entry_points={
                "torch.backends": [
                    "torch_spyre = torch_spyre:_autoload",
                ],
            },
        )
    else:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension

        if compile_aiupti:
            KINETO_INCLUDE_DIR = Path(torch.__path__[0]) / "include" / "kineto"
            PRIVATEUSE1_PROFILER_HEADER = (
                Path(torch.__path__[0])
                / "include"
                / "torch"
                / "csrc"
                / "profiler"
                / "standalone"
                / "privateuse1_profiler.h"
            )
            if not KINETO_INCLUDE_DIR.exists():
                raise RuntimeError(
                    f"Kineto include directory not found: {KINETO_INCLUDE_DIR}"
                )
            if not PRIVATEUSE1_PROFILER_HEADER.exists():
                raise RuntimeError(
                    "PrivateUse1 profiler header not found: "
                    f"{PRIVATEUSE1_PROFILER_HEADER}"
                )
            INCLUDE_DIRS += [KINETO_INCLUDE_DIR]

        profiler_sources = (
            list(PROFILER_SRC_DIR.glob("*.cpp")) if compile_aiupti else []
        )
        sources = list(CSRC_DIR.glob("*.cpp")) + profiler_sources
        distributed_sources = (
            list(DISTRIBUTED_SRC_DIR.glob("*.cpp")) if use_spyre_ccl else []
        )

        core_src_paths = [p.relative_to(ROOT_DIR).as_posix() for p in sorted(sources)]
        distributed_src_paths = [
            p.relative_to(ROOT_DIR).as_posix() for p in sorted(distributed_sources)
        ]

        # Build define_macros list conditionally
        base_define_macros = [
            ("PACKAGE_NAME", f'"{PACKAGE_NAME}"'),
            ("SPYRE_DEBUG_ENV", '"TORCH_SPYRE_DEBUG"'),
            ("SPYRE_DOWNCAST_ENV", '"TORCH_SPYRE_DOWNCAST_WARN"'),
            ("EAGER_MODE_ENV", '"EAGER_MODE"'),
            ("BOOST_ALL_DYN_LINK", None),  # avoid static link to boost
        ]
        if use_spyre_ccl:
            base_define_macros.append(("USE_SPYRE_CCL", None))
        if use_new_system:
            base_define_macros.append(("USE_FLEX_NAMESPACE", None))
        profiler_define_macros = (
            [
                ("HAS_AIUPTI", None),
                ("USE_KINETO", None),
                ("FMT_HEADER_ONLY", None),
            ]
            if compile_aiupti
            else []
        )

        ext_modules = [
            CppExtension(
                name=f"{PACKAGE_NAME}._C",
                sources=core_src_paths + distributed_src_paths,
                include_dirs=[str(p) for p in INCLUDE_DIRS],
                library_dirs=[str(p) for p in LIBRARY_DIRS],
                libraries=LIBRARIES,
                extra_compile_args={"cxx": EXTRA_CXX_FLAGS},
                define_macros=base_define_macros
                + profiler_define_macros
                + [
                    ("MODULE_NAME", f'"{PACKAGE_NAME}._C"'),
                ],
            ),
        ]

        BUILD_DIR = ROOT_DIR / "build"

        _BuildExtension = BuildExtension.with_options(
            no_python_abi_suffix=True, verbose=True
        )

        class PermanentBuildExtension(_BuildExtension):
            def finalize_options(self):
                super().finalize_options()
                self.build_temp = str(BUILD_DIR)

            def build_extension(self, ext):
                # Use a per-extension subdirectory so each gets its own build.ninja
                original_build_temp = self.build_temp
                self.build_temp = os.path.join(original_build_temp, ext.name)
                os.makedirs(self.build_temp, exist_ok=True)
                try:
                    super().build_extension(ext)
                finally:
                    self.build_temp = original_build_temp

        setup(
            ext_modules=ext_modules,
            cmdclass={
                "build_ext": PermanentBuildExtension,
                "clean": clean,
            },
            entry_points={
                "torch.backends": [
                    "torch_spyre = torch_spyre:_autoload",
                ],
            },
        )
