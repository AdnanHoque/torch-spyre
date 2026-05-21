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

import json
import os

from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.codegen.lx_neighbor_descriptor import (
    maybe_emit_lx_neighbor_descriptor,
)
from torch_spyre._inductor.codegen.restickify_lx_boundary import (
    patch_restickify_ddl_bridge_boundaries,
)
from torch_spyre._inductor.codegen.restickify_ptlx_boundary import (
    patch_restickify_ptlx_bridge_boundaries,
)
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import OpSpec
from torch_spyre._inductor.logging_utils import get_inductor_logger


logger = get_inductor_logger("sdsc_compile")


def generate_bundle(kernel_name: str, output_dir: str, specs: list[OpSpec]):
    """Output the SDSC Bundle for the OpSpecs in the given output_dir for the OpSpecs"""

    # 1. Generate SDSC.json for each OpSpec
    sdscs_json = []
    for idx, ks in enumerate(specs):
        allow_restickify_ddl_bridge = _allow_restickify_ddl_bridge_in_bundle(
            idx, ks, specs
        )
        sdsc_json = compile_op_spec(
            idx,
            ks,
            allow_restickify_ddl_bridge=allow_restickify_ddl_bridge,
        )
        sdscs_json.append(sdsc_json)

    if _spyre_config.restickify_ddl_bridge_boundary_patch:
        rows = patch_restickify_ddl_bridge_boundaries(sdscs_json, specs)
        for row in rows:
            logger.info("restickify DDL bridge boundary patch: %s", row)

    if _spyre_config.restickify_ptlx_bridge_e2e:
        rows = patch_restickify_ptlx_bridge_boundaries(sdscs_json, specs)
        for row in rows:
            logger.info("restickify PT-LX bridge boundary patch: %s", row)

    # Write JSON SDSCs to file system
    files = []
    for sdsc_json in sdscs_json:
        sdsc_name = next(iter(sdsc_json))
        file_name = f"sdsc_{sdsc_name}.json"
        files.append(file_name)
        with open(os.path.join(output_dir, file_name), "w") as file:
            logger.info(f"Generating {file.name}")
            json.dump(sdsc_json, file, indent=2)

    maybe_emit_lx_neighbor_descriptor(
        kernel_name,
        output_dir,
        files,
        specs,
        sdsc_payloads=sdscs_json,
    )

    # Generate bundle.mlir
    with open(os.path.join(output_dir, "bundle.mlir"), "w") as file:
        logger.info(f"Generating {file.name}")
        file.write("module {\n")
        file.write("\tfunc.func @sdsc_bundle() {\n")
        for f in files:
            file.write('\t\tsdscbundle.sdsc_execute () {sdsc_filename="' + f + '"}\n')
        file.write("\t\treturn\n")
        file.write("\t}\n")
        file.write("}\n")


def _allow_restickify_ddl_bridge_in_bundle(
    idx: int,
    spec: OpSpec,
    specs: list[OpSpec],
) -> bool:
    """Gate the probe-only DDL bridge to same-bundle internal edges.

    The DDL bridge prototype is only meaningful when a producer, restickify,
    and consumer are packaged in the same runtime bundle. A leading restickify
    can still have an in-graph producer at the Torch FX level, but if that
    producer lives in a previous runtime bundle then the bridge has no LX
    allocation to alias and must stay on the existing HBM path.
    """

    if spec.op != RESTICKIFY_OP:
        return True
    if idx == 0 or idx + 1 >= len(specs):
        return False
    return True
