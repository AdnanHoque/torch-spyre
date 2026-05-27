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

from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.op_spec import OpSpec
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.onchip_realize import (
    build_flash_attention_ifn_pair_tile_artifacts,
    build_flash_attention_layout_xform_pair_tile_artifacts,
    build_flash_attention_pipeline_artifact,
    build_flash_attention_pipeline_tile_artifacts,
    build_flash_attention_value_flow_tile_artifact,
    flash_attention_ifn_pair_tile_rejection_reasons,
    flash_attention_layout_xform_pair_rejection_reasons,
    flash_attention_value_flow_tile_rejection_reasons,
    LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE,
    realize_flash_attention_pointwise_handoffs,
    realize_onchip_handoff,
)


logger = get_inductor_logger("sdsc_compile")


def fold_onchip_handoff(sdsc_json: dict, realization) -> dict:
    """Fold a same-layout on-chip handoff into the consumer SDSC body in place.

    Installs the synthesized datadscs_/coreIdToDscSchedule/opFuncsUsed_ from
    ``realization`` (an onchip_realize.OnChipRealization) onto the single SDSC
    body. The producer-output and consumer-input LX flips are descriptors the
    caller must apply once labeledDs_ scaffolding is present; here we only fold
    the data-ops. Mirrors splice_2048_stcdp.patch_consumer_to_mixed.
    """
    body = sdsc_json[next(iter(sdsc_json))]
    body["coreIdToDscSchedule"] = realization.schedule
    body["datadscs_"] = realization.datadscs
    body["opFuncsUsed_"] = realization.opfuncs
    return sdsc_json


def generate_bundle(kernel_name: str, output_dir: str, specs: list[OpSpec]):
    """Output the SDSC Bundle for the OpSpecs in the given output_dir for the OpSpecs"""

    # 1. Generate SDSC.json for each OpSpec
    sdscs_json = []
    for idx, ks in enumerate(specs):
        sdsc_json = compile_op_spec(idx, ks)
        sdscs_json.append(sdsc_json)
    # When SPYRE_ONCHIP_HANDOFF_REALIZE is on, detect the eligible same-stick
    # same-shard producer->consumer edge among the SDSCs and turn the consumer
    # into a mixed DL+data-op SuperDSC (LX-resident handoff, no HBM round trip).
    # Default off -> output byte-identical to before. Needs the deeptools
    # Foundation gate + a device build to execute, so default fail-closed.
    if config.onchip_handoff_realize:
        if realize_onchip_handoff(
            sdscs_json,
            attention_score_handoff=config.onchip_attention_score_handoff,
            static_matmul_handoff=config.onchip_static_matmul_handoff,
            min_handoff_bytes=config.onchip_handoff_min_bytes,
        ):
            logger.info("Realized on-chip handoff")
    value_flow_tile = config.flash_attention_mixed_pipeline_value_flow_tile
    ifn_pair_tile = config.flash_attention_mixed_pipeline_ifn_pair_tile
    layout_xform_pair_tile = (
        config.flash_attention_mixed_pipeline_layout_xform_pair_tile
    )
    sidecar_sdscs = []
    sidecar_replacements = {}
    bundle_attrs_by_file = {}
    value_flow_rejections = {}
    emit_mixed_sidecars = (
        config.flash_attention_mixed_pipeline
        and (
            config.flash_attention_mixed_pipeline_artifact
            or config.flash_attention_mixed_pipeline_execute_tile >= 0
            or value_flow_tile >= 0
            or ifn_pair_tile >= 0
            or layout_xform_pair_tile != -1
        )
    )
    layout_xform_pair = None
    layout_xform_pair_rejections = None
    if emit_mixed_sidecars and layout_xform_pair_tile != -1:
        layout_xform_pair = (
            build_flash_attention_layout_xform_pair_tile_artifacts(
                sdscs_json,
                layout_xform_pair_tile,
            )
        )
        if layout_xform_pair is None:
            layout_xform_pair_rejections = (
                flash_attention_layout_xform_pair_rejection_reasons(
                    sdscs_json,
                    layout_xform_pair_tile,
                )
            )
    if (
        config.flash_attention_mixed_pipeline
        and config.flash_attention_pointwise_handoff
    ):
        pointwise_kwargs = {
            "score_scale_handoff": config.flash_attention_score_scale_handoff,
        }
        if layout_xform_pair is not None:
            pointwise_kwargs["pointwise_region0"] = (
                LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE
            )
            logger.info(
                "Realizing flash pointwise handoffs in a disjoint LX region "
                "while the layout-transform pair probe is active"
            )
        count = realize_flash_attention_pointwise_handoffs(
            sdscs_json,
            **pointwise_kwargs,
        )
        if count:
            logger.info(f"Realized {count} flash pointwise on-chip handoffs")
    if emit_mixed_sidecars:
        if ifn_pair_tile >= 0:
            ifn_pair = build_flash_attention_ifn_pair_tile_artifacts(
                sdscs_json,
                ifn_pair_tile,
            )
            if ifn_pair is not None:
                sidecar_sdscs.extend(ifn_pair["artifacts"])
                sidecar_replacements.update(ifn_pair["replacements"])
                bundle_attrs_by_file.update(ifn_pair["bundle_attrs"])
                logger.info(
                    "Executing explicit LX-copy flash attention pair sidecars"
                )
            else:
                logger.warning(
                    "Requested explicit LX-copy flash attention pair was "
                    "not realizable; keeping generated HBM-backed SDSCs: %s",
                    flash_attention_ifn_pair_tile_rejection_reasons(
                        sdscs_json,
                        ifn_pair_tile,
                    ),
                )
        if layout_xform_pair_tile != -1:
            if layout_xform_pair is not None:
                sidecar_sdscs.extend(layout_xform_pair["artifacts"])
                sidecar_replacements.update(layout_xform_pair["replacements"])
                bundle_attrs_by_file.update(layout_xform_pair["bundle_attrs"])
                logger.info(
                    "Executing experimental layout-transform flash attention "
                    "pair sidecars"
                )
            else:
                logger.warning(
                    "Requested layout-transform flash attention pair was not "
                    "realizable; keeping generated HBM-backed SDSCs: %s",
                    layout_xform_pair_rejections,
                )
        if value_flow_tile >= 0:
            value_flow = build_flash_attention_value_flow_tile_artifact(
                sdscs_json,
                value_flow_tile,
            )
            if value_flow is not None:
                artifact, replaced = value_flow
                sidecar_name = next(iter(artifact))
                sidecar_sdscs.append(artifact)
                if replaced not in sidecar_replacements:
                    sidecar_replacements[replaced] = sidecar_name
                    logger.info(
                        "Executing mixed flash attention value-flow tile sidecar"
                    )
            else:
                value_flow_rejections[value_flow_tile] = (
                    flash_attention_value_flow_tile_rejection_reasons(
                        sdscs_json,
                        value_flow_tile,
                    )
                )
                logger.warning(
                    "Requested mixed flash attention value-flow tile was not "
                    "realizable; keeping generated HBM-backed SDSC"
                )
        tile_artifacts = build_flash_attention_pipeline_tile_artifacts(
            sdscs_json,
            overlap_prefix=config.flash_attention_mixed_pipeline_overlap,
        )
        sidecar_sdscs.extend(tile_artifacts)
        execute_tile = config.flash_attention_mixed_pipeline_execute_tile
        for artifact in tile_artifacts:
            sidecar_name, sidecar_body = next(iter(artifact.items()))
            meta = sidecar_body.get("flashAttentionPipeline_", {})
            if meta.get("tile_index") != execute_tile:
                continue
            if meta.get("overlap_prefix") and meta.get("ifn_attached_input_idx") != 0:
                logger.warning(
                    "Requested mixed flash attention overlap tile is not "
                    "IFN-attached; keeping generated HBM-backed SDSC"
                )
                continue
            if meta.get("overlap_prefix") and not meta.get("ifn_runtime_safe", False):
                logger.warning(
                    "Requested mixed flash attention overlap tile is not "
                    "runtime-safe; keeping generated HBM-backed SDSC"
                )
                continue
            if meta.get("overlap_prefix_rejection_reasons"):
                continue
            replaced = meta.get("replaces_sdsc")
            if replaced is not None and replaced not in sidecar_replacements:
                sidecar_replacements[replaced] = sidecar_name
        for artifact in tile_artifacts:
            _sidecar_name, sidecar_body = next(iter(artifact.items()))
            meta = sidecar_body.get("flashAttentionPipeline_", {})
            tile_index = meta.get("tile_index")
            if tile_index in value_flow_rejections:
                meta["value_flow_requested"] = True
                meta["value_flow_rejection_reasons"] = (
                    value_flow_rejections[tile_index]
                )
        if config.flash_attention_mixed_pipeline_artifact:
            artifact = build_flash_attention_pipeline_artifact(
                sdscs_json,
                overlap=config.flash_attention_mixed_pipeline_overlap,
                name="mixed_flash_pipeline_full_artifact",
            )
            if artifact is not None:
                sidecar_sdscs.append(artifact)
                logger.info("Emitted mixed flash attention pipeline sidecar artifact")

    # Write JSON SDSCs to file system
    files = []
    for sdsc_json in sdscs_json:
        sdsc_name = next(iter(sdsc_json))
        bundle_sdsc_name = sidecar_replacements.get(sdsc_name, sdsc_name)
        file_name = f"sdsc_{bundle_sdsc_name}.json"
        files.append(file_name)
        file_name = f"sdsc_{sdsc_name}.json"
        with open(os.path.join(output_dir, file_name), "w") as file:
            logger.info(f"Generating {file.name}")
            json.dump(sdsc_json, file, indent=2)
    for sdsc_json in sidecar_sdscs:
        sdsc_name = next(iter(sdsc_json))
        file_name = f"sdsc_{sdsc_name}.json"
        with open(os.path.join(output_dir, file_name), "w") as file:
            logger.info(f"Generating sidecar {file.name}")
            json.dump(sdsc_json, file, indent=2)

    # Generate bundle.mlir
    with open(os.path.join(output_dir, "bundle.mlir"), "w") as file:
        logger.info(f"Generating {file.name}")
        file.write("module {\n")
        file.write("\tfunc.func @sdsc_bundle() {\n")
        for f in files:
            attrs = _bundle_attrs(f, bundle_attrs_by_file.get(f, {}))
            file.write(f"\t\tsdscbundle.sdsc_execute () {{{attrs}}}\n")
        file.write("\t\treturn\n")
        file.write("\t}\n")
        file.write("}\n")


def _bundle_attrs(file_name: str, attrs: dict) -> str:
    parts = [f"sdsc_filename={json.dumps(file_name)}"]
    for key, value in attrs.items():
        if value is None:
            parts.append(key)
        elif isinstance(value, str):
            parts.append(f"{key}={json.dumps(value)}")
        elif isinstance(value, bool):
            parts.append(f"{key}={'true' if value else 'false'}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
