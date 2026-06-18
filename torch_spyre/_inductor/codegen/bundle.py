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
from torch_spyre._inductor.codegen.causal_mask_dataop import (
    build_causal_idx_to_mask_emission_plan,
    causal_score_bias_contract_from_payload,
)
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.op_spec import OpSpec
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.onchip_realize import (
    build_flash_attention_ifn_pair_tile_artifacts,
    build_flash_attention_kv_repack_broadcast_copyback_artifacts,
    build_flash_attention_kv_repack_broadcast_pair_artifacts,
    build_flash_attention_kv_repack_broadcast_plan_artifact,
    build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts,
    build_flash_attention_kv_repack_hbm_staged_hoist_tile_artifacts,
    build_flash_attention_layout_xform_hoist_tile_artifacts,
    build_flash_attention_layout_xform_lookahead_tile_artifacts,
    build_flash_attention_layout_xform_pair_tile_artifacts,
    build_flash_attention_pipeline_artifact,
    build_flash_attention_pipeline_tile_artifacts,
    build_flash_attention_value_flow_tile_artifact,
    flash_attention_ifn_pair_tile_rejection_reasons,
    flash_attention_kv_repack_broadcast_copyback_rejection_reasons,
    flash_attention_kv_repack_broadcast_pair_rejection_reasons,
    flash_attention_kv_repack_hbm_prefetch_hoist_rejection_reasons,
    flash_attention_kv_repack_hbm_staged_hoist_rejection_reasons,
    flash_attention_layout_xform_hoist_rejection_reasons,
    flash_attention_layout_xform_lookahead_rejection_reasons,
    flash_attention_layout_xform_pair_rejection_reasons,
    flash_attention_value_flow_tile_rejection_reasons,
    realize_flash_attention_pointwise_handoffs,
    realize_onchip_handoff,
    realize_reduction_reshard_bundle,
)


logger = get_inductor_logger("sdsc_compile")


def _causal_idx_to_mask_plan_artifacts(specs: list[OpSpec], sdscs_json: list[dict]):
    artifacts = []
    for idx, (op_spec, sdsc_json) in enumerate(zip(specs, sdscs_json)):
        if getattr(op_spec, "op", None) != "causal_score_bias_like":
            continue
        constants = getattr(op_spec, "op_info", {}).get("constants", {})
        key_start = constants.get("keyStart")
        if key_start is None:
            continue
        contract = causal_score_bias_contract_from_payload(sdsc_json)
        artifacts.append(
            build_causal_idx_to_mask_emission_plan(
                contract,
                key_start=key_start,
                name=f"causal_idx_to_mask_plan_{idx}",
            )
        )
    return artifacts


def _flash_attention_kv_repack_broadcast_plan_artifacts(sdscs_json: list[dict]):
    artifacts = []
    for tile_index in range(len(sdscs_json)):
        for input_idx in (1, 2):
            artifact = build_flash_attention_kv_repack_broadcast_plan_artifact(
                sdscs_json,
                tile_index,
                input_idx=input_idx,
            )
            if artifact is not None:
                artifacts.append(artifact)
    return artifacts


def _mixed_sidecar_conflicts(
    result: dict,
    replacements: dict[str, str],
    omissions: set[str],
) -> list[str]:
    result_replacements = set(result.get("replacements", {}))
    result_omissions = set(result.get("omissions", ()))
    return sorted(
        result_replacements.intersection(replacements)
        | result_replacements.intersection(omissions)
        | result_omissions.intersection(replacements)
        | result_omissions.intersection(omissions)
    )


def _max_pointwise_lx_region0(*results: dict):
    regions = [
        result["pointwise_lx_region0"]
        for result in results
        if result is not None and "pointwise_lx_region0" in result
    ]
    return max(regions) if regions else None


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
    # Core-to-core reduction reshard (the genuine non-co-assignable move): the
    # SwiGLU mul -> down_proj K-reduction edge, gathered LX -> RIU ring -> LX.
    # Detects the edge by the producer-output reduction extent and inserts a
    # standalone pure-data-op STCDPOpLx SDSC between producer and consumer.
    # Default off -> output byte-identical to before.
    if config.onchip_reduction_reshard:
        if realize_reduction_reshard_bundle(
            sdscs_json,
            m_rows=config.onchip_reduction_reshard_m_rows,
            expected_k=config.onchip_reduction_reshard_k,
            m_split=config.onchip_reduction_reshard_m_split,
            n_split=config.onchip_reduction_reshard_n_split,
            num_cores=config.sencores,
            perband=config.onchip_reduction_reshard_perband,
        ):
            logger.info("Realized core-to-core reduction reshard")
    value_flow_tile = config.flash_attention_mixed_pipeline_value_flow_tile
    ifn_pair_tile = config.flash_attention_mixed_pipeline_ifn_pair_tile
    layout_xform_pair_tile = (
        config.flash_attention_mixed_pipeline_layout_xform_pair_tile
    )
    layout_xform_pair_overlap = getattr(
        config,
        "flash_attention_mixed_pipeline_layout_xform_pair_overlap",
        False,
    )
    layout_xform_lookahead_tile = getattr(
        config,
        "flash_attention_mixed_pipeline_layout_xform_lookahead_tile",
        -1,
    )
    layout_xform_hoist_tile = getattr(
        config,
        "flash_attention_mixed_pipeline_layout_xform_hoist_tile",
        -1,
    )
    kv_repack_pair_tile = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_tile",
        -1,
    )
    kv_repack_pair_ifn_transfer = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_ifn_transfer",
        True,
    )
    kv_repack_pair_subpiece_reuse = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_subpiece_reuse",
        True,
    )
    kv_repack_pair_group_size = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_group_size",
        0,
    )
    kv_repack_pair_self_resident_source = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_self_resident_source",
        False,
    )
    kv_repack_pair_hbm_source = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_hbm_source",
        False,
    )
    kv_repack_pair_hbm_direct_load = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_hbm_direct_load",
        False,
    )
    kv_repack_pair_hbm_staged = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_hbm_staged",
        False,
    )
    kv_repack_pair_consumer_core_state_init = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_consumer_core_state_init",
        True,
    )
    kv_repack_pair_consumer_ds_type = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_consumer_ds_type",
        "",
    )
    kv_repack_pair_consumer_lx_alloc_style = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_consumer_lx_alloc_style",
        "",
    )
    kv_repack_pair_use_unicast = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_use_unicast",
        -1,
    )
    kv_repack_pair_force_mc_mode = getattr(
        config,
        "flash_attention_kv_repack_broadcast_pair_force_mc_mode",
        -1,
    )
    kv_repack_hbm_staged_hoist_tile = getattr(
        config,
        "flash_attention_kv_repack_hbm_staged_hoist_tile",
        -1,
    )
    kv_repack_hbm_prefetch_hoist_tile = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_hoist_tile",
        -1,
    )
    kv_repack_hbm_prefetch_lx_base = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_lx_base",
        -1,
    )
    kv_repack_hbm_prefetch_serial = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_serial",
        False,
    )
    kv_repack_hbm_prefetch_prefill_current = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_prefill_current",
        False,
    )
    kv_repack_hbm_prefetch_redundant_future = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_redundant_future",
        False,
    )
    kv_repack_hbm_prefetch_serialize_current = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_serialize_current",
        False,
    )
    kv_repack_hbm_prefetch_external_future = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_external_future",
        False,
    )
    kv_repack_hbm_prefetch_overlap_after_sync = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_overlap_after_sync",
        True,
    )
    kv_repack_hbm_prefetch_tail_current = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_tail_current",
        False,
    )
    kv_repack_hbm_prefetch_source_fanout = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_source_fanout",
        False,
    )
    kv_repack_hbm_prefetch_loader_fanout = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_loader_fanout",
        False,
    )
    kv_repack_hbm_prefetch_loader_core = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_loader_core",
        0,
    )
    kv_repack_hbm_prefetch_loader_lx_base = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_loader_lx_base",
        -1,
    )
    kv_repack_hbm_prefetch_fanout_use_unicast = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_fanout_use_unicast",
        -1,
    )
    kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers",
        -1,
    )
    kv_repack_hbm_prefetch_fanout_copyback_core = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_fanout_copyback_core",
        -2,
    )
    kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core",
        False,
    )
    kv_repack_hbm_prefetch_loader_copyback_without_fanout = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_loader_copyback_without_fanout",
        False,
    )
    kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces",
        False,
    )
    kv_repack_hbm_prefetch_serialize_loader_core = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_serialize_loader_core",
        False,
    )
    kv_repack_hbm_prefetch_lx_roundtrip = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_lx_roundtrip",
        False,
    )
    kv_repack_hbm_prefetch_corelet1 = getattr(
        config,
        "flash_attention_kv_repack_hbm_prefetch_corelet1",
        False,
    )
    kv_repack_copyback_tile = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_tile",
        -1,
    )
    kv_repack_copyback_core = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_core",
        -1,
    )
    kv_repack_copyback_direct_source = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_direct_source",
        False,
    )
    kv_repack_copyback_hbm_roundtrip = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip",
        False,
    )
    kv_repack_copyback_hbm_source_fanout = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_hbm_source_fanout",
        False,
    )
    kv_repack_copyback_hbm_direct_load = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_hbm_direct_load",
        False,
    )
    kv_repack_copyback_hbm_roundtrip_load_only = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_load_only",
        False,
    )
    kv_repack_copyback_hbm_roundtrip_barrier_only = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_barrier_only",
        False,
    )
    kv_repack_copyback_data_only = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_data_only",
        False,
    )
    kv_repack_copyback_replace_consumer = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_replace_consumer",
        False,
    )
    kv_repack_copyback_compute_only = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_compute_only",
        False,
    )
    kv_repack_copyback_exact_clone = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_exact_clone",
        False,
    )
    kv_repack_copyback_preserve_consumer_name = getattr(
        config,
        "flash_attention_kv_repack_broadcast_copyback_preserve_consumer_name",
        False,
    )
    sidecar_sdscs = []
    sidecar_replacements = {}
    sidecar_insertions_before = {}
    sidecar_omissions = set()
    bundle_attrs_by_file = {}
    value_flow_rejections = {}
    causal_plan_artifacts = []
    kv_repack_plan_artifacts = []
    if getattr(config, "causal_idx_to_mask_plan_artifact", False):
        causal_plan_artifacts = _causal_idx_to_mask_plan_artifacts(
            specs,
            sdscs_json,
        )
        if causal_plan_artifacts:
            logger.info(
                "Emitted %d causal IdxToMask plan artifact(s)",
                len(causal_plan_artifacts),
            )
    if getattr(config, "flash_attention_kv_repack_broadcast_plan_artifact", False):
        kv_repack_plan_artifacts = (
            _flash_attention_kv_repack_broadcast_plan_artifacts(sdscs_json)
        )
        if kv_repack_plan_artifacts:
            logger.info(
                "Emitted %d flash attention K/V repack broadcast plan artifact(s)",
                len(kv_repack_plan_artifacts),
            )
    emit_mixed_sidecars = (
        config.flash_attention_mixed_pipeline
        and (
            config.flash_attention_mixed_pipeline_artifact
            or config.flash_attention_mixed_pipeline_execute_tile >= 0
            or value_flow_tile >= 0
            or ifn_pair_tile >= 0
            or layout_xform_pair_tile != -1
            or layout_xform_lookahead_tile != -1
            or layout_xform_hoist_tile != -1
            or kv_repack_pair_tile != -1
            or kv_repack_hbm_staged_hoist_tile != -1
            or kv_repack_hbm_prefetch_hoist_tile != -1
            or kv_repack_copyback_tile != -1
        )
    )
    kv_repack_copyback = None
    kv_repack_copyback_rejections = None
    if emit_mixed_sidecars and kv_repack_copyback_tile != -1:
        kv_repack_copyback = (
            build_flash_attention_kv_repack_broadcast_copyback_artifacts(
                sdscs_json,
                kv_repack_copyback_tile,
                stcdp_subpiece_reuse=kv_repack_pair_subpiece_reuse,
                broadcast_group_size=kv_repack_pair_group_size,
                self_resident_source=kv_repack_pair_self_resident_source,
                stcdp_use_unicast=kv_repack_pair_use_unicast,
                stcdp_force_mc_mode=kv_repack_pair_force_mc_mode,
                readback_core=kv_repack_copyback_core,
                direct_source=kv_repack_copyback_direct_source,
                hbm_roundtrip=kv_repack_copyback_hbm_roundtrip,
                hbm_source_fanout=kv_repack_copyback_hbm_source_fanout,
                hbm_direct_load=kv_repack_copyback_hbm_direct_load,
                hbm_roundtrip_load_only=(
                    kv_repack_copyback_hbm_roundtrip_load_only
                ),
                hbm_roundtrip_barrier_only=(
                    kv_repack_copyback_hbm_roundtrip_barrier_only
                ),
                data_only=kv_repack_copyback_data_only,
                replace_consumer=kv_repack_copyback_replace_consumer,
                compute_only=kv_repack_copyback_compute_only,
                exact_clone=kv_repack_copyback_exact_clone,
                preserve_consumer_name=kv_repack_copyback_preserve_consumer_name,
            )
        )
        if kv_repack_copyback is None:
            kv_repack_copyback_rejections = (
                flash_attention_kv_repack_broadcast_copyback_rejection_reasons(
                    sdscs_json,
                    kv_repack_copyback_tile,
                )
            )
    kv_repack_pair = None
    kv_repack_pair_rejections = None
    if (
        emit_mixed_sidecars
        and kv_repack_pair_tile != -1
        and kv_repack_copyback is None
    ):
        kv_repack_pair = build_flash_attention_kv_repack_broadcast_pair_artifacts(
            sdscs_json,
            kv_repack_pair_tile,
            include_input_fetch_transfer=kv_repack_pair_ifn_transfer,
            stcdp_subpiece_reuse=kv_repack_pair_subpiece_reuse,
            broadcast_group_size=kv_repack_pair_group_size,
            self_resident_source=kv_repack_pair_self_resident_source,
            hbm_source=kv_repack_pair_hbm_source,
            hbm_direct_load=kv_repack_pair_hbm_direct_load,
            hbm_staged=kv_repack_pair_hbm_staged,
            consumer_core_state_init=kv_repack_pair_consumer_core_state_init,
            consumer_ds_type=kv_repack_pair_consumer_ds_type,
            consumer_lx_alloc_style=kv_repack_pair_consumer_lx_alloc_style,
            stcdp_use_unicast=kv_repack_pair_use_unicast,
            stcdp_force_mc_mode=kv_repack_pair_force_mc_mode,
        )
        if kv_repack_pair is None:
            kv_repack_pair_rejections = (
                flash_attention_kv_repack_broadcast_pair_rejection_reasons(
                    sdscs_json,
                    kv_repack_pair_tile,
                )
            )
    layout_xform_hoist = None
    layout_xform_hoist_rejections = None
    if (
        emit_mixed_sidecars
        and layout_xform_hoist_tile != -1
        and kv_repack_copyback is None
        and kv_repack_pair is None
    ):
        layout_xform_hoist = (
            build_flash_attention_layout_xform_hoist_tile_artifacts(
                sdscs_json,
                layout_xform_hoist_tile,
            )
        )
        if layout_xform_hoist is None:
            layout_xform_hoist_rejections = (
                flash_attention_layout_xform_hoist_rejection_reasons(
                    sdscs_json,
                    layout_xform_hoist_tile,
                )
            )
    layout_xform_lookahead = None
    layout_xform_lookahead_rejections = None
    if (
        emit_mixed_sidecars
        and layout_xform_lookahead_tile != -1
        and kv_repack_copyback is None
        and kv_repack_pair is None
        and layout_xform_hoist is None
    ):
        layout_xform_lookahead = (
            build_flash_attention_layout_xform_lookahead_tile_artifacts(
                sdscs_json,
                layout_xform_lookahead_tile,
            )
        )
        if layout_xform_lookahead is None:
            layout_xform_lookahead_rejections = (
                flash_attention_layout_xform_lookahead_rejection_reasons(
                    sdscs_json,
                    layout_xform_lookahead_tile,
                )
            )
    layout_xform_pair = None
    layout_xform_pair_rejections = None
    if (
        emit_mixed_sidecars
        and layout_xform_pair_tile != -1
        and kv_repack_copyback is None
        and layout_xform_hoist is None
        and layout_xform_lookahead is None
    ):
        layout_xform_pair = (
            build_flash_attention_layout_xform_pair_tile_artifacts(
                sdscs_json,
                layout_xform_pair_tile,
                name_prefix=(
                    "mixed_flash_pipeline_tile_layout_xform_pair"
                    if layout_xform_pair_overlap
                    else "mixed_flash_layout_xform_pair_tile"
                ),
                overlap_consumer=layout_xform_pair_overlap,
            )
        )
        if layout_xform_pair is None:
            layout_xform_pair_rejections = (
                flash_attention_layout_xform_pair_rejection_reasons(
                    sdscs_json,
                    layout_xform_pair_tile,
                )
            )
    kv_repack_hbm_staged_hoist = None
    kv_repack_hbm_staged_hoist_rejections = None
    kv_repack_hbm_prefetch_hoist = None
    kv_repack_hbm_prefetch_hoist_rejections = None
    if (
        config.flash_attention_mixed_pipeline
        and config.flash_attention_pointwise_handoff
    ):
        pointwise_kwargs = {
            "score_scale_handoff": config.flash_attention_score_scale_handoff,
        }
        pointwise_region0 = _max_pointwise_lx_region0(
            layout_xform_hoist,
            kv_repack_copyback,
            kv_repack_pair,
            layout_xform_pair,
            layout_xform_lookahead,
            kv_repack_hbm_prefetch_hoist,
        )
        if pointwise_region0 is not None:
            pointwise_kwargs["pointwise_region0"] = pointwise_region0
            logger.info(
                "Realizing flash pointwise handoffs in a disjoint LX region "
                "while mixed flash attention sidecars are active"
            )
        count = realize_flash_attention_pointwise_handoffs(
            sdscs_json,
            **pointwise_kwargs,
        )
        if count:
            logger.info(f"Realized {count} flash pointwise on-chip handoffs")
            if (
                kv_repack_pair_tile != -1
                and kv_repack_pair is not None
                and kv_repack_copyback is None
            ):
                refreshed_kv_repack_pair = (
                    build_flash_attention_kv_repack_broadcast_pair_artifacts(
                        sdscs_json,
                        kv_repack_pair_tile,
                        include_input_fetch_transfer=kv_repack_pair_ifn_transfer,
                        stcdp_subpiece_reuse=kv_repack_pair_subpiece_reuse,
                        broadcast_group_size=kv_repack_pair_group_size,
                        self_resident_source=kv_repack_pair_self_resident_source,
                        hbm_source=kv_repack_pair_hbm_source,
                        hbm_direct_load=kv_repack_pair_hbm_direct_load,
                        hbm_staged=kv_repack_pair_hbm_staged,
                        consumer_core_state_init=(
                            kv_repack_pair_consumer_core_state_init
                        ),
                        consumer_ds_type=kv_repack_pair_consumer_ds_type,
                        consumer_lx_alloc_style=(
                            kv_repack_pair_consumer_lx_alloc_style
                        ),
                        stcdp_use_unicast=kv_repack_pair_use_unicast,
                        stcdp_force_mc_mode=kv_repack_pair_force_mc_mode,
                    )
                )
                if refreshed_kv_repack_pair is not None:
                    kv_repack_pair = refreshed_kv_repack_pair
                    logger.info(
                        "Rebuilt K/V repack broadcast pair after flash "
                        "pointwise handoff realization"
                    )
                else:
                    kv_repack_pair = None
                    kv_repack_pair_rejections = (
                        flash_attention_kv_repack_broadcast_pair_rejection_reasons(
                            sdscs_json,
                            kv_repack_pair_tile,
                        )
                    )
    if (
        emit_mixed_sidecars
        and kv_repack_hbm_prefetch_hoist_tile != -1
        and kv_repack_copyback is None
    ):
        kv_repack_hbm_prefetch_hoist = (
            build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
                sdscs_json,
                kv_repack_hbm_prefetch_hoist_tile,
                prefetch_lx_base=(
                    kv_repack_hbm_prefetch_lx_base
                    if kv_repack_hbm_prefetch_lx_base >= 0
                    else None
                ),
                serial_prefetch=kv_repack_hbm_prefetch_serial,
                prefill_current_input=kv_repack_hbm_prefetch_prefill_current,
                redundant_future_prefetch=(
                    kv_repack_hbm_prefetch_redundant_future
                ),
                serialize_current_prefetch=(
                    kv_repack_hbm_prefetch_serialize_current
                ),
                external_future_prefetch=kv_repack_hbm_prefetch_external_future,
                overlap_after_sync=kv_repack_hbm_prefetch_overlap_after_sync,
                tail_current_prefetch=kv_repack_hbm_prefetch_tail_current,
                prefetch_source_fanout=kv_repack_hbm_prefetch_source_fanout,
                prefetch_loader_fanout=kv_repack_hbm_prefetch_loader_fanout,
                prefetch_loader_core_id=kv_repack_hbm_prefetch_loader_core,
                prefetch_loader_lx_base=kv_repack_hbm_prefetch_loader_lx_base,
                prefetch_fanout_use_unicast=(
                    kv_repack_hbm_prefetch_fanout_use_unicast
                ),
                prefetch_fanout_use_lxsfp_lx_transfers=(
                    kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers
                ),
                prefetch_fanout_copyback_core=(
                    kv_repack_hbm_prefetch_fanout_copyback_core
                ),
                prefetch_fanout_restrict_to_copyback_core=(
                    kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core
                ),
                prefetch_loader_copyback_without_fanout=(
                    kv_repack_hbm_prefetch_loader_copyback_without_fanout
                ),
                prefetch_loader_fanout_full_tile_pieces=(
                    kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces
                ),
                serialize_loader_core_prefetch=(
                    kv_repack_hbm_prefetch_serialize_loader_core
                ),
                prefetch_lx_roundtrip=kv_repack_hbm_prefetch_lx_roundtrip,
                prefetch_corelet_id=1 if kv_repack_hbm_prefetch_corelet1 else None,
            )
        )
        if kv_repack_hbm_prefetch_hoist is None:
            kv_repack_hbm_prefetch_hoist_rejections = (
                flash_attention_kv_repack_hbm_prefetch_hoist_rejection_reasons(
                    sdscs_json,
                    kv_repack_hbm_prefetch_hoist_tile,
                )
            )
    if (
        emit_mixed_sidecars
        and kv_repack_hbm_staged_hoist_tile != -1
        and kv_repack_hbm_prefetch_hoist is None
        and kv_repack_copyback is None
    ):
        kv_repack_hbm_staged_hoist = (
            build_flash_attention_kv_repack_hbm_staged_hoist_tile_artifacts(
                sdscs_json,
                kv_repack_hbm_staged_hoist_tile,
            )
        )
        if kv_repack_hbm_staged_hoist is None:
            kv_repack_hbm_staged_hoist_rejections = (
                flash_attention_kv_repack_hbm_staged_hoist_rejection_reasons(
                    sdscs_json,
                    kv_repack_hbm_staged_hoist_tile,
                )
            )
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
        if kv_repack_copyback_tile != -1:
            if kv_repack_copyback is not None:
                sidecar_sdscs.extend(kv_repack_copyback["artifacts"])
                sidecar_replacements.update(kv_repack_copyback["replacements"])
                for key, values in kv_repack_copyback.get(
                    "insertions_before", {}
                ).items():
                    sidecar_insertions_before.setdefault(key, []).extend(values)
                bundle_attrs_by_file.update(kv_repack_copyback["bundle_attrs"])
                logger.info(
                    "Executing experimental K/V repack broadcast copyback "
                    "sidecars"
                )
            else:
                logger.warning(
                    "Requested K/V repack broadcast copyback probe was not "
                    "realizable; keeping generated HBM-backed SDSCs: %s",
                    kv_repack_copyback_rejections,
                )
        if kv_repack_hbm_staged_hoist_tile != -1:
            if kv_repack_hbm_staged_hoist is not None:
                sidecar_sdscs.extend(kv_repack_hbm_staged_hoist["artifacts"])
                sidecar_replacements.update(
                    kv_repack_hbm_staged_hoist["replacements"]
                )
                for key, values in kv_repack_hbm_staged_hoist.get(
                    "insertions_before", {}
                ).items():
                    sidecar_insertions_before.setdefault(key, []).extend(values)
                sidecar_omissions.update(
                    kv_repack_hbm_staged_hoist.get("omissions", ())
                )
                bundle_attrs_by_file.update(
                    kv_repack_hbm_staged_hoist["bundle_attrs"]
                )
                logger.info(
                    "Executing experimental K/V HBM-staged hoisted-producer "
                    "sidecars"
                )
            else:
                logger.warning(
                    "Requested K/V HBM-staged hoist probe was not realizable; "
                    "keeping generated HBM-backed SDSCs: %s",
                    kv_repack_hbm_staged_hoist_rejections,
                )
        if kv_repack_hbm_prefetch_hoist_tile != -1:
            if kv_repack_hbm_prefetch_hoist is not None:
                sidecar_sdscs.extend(kv_repack_hbm_prefetch_hoist["artifacts"])
                sidecar_replacements.update(
                    kv_repack_hbm_prefetch_hoist["replacements"]
                )
                for key, values in kv_repack_hbm_prefetch_hoist.get(
                    "insertions_before", {}
                ).items():
                    sidecar_insertions_before.setdefault(key, []).extend(values)
                sidecar_omissions.update(
                    kv_repack_hbm_prefetch_hoist.get("omissions", ())
                )
                bundle_attrs_by_file.update(
                    kv_repack_hbm_prefetch_hoist["bundle_attrs"]
                )
                logger.info(
                    "Executing experimental K/V HBM prefetch hoisted-producer "
                    "sidecars"
                )
            else:
                logger.warning(
                    "Requested K/V HBM prefetch hoist probe was not realizable; "
                    "keeping generated HBM-backed SDSCs: %s",
                    kv_repack_hbm_prefetch_hoist_rejections,
                )
        if (
            kv_repack_pair_tile != -1
            and kv_repack_hbm_staged_hoist is None
            and kv_repack_hbm_prefetch_hoist is None
        ):
            if kv_repack_pair is not None:
                sidecar_sdscs.extend(kv_repack_pair["artifacts"])
                sidecar_replacements.update(kv_repack_pair["replacements"])
                bundle_attrs_by_file.update(kv_repack_pair["bundle_attrs"])
                logger.info(
                    "Executing experimental K/V repack broadcast flash attention "
                    "pair sidecars"
                )
            else:
                logger.warning(
                    "Requested K/V repack broadcast flash attention pair was "
                    "not realizable; keeping generated HBM-backed SDSCs: %s",
                    kv_repack_pair_rejections,
                )
        if (
            layout_xform_hoist_tile != -1
            and kv_repack_copyback is None
            and kv_repack_pair is None
        ):
            if layout_xform_hoist is not None:
                sidecar_sdscs.extend(layout_xform_hoist["artifacts"])
                sidecar_replacements.update(layout_xform_hoist["replacements"])
                sidecar_omissions.update(layout_xform_hoist.get("omissions", ()))
                bundle_attrs_by_file.update(layout_xform_hoist["bundle_attrs"])
                logger.info(
                    "Executing experimental layout-transform flash attention "
                    "hoisted-future sidecars"
                )
            else:
                logger.warning(
                    "Requested layout-transform hoisted-future flash attention "
                    "pair was not realizable; keeping generated HBM-backed "
                    "SDSCs: %s",
                    layout_xform_hoist_rejections,
                )
        if (
            layout_xform_lookahead_tile != -1
            and kv_repack_copyback is None
            and kv_repack_pair is None
        ):
            if layout_xform_lookahead is not None:
                sidecar_sdscs.extend(layout_xform_lookahead["artifacts"])
                sidecar_replacements.update(layout_xform_lookahead["replacements"])
                bundle_attrs_by_file.update(layout_xform_lookahead["bundle_attrs"])
                logger.info(
                    "Executing experimental layout-transform flash attention "
                    "lookahead sidecars"
                )
            else:
                logger.warning(
                    "Requested layout-transform lookahead flash attention pair "
                    "was not realizable; keeping generated HBM-backed SDSCs: %s",
                    layout_xform_lookahead_rejections,
                )
        if (
            layout_xform_pair_tile != -1
            and kv_repack_copyback is None
            and layout_xform_hoist is None
            and layout_xform_lookahead is None
        ):
            if layout_xform_pair is not None:
                conflicts = _mixed_sidecar_conflicts(
                    layout_xform_pair,
                    sidecar_replacements,
                    sidecar_omissions,
                )
                if conflicts:
                    logger.warning(
                        "Requested layout-transform flash attention pair "
                        "conflicts with earlier sidecars for SDSCs %s; keeping "
                        "those generated HBM-backed SDSCs",
                        conflicts,
                    )
                else:
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
            force_ifn_prefix = getattr(
                config,
                "flash_attention_mixed_pipeline_ifn_prefix_force",
                False,
            )
            if (
                meta.get("overlap_prefix")
                and not meta.get("ifn_runtime_safe", False)
                and not force_ifn_prefix
            ):
                logger.warning(
                    "Requested mixed flash attention overlap tile is not "
                    "runtime-safe; keeping generated HBM-backed SDSC"
                )
                continue
            if (
                meta.get("overlap_prefix")
                and not meta.get("ifn_runtime_safe", False)
                and force_ifn_prefix
            ):
                meta["ifn_runtime_forced"] = True
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

    replacement_omission_conflicts = set(sidecar_replacements).intersection(
        sidecar_omissions
    )
    if replacement_omission_conflicts:
        raise ValueError(
            "SDSC sidecar replacement/omission conflict: "
            f"{sorted(replacement_omission_conflicts)}"
        )

    # Write JSON SDSCs to file system
    files = []
    for sdsc_json in sdscs_json:
        sdsc_name = next(iter(sdsc_json))
        for inserted_name in sidecar_insertions_before.get(sdsc_name, []):
            files.append(f"sdsc_{inserted_name}.json")
        if sdsc_name not in sidecar_omissions:
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
    for artifact in causal_plan_artifacts:
        artifact_name = next(iter(artifact))
        file_name = f"{artifact_name}.json"
        with open(os.path.join(output_dir, file_name), "w") as file:
            logger.info(f"Generating causal plan {file.name}")
            json.dump(artifact, file, indent=2)
    for artifact in kv_repack_plan_artifacts:
        artifact_name = next(iter(artifact))
        file_name = f"sdsc_{artifact_name}.json"
        with open(os.path.join(output_dir, file_name), "w") as file:
            logger.info(f"Generating K/V repack plan {file.name}")
            json.dump(artifact, file, indent=2)

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
