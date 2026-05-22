from tools.restickify_tile_ownership_probe import (
    default_core_mapping,
    no_tile_exchange_but_local_transpose,
    plan_streaming_ptlx_tiles,
    ring_distance,
    summarize,
    tile_exchange_but_no_local_transpose,
)


def test_ring_distance_uses_bidirectional_ring():
    assert ring_distance(0, 31, 32) == 1
    assert ring_distance(0, 16, 32) == 16
    assert ring_distance(7, 3, 32) == ring_distance(3, 7, 32)


def test_tile_summary_sees_no_movement_when_split_dimensions_match():
    summary = summarize(
        size=2048,
        tile_size=64,
        num_cores=32,
        source_split_dim="col",
        dest_split_dim="col",
    )

    assert summary.local_tiles == 1024
    assert summary.moving_tiles == 0
    assert summary.total_tile_hops == 0


def test_tile_summary_sees_all_to_all_when_split_dimensions_differ():
    summary = summarize(
        size=2048,
        tile_size=64,
        num_cores=32,
        source_split_dim="row",
        dest_split_dim="col",
    )

    assert summary.local_tiles == 32
    assert summary.moving_tiles == 992
    assert summary.max_tile_hops == 16
    assert summary.hop_histogram[0] == 32


def test_deterministic_fingerprints_distinguish_tile_exchange_from_local_transpose():
    tile = 64

    assert no_tile_exchange_but_local_transpose(0, 64, tile) == 0
    assert no_tile_exchange_but_local_transpose(64, 0, tile) == 64
    assert tile_exchange_but_no_local_transpose(0, 64, tile) == 64
    assert tile_exchange_but_no_local_transpose(127, 0, tile) == 63


def test_streaming_ptlx_plan_explains_small_shape_gather():
    source = {"mb": 32, "out": 1}
    dest = {"mb": 4, "out": 8}

    summary = plan_streaming_ptlx_tiles(
        size=512,
        source_work_slices=source,
        source_core_mapping=default_core_mapping(source),
        dest_work_slices=dest,
        dest_core_mapping=default_core_mapping(dest),
    )

    assert summary.total_tiles == 64
    assert summary.max_fan_in == 4
    assert summary.max_fan_out == 1
    assert summary.gather_tiles == 64
    assert "source-piece-smaller-than-tile" in summary.notes
    assert summary.tile_buffer_bytes == 64 * 64 * 2


def test_streaming_ptlx_plan_handles_one_to_one_2048_tiles():
    source = {"mb": 32, "out": 1}
    dest = {"mb": 1, "out": 32}

    summary = plan_streaming_ptlx_tiles(
        size=2048,
        source_work_slices=source,
        source_core_mapping=default_core_mapping(source),
        dest_work_slices=dest,
        dest_core_mapping=default_core_mapping(dest),
    )

    assert summary.total_tiles == 1024
    assert summary.local_tiles == 32
    assert summary.moving_tiles == 992
    assert summary.max_fan_in == 1
    assert summary.max_fan_out == 1
    assert summary.max_tile_hops == 16
    assert summary.tile_buffer_bytes == 8192


def test_streaming_ptlx_plan_models_core_count_mismatch():
    source = {"mb": 32, "out": 1}
    dest = {"mb": 24, "out": 1}

    summary = plan_streaming_ptlx_tiles(
        size=3072,
        source_work_slices=source,
        source_core_mapping=default_core_mapping(source),
        dest_work_slices=dest,
        dest_core_mapping=default_core_mapping(dest),
    )

    assert summary.source_core_count == 32
    assert summary.dest_core_count == 24
    assert "source-dest-core-count-mismatch" in summary.notes
    assert summary.total_tiles == 48 * 48


def test_streaming_ptlx_plan_reduces_4096_workspace_to_tile_buffer():
    source = {"mb": 32, "out": 1}
    dest = {"mb": 1, "out": 32}

    summary = plan_streaming_ptlx_tiles(
        size=4096,
        source_work_slices=source,
        source_core_mapping=default_core_mapping(source),
        dest_work_slices=dest,
        dest_core_mapping=default_core_mapping(dest),
    )

    assert summary.full_tensor_bytes_per_source_core == 1024 * 1024
    assert summary.full_tensor_bytes_per_dest_core == 1024 * 1024
    assert summary.tile_buffer_bytes == 8192
    assert summary.tile_buffer_bytes < summary.full_tensor_bytes_per_source_core
