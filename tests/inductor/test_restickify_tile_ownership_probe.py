from tools.restickify_tile_ownership_probe import (
    no_tile_exchange_but_local_transpose,
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
