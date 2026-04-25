import numpy as np

from wsi_inference import LevelSpec, TileSpec, build_level_tiles, is_background_tile


def test_build_level_tiles_covers_right_and_bottom_edges() -> None:
    level = LevelSpec(level=0, width=1000, height=900, downsample=1.0)

    tiles = build_level_tiles(level=level, tile_size=512, stride=512)

    assert tiles == [
        TileSpec(level=0, x=0, y=0, read_width=512, read_height=512),
        TileSpec(level=0, x=512, y=0, read_width=488, read_height=512),
        TileSpec(level=0, x=0, y=512, read_width=512, read_height=388),
        TileSpec(level=0, x=512, y=512, read_width=488, read_height=388),
    ]


def test_build_level_tiles_rejects_stride_larger_than_tile() -> None:
    level = LevelSpec(level=0, width=512, height=512, downsample=1.0)

    try:
        build_level_tiles(level=level, tile_size=256, stride=512)
    except ValueError as exc:
        assert "stride" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_is_background_tile_uses_white_fraction_rule() -> None:
    tile = np.full((4, 4, 3), 255, dtype=np.uint8)
    tile[0, 0] = [10, 10, 10]

    assert is_background_tile(tile, rgb_threshold=200, background_fraction=0.95) is False
    assert is_background_tile(tile, rgb_threshold=200, background_fraction=0.93) is True
