from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_build_level_tiles_rejects_nonpositive_tile_size() -> None:
    level = LevelSpec(level=0, width=512, height=512, downsample=1.0)

    for tile_size in (0, -1):
        try:
            build_level_tiles(level=level, tile_size=tile_size, stride=1)
        except ValueError as exc:
            assert "tile_size" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_build_level_tiles_rejects_nonpositive_stride() -> None:
    level = LevelSpec(level=0, width=512, height=512, downsample=1.0)

    for stride in (0, -1):
        try:
            build_level_tiles(level=level, tile_size=256, stride=stride)
        except ValueError as exc:
            assert "stride" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_is_background_tile_uses_white_fraction_rule() -> None:
    tile = np.full((4, 4, 3), 255, dtype=np.uint8)
    tile[0, 0] = [10, 10, 10]

    assert is_background_tile(tile, rgb_threshold=200, background_fraction=0.95) is False
    assert is_background_tile(tile, rgb_threshold=200, background_fraction=0.93) is True


def test_pad_tile_to_size_uses_white_pixels() -> None:
    from wsi_inference import pad_tile_to_size

    tile = np.zeros((2, 3, 3), dtype=np.uint8)

    padded = pad_tile_to_size(tile, tile_size=5)

    assert padded.shape == (5, 5, 3)
    assert np.all(padded[:2, :3] == 0)
    assert np.all(padded[2:, :, :] == 255)
    assert np.all(padded[:, 3:, :] == 255)


def test_stitch_tile_prediction_crops_back_to_valid_region() -> None:
    from wsi_inference import stitch_tile_prediction

    canvas = np.zeros((3, 6, 6), dtype=np.float32)
    prediction = np.ones((3, 4, 4), dtype=np.float32)
    tile = TileSpec(level=0, x=4, y=4, read_width=2, read_height=2)

    stitch_tile_prediction(canvas, tile, prediction)

    assert np.all(canvas[:, 4:6, 4:6] == 1.0)
    assert float(canvas[:, :4, :4].sum()) == 0.0


def test_quantize_global_scales_all_series_to_uint16() -> None:
    from wsi_inference import quantize_global

    arrays = [
        np.array([[[0.0, 1.0]]], dtype=np.float32),
        np.array([[[2.0, 3.0]]], dtype=np.float32),
    ]

    scaled = quantize_global(arrays)

    assert scaled[0].dtype == np.uint16
    assert scaled[1].dtype == np.uint16
    assert scaled[0][0, 0, 0] == 0
    assert scaled[1][0, 0, 1] == 65535


def test_quantize_tile_scales_each_tile_before_stitching() -> None:
    from wsi_inference import quantize_tile_prediction

    tile = np.array([[[1.0, 3.0], [5.0, 7.0]]], dtype=np.float32)

    scaled = quantize_tile_prediction(tile)

    assert scaled.dtype == np.uint16
    assert scaled.min() == 0
    assert scaled.max() == 65535


def test_write_ome_tiff_writes_one_series_per_level(tmp_path: Path) -> None:
    import tifffile

    from wsi_inference import write_ome_tiff

    path = tmp_path / "predictions.ome.tiff"
    write_ome_tiff(
        str(path),
        [
            np.zeros((3, 4, 5), dtype=np.uint16),
            np.ones((3, 2, 3), dtype=np.uint16),
        ],
    )

    with tifffile.TiffFile(path) as tif:
        assert len(tif.series) == 2
        assert tif.series[0].shape == (3, 4, 5)
        assert tif.series[1].shape == (3, 2, 3)
