from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import pytest

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


def test_read_level_tile_uses_level_zero_coordinates_for_downsampled_levels() -> None:
    from wsi_inference import read_level_tile

    class FakeSlide:
        level_downsamples = [1.0, 4.0]

        def __init__(self) -> None:
            self.read_calls = []

        def read_region(
            self,
            location: tuple[int, int],
            level: int,
            size: tuple[int, int],
        ) -> np.ndarray:
            self.read_calls.append((location, level, size))
            width, height = size
            return np.full((height, width, 4), 255, dtype=np.uint8)

    slide = FakeSlide()
    tile = TileSpec(level=1, x=112, y=56, read_width=32, read_height=16)

    output = read_level_tile(slide, tile, tile_size=32)

    assert slide.read_calls == [((448, 224), 1, (32, 16))]
    assert output.shape == (32, 32, 3)
    assert output.dtype == np.uint8


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
        assert tif.series[0].axes == "CYX"
        assert tif.series[1].axes == "CYX"
        assert tif.series[0].shape == (3, 4, 5)
        assert tif.series[1].shape == (3, 2, 3)


def test_prepare_tile_tensor_matches_predict_py_normalization() -> None:
    from wsi_inference import prepare_tile_tensor

    tile = np.full((2, 2, 3), 255, dtype=np.uint8)

    tensor = prepare_tile_tensor(tile)

    assert tuple(tensor.shape) == (3, 2, 2)
    assert float(tensor.max()) == 1.0
    assert float(tensor.min()) == 1.0


def test_run_tile_batch_returns_same_spatial_shape() -> None:
    import torch

    from wsi_inference import run_tile_batch

    class FakeModel(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (batch.shape[0], 3, batch.shape[2], batch.shape[3]),
                dtype=batch.dtype,
            )

    tiles = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),
    ]

    outputs = run_tile_batch(FakeModel(), tiles, device="cpu")

    assert len(outputs) == 2
    assert outputs[0].shape == (3, 4, 4)


def test_run_tile_batch_restores_training_state() -> None:
    import torch

    from wsi_inference import run_tile_batch

    class TrackingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_training_states = []

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            self.forward_training_states.append(self.training)
            return torch.zeros(
                (batch.shape[0], 3, batch.shape[2], batch.shape[3]),
                dtype=batch.dtype,
            )

    model = TrackingModel()
    model.train()

    outputs = run_tile_batch(model, [np.zeros((4, 4, 3), dtype=np.uint8)], device="cpu")

    assert len(outputs) == 1
    assert model.forward_training_states == [False]
    assert model.training is True


def test_run_level_inference_processes_each_level_independently() -> None:
    import torch

    from wsi_inference import run_level_inference

    class FakeSlide:
        level_downsamples = [1.0, 4.0]

        def __init__(self) -> None:
            self.level_calls = []
            self.location_calls = []

        def read_region(
            self,
            location: tuple[int, int],
            level: int,
            size: tuple[int, int],
        ) -> np.ndarray:
            self.location_calls.append(location)
            self.level_calls.append(level)
            width, height = size
            value = 20 if level == 0 else 40
            rgba = np.full((height, width, 4), 255, dtype=np.uint8)
            rgba[:, :, :3] = value
            return rgba

    class FakeModel(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return torch.ones(
                (batch.shape[0], 3, batch.shape[2], batch.shape[3]),
                dtype=batch.dtype,
            )

    level0 = LevelSpec(level=0, width=4, height=4, downsample=1.0)
    level1 = LevelSpec(level=1, width=2, height=2, downsample=4.0)
    slide = FakeSlide()

    out0 = run_level_inference(
        slide,
        FakeModel(),
        level0,
        tile_size=4,
        stride=4,
        device="cpu",
    )
    out1 = run_level_inference(
        slide,
        FakeModel(),
        level1,
        tile_size=4,
        stride=4,
        device="cpu",
    )

    assert out0.shape == (3, 4, 4)
    assert out1.shape == (3, 2, 2)
    assert slide.level_calls == [0, 1]
    assert slide.location_calls == [(0, 0), (0, 0)]


def test_run_level_inference_uses_bounded_mini_batches() -> None:
    import torch

    from wsi_inference import run_level_inference

    class FakeSlide:
        def read_region(
            self,
            location: tuple[int, int],
            level: int,
            size: tuple[int, int],
        ) -> np.ndarray:
            del location, level
            width, height = size
            rgba = np.full((height, width, 4), 255, dtype=np.uint8)
            rgba[:, :, :3] = 20
            return rgba

    class TrackingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes = []

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(int(batch.shape[0]))
            return torch.ones(
                (batch.shape[0], 3, batch.shape[2], batch.shape[3]),
                dtype=batch.dtype,
            )

    model = TrackingModel()
    level = LevelSpec(level=0, width=16, height=16, downsample=1.0)

    output = run_level_inference(
        FakeSlide(),
        model,
        level,
        tile_size=4,
        stride=4,
        device="cpu",
    )

    assert output.shape == (3, 16, 16)
    assert model.batch_sizes == [8, 8]


def test_run_level_inference_leaves_background_tiles_as_zero() -> None:
    import torch

    from wsi_inference import run_level_inference

    class FakeSlide:
        def read_region(
            self,
            location: tuple[int, int],
            level: int,
            size: tuple[int, int],
        ) -> np.ndarray:
            del location, level
            width, height = size
            return np.full((height, width, 4), 255, dtype=np.uint8)

    class FakeModel(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            del batch
            raise AssertionError("background tiles should not be inferred")

    level = LevelSpec(level=0, width=4, height=4, downsample=1.0)

    output = run_level_inference(
        FakeSlide(),
        FakeModel(),
        level,
        tile_size=4,
        stride=4,
        device="cpu",
    )

    assert float(output.sum()) == 0.0


def test_run_level_inference_for_ome_tile_quantization_scales_per_tile() -> None:
    import torch

    from wsi_inference import run_level_inference_for_ome

    class FakeSlide:
        def read_region(
            self,
            location: tuple[int, int],
            level: int,
            size: tuple[int, int],
        ) -> np.ndarray:
            del level
            width, height = size
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[:, :, 3] = 255
            rgba[:, :, :3] = 10 if location[0] == 0 else 20
            return rgba

    class FakeModel(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            del batch
            return torch.tensor(
                [
                    [[[1.0, 2.0], [3.0, 4.0]]],
                    [[[10.0, 20.0], [30.0, 40.0]]],
                ],
                dtype=torch.float32,
            )

    level = LevelSpec(level=0, width=4, height=2, downsample=1.0)

    output = run_level_inference_for_ome(
        slide=FakeSlide(),
        model=FakeModel(),
        level=level,
        tile_size=2,
        stride=2,
        device="cpu",
        out_channels=1,
        ome_quant_mode="tile",
    )

    expected_tile = np.array(
        [[[0, 21845], [43690, 65535]]],
        dtype=np.uint16,
    )
    assert output.dtype == np.uint16
    assert np.array_equal(output[:, :, 0:2], expected_tile)
    assert np.array_equal(output[:, :, 2:4], expected_tile)


def test_load_generator_uses_repo_defaults_and_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from wsi_inference import load_generator

    generator_calls = []
    torch_load_calls = []

    class FakeModel:
        def __init__(self) -> None:
            self.loaded = []
            self.eval_called = False
            self.training = True
            self.device = None

        def to(self, device: str) -> "FakeModel":
            self.device = device
            return self

        def load_state_dict(self, state_dict: dict, strict: bool = False) -> None:
            self.loaded.append((state_dict, strict))

        def eval(self) -> "FakeModel":
            self.eval_called = True
            self.training = False
            return self

    fake_model = FakeModel()

    def fake_generator(**kwargs: Any) -> FakeModel:
        generator_calls.append(kwargs)
        return fake_model

    def fake_torch_load(path: str, map_location: str) -> Dict[str, int]:
        torch_load_calls.append((path, map_location))
        return {"weights": 1}

    monkeypatch.setattr("wsi_inference.Attention_GAN.Generator", fake_generator)
    monkeypatch.setattr("wsi_inference.torch.load", fake_torch_load)

    model = load_generator("checkpoint.pt", out_channels=5, device="cpu")

    assert model is fake_model
    assert generator_calls == [
        {
            "n_channels": 64,
            "in_channels": 3,
            "batch_norm": False,
            "out_channels": 5,
            "padding": 1,
            "pooling_mode": "maxpool",
        }
    ]
    assert torch_load_calls == [("checkpoint.pt", "cpu")]
    assert fake_model.loaded == [({"weights": 1}, True)]
    assert fake_model.eval_called is True
    assert fake_model.training is False
