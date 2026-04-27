# WSI Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `.svs` whole-slide inference pipeline that tiles selected WSI levels, skips background tiles, stitches raw float predictions into per-level `.npy` files, and writes all selected levels into one OME-TIFF as separate image series.

**Architecture:** Add a new top-level CLI script `infer_wsi.py` and a focused helper module `wsi_inference.py`. Keep the new pipeline isolated from the existing `predict.py` path, importing only the existing generator definition from `Attention_GAN.py` and duplicating small preprocessing or checkpoint-loading logic when that lowers regression risk.

**Tech Stack:** Python 3.9, PyTorch, OpenSlide (`openslide-python`), `tifffile`, NumPy, `pytest`

---

## File Structure

- Create: `infer_wsi.py`
  - CLI entry point, argument parsing, orchestration, logging, output naming
- Create: `wsi_inference.py`
  - slide metadata access, tile grid generation, white padding, background rule, batching, stitching, quantization, OME writing
- Create: `tests/test_wsi_inference_core.py`
  - unit tests for grid, padding, background filtering, stitching, quantization, and OME writer behavior
- Create: `tests/test_infer_wsi_cli.py`
  - CLI and end-to-end orchestration tests using a fake slide reader and fake model
- Modify: `requirements.txt`
  - add runtime dependencies for OpenSlide and OME-TIFF export
- Modify: `README.md`
  - add WSI inference install notes and example usage

## Task 1: Add Dependencies And Core Geometry Helpers

**Files:**
- Modify: `requirements.txt`
- Create: `wsi_inference.py`
- Test: `tests/test_wsi_inference_core.py`

- [ ] **Step 1: Write the failing tests for level tiling and background detection**

```python
# tests/test_wsi_inference_core.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_wsi_inference_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wsi_inference'`

- [ ] **Step 3: Add the new dependencies**

```text
# requirements.txt
numpy==1.24.4
opencv_python==4.8.0.76
opencv_python_headless==4.10.0.84
openslide-python
Pillow>=9.2.0
scikit-image
tifffile
timm==0.9.12
torch==2.1.2
torchvision==0.16.2
tqdm==4.63.0
```

- [ ] **Step 4: Write the minimal geometry and background helpers**

```python
# wsi_inference.py
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class LevelSpec:
    level: int
    width: int
    height: int
    downsample: float


@dataclass(frozen=True)
class TileSpec:
    level: int
    x: int
    y: int
    read_width: int
    read_height: int


def build_level_tiles(level: LevelSpec, tile_size: int, stride: int) -> List[TileSpec]:
    if stride > tile_size:
        raise ValueError("stride must be less than or equal to tile_size")

    tiles: List[TileSpec] = []
    y = 0
    while y < level.height:
        x = 0
        while x < level.width:
            read_width = min(tile_size, level.width - x)
            read_height = min(tile_size, level.height - y)
            tiles.append(
                TileSpec(
                    level=level.level,
                    x=x,
                    y=y,
                    read_width=read_width,
                    read_height=read_height,
                )
            )
            x += stride
        y += stride
    return tiles


def is_background_tile(
    tile_rgb: np.ndarray,
    rgb_threshold: int = 200,
    background_fraction: float = 0.995,
) -> bool:
    white_mask = np.all(tile_rgb > rgb_threshold, axis=-1)
    return float(white_mask.mean()) >= background_fraction
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_wsi_inference_core.py -v`
Expected: PASS for the three tests above

- [ ] **Step 6: Commit**

```bash
git add requirements.txt wsi_inference.py tests/test_wsi_inference_core.py
git commit -m "feat: add WSI tiling and background helpers"
```

## Task 2: Add White Padding, Edge Cropping, And Raw Stitching

**Files:**
- Modify: `wsi_inference.py`
- Modify: `tests/test_wsi_inference_core.py`

- [ ] **Step 1: Write the failing tests for white padding and stitched placement**

```python
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
    pred = np.ones((3, 4, 4), dtype=np.float32)
    tile = TileSpec(level=0, x=4, y=4, read_width=2, read_height=2)

    stitch_tile_prediction(canvas, tile, pred)

    assert np.all(canvas[:, 4:6, 4:6] == 1.0)
    assert float(canvas[:, :4, :4].sum()) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_wsi_inference_core.py::test_pad_tile_to_size_uses_white_pixels tests/test_wsi_inference_core.py::test_stitch_tile_prediction_crops_back_to_valid_region -v`
Expected: FAIL with `ImportError` or `AttributeError` for missing helpers

- [ ] **Step 3: Write the minimal padding and stitching code**

```python
def pad_tile_to_size(tile_rgb: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = tile_rgb.shape[:2]
    padded = np.full((tile_size, tile_size, 3), 255, dtype=tile_rgb.dtype)
    padded[:height, :width, :] = tile_rgb
    return padded


def stitch_tile_prediction(
    canvas: np.ndarray,
    tile: TileSpec,
    prediction: np.ndarray,
) -> None:
    valid_prediction = prediction[:, : tile.read_height, : tile.read_width]
    canvas[
        :,
        tile.y : tile.y + tile.read_height,
        tile.x : tile.x + tile.read_width,
    ] = valid_prediction
```

- [ ] **Step 4: Add a stitched tensor allocator helper**

```python
def allocate_level_canvas(out_channels: int, level: LevelSpec) -> np.ndarray:
    return np.zeros((out_channels, level.height, level.width), dtype=np.float32)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_wsi_inference_core.py -v`
Expected: PASS with the new padding and stitching tests included

- [ ] **Step 6: Commit**

```bash
git add wsi_inference.py tests/test_wsi_inference_core.py
git commit -m "feat: add WSI padding and raw stitching"
```

## Task 3: Add Quantization And OME Export Helpers

**Files:**
- Modify: `wsi_inference.py`
- Modify: `tests/test_wsi_inference_core.py`

- [ ] **Step 1: Write the failing tests for quantization modes**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_wsi_inference_core.py::test_quantize_global_scales_all_series_to_uint16 tests/test_wsi_inference_core.py::test_quantize_tile_scales_each_tile_before_stitching -v`
Expected: FAIL because quantization helpers do not exist yet

- [ ] **Step 3: Implement the quantization helpers**

```python
from typing import List, Sequence


def _scale_to_uint16(array: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint16)
    scaled = (array - min_value) / (max_value - min_value)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 65535.0).astype(np.uint16)


def quantize_global(level_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    non_empty = [arr for arr in level_arrays if arr.size > 0]
    if not non_empty:
        raise ValueError("global quantization requires at least one array")
    global_min = min(float(arr.min()) for arr in non_empty)
    global_max = max(float(arr.max()) for arr in non_empty)
    return [_scale_to_uint16(arr, global_min, global_max) for arr in level_arrays]


def quantize_tile_prediction(tile_prediction: np.ndarray) -> np.ndarray:
    return _scale_to_uint16(
        tile_prediction,
        float(tile_prediction.min()),
        float(tile_prediction.max()),
    )
```

- [ ] **Step 4: Add a simple OME writer wrapper**

```python
import tifffile


def write_ome_tiff(output_path: str, series_arrays: Sequence[np.ndarray]) -> None:
    with tifffile.TiffWriter(output_path, ome=True) as tif:
        for series in series_arrays:
            tif.write(series, metadata={"axes": "CYX"})
```

- [ ] **Step 5: Add a writer smoke test**

```python
def test_write_ome_tiff_writes_one_series_per_level(tmp_path) -> None:
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_wsi_inference_core.py -v`
Expected: PASS including quantization and OME writer coverage

- [ ] **Step 7: Commit**

```bash
git add wsi_inference.py tests/test_wsi_inference_core.py
git commit -m "feat: add quantization and OME export helpers"
```

## Task 4: Add Generator Loading And Tile Batch Inference

**Files:**
- Modify: `wsi_inference.py`
- Modify: `tests/test_wsi_inference_core.py`

- [ ] **Step 1: Write the failing tests for tensor normalization and output shape**

```python
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
            return torch.zeros((batch.shape[0], 3, batch.shape[2], batch.shape[3]), dtype=batch.dtype)

    tiles = [np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8)]
    outputs = run_tile_batch(FakeModel(), tiles, device="cpu")

    assert len(outputs) == 2
    assert outputs[0].shape == (3, 4, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_wsi_inference_core.py::test_prepare_tile_tensor_matches_predict_py_normalization tests/test_wsi_inference_core.py::test_run_tile_batch_returns_same_spatial_shape -v`
Expected: FAIL because tile inference helpers do not exist

- [ ] **Step 3: Implement normalization and model execution**

```python
import torch


def prepare_tile_tensor(tile_rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(tile_rgb.transpose(2, 0, 1)).float() / 255.0
    return (tensor - 0.5) * 2.0


from typing import List


def run_tile_batch(model: torch.nn.Module, tiles: List[np.ndarray], device: str) -> List[np.ndarray]:
    batch = torch.stack([prepare_tile_tensor(tile) for tile in tiles], dim=0).to(device)
    with torch.no_grad():
        outputs = model(batch).detach().cpu().numpy().astype(np.float32)
    return [outputs[i] for i in range(outputs.shape[0])]
```

- [ ] **Step 4: Add generator loading with current repo defaults**

```python
import Attention_GAN


def load_generator(checkpoint_path: str, out_channels: int, device: str) -> torch.nn.Module:
    model = Attention_GAN.Generator(
        n_channels=64,
        in_channels=3,
        batch_norm=False,
        out_channels=out_channels,
        padding=1,
        pooling_mode="maxpool",
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_wsi_inference_core.py -v`
Expected: PASS with inference-helper tests added

- [ ] **Step 6: Commit**

```bash
git add wsi_inference.py tests/test_wsi_inference_core.py
git commit -m "feat: add WSI model loading and batch inference"
```

## Task 5: Add Slide Reader And Per-Level Pipeline Execution

**Files:**
- Modify: `wsi_inference.py`
- Modify: `tests/test_wsi_inference_core.py`

- [ ] **Step 1: Write the failing tests for per-level independence and zero-filled skipped tiles**

```python
def test_run_level_inference_processes_each_level_independently() -> None:
    import numpy as np
    import torch
    from wsi_inference import LevelSpec, run_level_inference

    class FakeSlide:
        def read_region(self, location, level, size):
            width, height = size
            value = 20 if level == 0 else 40
            rgba = np.full((height, width, 4), 255, dtype=np.uint8)
            rgba[:, :, :3] = value
            return rgba

    class FakeModel(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return torch.ones((batch.shape[0], 3, batch.shape[2], batch.shape[3]), dtype=batch.dtype)

    level0 = LevelSpec(level=0, width=4, height=4, downsample=1.0)
    level1 = LevelSpec(level=1, width=2, height=2, downsample=4.0)

    out0 = run_level_inference(FakeSlide(), FakeModel(), level0, tile_size=4, stride=4, device="cpu")
    out1 = run_level_inference(FakeSlide(), FakeModel(), level1, tile_size=4, stride=4, device="cpu")

    assert out0.shape == (3, 4, 4)
    assert out1.shape == (3, 2, 2)


def test_run_level_inference_leaves_background_tiles_as_zero() -> None:
    import numpy as np
    import torch
    from wsi_inference import LevelSpec, run_level_inference

    class FakeSlide:
        def read_region(self, location, level, size):
            width, height = size
            rgba = np.full((height, width, 4), 255, dtype=np.uint8)
            return rgba

    class FakeModel(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            raise AssertionError("background tiles should not be inferred")

    level = LevelSpec(level=0, width=4, height=4, downsample=1.0)
    output = run_level_inference(FakeSlide(), FakeModel(), level, tile_size=4, stride=4, device="cpu")

    assert float(output.sum()) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_wsi_inference_core.py::test_run_level_inference_processes_each_level_independently tests/test_wsi_inference_core.py::test_run_level_inference_leaves_background_tiles_as_zero -v`
Expected: FAIL because the level runner does not exist

- [ ] **Step 3: Add slide metadata and tile reading helpers**

```python
from openslide import OpenSlide


def open_slide(slide_path: str) -> OpenSlide:
    if not slide_path.lower().endswith(".svs"):
        raise ValueError("slide path must point to a .svs file")
    return OpenSlide(slide_path)


from typing import List


def get_level_specs(slide: OpenSlide) -> List[LevelSpec]:
    specs: List[LevelSpec] = []
    for level in range(slide.level_count):
        width, height = slide.level_dimensions[level]
        specs.append(
            LevelSpec(
                level=level,
                width=width,
                height=height,
                downsample=float(slide.level_downsamples[level]),
            )
        )
    return specs


def read_level_tile(slide, tile: TileSpec, tile_size: int) -> np.ndarray:
    rgba = np.asarray(
        slide.read_region((tile.x, tile.y), tile.level, (tile.read_width, tile.read_height))
    )
    rgb = rgba[:, :, :3].astype(np.uint8)
    return pad_tile_to_size(rgb, tile_size=tile_size)
```

- [ ] **Step 4: Add the per-level runner**

```python
def run_level_inference(
    slide,
    model: torch.nn.Module,
    level: LevelSpec,
    tile_size: int,
    stride: int,
    device: str,
    out_channels: int = 3,
    rgb_threshold: int = 200,
    background_fraction: float = 0.995,
) -> np.ndarray:
    canvas = allocate_level_canvas(out_channels=out_channels, level=level)
    tiles = build_level_tiles(level=level, tile_size=tile_size, stride=stride)

    batch_specs: List[TileSpec] = []
    batch_tiles: List[np.ndarray] = []
    for tile in tiles:
        padded_tile = read_level_tile(slide, tile, tile_size=tile_size)
        valid_tile = padded_tile[: tile.read_height, : tile.read_width, :]
        if is_background_tile(valid_tile, rgb_threshold=rgb_threshold, background_fraction=background_fraction):
            continue
        batch_specs.append(tile)
        batch_tiles.append(padded_tile)

    if batch_tiles:
        outputs = run_tile_batch(model, batch_tiles, device=device)
        for tile, prediction in zip(batch_specs, outputs):
            stitch_tile_prediction(canvas, tile, prediction)
    return canvas
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_wsi_inference_core.py -v`
Expected: PASS with per-level runner coverage

- [ ] **Step 6: Commit**

```bash
git add wsi_inference.py tests/test_wsi_inference_core.py
git commit -m "feat: add per-level WSI inference runner"
```

## Task 6: Add The Standalone CLI And Output Naming

**Files:**
- Create: `infer_wsi.py`
- Create: `tests/test_infer_wsi_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
# tests/test_infer_wsi_cli.py
import numpy as np

from infer_wsi import build_parser, main, resolve_stride


def test_build_parser_has_expected_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["--slide", "sample.svs", "--output-dir", "out"])

    assert args.levels == [0]
    assert args.tile_size == 512
    assert resolve_stride(args.tile_size, args.stride) == 512
    assert args.batch_size == 4
    assert args.out_channels == 3
    assert args.device == "cuda:0"
    assert args.ome_quant_mode == "tile"
    assert args.checkpoint == "weights/hemit_weight.pth"


def test_main_saves_npy_per_level_and_one_ome_tiff(tmp_path, monkeypatch) -> None:
    saved = {"npy": [], "ome": None}

    monkeypatch.setattr("infer_wsi.open_slide", lambda path: object())
    monkeypatch.setattr(
        "infer_wsi.get_level_specs",
        lambda slide: [
            type("L", (), {"level": 0, "width": 4, "height": 4, "downsample": 1.0})(),
            type("L", (), {"level": 1, "width": 2, "height": 2, "downsample": 4.0})(),
        ],
    )
    monkeypatch.setattr("infer_wsi.load_generator", lambda checkpoint_path, out_channels, device: object())
    monkeypatch.setattr(
        "infer_wsi.run_level_inference",
        lambda **kwargs: np.zeros((3, kwargs["level"].height, kwargs["level"].width), dtype=np.float32),
    )
    monkeypatch.setattr("numpy.save", lambda path, array: saved["npy"].append((str(path), array.shape)))
    monkeypatch.setattr("infer_wsi.write_ome_tiff", lambda path, arrays: saved.__setitem__("ome", (str(path), len(arrays))))

    exit_code = main(
        [
            "--slide", "sample.svs",
            "--output-dir", str(tmp_path),
            "--levels", "0", "1",
        ]
    )

    assert exit_code == 0
    assert len(saved["npy"]) == 2
    assert saved["ome"][1] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_infer_wsi_cli.py -v`
Expected: FAIL with `ModuleNotFoundError` for `infer_wsi`

- [ ] **Step 3: Implement the CLI parser**

```python
# infer_wsi.py
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from wsi_inference import (
    get_level_specs,
    load_generator,
    open_slide,
    quantize_global,
    run_level_inference,
    write_ome_tiff,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WSI inference on selected SVS pyramid levels.")
    parser.add_argument("--slide", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="weights/hemit_weight.pth")
    parser.add_argument("--levels", nargs="+", type=int, default=[0])
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ome-quant-mode", choices=["global", "tile", "none"], default="tile")
    parser.add_argument("--background-threshold-rgb", type=int, default=200)
    parser.add_argument("--background-fraction", type=float, default=0.995)
    return parser


def resolve_stride(tile_size: int, stride: Optional[int]) -> int:
    return tile_size if stride is None else stride
```

- [ ] **Step 4: Implement the orchestration**

```python
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stride = resolve_stride(args.tile_size, args.stride)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide = open_slide(args.slide)
    level_specs = {spec.level: spec for spec in get_level_specs(slide)}
    missing = [level for level in args.levels if level not in level_specs]
    if missing:
        raise ValueError(f"Requested levels not found: {missing}")

    model = load_generator(args.checkpoint, out_channels=args.out_channels, device=args.device)

    raw_level_arrays: List[np.ndarray] = []
    for level_index in args.levels:
        level = level_specs[level_index]
        prediction = run_level_inference(
            slide=slide,
            model=model,
            level=level,
            tile_size=args.tile_size,
            stride=stride,
            device=args.device,
            out_channels=args.out_channels,
            rgb_threshold=args.background_threshold_rgb,
            background_fraction=args.background_fraction,
        )
        raw_level_arrays.append(prediction)
        np.save(output_dir / f"{Path(args.slide).stem}.level-{level_index}.npy", prediction)

    ome_arrays = raw_level_arrays if args.ome_quant_mode == "none" else quantize_global(raw_level_arrays)
    write_ome_tiff(output_dir / f"{Path(args.slide).stem}.predictions.ome.tiff", ome_arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_infer_wsi_cli.py -v`
Expected: PASS for parser defaults and CLI output behavior

- [ ] **Step 6: Commit**

```bash
git add infer_wsi.py tests/test_infer_wsi_cli.py
git commit -m "feat: add standalone WSI inference CLI"
```

## Task 7: Finish `tile` Quantization Stitching Behavior

**Files:**
- Modify: `wsi_inference.py`
- Modify: `infer_wsi.py`
- Modify: `tests/test_wsi_inference_core.py`

- [ ] **Step 1: Write the failing test for tile-wise TIFF scaling**

```python
def test_run_level_inference_tile_quantization_scales_per_tile() -> None:
    import numpy as np
    import torch
    from wsi_inference import LevelSpec, run_level_inference_for_ome

    class FakeSlide:
        def read_region(self, location, level, size):
            width, height = size
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[:, :, 3] = 255
            return rgba

    class FakeModel(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((batch.shape[0], 1, batch.shape[2], batch.shape[3]), dtype=batch.dtype)
            out[:, :, :, :] = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
            return out

    level = LevelSpec(level=0, width=2, height=2, downsample=1.0)
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

    assert output.dtype == np.uint16
    assert output.min() == 0
    assert output.max() == 65535
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_wsi_inference_core.py::test_run_level_inference_tile_quantization_scales_per_tile -v`
Expected: FAIL because OME tile mode is not wired separately yet

- [ ] **Step 3: Implement a dedicated OME-level runner for `tile` mode**

```python
def run_level_inference_for_ome(
    slide,
    model: torch.nn.Module,
    level: LevelSpec,
    tile_size: int,
    stride: int,
    device: str,
    out_channels: int,
    ome_quant_mode: str,
    rgb_threshold: int = 200,
    background_fraction: float = 0.995,
) -> np.ndarray:
    raw_canvas = allocate_level_canvas(out_channels=out_channels, level=level)
    ome_canvas = np.zeros((out_channels, level.height, level.width), dtype=np.uint16 if ome_quant_mode != "none" else np.float32)
    tiles = build_level_tiles(level=level, tile_size=tile_size, stride=stride)

    for tile in tiles:
        padded_tile = read_level_tile(slide, tile, tile_size=tile_size)
        valid_tile = padded_tile[: tile.read_height, : tile.read_width, :]
        if is_background_tile(valid_tile, rgb_threshold=rgb_threshold, background_fraction=background_fraction):
            continue
        prediction = run_tile_batch(model, [padded_tile], device=device)[0]
        stitch_tile_prediction(raw_canvas, tile, prediction)
        if ome_quant_mode == "tile":
            stitch_tile_prediction(ome_canvas, tile, quantize_tile_prediction(prediction))

    if ome_quant_mode == "none":
        return raw_canvas
    if ome_quant_mode == "tile":
        return ome_canvas
    raise ValueError("run_level_inference_for_ome only handles tile or none modes")
```

- [ ] **Step 4: Update the CLI to use the correct OME export path**

```python
    ome_arrays: List[np.ndarray]
    if args.ome_quant_mode == "global":
        ome_arrays = quantize_global(raw_level_arrays)
    else:
        ome_arrays = []
        for level_index in args.levels:
            level = level_specs[level_index]
            ome_arrays.append(
                run_level_inference_for_ome(
                    slide=slide,
                    model=model,
                    level=level,
                    tile_size=args.tile_size,
                    stride=stride,
                    device=args.device,
                    out_channels=args.out_channels,
                    ome_quant_mode=args.ome_quant_mode,
                    rgb_threshold=args.background_threshold_rgb,
                    background_fraction=args.background_fraction,
                )
            )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_wsi_inference_core.py tests/test_infer_wsi_cli.py -v`
Expected: PASS with correct `tile`-mode OME behavior

- [ ] **Step 6: Commit**

```bash
git add infer_wsi.py wsi_inference.py tests/test_wsi_inference_core.py
git commit -m "feat: implement tile-wise OME quantization"
```

## Task 8: Add User-Facing Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a WSI inference section to the README**

```markdown
## WSI Inference

The repository also supports whole-slide inference from `.svs` files.

### Additional dependencies

Install runtime dependencies for slide reading and OME-TIFF writing:

~~~bash
pip install openslide-python tifffile
~~~

You also need the OpenSlide shared library installed on your system.

### Example

~~~bash
python infer_wsi.py \
  --slide path/to/sample.svs \
  --output-dir outputs/sample \
  --checkpoint weights/hemit_weight.pth \
  --levels 0 1 \
  --tile-size 512 \
  --device cuda:0 \
  --ome-quant-mode tile
~~~

This writes:

- one raw `.npy` per selected level
- one OME-TIFF containing one series per selected level
```

- [ ] **Step 2: Run a formatting sanity check**

Run: `sed -n '1,260p' README.md`
Expected: the new WSI section renders as plain Markdown with no broken fences

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add WSI inference usage"
```

## Spec Coverage Check

- Standalone entry point: covered by Task 6.
- `OpenSlide` `.svs` reading and explicit WSI levels: covered by Task 5 and Task 6.
- Background skip rule using `99.5%` white pixels above `200`: covered by Task 1 and Task 5.
- White edge padding and crop-back stitching: covered by Task 2.
- Same-size output with configurable channels defaulting to `3`: covered by Task 4 and Task 6.
- Per-level raw float `.npy` outputs: covered by Task 2, Task 5, and Task 6.
- One OME-TIFF with separate image series per selected level: covered by Task 3 and Task 6.
- `--ome-quant-mode {global,tile,none}` semantics: covered by Task 3 and Task 7.
- CLI defaults including `levels=0`, `ome-quant-mode=tile`, `device=cuda:0`: covered by Task 6.
- Dependency and user docs update: covered by Task 8.
