from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Sequence

import Attention_GAN
import numpy as np
import tifffile
import torch

if TYPE_CHECKING:
    from openslide import OpenSlide

LEVEL_INFERENCE_BATCH_SIZE = 8


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
    if tile_size <= 0:
        raise ValueError("tile_size must be greater than 0")
    if stride <= 0:
        raise ValueError("stride must be greater than 0")
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


def allocate_level_canvas(out_channels: int, level: LevelSpec) -> np.ndarray:
    return np.zeros((out_channels, level.height, level.width), dtype=np.float32)


def open_slide(slide_path: str) -> "OpenSlide":
    if not slide_path.lower().endswith(".svs"):
        raise ValueError("slide path must point to a .svs file")

    from openslide import OpenSlide

    return OpenSlide(slide_path)


def get_level_specs(slide: Any) -> List[LevelSpec]:
    specs: List[LevelSpec] = []
    for level_index in range(slide.level_count):
        width, height = slide.level_dimensions[level_index]
        specs.append(
            LevelSpec(
                level=level_index,
                width=width,
                height=height,
                downsample=float(slide.level_downsamples[level_index]),
            )
        )
    return specs


def read_level_tile(slide: Any, tile: TileSpec, tile_size: int) -> np.ndarray:
    rgba = np.asarray(
        slide.read_region((tile.x, tile.y), tile.level, (tile.read_width, tile.read_height))
    )
    rgb = rgba[:, :, :3].astype(np.uint8)
    return pad_tile_to_size(rgb, tile_size=tile_size)


def _scale_to_uint16(
    array: np.ndarray,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint16)

    scaled = (array - min_value) / (max_value - min_value)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 65535.0).astype(np.uint16)


def quantize_global(level_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    non_empty = [array for array in level_arrays if array.size > 0]
    if not non_empty:
        raise ValueError("global quantization requires at least one array")

    global_min = min(float(array.min()) for array in non_empty)
    global_max = max(float(array.max()) for array in non_empty)
    return [_scale_to_uint16(array, global_min, global_max) for array in level_arrays]


def quantize_tile_prediction(tile_prediction: np.ndarray) -> np.ndarray:
    return _scale_to_uint16(
        tile_prediction,
        float(tile_prediction.min()),
        float(tile_prediction.max()),
    )


def write_ome_tiff(output_path: str, series_arrays: Sequence[np.ndarray]) -> None:
    with tifffile.TiffWriter(output_path, ome=True, bigtiff=True) as tif:
        for series in series_arrays:
            tif.write(
                series,
                metadata={"axes": "CYX"},
                photometric="MINISBLACK",
            )


def prepare_tile_tensor(tile_rgb: np.ndarray) -> torch.Tensor:
    # Input tiles are HWC uint8 RGB; the model expects CHW float32 in [-1, 1].
    tensor = torch.from_numpy(tile_rgb.transpose(2, 0, 1)).to(dtype=torch.float32)
    tensor = tensor / 255.0
    return (tensor - 0.5) * 2.0


def run_tile_batch(
    model: torch.nn.Module,
    tiles: List[np.ndarray],
    device: str,
) -> List[np.ndarray]:
    batch = torch.stack([prepare_tile_tensor(tile) for tile in tiles], dim=0).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(batch).detach().cpu().numpy().astype(np.float32)
    finally:
        model.train(was_training)
    return [outputs[index] for index in range(outputs.shape[0])]


def run_level_inference(
    slide: Any,
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

    def flush_batch() -> None:
        if not batch_tiles:
            return

        outputs = run_tile_batch(model, batch_tiles, device=device)
        for batch_tile, prediction in zip(batch_specs, outputs):
            stitch_tile_prediction(canvas, batch_tile, prediction)
        batch_specs.clear()
        batch_tiles.clear()

    for tile in tiles:
        padded_tile = read_level_tile(slide, tile, tile_size=tile_size)
        valid_tile = padded_tile[: tile.read_height, : tile.read_width, :]
        if is_background_tile(
            valid_tile,
            rgb_threshold=rgb_threshold,
            background_fraction=background_fraction,
        ):
            continue
        batch_specs.append(tile)
        batch_tiles.append(padded_tile)
        if len(batch_tiles) >= LEVEL_INFERENCE_BATCH_SIZE:
            flush_batch()

    flush_batch()

    return canvas


def run_level_inference_for_ome(
    slide: Any,
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
    if ome_quant_mode not in {"tile", "none"}:
        raise ValueError("run_level_inference_for_ome only handles tile or none modes")

    # Canvas is CHW; `tile` stores uint16-scaled predictions, `none` preserves float32.
    canvas_dtype = np.uint16 if ome_quant_mode == "tile" else np.float32
    canvas = np.zeros((out_channels, level.height, level.width), dtype=canvas_dtype)
    tiles = build_level_tiles(level=level, tile_size=tile_size, stride=stride)

    batch_specs: List[TileSpec] = []
    batch_tiles: List[np.ndarray] = []

    def flush_batch() -> None:
        if not batch_tiles:
            return

        outputs = run_tile_batch(model, batch_tiles, device=device)
        for batch_tile, prediction in zip(batch_specs, outputs):
            ome_prediction = (
                quantize_tile_prediction(prediction)
                if ome_quant_mode == "tile"
                else prediction
            )
            stitch_tile_prediction(canvas, batch_tile, ome_prediction)
        batch_specs.clear()
        batch_tiles.clear()

    for tile in tiles:
        padded_tile = read_level_tile(slide, tile, tile_size=tile_size)
        valid_tile = padded_tile[: tile.read_height, : tile.read_width, :]
        if is_background_tile(
            valid_tile,
            rgb_threshold=rgb_threshold,
            background_fraction=background_fraction,
        ):
            continue
        batch_specs.append(tile)
        batch_tiles.append(padded_tile)
        if len(batch_tiles) >= LEVEL_INFERENCE_BATCH_SIZE:
            flush_batch()

    flush_batch()

    return canvas


def load_generator(
    checkpoint_path: str,
    out_channels: int,
    device: str,
) -> torch.nn.Module:
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
