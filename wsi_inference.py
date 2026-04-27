from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Sequence

import Attention_GAN
import numpy as np
import tifffile
import torch

if TYPE_CHECKING:
    from openslide import OpenSlide

LEVEL_INFERENCE_BATCH_SIZE = 8
DEFAULT_BACKGROUND_FRACTION = 0.999


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
    background_fraction: float = DEFAULT_BACKGROUND_FRACTION,
) -> bool:
    white_mask = np.all(tile_rgb > rgb_threshold, axis=-1)
    return float(white_mask.mean()) >= background_fraction


def _tile_level0_location(slide: Any, tile: TileSpec) -> tuple[int, int]:
    location = (tile.x, tile.y)
    if tile.level == 0:
        return location

    level_downsamples = getattr(slide, "level_downsamples", None)
    if level_downsamples is None:
        raise ValueError("slide must expose level_downsamples for levels above 0")

    try:
        downsample = float(level_downsamples[tile.level])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"slide must expose a valid downsample for level {tile.level}"
        ) from exc

    if not np.isfinite(downsample) or downsample <= 0.0:
        raise ValueError(f"slide must expose a valid downsample for level {tile.level}")

    # OpenSlide locations are always expressed in level-0 coordinates.
    return (int(round(tile.x * downsample)), int(round(tile.y * downsample)))


def _log_skipped_tile(
    tile: TileSpec,
    slide_location: tuple[int, int],
    tile_rgb: np.ndarray,
    rgb_threshold: int,
) -> None:
    white_mask = np.all(tile_rgb > rgb_threshold, axis=-1)
    non_white_fraction = float((~white_mask).mean())
    channel_min = tile_rgb.min(axis=(0, 1))
    channel_max = tile_rgb.max(axis=(0, 1))
    channel_mean = tile_rgb.mean(axis=(0, 1))

    print(
        "Skipped background tile: "
        f"level={tile.level} "
        f"slide_xy=({slide_location[0]}, {slide_location[1]}) "
        f"rgb_min=({channel_min[0]:.3f}, {channel_min[1]:.3f}, {channel_min[2]:.3f}) "
        f"rgb_max=({channel_max[0]:.3f}, {channel_max[1]:.3f}, {channel_max[2]:.3f}) "
        f"rgb_mean=({channel_mean[0]:.3f}, {channel_mean[1]:.3f}, {channel_mean[2]:.3f}) "
        f"non_white_fraction={non_white_fraction:.6f}"
    )


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


def allocate_level_canvas(
    out_channels: int,
    level: LevelSpec,
    fill_value: float = np.nan,
) -> np.ndarray:
    return np.full(
        (out_channels, level.height, level.width),
        fill_value,
        dtype=np.float32,
    )


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
    location = _tile_level0_location(slide, tile)
    rgba = np.asarray(
        slide.read_region(location, tile.level, (tile.read_width, tile.read_height))
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


def _global_finite_min(level_arrays: Sequence[np.ndarray]) -> float:
    finite_mins: List[float] = []
    for array in level_arrays:
        if np.isinf(array).any():
            raise ValueError("predictions must not contain infinite values")

        finite_mask = np.isfinite(array)
        if finite_mask.any():
            finite_mins.append(float(array[finite_mask].min()))

    if not finite_mins:
        return 0.0
    return min(finite_mins)


def fill_nan_with_global_min(level_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    global_min = _global_finite_min(level_arrays)
    return [
        np.nan_to_num(array, nan=global_min).astype(array.dtype, copy=False)
        for array in level_arrays
    ]


def quantize_global(level_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    non_empty = [array for array in level_arrays if array.size > 0]
    if not non_empty:
        raise ValueError("global quantization requires at least one array")

    if any(np.isinf(array).any() for array in non_empty):
        raise ValueError("predictions must not contain infinite values")

    finite_arrays = [array[np.isfinite(array)] for array in non_empty]
    finite_arrays = [array for array in finite_arrays if array.size > 0]
    if not finite_arrays:
        return [np.zeros_like(array, dtype=np.uint16) for array in level_arrays]

    global_min = min(float(array.min()) for array in finite_arrays)
    global_max = max(float(array.max()) for array in finite_arrays)
    filled_arrays = [np.nan_to_num(array, nan=global_min) for array in level_arrays]
    return [_scale_to_uint16(array, global_min, global_max) for array in filled_arrays]


def _fill_nan_with_min(array: np.ndarray) -> np.ndarray:
    if np.isinf(array).any():
        raise ValueError("predictions must not contain infinite values")

    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros_like(array, dtype=np.float32)

    min_value = float(array[finite_mask].min())
    return np.nan_to_num(array, nan=min_value).astype(np.float32, copy=False)


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
    background_fraction: float = DEFAULT_BACKGROUND_FRACTION,
    log_skipped_tiles: bool = True,
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
            if log_skipped_tiles:
                _log_skipped_tile(
                    tile,
                    _tile_level0_location(slide, tile),
                    valid_tile,
                    rgb_threshold,
                )
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
    background_fraction: float = DEFAULT_BACKGROUND_FRACTION,
    log_skipped_tiles: bool = True,
) -> np.ndarray:
    if ome_quant_mode not in {"tile", "none"}:
        raise ValueError("run_level_inference_for_ome only handles tile or none modes")

    # Canvas is CHW; skipped tile mode regions remain black, raw float regions stay NaN until resolved.
    if ome_quant_mode == "tile":
        canvas = np.zeros((out_channels, level.height, level.width), dtype=np.uint16)
    else:
        canvas = allocate_level_canvas(out_channels=out_channels, level=level)
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
            if log_skipped_tiles:
                _log_skipped_tile(
                    tile,
                    _tile_level0_location(slide, tile),
                    valid_tile,
                    rgb_threshold,
                )
            continue
        batch_specs.append(tile)
        batch_tiles.append(padded_tile)
        if len(batch_tiles) >= LEVEL_INFERENCE_BATCH_SIZE:
            flush_batch()

    flush_batch()

    if ome_quant_mode == "none":
        return _fill_nan_with_min(canvas)

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
