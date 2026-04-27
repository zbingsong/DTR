import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

import wsi_inference
from wsi_inference import (
    fill_nan_with_global_min,
    get_level_specs,
    load_generator,
    open_slide,
    quantize_global,
    run_level_inference,
    run_level_inference_for_ome,
    write_ome_tiff,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WSI inference on selected SVS pyramid levels."
    )
    parser.add_argument("--slide", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="weights/hemit_weight.pth")
    parser.add_argument("--levels", nargs="+", type=int, default=[0])
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ome-quant-mode",
        choices=["global", "tile", "none"],
        default="tile",
    )
    parser.add_argument("--background-threshold-rgb", type=int, default=200)
    parser.add_argument("--background-fraction", type=float, default=0.999)
    return parser


def resolve_stride(tile_size: int, stride: Optional[int]) -> int:
    return tile_size if stride is None else stride


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stride = resolve_stride(args.tile_size, args.stride)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide = open_slide(args.slide)
    slide_close = getattr(slide, "close", None)
    previous_batch_size = wsi_inference.LEVEL_INFERENCE_BATCH_SIZE
    wsi_inference.LEVEL_INFERENCE_BATCH_SIZE = args.batch_size

    try:
        level_specs = {spec.level: spec for spec in get_level_specs(slide)}
        missing = [level for level in args.levels if level not in level_specs]
        if missing:
            raise ValueError(f"Requested levels not found: {missing}")

        model = load_generator(
            args.checkpoint,
            out_channels=args.out_channels,
            device=args.device,
        )

        raw_level_arrays: List[np.ndarray] = []
        slide_stem = Path(args.slide).stem
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
            np.save(output_dir / f"{slide_stem}.level-{level_index}.npy", prediction)

        if args.ome_quant_mode == "global":
            ome_arrays = quantize_global(raw_level_arrays)
        elif args.ome_quant_mode == "tile":
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
                        log_skipped_tiles=False,
                    )
                )
        else:
            ome_arrays = fill_nan_with_global_min(raw_level_arrays)
        write_ome_tiff(output_dir / f"{slide_stem}.predictions.ome.tiff", ome_arrays)
    finally:
        wsi_inference.LEVEL_INFERENCE_BATCH_SIZE = previous_batch_size
        if callable(slide_close):
            slide_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
