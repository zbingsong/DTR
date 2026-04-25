# WSI Inference Design

Date: 2026-04-25
Repo: `DTR`
Status: Draft approved for planning

## Summary

This design adds a new standalone WSI inference entry point for H&E-to-mIHC prediction from `.svs` slides without changing the existing patch-based inference path in `predict.py`.

The new pipeline will:

- open a `.svs` file with `OpenSlide`
- run inference on one or more explicitly selected WSI pyramid levels
- split each level into tiles
- skip background tiles using a fixed white-pixel rule
- run the current generator on non-background tiles
- stitch raw float predictions per level
- save one raw `.npy` per selected level
- save all selected levels into one OME-TIFF file as separate image series

The implementation should prefer isolation over reuse. Small amounts of duplication from the current inference code are acceptable if they reduce the risk of breaking the existing codebase.

## Goals

- Add a dedicated WSI inference workflow for `.svs` inputs.
- Keep each selected WSI level independent during tiling, inference, stitching, and raw output saving.
- Preserve raw model predictions in per-level `.npy` files.
- Export all selected levels into one ImageJ-readable OME-TIFF file.
- Support configurable output channel count, defaulting to the current model behavior of `3`.
- Avoid breaking current training and patch inference workflows.

## Non-Goals

- Refactoring `predict.py` into a shared framework.
- Supporting non-`.svs` input formats in v1.
- Supporting associated slide images such as label, thumbnail, or macro images for inference.
- Supporting model architectures whose output spatial size differs from the input tile size.
- Adding a learned tissue-segmentation model.
- Forcing selected levels into a single multiscale pyramid in the output OME-TIFF.

## Current Repo Constraints

Observed from the existing repo:

- `predict.py` performs patch inference only.
- `Attention_GAN.Generator` is instantiated with `in_channels=3` and `out_channels=3` in current training and inference paths.
- The generator preserves spatial size from input tile to output tile.
- No WSI reader, tiler, tissue masker, stitcher, or OME-TIFF writer exists today.

These constraints make a new standalone entry point the safest design for v1.

## Proposed Entry Point

Add a new top-level script:

- `infer_wsi.py`

This script will orchestrate the full WSI pipeline and will not modify the behavior of `predict.py`.

The script may directly import useful existing objects such as:

- `Attention_GAN.Generator`

It may also duplicate a small amount of preprocessing or checkpoint-loading logic if that is safer than refactoring existing code.

## High-Level Architecture

The pipeline is organized into six stages:

1. `Slide open and level validation`
2. `Per-level tile grid creation`
3. `Background filtering`
4. `Tile inference`
5. `Per-level stitching and raw `.npy` save`
6. `OME-TIFF export`

Suggested internal helper boundaries:

- `load_generator(...)`
  - loads checkpoint and prepares the generator on the requested device
- `open_slide(...)`
  - opens the `.svs` and exposes valid WSI pyramid level metadata
- `iter_level_tiles(...)`
  - yields tile coordinates and valid crop extents for a given level
- `is_background_tile(...)`
  - applies the fixed white-pixel rule
- `infer_tile_batch(...)`
  - normalizes input tiles and runs model inference on a batch
- `stitch_level_predictions(...)`
  - writes predictions into a stitched per-level tensor
- `write_ome_tiff(...)`
  - writes one OME series per selected level

These helpers can live inside `infer_wsi.py` in v1 or move into a small WSI-specific helper module if the file becomes too large. They should not force a repo-wide inference refactor.

## CLI Design

### Required arguments

- `--slide`
  - input `.svs` path
- `--output-dir`
  - destination directory for per-level `.npy` files and the final OME-TIFF

All arguments other than `--slide` and `--output-dir` have defaults.

### Optional arguments and defaults

- `--levels`
  - type: one or more integer level indices
  - default: `0`
- `--checkpoint`
  - default: `weights/hemit_weight.pth`
- `--tile-size`
  - default: `512`
- `--stride`
  - default: equal to `tile-size`
- `--batch-size`
  - default: `4`
- `--out-channels`
  - default: `3`
- `--device`
  - default: `cuda:0`
- `--ome-quant-mode`
  - choices: `global`, `tile`, `none`
  - default: `tile`
- `--background-threshold-rgb`
  - default: `200`
- `--background-fraction`
  - default: `0.995`

### Example invocation

```bash
python infer_wsi.py \
  --slide path/to/sample.svs \
  --output-dir outputs/sample \
  --checkpoint weights/hemit_weight.pth \
  --levels 0 1 2 \
  --tile-size 512 \
  --batch-size 4 \
  --out-channels 3 \
  --device cuda:0 \
  --ome-quant-mode tile
```

## Input Semantics

### Supported source

Only `.svs` slides are in scope for v1.

The script must use `OpenSlide` to:

- open the slide
- enumerate true WSI pyramid levels
- read image regions from a requested level
- ignore associated images such as slide label or macro image

### WSI levels

`--levels` must refer to explicit WSI pyramid level indices from the main slide image.

Each requested level is treated as an independent image for:

- tiling
- background filtering
- inference
- stitching
- `.npy` save
- OME series export

No assumption is made that the chosen levels form a complete or regular pyramid.

## Tile Grid And Edge Handling

For each selected level:

- use that level’s native dimensions
- generate a level-local tile grid using `tile-size` and `stride`
- if width or height is not divisible by `tile-size`, include right and bottom edge tiles so that the entire level is covered

### Edge tile policy

When an edge tile is smaller than `tile-size`:

- read the valid image region from the slide
- pad to full tile size using white pixels `255,255,255`
- run inference on the padded tile
- crop the predicted tile back to the valid region before stitching

White padding is required so edge context behaves like normal slide background before normalization and background checking.

## Background Filtering

The fixed background rule for v1 is:

- classify a tile as background if at least `99.5%` of its pixels satisfy:
  - `R > 200`
  - `G > 200`
  - `B > 200`

Implications:

- background filtering is heuristic and self-contained
- no external segmentation model is required
- skipped tiles are not sent through the model

### Output for skipped tiles

Skipped tiles must contribute zeros in every output channel in the stitched raw prediction.

This applies to:

- per-level raw `.npy`
- OME-TIFF export

## Model Inference Contract

### Input contract

Input tile:

- shape: `3 x H x W`
- RGB
- normalized exactly like current inference:
  - convert to tensor in `[0, 1]`
  - scale to `[-1, 1]`

### Output contract

For v1:

- output spatial size must equal input tile spatial size
- output channel count is configurable
- default `out_channels=3`

Expected stitched tensor shape per level:

- `(C, H, W)`

Where:

- `C` is `out_channels`
- `H` and `W` are the selected WSI level dimensions

The implementation should fail early if the loaded checkpoint is incompatible with the requested output channel count.

## Stitching Strategy

### Baseline v1 default

Use:

- `stride = tile-size`

This means no overlap by default.

Rationale:

- the user explicitly tied overlap policy to the actual model behavior
- current generator output matches input tile size exactly
- no-overlap is simplest and lowest-risk for the first WSI pipeline

### Stitching behavior

- allocate one raw float tensor per selected level with shape `(C, H, W)`
- for each inferred tile:
  - crop away padded border predictions if present
  - place the valid predicted region into the stitched tensor at the tile’s level-local coordinates
- for skipped background tiles:
  - leave zeros in the corresponding output region

No blending is required in v1 because default stride equals tile size.

If overlap is added later, the stitcher should be extendable to support accumulation and averaging.

## Raw Per-Level Output

For each selected level, save one raw `.npy` file containing the stitched float prediction tensor.

Suggested naming:

- `<slide_stem>.level-0.npy`
- `<slide_stem>.level-1.npy`
- `<slide_stem>.level-2.npy`

Shape:

- `(C, H, W)`

Data type:

- floating-point model output, before any TIFF quantization

The `.npy` files are the authoritative raw outputs of the pipeline.

## OME-TIFF Export

### Container layout

Write one OME-TIFF file containing:

- one independent OME image series per selected WSI level

Do not attempt to encode the selected levels as a single multiscale pyramid, even if they originated from the same source slide.

This matches the user requirement that each selected WSI level should be treated as an independent picture while still being stored in one file.

### Series content

Each series stores the stitched prediction for exactly one selected level.

Axes:

- `CYX`

Series ordering:

- preserve the order supplied by `--levels`

### ImageJ readability

Use `tifffile` to write an OME-TIFF that is readable by ImageJ.

The implementation should prefer straightforward OME series layout over advanced vendor-specific TIFF features.

## `--ome-quant-mode`

This argument controls how raw float stitched predictions are converted for OME-TIFF output.

Choices:

- `global`
- `tile`
- `none`

Default:

- `tile`

### `global`

Behavior:

- collect the raw stitched predictions that will be exported
- compute one global linear scaling to fill the full `uint16` range
- apply that scaling to all exported levels
- write `uint16` OME-TIFF series

Important:

- `.npy` outputs remain raw float
- all levels share the same scale mapping

### `tile`

Behavior:

- scale each predicted tile independently to the full `uint16` range before stitching
- stitch those already-scaled tiles into the OME-TIFF output
- raw `.npy` still stores the stitched float predictions before any scaling

Important:

- scaling is per tile, not per level
- background tiles still remain zero-filled in the stitched result

### `none`

Behavior:

- write raw float predictions to OME-TIFF without scaling

Important:

- no quantization is applied
- `.npy` and OME-TIFF store the same numeric prediction values apart from TIFF container representation

## Error Handling

The script should fail early and clearly for:

- input file not ending in `.svs`
- `OpenSlide` open failure
- requested `--levels` not present in the slide pyramid
- invalid `stride` or `tile-size`
- checkpoint load failure
- checkpoint incompatible with requested `out_channels`
- model output spatial shape not matching input tile shape
- OME-TIFF write failure

Additional behavior:

- associated images like label or macro must never be treated as valid inference levels
- if `ome-quant-mode=global` and there are no non-background inferred tiles across all selected levels, the script should fail with a clear error rather than silently inventing a scaling range

## Logging And Progress Reporting

The script should print or log, at minimum:

- slide path
- selected levels
- model checkpoint
- device
- per-level dimensions
- per-level tile count
- per-level kept vs skipped background tile count
- output tensor shape per level
- saved `.npy` path per level
- final OME-TIFF path

Progress should be level-aware so users can tell which level is currently running.

## Dependencies

New dependencies allowed for this feature:

- `openslide-python`
- `tifffile`

System dependency expected:

- OpenSlide shared library required by `openslide-python`

The implementation should update dependency documentation or install requirements as part of the feature work.

## Testing Strategy

### Unit tests

Add focused tests for:

- level tile grid generation
- edge tile padding with white pixels
- crop-back behavior for padded edge predictions
- background tile rule
- zero-fill behavior for skipped tiles
- per-level stitched tensor shape
- quantization behavior for `global`, `tile`, and `none`

### Integration-style tests

Use a mocked slide reader abstraction or `OpenSlide` stub for CI-friendly tests covering:

- level selection validation
- per-level independence
- `.npy` output naming and shape
- OME series count and ordering

### Manual validation

Run at least one manual smoke test with a real `.svs` slide in the `DTR` environment to verify:

- level metadata is read correctly
- background skipping behaves as expected
- `.npy` outputs have expected dimensions
- OME-TIFF opens in ImageJ and exposes one series per selected level

## Compatibility And Risk Control

To avoid breaking the current codebase:

- do not change `predict.py` behavior
- do not alter existing dataset adapters
- do not refactor training or patch inference into a new abstraction unless strictly necessary
- keep WSI-specific logic isolated to the new standalone path

Small code duplication is acceptable if it prevents regressions in the existing patch workflow.

## Open Implementation Notes

These notes are intentional implementation guidance, not unresolved requirements:

- keep `infer_wsi.py` self-contained in v1
- if helper extraction is needed, extract only WSI-local helpers
- prefer robust coordinate bookkeeping over aggressive optimization
- prioritize correctness of per-level output dimensions and raw-value handling over clever abstractions

## Accepted Decisions

- standalone script rather than extending `predict.py`
- explicit level indices, default `0`
- `OpenSlide` for `.svs` reading
- heuristic white-background rule
- white padding on edge tiles
- per-level independence for inference and output
- separate OME series per selected level in one file
- configurable channel count with default `3`
- raw `.npy` always stores unscaled floats
- OME export controlled by `--ome-quant-mode`
- default `--ome-quant-mode tile`
- default `--device cuda:0`
