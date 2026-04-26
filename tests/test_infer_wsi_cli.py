from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_main_rejects_missing_levels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("infer_wsi.open_slide", lambda path: object())
    monkeypatch.setattr(
        "infer_wsi.get_level_specs",
        lambda slide: [type("L", (), {"level": 0, "width": 4, "height": 4, "downsample": 1.0})()],
    )

    with pytest.raises(ValueError, match="Requested levels not found"):
        main(
            [
                "--slide",
                "sample.svs",
                "--output-dir",
                str(tmp_path),
                "--levels",
                "1",
            ]
        )


def test_main_saves_npy_per_level_and_one_ome_tiff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = {"npy": [], "ome": None}

    monkeypatch.setattr("infer_wsi.open_slide", lambda path: object())
    monkeypatch.setattr(
        "infer_wsi.get_level_specs",
        lambda slide: [
            type("L", (), {"level": 0, "width": 4, "height": 4, "downsample": 1.0})(),
            type("L", (), {"level": 1, "width": 2, "height": 2, "downsample": 4.0})(),
        ],
    )
    monkeypatch.setattr(
        "infer_wsi.load_generator",
        lambda checkpoint_path, out_channels, device: object(),
    )
    monkeypatch.setattr(
        "infer_wsi.run_level_inference",
        lambda **kwargs: np.zeros(
            (3, kwargs["level"].height, kwargs["level"].width),
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        "numpy.save",
        lambda path, array: saved["npy"].append((str(path), array.shape)),
    )
    monkeypatch.setattr(
        "infer_wsi.write_ome_tiff",
        lambda path, arrays: saved.__setitem__("ome", (str(path), len(arrays))),
    )

    exit_code = main(
        [
            "--slide",
            "sample.svs",
            "--output-dir",
            str(tmp_path),
            "--levels",
            "0",
            "1",
        ]
    )

    assert exit_code == 0
    assert saved["npy"] == [
        (str(tmp_path / "sample.level-0.npy"), (3, 4, 4)),
        (str(tmp_path / "sample.level-1.npy"), (3, 2, 2)),
    ]
    assert saved["ome"] == (str(tmp_path / "sample.predictions.ome.tiff"), 2)
