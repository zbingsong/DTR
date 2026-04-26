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
    assert args.background_fraction == 0.999


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
        "infer_wsi.run_level_inference_for_ome",
        lambda **kwargs: np.zeros(
            (3, kwargs["level"].height, kwargs["level"].width),
            dtype=np.uint16,
        ),
        raising=False,
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


def _configure_main_mode_test(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls = {
        "quantize_global": [],
        "run_level_inference": [],
        "run_level_inference_for_ome": [],
        "saved_npy": [],
        "ome_arrays": None,
    }

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

    def fake_run_level_inference(**kwargs: object) -> np.ndarray:
        level = kwargs["level"]
        calls["run_level_inference"].append(level.level)
        return np.full(
            (3, level.height, level.width),
            fill_value=level.level + 1.5,
            dtype=np.float32,
        )

    def fake_quantize_global(arrays: list[np.ndarray]) -> list[np.ndarray]:
        calls["quantize_global"].append([array.copy() for array in arrays])
        return [
            np.full_like(array, fill_value=(index + 1) * 101, dtype=np.uint16)
            for index, array in enumerate(arrays)
        ]

    def fake_run_level_inference_for_ome(**kwargs: object) -> np.ndarray:
        level = kwargs["level"]
        calls["run_level_inference_for_ome"].append((level.level, kwargs["ome_quant_mode"]))
        return np.full(
            (3, level.height, level.width),
            fill_value=(level.level + 1) * 11,
            dtype=np.uint16,
        )

    monkeypatch.setattr("infer_wsi.run_level_inference", fake_run_level_inference)
    monkeypatch.setattr("infer_wsi.quantize_global", fake_quantize_global)
    monkeypatch.setattr(
        "infer_wsi.run_level_inference_for_ome",
        fake_run_level_inference_for_ome,
        raising=False,
    )
    monkeypatch.setattr(
        "numpy.save",
        lambda path, array: calls["saved_npy"].append((str(path), array.copy())),
    )
    monkeypatch.setattr(
        "infer_wsi.write_ome_tiff",
        lambda path, arrays: calls.__setitem__("ome_arrays", (str(path), list(arrays))),
    )

    return calls


def test_main_global_mode_uses_global_quantization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _configure_main_mode_test(monkeypatch)

    exit_code = main(
        [
            "--slide",
            "sample.svs",
            "--output-dir",
            str(tmp_path),
            "--levels",
            "0",
            "1",
            "--ome-quant-mode",
            "global",
        ]
    )

    assert exit_code == 0
    assert calls["run_level_inference"] == [0, 1]
    assert calls["run_level_inference_for_ome"] == []
    assert len(calls["quantize_global"]) == 1
    assert calls["ome_arrays"][1][0].dtype == np.uint16
    assert int(calls["ome_arrays"][1][0][0, 0, 0]) == 101
    assert int(calls["ome_arrays"][1][1][0, 0, 0]) == 202
    assert calls["saved_npy"][0][1].dtype == np.float32
    assert float(calls["saved_npy"][0][1][0, 0, 0]) == 1.5


def test_main_tile_mode_uses_tile_quantized_ome_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _configure_main_mode_test(monkeypatch)

    exit_code = main(
        [
            "--slide",
            "sample.svs",
            "--output-dir",
            str(tmp_path),
            "--levels",
            "0",
            "1",
            "--ome-quant-mode",
            "tile",
        ]
    )

    assert exit_code == 0
    assert calls["run_level_inference"] == [0, 1]
    assert calls["run_level_inference_for_ome"] == [(0, "tile"), (1, "tile")]
    assert calls["quantize_global"] == []
    assert calls["ome_arrays"][1][0].dtype == np.uint16
    assert int(calls["ome_arrays"][1][0][0, 0, 0]) == 11
    assert int(calls["ome_arrays"][1][1][0, 0, 0]) == 22
    assert calls["saved_npy"][1][1].dtype == np.float32
    assert float(calls["saved_npy"][1][1][0, 0, 0]) == 2.5


def test_main_none_mode_writes_raw_float_ome_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _configure_main_mode_test(monkeypatch)

    exit_code = main(
        [
            "--slide",
            "sample.svs",
            "--output-dir",
            str(tmp_path),
            "--levels",
            "0",
            "1",
            "--ome-quant-mode",
            "none",
        ]
    )

    assert exit_code == 0
    assert calls["run_level_inference"] == [0, 1]
    assert calls["run_level_inference_for_ome"] == []
    assert calls["quantize_global"] == []
    assert calls["ome_arrays"][1][0].dtype == np.float32
    assert float(calls["ome_arrays"][1][0][0, 0, 0]) == 1.5
    assert float(calls["ome_arrays"][1][1][0, 0, 0]) == 2.5
