"""Command line and TOML configuration behaviour."""

from pathlib import Path
from types import SimpleNamespace

import pytest


import cli


def test_cli_and_toml_build_equivalent_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "source.mkv").touch()
    (tmp_path / "tiles.pkl").touch()
    config_file = tmp_path / "run.toml"
    config_file.write_text(
        'source = "source.mkv"\n'
        'gallery = "tiles.pkl"\n'
        'grid_size = 4\n'
        'contrast = 0.6\n'
        'candidates = 20\n'
        'stochastic = true\n'
        'use_cache = false\n'
        'segment = 1200\n'
    )

    from_file = cli.parse_args(["--config", str(config_file)])
    from_cli = cli.parse_args([
        "--source", "source.mkv", "--gallery", "tiles.pkl", "--grid-size", "4",
        "--contrast", "0.6", "--candidates", "20", "--stochastic",
        "--no-cache", "--segment", "1200",
    ])
    assert from_file.config == from_cli.config
    assert from_file.gallery.fingerprint == from_cli.gallery.fingerprint


def test_cli_precedence_and_boolean_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/source.mkv").touch()
    (tmp_path / "assets/tiles.pkl").touch()
    (tmp_path / "config.toml").write_text(
        'source = "assets/source.mkv"\n'
        'gallery = "assets/tiles.pkl"\n'
        'contrast = 0.4\n'
        'use_cache = false\n'
        'hold_tiles = false\n'
    )
    options = cli.parse_args(["--contrast", "0.9", "--cache", "--hold-tiles"])
    assert options.config.contrast == 0.9
    assert options.config.use_cache is True
    assert options.config.hold_tiles is True
    assert options.config.tile_fit == "native"
    assert options.config.grid_size == 8


def test_no_stochastic_forces_one_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    opts = cli.parse_args(["--source", "s.mp4", "--gallery", "g.pkl", "--no-stochastic"])
    assert opts.config.candidates == 1


def test_zero_segment_disables_segmentation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    opts = cli.parse_args(["--source", "s.mp4", "--gallery", "g.pkl", "--segment", "0"])
    assert opts.config.segment_frames == 0


def test_grid_and_binary_budget_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    opts = cli.parse_args([
        "--source", "s.mp4", "--gallery", "g.pkl", "--grid", "30x20",
        "--gallery-budget", "1.5G",
    ])
    assert opts.config.grid == (30, 20)
    assert opts.config.gallery_budget == int(1.5 * (1 << 30))


def test_unknown_key_and_missing_explicit_config_are_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "bad.toml"
    bad.write_text('source = "s.mp4"\ngallery = "g.pkl"\nwat = 1\n')
    with pytest.raises(ValueError, match="unknown configuration key"):
        cli.parse_args(["--config", str(bad)])
    with pytest.raises(ValueError, match="config file not found"):
        cli.parse_args(["--config", "missing.toml"])


def test_help_does_not_read_malformed_default_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text("not = [valid TOML")
    assert cli.cli_main(["--help"]) == 0


@pytest.mark.parametrize("body", [
    'source = 4\ngallery = "g.pkl"\n',
    'source = "s.mp4"\ngallery = "g.pkl"\nuse_cache = 1\n',
    'source = "s.mp4"\ngallery = "g.pkl"\ncontrast = true\n',
    'source = "s.mp4"\ngallery = "g.pkl"\nstride = true\n',
])
def test_invalid_toml_types_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "bad.toml"
    config.write_text(body)
    with pytest.raises(ValueError, match="must be"):
        cli.parse_args(["--config", str(config)])


@pytest.mark.parametrize("value", ["0", "NaN", "inf", "-2G", "xG"])
def test_invalid_budget_fails(value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises((ValueError, SystemExit)):
        cli.parse_args(["--source", "s", "--gallery", "g.pkl", "--gallery-budget", value])


def test_dry_run_only_probes_and_estimates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    derived = SimpleNamespace(cell_size=(16, 16))
    seen: list[str] = []
    monkeypatch.setattr(cli.video, "probe_video", lambda config, tile_aspect=None: (seen.append("probe") or derived))
    monkeypatch.setattr(cli.gallery, "check_gallery_budget", lambda *args: seen.append("estimate"))
    monkeypatch.setattr(cli.pipeline, "main", lambda *args: pytest.fail("full pipeline called"))
    assert cli.cli_main(["--source", "s.mp4", "--gallery", "g.pkl", "--dry-run"]) == 0
    assert seen == ["probe", "estimate"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("gallery_kind", ["missing", "empty", "broken"])
def test_dry_run_rejects_missing_or_unreadable_video_gallery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    gallery_kind: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    gallery_path = tmp_path / "gallery.mkv"
    if gallery_kind == "empty":
        gallery_path = tmp_path / "empty"
        gallery_path.mkdir()
    elif gallery_kind == "broken":
        gallery_path.write_bytes(b"not a video")
    monkeypatch.setattr(
        cli.video, "probe_video",
        lambda config, tile_aspect=None: SimpleNamespace(cell_size=(16, 16)),
    )

    result = cli.cli_main([
        "--source", "source.mp4", "--gallery", str(gallery_path),
        "--gallery-type", "video", "--tile-fit", "crop", "--dry-run",
    ])

    assert result == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("budget", ["abc", "1e308G"])
def test_invalid_toml_budget_reports_an_error(tmp_path, monkeypatch, capsys, budget):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        f'source = "source.mp4"\ngallery = "train"\ngallery_budget = "{budget}"\n')
    assert cli.cli_main([]) == 2
    assert "gallery_budget" in capsys.readouterr().err


def test_overflowing_cli_budget_reports_an_error(capsys):
    assert cli.cli_main(["--source", "source.mp4", "--gallery", "train",
                         "--gallery-budget", "1e308G"]) == 2
    assert "size is too large" in capsys.readouterr().err
