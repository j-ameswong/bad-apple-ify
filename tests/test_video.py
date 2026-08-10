import numpy as np
import pytest

from main import Config, probe_video, stream_frames


def test_probe_video_reads_metadata(video):
    path, frames = video
    config = Config(input_dir=str(path), output_dir="")

    probe_video(config)

    assert config.src_fps == 30
    assert config.output_fps == 30
    assert config.src_dimensions == (64, 48)
    assert config.src_frame_count == len(frames)


def test_probe_video_derives_aspect_ratio(video_factory):
    """The grid follows the source's own ratio; no allowlist, no stretching."""
    path, _ = video_factory(count=2, width=64, height=48)
    config = Config(input_dir=str(path), output_dir="")

    probe_video(config)

    assert config.aspect_ratio == (4, 3)


def test_probe_video_keeps_unusual_aspect_ratio(video_factory):
    """A 2.39:1-ish source must not be snapped to 16:9."""
    path, _ = video_factory(count=2, width=478, height=200)
    config = Config(input_dir=str(path), output_dir="")

    probe_video(config)

    num, den = config.aspect_ratio
    assert num / den == pytest.approx(478 / 200, rel=0.02)
    assert den <= 16  # limit_denominator keeps the pair usable as a multiplier


def test_target_dimensions_are_grid_multiples(video):
    path, _ = video
    config = Config(input_dir=str(path), output_dir="", grid_size=2)

    probe_video(config)

    cell_w, cell_h = config.cell_size()
    assert config.target_dimensions == (config.grid_x * cell_w, config.grid_y * cell_h)
    assert cell_w >= 1 and cell_h >= 1


def test_target_dimensions_never_degenerate(video):
    """A grid finer than the source still gets at least one pixel per cell."""
    path, _ = video
    config = Config(input_dir=str(path), output_dir="", grid_size=64)

    probe_video(config)

    assert min(config.cell_size()) >= 1


def test_stream_frames_yields_every_frame(video):
    path, frames = video
    config = Config(input_dir=str(path), output_dir="", grid_size=2)
    probe_video(config)

    streamed = list(stream_frames(config))

    assert len(streamed) == len(frames)
    for frame in streamed:
        assert frame.shape[1::-1] == config.target_dimensions


def test_stream_frames_preserves_content(video):
    """Target dimensions equal the source here, so frames must survive intact."""
    path, frames = video
    config = Config(input_dir=str(path), output_dir="", grid_size=2)
    probe_video(config)
    assert config.target_dimensions == config.src_dimensions

    np.testing.assert_array_equal(np.array(list(stream_frames(config))), frames)


def test_stream_frames_is_lazy(video):
    """Nothing is decoded until iteration, and only one frame is held at a time."""
    path, _ = video
    config = Config(input_dir=str(path), output_dir="", grid_size=2)
    probe_video(config)

    stream = stream_frames(config)
    first = next(stream)

    assert first.nbytes < 1024 * 1024
    assert next(stream) is not first


def test_stream_frames_survives_bad_frame_count(video, monkeypatch):
    """CAP_PROP_FRAME_COUNT is a container guess; the generator must not trust it."""
    import cv2

    path, frames = video
    config = Config(input_dir=str(path), output_dir="", grid_size=2)
    probe_video(config)

    real_get = cv2.VideoCapture.get
    monkeypatch.setattr(
        cv2.VideoCapture, "get",
        lambda self, prop: 9999.0 if prop == cv2.CAP_PROP_FRAME_COUNT else real_get(self, prop),
    )

    assert len(list(stream_frames(config))) == len(frames)


def test_missing_video_raises(tmp_path):
    config = Config(input_dir=str(tmp_path / "nope.mkv"), output_dir="")
    with pytest.raises(ValueError):
        probe_video(config)
