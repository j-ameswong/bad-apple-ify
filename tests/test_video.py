import numpy as np
import pytest

from main import UserConfig, probe_video, stream_frames


def test_probe_video_reads_metadata(video):
    path, frames = video
    derived = probe_video(UserConfig(input_dir=str(path), output_dir=""))

    assert derived.src_fps == 30
    assert derived.output_fps == 30
    assert derived.src_dimensions == (64, 48)
    assert derived.src_frame_count == len(frames)


def test_probe_video_derives_the_grid(video_factory):
    """The grid follows the source's own ratio; no allowlist, no stretching."""
    path, _ = video_factory(count=2, width=64, height=48)

    derived = probe_video(UserConfig(input_dir=str(path), output_dir="", grid_size=2))

    assert derived.aspect_ratio == (4, 3)
    assert (derived.grid_x, derived.grid_y) == (8, 6)
    assert derived.cell_size == (8, 8)


def test_stream_frames_yields_every_frame(video):
    path, frames = video
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2)
    derived = probe_video(config)

    streamed = list(stream_frames(config, derived))

    assert len(streamed) == len(frames)
    for frame in streamed:
        assert frame.shape[1::-1] == derived.target_dimensions


def test_stream_frames_preserves_content(video):
    """Target dimensions equal the source here, so frames must survive intact."""
    path, frames = video
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2)
    derived = probe_video(config)
    assert derived.target_dimensions == derived.src_dimensions

    np.testing.assert_array_equal(np.array(list(stream_frames(config, derived))),
                                  frames)


def test_stream_frames_is_lazy(video):
    """Nothing is decoded until iteration, and only one frame is held at a time."""
    path, _ = video
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2)
    derived = probe_video(config)

    stream = stream_frames(config, derived)
    first = next(stream)

    assert first.nbytes < 1024 * 1024
    assert next(stream) is not first


def test_stream_frames_survives_bad_frame_count(video, monkeypatch):
    """CAP_PROP_FRAME_COUNT is a container guess; the generator must not trust it."""
    import cv2

    path, frames = video
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2)
    derived = probe_video(config)

    real_get = cv2.VideoCapture.get
    monkeypatch.setattr(
        cv2.VideoCapture, "get",
        lambda self, prop: 9999.0 if prop == cv2.CAP_PROP_FRAME_COUNT else real_get(self, prop),
    )

    assert len(list(stream_frames(config, derived))) == len(frames)


def test_missing_video_raises(tmp_path):
    config = UserConfig(input_dir=str(tmp_path / "nope.mkv"), output_dir="")
    with pytest.raises(ValueError):
        probe_video(config)
