import shutil
from fractions import Fraction

import numpy as np
import pytest

from conftest import write_rated_video
import bad_apple.video as video_io
from main import UserConfig, probe_video, stream_frames

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg not on PATH")


def test_probe_video_reads_metadata(video):
    path, frames = video
    derived = probe_video(UserConfig(input_dir=str(path), output_dir=""))

    assert derived.src_fps == 30
    assert derived.output_fps == 30
    assert derived.src_dimensions == (64, 48)
    assert derived.src_frame_count == len(frames)


@needs_ffmpeg
@pytest.mark.parametrize("rate, expected", [
    ("30000/1001", Fraction(30000, 1001)),
    ("24000/1001", Fraction(24000, 1001)),
    ("60000/1001", Fraction(60000, 1001)),
    ("2997/100", Fraction(2997, 100)),
    ("25", Fraction(25)),
])
def test_probe_video_keeps_fractional_rates(tmp_path, rate, expected):
    """29.97 rounded to 30 runs the mosaic 0.1% fast: a frame adrift every 33s."""
    path = tmp_path / "ntsc.mp4"
    write_rated_video(path, rate, count=10, audio=False)

    derived = probe_video(UserConfig(input_dir=str(path), output_dir=""))

    assert derived.src_fps == expected
    assert derived.output_fps == expected


def test_stream_frames_preserves_content(video):
    """Target dimensions equal the source here, so frames must survive intact."""
    path, frames = video
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2)
    derived = probe_video(config)
    assert derived.target_dimensions == derived.src_dimensions

    np.testing.assert_array_equal(np.array(list(stream_frames(config, derived))),
                                  frames)


def test_stream_frames_reads_only_as_far_as_asked(video, monkeypatch):
    """Two frames asked for, two decoded: nothing is read ahead of the consumer,
    so a feature-length source never sits in RAM."""
    path, _ = video
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2)
    derived = probe_video(config)
    reads = []
    real_capture = video_io.cv2.VideoCapture

    class CountingCapture:
        def __init__(self, name):
            self._cap = real_capture(name)

        def __getattr__(self, attr):
            return getattr(self._cap, attr)

        def read(self):
            reads.append(True)
            return self._cap.read()

    monkeypatch.setattr(video_io.cv2, "VideoCapture", CountingCapture)
    stream = stream_frames(config, derived)

    assert reads == []
    next(stream)
    next(stream)
    assert len(reads) == 2


def test_stream_frames_survives_bad_frame_count(video, monkeypatch):
    """CAP_PROP_FRAME_COUNT is a container guess; the generator must not trust it."""
    import cv2

    path, frames = video
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2)
    derived = probe_video(config)

    real_get = video_io.cv2.VideoCapture.get
    monkeypatch.setattr(
        video_io.cv2.VideoCapture, "get",
        lambda self, prop: 9999.0 if prop == cv2.CAP_PROP_FRAME_COUNT else real_get(self, prop),
    )

    assert len(list(stream_frames(config, derived))) == len(frames)


def test_missing_video_raises(tmp_path):
    config = UserConfig(input_dir=str(tmp_path / "nope.mkv"), output_dir="")
    with pytest.raises(ValueError):
        probe_video(config)


def test_probe_video_releases_capture_when_metadata_is_invalid(video, monkeypatch):
    import cv2

    path, _ = video
    real_capture = cv2.VideoCapture
    released = []

    class InvalidRateCapture:
        def __init__(self, name):
            self._capture = real_capture(name)

        def isOpened(self):
            return self._capture.isOpened()

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return float("nan")
            return self._capture.get(prop)

        def release(self):
            released.append(True)
            self._capture.release()

    monkeypatch.setattr(video_io.cv2, "VideoCapture", InvalidRateCapture)
    with pytest.raises(ValueError, match="invalid frame rate"):
        probe_video(UserConfig(input_dir=str(path), output_dir=""))
    assert released == [True]
