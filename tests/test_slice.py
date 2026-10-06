"""2.7: a source slice keeps the full run's frames, tile choices and audio."""

from dataclasses import replace
from fractions import Fraction
import shutil
import subprocess

import numpy as np
import pytest

import bad_apple.video as video_io
from conftest import probe_stream, read_video
from main import (CifarGallery, DerivedConfig, UserConfig, build_metric,
                  build_mosaics, encode_video, main, parse_config, probe_video,
                  stream_frames)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg not on PATH")


@pytest.mark.parametrize("field, value", [
    ("start", -1), ("start", float("inf")), ("start", float("nan")),
    ("duration", 0), ("duration", -1), ("duration", float("inf")),
    ("duration", float("nan")),
])
def test_invalid_times_are_rejected_by_config_and_cli(field, value, capsys):
    with pytest.raises(ValueError, match=field):
        UserConfig(input_dir="", output_dir="", **{field: value})
    with pytest.raises(SystemExit) as error:
        parse_config([f"--{field}", str(value)])
    assert error.value.code == 2
    assert field in capsys.readouterr().err


def test_cli_slice_flags_and_defaults():
    default = parse_config([])
    assert default.start == 0
    assert default.duration is None
    assert parse_config(["--start", "60", "--duration", "10"]) == replace(
        default, start=60, duration=10)


@pytest.mark.parametrize("rate, start, duration, first, stop", [
    (Fraction(30), 60, 10, 1800, 2100),
    (Fraction(30), 0.1, 0.2, 3, 9),
    (Fraction(30), 0.01, 0.1, 1, 4),
    (Fraction(30000, 1001), 60, 10, 1799, 2098),
    (Fraction(30000, 1001), 0, 10, 0, 300),
])
def test_frame_boundaries_use_the_exact_rate(rate, start, duration, first, stop):
    config = UserConfig(input_dir="", output_dir="", start=start, duration=duration)
    derived = DerivedConfig.from_source(config, fps=rate, dimensions=(64, 48),
                                        frame_count=9999)
    assert (derived.start_frame, derived.stop_frame) == (first, stop)
    assert derived.output_frame_count == stop - first


def test_interval_between_two_frames_is_empty():
    config = UserConfig(input_dir="", output_dir="", start=0.01, duration=0.001)
    with pytest.raises(ValueError, match="contains no frames"):
        DerivedConfig.from_source(config, fps=Fraction(30), dimensions=(64, 48),
                                  frame_count=10)


@pytest.mark.parametrize("start, duration, first, stop", [
    (0, 0.1, 0, 3), (0.1, None, 3, 10), (0.1, 0.1, 3, 6),
    (0.1, 10, 3, 10), (10, None, 10, 10),
])
def test_stream_selects_the_slice_until_real_eof(video, start, duration, first, stop):
    path, frames = video
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2,
                        start=start, duration=duration)
    derived = probe_video(config)
    selected = list(stream_frames(config, derived))
    assert len(selected) == stop - first
    for got, expected in zip(selected, frames[first:stop]):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("count", [0, 1, 99999])
def test_bad_metadata_never_limits_the_slice(video, count):
    path, frames = video
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2,
                        start=0.1, duration=10)
    derived = replace(probe_video(config), src_frame_count=count)
    np.testing.assert_array_equal(list(stream_frames(config, derived)), frames[3:])
    if count == 0:
        assert derived.output_frame_count == 300


def test_stream_stops_reading_and_releases_capture(video, monkeypatch):
    path, _ = video
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2,
                        start=0.1, duration=0.1)
    derived = probe_video(config)
    real_capture = video_io.cv2.VideoCapture
    calls = []

    class Capture:
        def __init__(self, name):
            self.cap = real_capture(name)

        def isOpened(self):
            return self.cap.isOpened()

        def grab(self):
            calls.append("grab")
            return self.cap.grab()

        def read(self):
            calls.append("read")
            return self.cap.read()

        def release(self):
            calls.append("release")
            self.cap.release()

    monkeypatch.setattr(video_io.cv2, "VideoCapture", Capture)
    assert len(list(stream_frames(config, derived))) == 3
    assert calls == ["grab"] * 3 + ["read"] * 3 + ["release"]

    calls.clear()
    stream = stream_frames(config, derived)
    next(stream)
    stream.close()
    assert calls == ["grab"] * 3 + ["read", "release"]


@needs_ffmpeg
def test_sixty_second_start_produces_the_full_runs_next_300_frames(
        tmp_path, video_factory, cifar_pickle, monkeypatch):
    """Compare raw mosaics before lossy encoding, then count both encoded outputs."""
    monkeypatch.chdir(tmp_path)
    path, source_frames = video_factory(count=2110, width=32, height=24)
    gallery = CifarGallery(cifar_pickle[0])
    config = UserConfig(input_dir=str(path), output_dir=str(tmp_path / "out"),
                        grid_size=2, contrast=1.0, start=60, duration=10)
    full_config = replace(config, start=0, duration=None)
    full_derived = probe_video(full_config, gallery.native_aspect)
    metric = build_metric(gallery.load(full_derived.cell_size), full_config, full_derived)
    expected = list(build_mosaics(iter(source_frames), metric, full_derived))[1800:2100]
    captured = []

    def record_encode(mosaics, derived, output_path):
        def record():
            for mosaic in mosaics:
                captured.append(mosaic)
                yield mosaic
        return encode_video(record(), derived, output_path)

    monkeypatch.setattr(video_io, "encode_video", record_encode)
    combined = main(gallery, config)

    np.testing.assert_array_equal(captured, expected)
    np.testing.assert_array_equal(list(stream_frames(config, probe_video(config))),
                                  source_frames[1800:2100])
    for output in (tmp_path / "out" / "output.mp4", combined):
        assert probe_stream(output)["nb_frames"] == "300"
        assert float(probe_stream(output)["duration"]) == pytest.approx(10)


@pytest.mark.parametrize("metric", ["brightness", "colour"])
@pytest.mark.parametrize("hold_tiles", [False, True])
def test_prefix_replays_seeded_choices(video_factory, cifar_pickle, metric, hold_tiles):
    path, frames = video_factory(count=30)
    config = UserConfig(input_dir=str(path), output_dir="", grid_size=2,
                        contrast=1.0, metric=metric, hold_tiles=hold_tiles,
                        candidates=16, epsilon=0.5, colour_bins=8)
    derived = probe_video(config)
    gallery = CifarGallery(cifar_pickle[0]).load(derived.cell_size)
    expected = list(build_mosaics(iter(frames), build_metric(gallery, config, derived),
                                  derived))[9:15]
    sliced = replace(config, start=0.3, duration=0.2)
    selection = probe_video(sliced)
    actual = list(build_mosaics(
        stream_frames(sliced, selection, include_prefix=True),
        build_metric(gallery, sliced, selection), selection,
        warmup_frames=selection.start_frame))
    np.testing.assert_array_equal(actual, expected)


def test_empty_slice_does_not_launch_an_encoder(tmp_path, video, monkeypatch):
    path, _ = video
    config = UserConfig(input_dir=str(path), output_dir="", start=10)
    derived = probe_video(config)

    def unexpected_encoder(*args, **kwargs):
        pytest.fail("ffmpeg must not run for an empty slice")

    monkeypatch.setattr(video_io.subprocess, "Popen", unexpected_encoder)
    with pytest.raises(ValueError, match="contains no frames"):
        encode_video(stream_frames(config, derived), derived, tmp_path / "out.mp4")
    assert not (tmp_path / "out.mp4").exists()


def audio_samples(path):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
        check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


@needs_ffmpeg
@pytest.mark.parametrize("rate", ["30", "30000/1001"])
@pytest.mark.parametrize("duration", [0.5, None, 10])
def test_combined_picture_and_audio_share_the_slice(
        tmp_path, cifar_pickle, monkeypatch, rate, duration):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.mp4"
    source_duration = float(90 / Fraction(rate))
    # A chirp changes pitch continuously, so untrimmed audio cannot pass.
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"testsrc=size=64x48:rate={rate}", "-f", "lavfi", "-i",
        "aevalsrc=sin(2*PI*(220*t+110*t*t)):s=48000",
        "-frames:v", "90", "-t", str(source_duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)
    ], check=True)
    config = UserConfig(input_dir=str(source), output_dir=str(tmp_path / "out"),
                        grid_size=2, contrast=1.0, candidates=1,
                        start=1, duration=duration)
    combined = main(CifarGallery(cifar_pickle[0]), config)
    derived = probe_video(config)
    expected_frames = read_video(source)[derived.start_frame:derived.stop_frame]
    actual_frames = read_video(combined)
    assert len(actual_frames) == len(expected_frames)
    # The left pane is re-encoded, so compare with a small compression tolerance.
    left = actual_frames[:, :, :64].astype(float)
    assert np.abs(left - expected_frames.astype(float)).mean() < 3

    expected_duration = float(len(expected_frames) / Fraction(rate))
    assert probe_stream(combined)["r_frame_rate"] == rate + ("/1" if rate == "30" else "")
    assert float(probe_stream(combined)["duration"]) == pytest.approx(expected_duration, abs=1e-5)
    assert float(probe_stream(combined, "a:0")["duration"]) == pytest.approx(expected_duration, abs=0.025)

    first_sample = round(float(derived.start_frame / derived.src_fps) * 48000)
    count = round(expected_duration * 48000)
    expected_audio = audio_samples(source)[first_sample:first_sample + count]
    actual_audio = audio_samples(combined)[:len(expected_audio)]
    assert len(actual_audio) == len(expected_audio)
    # Ignore the encoder's edge padding; the middle must be the same audio.
    assert np.corrcoef(expected_audio[1024:-1024], actual_audio[1024:-1024])[0, 1] > 0.98
