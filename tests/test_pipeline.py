"""1.3: each pipeline stage standing on its own.

The point of the split is that a stage can be driven with hand-made inputs and
checked without running the ones around it, so that is how these test them —
`build_mosaics` off a plain list of frames, `encode_video` off a plain list of
mosaics, `combine_videos` off two files it did not produce.
"""

import shutil
import subprocess

import numpy as np
import pytest


import bad_apple.video as video_io

from conftest import (CELL, make_frames, probe_stream, read_video,
                      write_rated_video, write_video)
from main import (BrightnessMetric, CifarGallery, DerivedConfig, UserConfig,
                  VideoGallery,
                  build_metric, build_mosaics, combine_videos, encode_video,
                  gallery_brightness, main, mosaic_frame, probe_video)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg not on PATH")


def make_metric(cell_gallery, candidates: int = 4, epsilon: float = 0.1):
    metric = BrightnessMetric(candidates=candidates, epsilon=epsilon, seed=0)
    metric.precompute(cell_gallery, CELL, gallery_brightness(cell_gallery))
    return metric


def test_build_metric_trims_the_gallery(cell_gallery):
    """contrast < 1 keeps only a percentile band, so fewer tiles survive."""
    derived = _derived_for(CELL)
    wide = build_metric(cell_gallery, _config(contrast=1.0), derived)
    narrow = build_metric(cell_gallery, _config(contrast=0.2), derived)

    assert len(narrow.tiles) < len(wide.tiles)
    # The trimmed band is centred, so its tiles avoid both extremes.
    assert narrow.tiles.mean() == pytest.approx(cell_gallery.mean(), abs=8)


@pytest.mark.parametrize("metric", ["brightness", "colour"])
def test_a_gallery_trimmed_to_nothing_says_so(metric):
    """Four tiles at the default contrast=0.1 leave none in the band. That has
    to be a plain error, not an IndexError from deep in the brightness lookup."""
    tiles = np.stack([np.full((CELL[1], CELL[0], 3), v, dtype=np.uint8)
                      for v in (10, 80, 160, 240)])

    with pytest.raises(ValueError, match="empty"):
        build_metric(tiles, _config(metric=metric), _derived_for(CELL))


def test_build_mosaics_matches_frame_by_frame(cell_gallery):
    """The stream is exactly `mosaic_frame` applied in order, nothing more."""
    frames = list(make_frames(5, CELL[0] * 4, CELL[1] * 3, seed=7))

    # One metric per side, not one per frame: sampling advances the RNG, so the
    # two runs only agree if they see the same frames in the same order.
    reference = make_metric(cell_gallery)
    expected = [mosaic_frame(f, reference) for f in frames]
    got = list(build_mosaics(iter(frames), make_metric(cell_gallery),
                             _derived_for(CELL)))

    assert len(got) == len(frames)
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)


def test_build_mosaics_is_lazy(cell_gallery):
    """Nothing is decoded or matched until the consumer asks for a frame."""
    consumed = []

    def frames():
        for frame in make_frames(4, CELL[0] * 2, CELL[1] * 2, seed=1):
            consumed.append(frame)
            yield frame

    stream = build_mosaics(frames(), make_metric(cell_gallery), _derived_for(CELL))
    assert consumed == []
    next(stream)
    assert len(consumed) == 1


@needs_ffmpeg
def test_encode_video_writes_every_frame(tmp_path, cell_gallery):
    derived = _derived_for(CELL, grid=(4, 3), fps=10)
    metric = make_metric(cell_gallery)
    width, height = derived.target_dimensions
    mosaics = [mosaic_frame(f, metric)
               for f in make_frames(6, width, height, seed=3)]

    out = encode_video(iter(mosaics), derived, tmp_path / "mosaic.mp4")

    assert out.exists()
    decoded = read_video(out)
    assert len(decoded) == len(mosaics)
    assert decoded.shape[1:3] == (height, width)


@needs_ffmpeg
def test_encode_video_raises_when_ffmpeg_fails(tmp_path, cell_gallery):
    derived = _derived_for(CELL)
    metric = make_metric(cell_gallery)
    width, height = derived.target_dimensions
    mosaics = [mosaic_frame(f, metric) for f in make_frames(2, width, height)]

    # ffmpeg cannot infer an output muxer from this suffix.
    output = tmp_path / "out.invalid"
    output.write_bytes(b"old complete output")
    with pytest.raises(RuntimeError, match="ffmpeg encode failed"):
        encode_video(iter(mosaics), derived, output)
    assert output.read_bytes() == b"old complete output"
    assert list(tmp_path.glob(".out.*.invalid")) == []


@needs_ffmpeg
def test_encode_video_iterator_error_keeps_old_output_and_closes_iterator(
        tmp_path, cell_gallery):
    derived = _derived_for(CELL)
    metric = make_metric(cell_gallery)
    width, height = derived.target_dimensions
    frame = mosaic_frame(make_frames(1, width, height)[0], metric)
    closed = []
    output = tmp_path / "mosaic.mp4"
    output.write_bytes(b"old complete output")

    def mosaics():
        try:
            yield frame
            raise LookupError("mosaic generation failed")
        finally:
            closed.append(True)

    with pytest.raises(LookupError, match="mosaic generation failed"):
        encode_video(mosaics(), derived, output)

    assert closed == [True]
    assert output.read_bytes() == b"old complete output"
    assert list(tmp_path.glob(".mosaic.*.mp4")) == []


def test_encode_video_closes_iterator_if_output_setup_fails(tmp_path):
    derived = _derived_for(CELL)
    parent = tmp_path / "not-a-directory"
    parent.write_text("block mkdir")
    closed = []
    frame = np.zeros((*derived.target_dimensions[::-1], 3), dtype=np.uint8)

    def mosaics():
        try:
            yield frame
            yield frame
        finally:
            closed.append(True)

    with pytest.raises(FileExistsError):
        encode_video(mosaics(), derived, parent / "mosaic.mp4")

    assert closed == [True]


def test_encode_video_reaps_ffmpeg_after_broken_pipe_on_close(
        tmp_path, monkeypatch):
    derived = _derived_for(CELL)
    waited = []

    class BrokenPipe:
        def write(self, _data):
            raise BrokenPipeError("write failed")

        def close(self):
            raise BrokenPipeError("close failed")

    class Process:
        stdin = BrokenPipe()
        returncode = None

        def wait(self):
            waited.append(True)
            self.returncode = 1
            return 1

        def poll(self):
            return self.returncode

    process = Process()
    monkeypatch.setattr(video_io.subprocess, "Popen", lambda *args, **kwargs: process)
    output = tmp_path / "mosaic.mp4"
    output.write_bytes(b"old complete output")
    frame = np.zeros((*derived.target_dimensions[::-1], 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="ffmpeg encode failed"):
        encode_video(iter([frame]), derived, output)

    assert waited == [True]
    assert output.read_bytes() == b"old complete output"


@needs_ffmpeg
def test_combine_videos_stacks_side_by_side(tmp_path, video_factory):
    """The source is scaled to the mosaic's size, so mismatched inputs stack.

    This is the 0.5 fix under test: the two inputs differ in size here, which
    `hstack` alone would reject outright.
    """
    source, _ = video_factory(count=5, width=64, height=48)
    mosaic_path = tmp_path / "mosaic.mkv"
    write_video(mosaic_path, make_frames(5, 32, 24, seed=5))

    out = combine_videos(source, mosaic_path, tmp_path / "combined.mkv", (32, 24))

    decoded = read_video(out)
    assert len(decoded) == 5
    assert decoded.shape[1:3] == (24, 64)  # one 32x24 pane beside the other


@needs_ffmpeg
def test_combine_videos_raises_when_ffmpeg_fails(tmp_path, video_factory):
    source, _ = video_factory(count=2, width=64, height=48)
    mosaic_path = tmp_path / "mosaic.mkv"
    write_video(mosaic_path, make_frames(2, 32, 24, seed=5))

    output = tmp_path / "combined.invalid"
    output.write_bytes(b"old complete output")
    with pytest.raises(RuntimeError, match="ffmpeg combine failed"):
        combine_videos(source, mosaic_path, output, (32, 24))
    assert output.read_bytes() == b"old complete output"
    assert list(tmp_path.glob(".combined.*.invalid")) == []


@needs_ffmpeg
def test_combine_videos_refuses_to_overwrite_input(tmp_path, video_factory):
    source, _ = video_factory(count=2, width=64, height=48)
    mosaic_path = tmp_path / "mosaic.mkv"
    write_video(mosaic_path, make_frames(2, 32, 24, seed=5))
    source_before = source.read_bytes()

    with pytest.raises(ValueError, match="would overwrite an input"):
        combine_videos(source, mosaic_path, source, (32, 24))

    assert source.read_bytes() == source_before


@needs_ffmpeg
def test_main_orchestrates_end_to_end(tmp_path, video, cifar_pickle, monkeypatch):
    """main() produces both videos from a source and a GallerySource alone."""
    monkeypatch.chdir(tmp_path)  # keep the gallery cache out of the repo
    path, frames = video
    config = UserConfig(input_dir=str(path), output_dir=str(tmp_path / "out"),
                        grid_size=2, contrast=1.0, candidates=8, epsilon=0.1)

    combined = main(CifarGallery(cifar_pickle[0]), config)

    derived = probe_video(config)
    width, height = derived.target_dimensions
    assert len(read_video(tmp_path / "out" / "output.mp4")) == len(frames)

    decoded = read_video(combined)
    # Both panes use the same frame clock, even across different containers.
    assert len(decoded) == len(frames)
    assert decoded.shape[1:3] == (height, width * 2)


@needs_ffmpeg
def test_main_encodes_1080p_against_widescreen_tiles(tmp_path, video_factory,
                                                    monkeypatch):
    """1080p at grid_size=8 against 16:9 tiles once sized to 1917x1080, and
    libx264 won't open on an odd width. Real 1080p, since the odd side only
    turns up at real resolutions."""
    monkeypatch.chdir(tmp_path)  # keep the gallery cache out of the repo
    source, frames = video_factory(count=3, width=1920, height=1080)
    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()
    write_video(gallery_dir / "ep1.mkv", make_frames(20, 64, 36, seed=3))
    config = UserConfig(input_dir=str(source), output_dir=str(tmp_path / "out"),
                        grid_size=8)

    combined = main(VideoGallery(gallery_dir, stride=1), config)

    assert read_video(tmp_path / "out" / "output.mp4").shape == (3, 1080, 1944, 3)
    assert read_video(combined).shape[1:3] == (1080, 1944 * 2)


@needs_ffmpeg
def test_main_under_stretch_keeps_the_square_grid(tmp_path, video, video_factory,
                                                  monkeypatch):
    """Whether the gallery's shape reaches the probe is `main()`'s call, so it
    has to run to be tested. 16:9 tiles would widen this 4:3 grid under
    `native`; `stretch` has to leave it exactly as it was."""
    monkeypatch.chdir(tmp_path)  # keep the gallery cache out of the repo
    source, _ = video
    gallery_path, _ = video_factory(count=4, width=96, height=54, seed=3)
    gallery = VideoGallery(gallery_path, stride=1)
    config = UserConfig(input_dir=str(source), output_dir=str(tmp_path / "out"),
                        grid_size=2, contrast=1.0, tile_fit="stretch")
    square = probe_video(config).target_dimensions
    # The control: this gallery's shape would change the grid if it got through.
    assert probe_video(config, gallery.native_aspect).target_dimensions != square

    main(gallery, config)

    height, width = read_video(tmp_path / "out" / "output.mp4").shape[1:3]
    assert (width, height) == square


@needs_ffmpeg
def test_main_keeps_an_ntsc_rate_through_to_the_output(tmp_path, cifar_pickle,
                                                       monkeypatch):
    """Rounded to 30, 60 frames of 29.97 came out 2.000s against 2.002s of
    source and audio, and the gap grows a frame every 33s from there."""
    monkeypatch.chdir(tmp_path)  # keep the gallery cache out of the repo
    source = tmp_path / "ntsc.mp4"
    write_rated_video(source, "30000/1001", count=60)
    config = UserConfig(input_dir=str(source), output_dir=str(tmp_path / "out"),
                        grid_size=2, contrast=1.0)

    combined = main(CifarGallery(cifar_pickle[0]), config)

    original = probe_stream(source)
    for encoded in (tmp_path / "out" / "output.mp4", combined):
        video = probe_stream(encoded)
        assert video["r_frame_rate"] == "30000/1001"
        assert video["nb_frames"] == original["nb_frames"] == "60"
        assert video["duration"] == original["duration"]
    # The audio is the source's, so it has to still line up with the picture.
    assert float(probe_stream(combined, "a:0")["duration"]) == pytest.approx(
        float(original["duration"]), abs=0.05)


def _config(**kwargs) -> UserConfig:
    return UserConfig(input_dir="", output_dir="", **kwargs)


def _derived_for(cell, grid=(4, 4), fps=30) -> DerivedConfig:
    """A DerivedConfig for a source that would produce this cell and grid.

    Built directly rather than through `probe_video`, so a stage can be tested
    without a video file behind it.
    """
    dimensions = (grid[0] * cell[0], grid[1] * cell[1])
    return DerivedConfig(src_fps=fps, src_dimensions=dimensions,
                         src_frame_count=0, aspect_ratio=grid,
                         grid=grid, cell_size=cell)


@pytest.mark.parametrize("name", ["output.mp4", "combined.mp4"])
def test_pipeline_refuses_to_overwrite_source(tmp_path, name):
    source = tmp_path / name
    source.write_bytes(b"original input")
    config = UserConfig(input_dir=str(source), output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="overwrite an input"):
        main(CifarGallery(tmp_path / "train"), config)
    assert source.read_bytes() == b"original input"


def test_pipeline_refuses_to_overwrite_gallery(tmp_path, video):
    from main import VideoGallery
    path, _ = video
    gallery_path = tmp_path / "output.mp4"
    gallery_path.write_bytes(b"original gallery")
    config = UserConfig(input_dir=str(path), output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="overwrite an input"):
        main(VideoGallery(gallery_path), config)
    assert gallery_path.read_bytes() == b"original gallery"
