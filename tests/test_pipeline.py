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

from conftest import CELL, make_frames, read_video, write_video
from main import (BrightnessMetric, CifarGallery, DerivedConfig, UserConfig,
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

    # A directory that does not exist: ffmpeg cannot open the output.
    with pytest.raises(RuntimeError, match="ffmpeg encode failed"):
        encode_video(iter(mosaics), derived, tmp_path / "nope" / "out.mp4")


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
    # hstack holds the last frame of whichever input runs out first, so the
    # combined length is only bounded below by the source's — the exact count
    # depends on how the two containers' timestamps line up.
    assert len(decoded) >= len(frames)
    assert decoded.shape[1:3] == (height, width * 2)


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
