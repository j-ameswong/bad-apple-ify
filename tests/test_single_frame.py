"""2.4: grid_size=1 as single-frame mode.

No separate code path — the grid collapses to 1x1 and the existing match /
mosaic / encode stages carry it, so what these test is that the *output* is a
whole gallery image per source frame rather than a composite of tiles.
"""

import shutil

import numpy as np
import pytest

from conftest import read_video
from main import (CifarGallery, UserConfig, build_metric, build_mosaics,
                  load_gallery, main, mosaic_frame, probe_video,
                  resize_gallery_to_cells, stream_frames)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg not on PATH")


def single_frame_setup(video, gallery, tmp_path):
    """Probe a source at grid_size=1 and build a metric over the gallery."""
    path, frames = video
    config = UserConfig(input_dir=str(path), output_dir=str(tmp_path / "out"),
                        grid_size=1, contrast=1.0, candidates=8, epsilon=0.5)
    derived = probe_video(config)
    tiles = load_gallery(_source(gallery), derived, use_cache=False)
    return config, derived, build_metric(tiles, config, derived), frames


class _source:
    """A `GallerySource` over an in-memory array, resized on load like any other."""

    def __init__(self, gallery: np.ndarray):
        self._gallery = gallery

    @property
    def fingerprint(self) -> str:
        return "test-array"

    def estimate_count(self) -> int:
        return len(self._gallery)

    def load(self, cell_size):
        return resize_gallery_to_cells(self._gallery, cell_size)


def test_every_output_frame_is_one_whole_gallery_image(video, gallery, tmp_path):
    config, derived, metric, frames = single_frame_setup(video, gallery, tmp_path)

    mosaics = list(build_mosaics(stream_frames(config, derived), metric, derived))

    assert len(mosaics) == len(frames)
    width, height = derived.target_dimensions
    for mosaic in mosaics:
        assert mosaic.shape == (height, width, 3)
        # Not a composite: the frame is byte-identical to a single tile.
        assert any(np.array_equal(mosaic, tile) for tile in metric.tiles)


def test_single_frame_tracks_source_brightness(video, gallery, tmp_path):
    """The one cell still matches — a dark frame picks a darker image than a bright one."""
    config, derived, metric, _ = single_frame_setup(video, gallery, tmp_path)
    width, height = derived.target_dimensions
    dark = np.full((height, width, 3), 30, dtype=np.uint8)
    bright = np.full((height, width, 3), 220, dtype=np.uint8)

    assert mosaic_frame(dark, metric).mean() < mosaic_frame(bright, metric).mean()


def test_single_frame_mode_uses_the_whole_gallery(video, gallery, tmp_path):
    """Distinct source frames still draw distinct images, not one repeated tile."""
    config, derived, metric, frames = single_frame_setup(video, gallery, tmp_path)

    mosaics = list(build_mosaics(stream_frames(config, derived), metric, derived))
    distinct = {mosaic.tobytes() for mosaic in mosaics}

    assert len(distinct) > 1


@needs_ffmpeg
def test_main_at_grid_size_one_keeps_the_frame_count(tmp_path, video, cifar_pickle,
                                                     monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep the gallery cache out of the repo
    path, frames = video
    config = UserConfig(input_dir=str(path), output_dir=str(tmp_path / "out"),
                        grid_size=1, contrast=1.0, candidates=8, epsilon=0.5)

    main(CifarGallery(cifar_pickle[0]), config)

    decoded = read_video(tmp_path / "out" / "output.mp4")
    assert len(decoded) == len(frames)
    width, height = probe_video(config).target_dimensions
    assert decoded.shape[1:3] == (height, width)
