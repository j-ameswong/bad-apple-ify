"""2.1: tiles decoded from video.

The fixture videos are FFV1, so a frame decodes back to exactly the bytes that
went in and these can assert on pixels rather than tolerances. Nothing here
asserts against `CAP_PROP_FRAME_COUNT` — that metadata is a hint, and the point
of the tests is what `load()` actually kept.
"""

import shutil

import numpy as np
import pytest

from conftest import CELL, read_video, write_video
from main import (UserConfig, VideoGallery, crop_to_aspect, fit_to_cell,
                  main, probe_video, resize_gallery_to_cells)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg not on PATH")


def test_stride_keeps_every_nth_frame(video_factory):
    path, frames = video_factory(count=12)

    tiles = VideoGallery(path, stride=3).load(CELL, fit="stretch")

    np.testing.assert_array_equal(
        tiles, resize_gallery_to_cells(frames[::3], CELL, "stretch"))


def test_stride_one_keeps_everything(video_factory):
    path, frames = video_factory(count=6)

    assert len(VideoGallery(path, stride=1).load(CELL)) == len(frames)


def test_tiles_arrive_at_cell_size(video_factory):
    """Full-resolution frames are never stored, which is the whole protocol."""
    path, _ = video_factory(count=6, width=128, height=96)
    cell = (5, 3)

    tiles = VideoGallery(path, stride=2).load(cell)

    assert tiles.shape[1:] == (cell[1], cell[0], 3)
    assert tiles.dtype == np.uint8


def test_held_frames_are_deduped(tmp_path, video_factory):
    """Animation holds cels for several ticks; those are one tile, not four."""
    _, frames = video_factory(count=4)
    held = np.repeat(frames, 3, axis=0)  # every frame held for three ticks
    path = tmp_path / "held.mkv"
    write_video(path, held)

    tiles = VideoGallery(path, stride=1).load(CELL, fit="stretch")

    np.testing.assert_array_equal(
        tiles, resize_gallery_to_cells(frames, CELL, "stretch"))


def test_dedupe_is_on_the_downscaled_tile(tmp_path, video_factory):
    """Frames that differ only below cell resolution collapse to one tile."""
    _, frames = video_factory(count=4, width=64, height=48)
    twin = frames[1].copy()
    twin[0, 0] = 255 - twin[0, 0]  # one pixel, gone by the time it is 4x4
    path = tmp_path / "twins.mkv"
    write_video(path, np.stack([frames[1], twin]))

    assert len(VideoGallery(path, stride=1).load(CELL)) == 1


def test_a_directory_is_one_gallery(tmp_path, video_factory):
    """A season is one argument, read in a stable order."""
    gallery_dir = tmp_path / "season"
    gallery_dir.mkdir()
    episodes = []
    for i in range(3):
        _, frames = video_factory(count=4, seed=i + 1)
        write_video(gallery_dir / f"ep{i:02d}.mkv", frames)
        episodes.append(frames)

    tiles = VideoGallery(gallery_dir, stride=2).load(CELL, fit="stretch")

    expected = np.concatenate([frames[::2] for frames in episodes])
    np.testing.assert_array_equal(
        tiles, resize_gallery_to_cells(expected, CELL, "stretch"))


def test_appledouble_stubs_are_skipped(tmp_path, video_factory):
    """`._name.mkv` next to `name.mkv` is a resource fork, not a video."""
    gallery_dir = tmp_path / "season"
    gallery_dir.mkdir()
    _, frames = video_factory(count=4)
    write_video(gallery_dir / "ep01.mkv", frames)
    (gallery_dir / "._ep01.mkv").write_bytes(b"\x00\x05\x16\x07not a video")

    assert len(VideoGallery(gallery_dir, stride=1).load(CELL)) == len(frames)


def test_missing_video_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        VideoGallery(tmp_path / "nothing.mkv").load(CELL)


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        VideoGallery(tmp_path).load(CELL)


def test_native_aspect_is_the_frame_shape(video_factory):
    path, _ = video_factory(count=2, width=64, height=48)

    assert VideoGallery(path).native_aspect == (4, 3)


def test_native_aspect_of_nothing_is_none(tmp_path):
    assert VideoGallery(tmp_path).native_aspect is None


def test_estimate_is_an_upper_bound(tmp_path, video_factory):
    """Dedupe only ever lowers the real count, so the estimate stays above it."""
    _, frames = video_factory(count=6)
    path = tmp_path / "held.mkv"
    write_video(path, np.repeat(frames, 2, axis=0))
    gallery = VideoGallery(path, stride=2)

    assert gallery.estimate_count() >= len(gallery.load(CELL))


def test_fingerprint_tracks_stride_and_files(tmp_path, video_factory):
    path, _ = video_factory(count=2)
    other, _ = video_factory(count=2, seed=9)

    assert VideoGallery(path, 10).fingerprint != VideoGallery(path, 5).fingerprint
    assert VideoGallery(path, 10).fingerprint != VideoGallery(other, 10).fingerprint


# --- fitting a 16:9 frame into a cell that isn't 16:9 -----------------------


def test_crop_takes_the_middle_at_the_cells_ratio(video_factory):
    _, frames = video_factory(count=4, width=64, height=48)

    cropped = crop_to_aspect(frames[1], (4, 4))

    assert cropped.shape == (48, 48, 3)
    np.testing.assert_array_equal(cropped, frames[1][:, 8:56])


def test_crop_of_a_matching_ratio_is_the_whole_image(video_factory):
    _, frames = video_factory(count=4, width=64, height=48)

    np.testing.assert_array_equal(crop_to_aspect(frames[1], (8, 6)), frames[1])


def test_stretch_uses_the_whole_frame(video_factory):
    """The two fits differ, and only `stretch` sees the edges of the frame."""
    _, frames = video_factory(count=4, width=64, height=48)

    crop = fit_to_cell(frames[1], (4, 4), "crop")
    stretch = fit_to_cell(frames[1], (4, 4), "stretch")

    assert not np.array_equal(crop, stretch)
    np.testing.assert_array_equal(stretch, fit_to_cell(frames[1], (4, 4), "stretch"))


def test_native_fit_is_a_crop(video_factory):
    """`native` only reaches a mismatched cell in single-frame mode; crop there."""
    _, frames = video_factory(count=4, width=64, height=48)

    np.testing.assert_array_equal(fit_to_cell(frames[1], (4, 4), "native"),
                                  fit_to_cell(frames[1], (4, 4), "crop"))


def test_a_big_downscale_averages_rather_than_samples():
    """A bright patch has to survive 1080p -> 14x8, or its tile lands in the
    wrong brightness bucket.

    Bilinear reads a 2x2 neighbourhood, so at a 130x reduction it misses the
    patch entirely and the tile comes back pure black.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[500:560, 900:1000] = 255
    fraction = (60 * 100) / (1080 * 1920)

    tile = fit_to_cell(frame, (14, 8), "stretch")

    assert tile.mean() / 255 == pytest.approx(fraction, rel=0.3)


def test_upscaling_still_interpolates():
    """AREA is nearest-neighbour going up, so single-frame mode keeps bilinear."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:2] = 255

    blown_up = fit_to_cell(image, (64, 64), "stretch")

    # Nearest would give two flat bands and nothing in between.
    assert len(np.unique(blown_up[:, 0, 0])) > 2


# --- through the pipeline ---------------------------------------------------


@needs_ffmpeg
def test_a_16_9_gallery_over_a_4_3_source(tmp_path, video, video_factory,
                                          monkeypatch):
    """End to end on the shape that motivated `native`: wide tiles, square-ish frame."""
    monkeypatch.chdir(tmp_path)  # keep the gallery cache out of the repo
    source, frames = video  # 64x48, 4:3
    gallery_path, _ = video_factory(count=30, width=96, height=54, seed=3)
    config = UserConfig(input_dir=str(source), output_dir=str(tmp_path / "out"),
                        grid_size=2, contrast=1.0, candidates=4, epsilon=0.5)
    gallery = VideoGallery(gallery_path, stride=1)

    main(gallery, config)

    derived = probe_video(config, gallery.native_aspect)
    cell_w, cell_h = derived.cell_size
    assert cell_w > cell_h  # the tiles kept their shape
    assert len(read_video(tmp_path / "out" / "output.mp4")) == len(frames)


def test_stretch_gives_back_the_square_grid(video, video_factory):
    """The flag is a flag: `stretch` derives exactly what it did before 2.1."""
    source, _ = video
    config = UserConfig(input_dir=str(source), output_dir="", grid_size=2,
                        tile_fit="stretch")

    assert probe_video(config).cell_size == (8, 8)
