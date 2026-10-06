"""1.2c: estimating the tile array before building it.

The whole point is that nothing gets decoded or allocated, so most of what
follows asserts that `load()` was never entered — not that it was quick.
"""

from pathlib import Path

import numpy as np
import pytest

from conftest import CELL, make_frames, write_video
import main
from main import (CifarGallery, DerivedConfig, GalleryTooLarge, HARD_BUDGET,
                  TileBuffer, VideoGallery, check_gallery_budget, format_bytes,
                  load_gallery)


class ExplodingSource:
    """Claims a tile count and refuses to be loaded. If `load()` runs, the test
    was meant to have stopped before it."""

    def __init__(self, count: int | None, fingerprint: str = "boom:1"):
        self._count = count
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def native_aspect(self) -> tuple[int, int] | None:
        return (1, 1)

    def estimate_count(self) -> int | None:
        return self._count

    def load(self, cell_size: tuple[int, int], fit: str = "native",
             budget: int = HARD_BUDGET) -> np.ndarray:
        raise AssertionError("load() was entered despite the budget check")


class TinySource(ExplodingSource):
    """Same, but it will actually hand over a (count, cell) array."""

    def load(self, cell_size: tuple[int, int], fit: str = "native",
             budget: int = HARD_BUDGET) -> np.ndarray:
        cell_w, cell_h = cell_size
        return np.zeros((self._count, cell_h, cell_w, 3), dtype=np.uint8)


def derived_for(cell_size: tuple[int, int]) -> DerivedConfig:
    return DerivedConfig(src_fps=30, src_dimensions=(64, 48), src_frame_count=10,
                         aspect_ratio=(4, 3), grid=(8, 6), cell_size=cell_size)


# --- the estimators ---------------------------------------------------------


def test_cifar_estimate_is_exact_on_a_synthetic_batch(cifar_pickle):
    path, expected = cifar_pickle
    assert CifarGallery(path).estimate_count() == len(expected)


REAL_CIFAR = Path("assets/gallery/train")


@pytest.mark.skipif(not REAL_CIFAR.exists(), reason="real CIFAR batch not present")
def test_cifar_estimate_is_close_on_the_real_batch():
    """Labels and filenames pad the pickle, so the divisor reads a little high."""
    gallery = CifarGallery(REAL_CIFAR)
    estimate, real = gallery.estimate_count(), len(gallery.load((1, 1)))

    assert real <= estimate <= real * 1.05


def test_video_estimate_divides_by_stride(tmp_path):
    path = tmp_path / "gallery.mkv"
    write_video(path, make_frames(20, 16, 16))

    assert VideoGallery(path, stride=1).estimate_count() == 20
    assert VideoGallery(path, stride=10).estimate_count() == 2
    # A part-full final chunk still contributes a frame.
    assert VideoGallery(path, stride=7).estimate_count() == 3


def test_video_estimate_sums_over_a_directory(tmp_path):
    for i in range(3):
        write_video(tmp_path / f"ep{i}.mkv", make_frames(10, 16, 16))
    (tmp_path / "notes.txt").write_text("not a video")

    assert VideoGallery(tmp_path, stride=5).estimate_count() == 6


def test_video_estimate_rejects_an_unreadable_video(tmp_path):
    (tmp_path / "broken.mkv").write_bytes(b"not a video at all")

    with pytest.raises(ValueError, match="could not be opened"):
        VideoGallery(tmp_path / "broken.mkv").estimate_count()


def test_a_lying_estimate_is_caught_on_the_way_up(tmp_path, monkeypatch):
    """The estimate is container metadata, so the growth path re-prices.

    Otherwise a file that under-reports its frame count sails past the budget
    that was checked, doubling its way to an OOM.
    """
    path = tmp_path / "gallery.mkv"
    write_video(path, make_frames(20, 16, 16))
    source = VideoGallery(path, stride=1)
    monkeypatch.setattr(source, "estimate_count", lambda: 1)

    # One 4x4 tile is 48 B, so a budget of 96 B stops the first doubling.
    with pytest.raises(GalleryTooLarge):
        source.load(CELL, "stretch", budget=96)


def test_a_refused_load_still_releases_the_capture(tmp_path, monkeypatch):
    """The refusal comes from inside the decode loop, so it has to go through
    the `finally` rather than leak an open file."""
    path = tmp_path / "gallery.mkv"
    write_video(path, make_frames(20, 16, 16))
    released = []
    real_capture = main.cv2.VideoCapture

    class TrackingCapture:
        def __init__(self, name):
            self._cap = real_capture(name)

        def __getattr__(self, attr):
            return getattr(self._cap, attr)

        def release(self):
            released.append(True)
            self._cap.release()

    monkeypatch.setattr(main.cv2, "VideoCapture", TrackingCapture)
    source = VideoGallery(path, stride=1)
    monkeypatch.setattr(source, "estimate_count", lambda: 1)

    with pytest.raises(GalleryTooLarge):
        source.load(CELL, "stretch", budget=96)
    assert released == [True]


def test_growth_stops_at_the_budget_rather_than_overshooting():
    """Doubling 3 tiles to 6 would be 288 B against a 200 B budget, but a
    fourth tile (192 B) fits, so it grows to that instead of refusing."""
    buffer = TileBuffer(3, CELL, budget=200)
    for _ in range(4):
        buffer.next_slot()[:] = 7
        buffer.keep()

    assert buffer.trimmed().shape == (4, 4, 4, 3)
    buffer.next_slot()[:] = 7
    with pytest.raises(GalleryTooLarge):
        buffer.keep()


def test_growth_carries_the_tiles_already_kept():
    buffer = TileBuffer(1, CELL, budget=HARD_BUDGET)
    for value in range(5):
        buffer.next_slot()[:] = value
        buffer.keep()

    assert [int(tile[0, 0, 0]) for tile in buffer.trimmed()] == list(range(5))


def solid_frames(*values: int) -> np.ndarray:
    """One flat 16x16 frame per value, so equal values dedupe to one tile."""
    return np.stack([np.full((16, 16, 3), v, dtype=np.uint8) for v in values])


def unknown_count_gallery(tmp_path, monkeypatch, frames) -> VideoGallery:
    path = tmp_path / "gallery.mkv"
    write_video(path, frames)
    source = VideoGallery(path, stride=1)
    monkeypatch.setattr(source, "estimate_count", lambda: None)
    return source


def test_duplicates_at_the_budget_ceiling_still_load(tmp_path, monkeypatch):
    """One 4x4 tile is 48 B, so 49 B holds exactly one. The repeats are
    dropped before they need a slot, so they can't trip the budget."""
    source = unknown_count_gallery(tmp_path, monkeypatch, solid_frames(9, 9, 9))

    assert len(source.load(CELL, "stretch", budget=49)) == 1


def test_a_new_tile_past_the_budget_ceiling_is_refused(tmp_path, monkeypatch):
    source = unknown_count_gallery(tmp_path, monkeypatch, solid_frames(9, 9, 200))

    with pytest.raises(GalleryTooLarge):
        source.load(CELL, "stretch", budget=49)


def test_an_unknown_count_loads_under_a_budget_its_fallback_would_blow(
        tmp_path, monkeypatch):
    """FALLBACK_CAPACITY tiles would be 48 KB; 20 tiles fit in under 1 KB."""
    source = unknown_count_gallery(tmp_path, monkeypatch,
                                   make_frames(20, 16, 16))

    assert len(source.load(CELL, "stretch", budget=20 * 48 + 1)) == 20


def test_an_unknown_count_is_refused_when_not_one_tile_fits(tmp_path,
                                                            monkeypatch):
    source = unknown_count_gallery(tmp_path, monkeypatch, solid_frames(9))

    with pytest.raises(GalleryTooLarge):
        source.load(CELL, "stretch", budget=48)


def test_the_estimate_is_only_scanned_once(tmp_path):
    """Three call sites per load, each opening every file in the directory."""
    for i in range(3):
        write_video(tmp_path / f"ep{i}.mkv", make_frames(10, 16, 16))
    source = VideoGallery(tmp_path, stride=5)

    scans = 0
    original = source._scan_count

    def counting_scan() -> int | None:
        nonlocal scans
        scans += 1
        return original()

    setattr(source, "_scan_count", counting_scan)
    source.load(CELL, "stretch")

    assert scans == 1


# --- acting on the estimate -------------------------------------------------


def test_over_the_hard_budget_refuses_before_any_load():
    with pytest.raises(GalleryTooLarge):
        load_gallery(ExplodingSource(50_000), derived_for((512, 384)),
                     use_cache=False)


def test_the_refusal_spells_out_the_arithmetic_and_the_knobs():
    with pytest.raises(GalleryTooLarge) as excinfo:
        check_gallery_budget(ExplodingSource(50_000), (512, 384))

    message = str(excinfo.value)
    assert "50000 tiles" in message and "512x384x3 B" in message
    assert "27.5 GB" in message
    assert "grid_size" in message and "stride" in message
    assert "gallery_budget" in message


def test_a_raised_budget_lets_it_through():
    """The override is the whole reason the refusal is tolerable."""
    source, derived = TinySource(4), derived_for(CELL)
    with pytest.raises(GalleryTooLarge):
        load_gallery(source, derived, use_cache=False, budget=8)  # 192 B of tiles

    tiles = load_gallery(source, derived, use_cache=False, budget=1 << 30)

    assert len(tiles) == 4


def test_under_the_soft_budget_is_an_ordinary_line(capsys):
    check_gallery_budget(TinySource(1000), CELL)

    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "~1000 tiles x 4x4x3 B" in out


def test_over_the_soft_budget_warns_loudly(capsys):
    # 100k tiles at 64x64 is ~1.2 GB: noisy, not fatal.
    check_gallery_budget(TinySource(100_000), (64, 64))

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "grid_size" in out and "stride" in out


def test_an_unknown_count_neither_warns_nor_refuses(capsys):
    check_gallery_budget(ExplodingSource(None), (512, 384))

    out = capsys.readouterr().out
    assert "unknown" in out.lower()
    assert "WARNING" not in out


def test_a_cache_hit_skips_the_estimate_entirely(tmp_path, cell_gallery):
    """The cached array's real size is already known, so no guess is needed."""
    cache_dir = tmp_path / "cache"
    load_gallery(TinySource(4, fingerprint="cached:1"), derived_for(CELL),
                 cache_dir=cache_dir)

    # Same fingerprint, now claiming a hopeless size and refusing to load.
    tiles = load_gallery(ExplodingSource(10 ** 9, fingerprint="cached:1"),
                         derived_for(CELL), cache_dir=cache_dir)

    assert tiles.shape == (4, CELL[1], CELL[0], 3)


def test_a_cache_hit_over_budget_is_refused_before_it_loads(tmp_path, monkeypatch):
    """A cache written under a bigger budget still has to fit this one's."""
    cache_dir = tmp_path / "cache"
    load_gallery(TinySource(4, fingerprint="cached:1"), derived_for(CELL),
                 cache_dir=cache_dir)

    def no_alloc(*args, **kwargs):
        raise AssertionError("tiles were allocated before the budget check")
    monkeypatch.setattr(np, "empty", no_alloc)

    with pytest.raises(GalleryTooLarge, match="4 tiles"):
        load_gallery(ExplodingSource(4, fingerprint="cached:1"),
                     derived_for(CELL), cache_dir=cache_dir, budget=8)


def test_a_mangled_count_is_a_miss_not_a_refusal(tmp_path):
    """The count is priced only once the file size backs it up. Otherwise one
    flipped bit turning 4 tiles into 6 reads as a gallery over budget."""
    cache_dir = tmp_path / "cache"
    source = TinySource(4, fingerprint="cached:1")
    load_gallery(source, derived_for(CELL), cache_dir=cache_dir)

    cached_file, = cache_dir.glob("*.npy")
    data = cached_file.read_bytes()
    cached_file.write_bytes(data.replace(b"(4, 4, 4, 3)", b"(6, 4, 4, 3)"))

    # 4 tiles are 192 B and fit; 6 would be 288 B and wouldn't.
    tiles = load_gallery(source, derived_for(CELL), cache_dir=cache_dir,
                         budget=200)

    assert tiles.shape == (4, CELL[1], CELL[0], 3)


def test_format_bytes_reads_like_a_person_wrote_it():
    assert format_bytes(512) == "512 B"
    assert format_bytes(38 * 1024 ** 2) == "38 MB"
    assert format_bytes(8 * 1024 ** 3) == "8 GB"
    assert format_bytes(1.3 * 1024 ** 3) == "1.3 GB"
