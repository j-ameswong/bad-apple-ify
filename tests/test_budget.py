"""1.2c: estimating the tile array before building it.

The whole point is that nothing gets decoded or allocated, so most of what
follows asserts that `load()` was never entered — not that it was quick.
"""

from pathlib import Path

import numpy as np
import pytest

from conftest import CELL, make_frames, write_video
from main import (CifarGallery, DerivedConfig, GalleryTooLarge, VideoGallery,
                  check_gallery_budget, format_bytes, load_gallery)


class ExplodingSource:
    """Claims a tile count and refuses to be loaded. If `load()` runs, the test
    was meant to have stopped before it."""

    def __init__(self, count: int | None, fingerprint: str = "boom:1"):
        self._count = count
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def estimate_count(self) -> int | None:
        return self._count

    def load(self, cell_size: tuple[int, int]) -> np.ndarray:
        raise AssertionError("load() was entered despite the budget check")


class TinySource(ExplodingSource):
    """Same, but it will actually hand over a (count, cell) array."""

    def load(self, cell_size: tuple[int, int]) -> np.ndarray:
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


def test_video_estimate_is_none_when_nothing_can_be_read(tmp_path):
    (tmp_path / "broken.mkv").write_bytes(b"not a video at all")

    assert VideoGallery(tmp_path / "broken.mkv").estimate_count() is None


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


def test_format_bytes_reads_like_a_person_wrote_it():
    assert format_bytes(512) == "512 B"
    assert format_bytes(38 * 1024 ** 2) == "38 MB"
    assert format_bytes(8 * 1024 ** 3) == "8 GB"
    assert format_bytes(1.3 * 1024 ** 3) == "1.3 GB"
