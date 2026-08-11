"""1.2b: the gallery cache.

A gallery load is one decode pass over the whole source — minutes for a video
gallery — so the tests here are mostly about *not* calling `load()`.
"""

import numpy as np
import pytest

from conftest import CELL
from main import CifarGallery, DerivedConfig, cache_key, load_gallery


class CountingSource:
    """A `GallerySource` that records how many times it was actually loaded."""

    def __init__(self, tiles: np.ndarray, fingerprint: str = "fake:1"):
        self._tiles = tiles
        self._fingerprint = fingerprint
        self.loads = 0

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def load(self, cell_size: tuple[int, int]) -> np.ndarray:
        self.loads += 1
        cell_w, cell_h = cell_size
        return np.broadcast_to(self._tiles[:, :1, :1],
                               (len(self._tiles), cell_h, cell_w, 3)).copy()


def derived_for(cell_size: tuple[int, int]) -> DerivedConfig:
    """A `DerivedConfig` with the given cell size; nothing else is read here."""
    return DerivedConfig(src_fps=30, src_dimensions=(64, 48), src_frame_count=10,
                         aspect_ratio=(4, 3), grid=(8, 6), cell_size=cell_size)


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


def test_second_run_skips_the_load(cache_dir, cell_gallery):
    source = CountingSource(cell_gallery)
    derived = derived_for(CELL)

    first = load_gallery(source, derived, cache_dir=cache_dir)
    second = load_gallery(source, derived, cache_dir=cache_dir)

    assert source.loads == 1
    np.testing.assert_array_equal(first, second)


def test_cache_hit_is_byte_identical(cache_dir, cifar_pickle):
    """The cached array must equal what the source would have produced."""
    path, _ = cifar_pickle
    derived = derived_for(CELL)

    load_gallery(CifarGallery(path), derived, cache_dir=cache_dir)
    cached = load_gallery(CifarGallery(path), derived, cache_dir=cache_dir)

    np.testing.assert_array_equal(cached, CifarGallery(path).load(CELL))
    assert cached.dtype == np.uint8


def test_changing_cell_size_misses(cache_dir, cell_gallery):
    source = CountingSource(cell_gallery)

    load_gallery(source, derived_for(CELL), cache_dir=cache_dir)
    tiles = load_gallery(source, derived_for((8, 6)), cache_dir=cache_dir)

    assert source.loads == 2
    assert tiles.shape[1:] == (6, 8, 3)


def test_changing_the_source_misses(cache_dir, cell_gallery):
    """A different fingerprint — a different file, mtime, or stride — re-loads."""
    first = CountingSource(cell_gallery, fingerprint="fake:1")
    second = CountingSource(cell_gallery, fingerprint="fake:2")
    derived = derived_for(CELL)

    load_gallery(first, derived, cache_dir=cache_dir)
    load_gallery(second, derived, cache_dir=cache_dir)

    assert (first.loads, second.loads) == (1, 1)


def test_touching_the_gallery_file_misses(cache_dir, cifar_pickle, tmp_path):
    """mtime is in the key, so an edited gallery is never served stale."""
    path, _ = cifar_pickle
    derived = derived_for(CELL)
    before = cache_key(CifarGallery(path), CELL)

    load_gallery(CifarGallery(path), derived, cache_dir=cache_dir)
    path.touch()

    assert cache_key(CifarGallery(path), CELL) != before


def test_no_cache_bypasses_it_entirely(cache_dir, cell_gallery):
    source = CountingSource(cell_gallery)
    derived = derived_for(CELL)

    load_gallery(source, derived, cache_dir=cache_dir, use_cache=False)
    load_gallery(source, derived, cache_dir=cache_dir, use_cache=False)

    assert source.loads == 2
    assert not cache_dir.exists()


def test_malformed_cache_falls_back_to_a_reload(cache_dir, cell_gallery):
    source = CountingSource(cell_gallery)
    derived = derived_for(CELL)
    load_gallery(source, derived, cache_dir=cache_dir)

    cached_file, = cache_dir.glob("*.npy")
    np.save(cached_file, np.zeros((3, 2, 2, 3), dtype=np.uint8))

    tiles = load_gallery(source, derived, cache_dir=cache_dir)

    assert source.loads == 2
    assert tiles.shape[1:] == (CELL[1], CELL[0], 3)


def test_no_temp_files_are_left_behind(cache_dir, cell_gallery):
    load_gallery(CountingSource(cell_gallery), derived_for(CELL), cache_dir=cache_dir)

    assert [p.name for p in cache_dir.iterdir()] == [
        p.name for p in cache_dir.glob("*.npy")]
    assert not list(cache_dir.glob("*.tmp.npy"))


def test_cache_key_is_filename_safe(cell_gallery):
    key = cache_key(CountingSource(cell_gallery, fingerprint="/a b/c:1"), CELL)
    assert key.isalnum()
