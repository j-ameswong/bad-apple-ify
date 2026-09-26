"""1.2b: the gallery cache.

A gallery load is one decode pass over the whole source — minutes for a video
gallery — so the tests here are mostly about *not* calling `load()`.
"""

import errno
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import CELL
import main
from main import (CifarGallery, DerivedConfig, HARD_BUDGET, cache_key,
                  load_gallery, read_cached_tiles)


class CountingSource:
    """A `GallerySource` that records how many times it was actually loaded."""

    def __init__(self, tiles: np.ndarray, fingerprint: str = "fake:1"):
        self._tiles = tiles
        self._fingerprint = fingerprint
        self.loads = 0

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def native_aspect(self) -> tuple[int, int] | None:
        return (1, 1)

    def estimate_count(self) -> int:
        return len(self._tiles)

    def load(self, cell_size: tuple[int, int], fit: str = "native",
             budget: int = HARD_BUDGET) -> np.ndarray:
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


def test_truncated_cache_falls_back_to_a_reload(cache_dir, cell_gallery):
    """Right header, missing tail — the header alone would call it a hit."""
    source = CountingSource(cell_gallery)
    derived = derived_for(CELL)
    load_gallery(source, derived, cache_dir=cache_dir)

    cached_file, = cache_dir.glob("*.npy")
    data = cached_file.read_bytes()
    cached_file.write_bytes(data[:-100])

    tiles = load_gallery(source, derived, cache_dir=cache_dir)

    assert source.loads == 2
    assert tiles.shape == (len(cell_gallery), CELL[1], CELL[0], 3)


def test_wrong_dtype_cache_falls_back_to_a_reload(cache_dir, cell_gallery):
    source = CountingSource(cell_gallery)
    derived = derived_for(CELL)
    load_gallery(source, derived, cache_dir=cache_dir)

    cached_file, = cache_dir.glob("*.npy")
    np.save(cached_file, np.zeros((3, CELL[1], CELL[0], 3), dtype=np.float32))

    tiles = load_gallery(source, derived, cache_dir=cache_dir)

    assert source.loads == 2
    assert tiles.dtype == np.uint8


def flip(data: bytes, index: int, bit: int) -> bytes:
    mangled = bytearray(data)
    mangled[index] ^= 1 << bit
    return bytes(mangled)


# ValueError, TypeError and SyntaxError out of numpy's header parser, in that
# order. The bit-flip sweep below gets TokenError. See docs/gallery-cache.md.
MANGLED_HEADERS = {
    "empty file": lambda data: b"",
    "bytes key": lambda data: data.replace(b"'shape'", b"b'shape'"),
    "comma in the dtype": lambda data: data.replace(b"'|u1'", b"',u1'"),
}


@pytest.mark.parametrize("mangling", MANGLED_HEADERS)
def test_a_mangled_header_falls_back_to_a_reload(cache_dir, cell_gallery, mangling):
    source = CountingSource(cell_gallery)
    derived = derived_for(CELL)
    load_gallery(source, derived, cache_dir=cache_dir)

    cached_file, = cache_dir.glob("*.npy")
    cached_file.write_bytes(MANGLED_HEADERS[mangling](cached_file.read_bytes()))

    tiles = load_gallery(source, derived, cache_dir=cache_dir)

    assert source.loads == 2
    assert tiles.shape == (len(cell_gallery), CELL[1], CELL[0], 3)


def test_any_one_bit_flipped_in_the_header_is_a_miss_or_the_same_tiles(
        tmp_path, cell_gallery):
    """Never an exception and never the wrong tiles. Over a quarter of these
    flips get TokenError out of numpy rather than ValueError."""
    path = tmp_path / "tiles.npy"
    np.save(path, cell_gallery)
    data = path.read_bytes()

    for index in range(data.index(b"\n") + 1):
        for bit in range(8):
            path.write_bytes(flip(data, index, bit))
            tiles = read_cached_tiles(path, CELL)
            assert tiles is None or np.array_equal(tiles, cell_gallery), (index, bit)


def test_a_fortran_order_cache_never_comes_back_scrambled(cache_dir, cell_gallery):
    """Right shape, right dtype, bytes in the other order. Read as C order
    they'd be scrambled tiles, so it has to be a miss."""
    source = CountingSource(cell_gallery)
    derived = derived_for(CELL)
    first = load_gallery(source, derived, cache_dir=cache_dir)

    cached_file, = cache_dir.glob("*.npy")
    np.save(cached_file, np.asfortranarray(first))

    np.testing.assert_array_equal(
        load_gallery(source, derived, cache_dir=cache_dir), first)


def test_a_cache_cut_short_after_its_size_check_falls_back_to_a_reload(
        cache_dir, cell_gallery, monkeypatch):
    """The stat and the read are separate calls, and anything truncating the
    file in place between them (`cp` over it, say) leaves the read short."""
    source = CountingSource(cell_gallery)
    derived = derived_for(CELL)
    load_gallery(source, derived, cache_dir=cache_dir)

    cached_file, = cache_dir.glob("*.npy")
    whole = cached_file.stat().st_size
    cached_file.write_bytes(cached_file.read_bytes()[:-100])
    monkeypatch.setattr(main.os, "fstat", lambda fd: SimpleNamespace(st_size=whole))

    tiles = load_gallery(source, derived, cache_dir=cache_dir)

    assert source.loads == 2
    assert tiles.shape == (len(cell_gallery), CELL[1], CELL[0], 3)


def test_a_failed_write_leaves_nothing_behind(cache_dir, cell_gallery, monkeypatch):
    """Disk full halfway through `np.save`: the error still surfaces, but neither
    the half-written temp nor a half cache is left for the next run."""
    def fills_the_disk(file, arr, allow_pickle=True):
        Path(file).write_bytes(b"\x93NUMPY")
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(np, "save", fills_the_disk)

    with pytest.raises(OSError):
        load_gallery(CountingSource(cell_gallery), derived_for(CELL),
                     cache_dir=cache_dir)

    assert list(cache_dir.iterdir()) == []


def test_cache_key_is_filename_safe(cell_gallery):
    key = cache_key(CountingSource(cell_gallery, fingerprint="/a b/c:1"), CELL)
    assert key.isalnum()
