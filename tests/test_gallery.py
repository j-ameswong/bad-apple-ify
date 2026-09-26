import cv2
import numpy as np
import pytest

from conftest import CELL
from main import (CifarGallery, UserConfig, VideoGallery, gallery_brightness,
                  load_gallery, probe_video, read_cifar_batch,
                  resize_gallery_to_cells, shrink_gallery)


def test_read_cifar_batch_shape_and_channel_order(cifar_pickle):
    path, expected = cifar_pickle
    loaded = read_cifar_batch(path)

    assert loaded.shape == expected.shape
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, expected)


def test_read_cifar_batch_is_contiguous(cifar_pickle):
    """Not for cv2's sake, which takes the reversed view: a copy resizes faster."""
    path, _ = cifar_pickle
    assert read_cifar_batch(path).flags["C_CONTIGUOUS"]


def test_cifar_gallery_loads_at_cell_size(cifar_pickle):
    """1.2: `load()` returns tiles already at cell size, never full-resolution."""
    path, expected = cifar_pickle
    cell = (5, 3)

    tiles = CifarGallery(path).load(cell)

    assert tiles.shape == (len(expected), cell[1], cell[0], 3)
    assert tiles.dtype == np.uint8


def test_cifar_gallery_matches_raw_load_then_resize(cifar_pickle):
    """The abstraction must not change the pixels, only where the resize lives."""
    path, expected = cifar_pickle

    tiles = CifarGallery(path).load(CELL)

    np.testing.assert_array_equal(tiles, resize_gallery_to_cells(expected, CELL))


@pytest.fixture(params=["cifar", "video"])
def fresh_source(request):
    """Each real `GallerySource`, as a new instance over the same file per
    call: how the next run would build it."""
    if request.param == "cifar":
        path, _ = request.getfixturevalue("cifar_pickle")
        return lambda: CifarGallery(path)
    # 16:9 against the 4:3 `video`, so the native cell isn't square.
    path, _ = request.getfixturevalue("video_factory")(width=96, height=54)
    return lambda: VideoGallery(path, stride=2)


def test_gallery_sources_satisfy_the_protocol(fresh_source, video, tmp_path,
                                              capsys):
    """Through the calls `main()` makes before it encodes anything: probe on
    `native_aspect`, then `load_gallery()`, which prices the estimate, keys the
    cache on the fingerprint and loads. A second instance has to hit the first
    one's cache, or the fingerprint isn't identifying anything."""
    config = UserConfig(input_dir=str(video[0]), output_dir="", grid_size=2)
    derived = probe_video(config, fresh_source().native_aspect)
    cache_dir = tmp_path / "cache"
    # Both alive at once, so they can't share an id() and pass by accident.
    first, second = fresh_source(), fresh_source()

    decoded = load_gallery(first, derived, cache_dir=cache_dir)
    capsys.readouterr()
    cached = load_gallery(second, derived, cache_dir=cache_dir)

    cell_w, cell_h = derived.cell_size
    assert decoded.shape[1:] == (cell_h, cell_w, 3)
    assert decoded.dtype == np.uint8
    assert len(decoded) <= fresh_source().estimate_count()
    assert "Gallery cache hit" in capsys.readouterr().out
    np.testing.assert_array_equal(cached, decoded)


def test_gallery_brightness_matches_cvtcolor(cell_gallery):
    """The dot-product form must agree with the per-image cvtColor it replaced."""
    reference = np.array([
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0 for img in cell_gallery
    ])
    np.testing.assert_allclose(gallery_brightness(cell_gallery), reference, atol=1 / 255)


def test_shrink_gallery_keeps_middle_band(cell_gallery):
    brightness = gallery_brightness(cell_gallery)
    config = UserConfig(input_dir="", output_dir="", contrast=0.5)

    kept, kept_brightness = shrink_gallery(cell_gallery, brightness, config)

    assert len(kept) == len(kept_brightness)
    # Half the band around the median, so roughly half the images survive.
    assert 0.4 * len(cell_gallery) <= len(kept) <= 0.6 * len(cell_gallery)
    # And they are the middle ones: nothing at either extreme.
    assert kept_brightness.min() > brightness.min()
    assert kept_brightness.max() < brightness.max()


def test_shrink_gallery_full_contrast_keeps_everything(cell_gallery):
    brightness = gallery_brightness(cell_gallery)
    config = UserConfig(input_dir="", output_dir="", contrast=1.0)

    kept, kept_brightness = shrink_gallery(cell_gallery, brightness, config)

    np.testing.assert_array_equal(kept, cell_gallery)
    np.testing.assert_array_equal(kept_brightness, brightness)


def test_shrink_gallery_returns_matching_brightnesses(cell_gallery):
    """The returned brightnesses must be the survivors' own, not a stale slice."""
    brightness = gallery_brightness(cell_gallery)
    config = UserConfig(input_dir="", output_dir="", contrast=0.3)

    kept, kept_brightness = shrink_gallery(cell_gallery, brightness, config)

    np.testing.assert_allclose(gallery_brightness(kept), kept_brightness)
