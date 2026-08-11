import cv2
import numpy as np
import pytest

from conftest import CELL
from main import (CifarGallery, GallerySource, UserConfig, VideoGallery,
                  gallery_brightness, read_cifar_batch, resize_gallery_to_cells,
                  shrink_gallery)


def test_read_cifar_batch_shape_and_channel_order(cifar_pickle):
    path, expected = cifar_pickle
    loaded = read_cifar_batch(path)

    assert loaded.shape == expected.shape
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, expected)


def test_read_cifar_batch_is_contiguous(cifar_pickle):
    """cv2.resize refuses a reverse-strided view, so the RGB->BGR flip must copy."""
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


def test_gallery_sources_satisfy_the_protocol(tmp_path):
    """Both implementations are usable anywhere a `GallerySource` is asked for."""

    def takes_a_source(source: GallerySource) -> GallerySource:
        return source

    assert callable(takes_a_source(CifarGallery(tmp_path / "train")).load)
    assert callable(takes_a_source(VideoGallery(tmp_path / "gallery.mkv")).load)


def test_video_gallery_is_not_implemented_yet(tmp_path):
    with pytest.raises(NotImplementedError):
        VideoGallery(tmp_path / "gallery.mkv", stride=10).load(CELL)


def test_gallery_brightness_matches_cvtcolor(cell_gallery):
    """The dot-product form must agree with the per-image cvtColor it replaced."""
    reference = np.array([
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0 for img in cell_gallery
    ])
    np.testing.assert_allclose(gallery_brightness(cell_gallery), reference, atol=1 / 255)


def test_gallery_brightness_range(cell_gallery):
    brightness = gallery_brightness(cell_gallery)
    assert brightness.shape == (len(cell_gallery),)
    assert np.all((brightness >= 0) & (brightness <= 1))


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
