import cv2
import numpy as np

from main import UserConfig, get_gallery, gallery_brightness, shrink_gallery


def test_get_gallery_shape_and_channel_order(cifar_pickle):
    path, expected = cifar_pickle
    loaded = get_gallery(str(path))

    assert loaded.shape == expected.shape
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, expected)


def test_get_gallery_is_contiguous(cifar_pickle):
    """cv2.resize refuses a reverse-strided view, so the RGB->BGR flip must copy."""
    path, _ = cifar_pickle
    assert get_gallery(str(path)).flags["C_CONTIGUOUS"]


def test_gallery_brightness_matches_cvtcolor(gallery):
    """The dot-product form must agree with the per-image cvtColor it replaced."""
    reference = np.array([
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0 for img in gallery
    ])
    np.testing.assert_allclose(gallery_brightness(gallery), reference, atol=1 / 255)


def test_gallery_brightness_range(gallery):
    brightness = gallery_brightness(gallery)
    assert brightness.shape == (len(gallery),)
    assert np.all((brightness >= 0) & (brightness <= 1))


def test_shrink_gallery_keeps_middle_band(gallery):
    brightness = gallery_brightness(gallery)
    config = UserConfig(input_dir="", output_dir="", contrast=0.5)

    kept, kept_brightness = shrink_gallery(gallery, brightness, config)

    assert len(kept) == len(kept_brightness)
    # Half the band around the median, so roughly half the images survive.
    assert 0.4 * len(gallery) <= len(kept) <= 0.6 * len(gallery)
    # And they are the middle ones: nothing at either extreme.
    assert kept_brightness.min() > brightness.min()
    assert kept_brightness.max() < brightness.max()


def test_shrink_gallery_full_contrast_keeps_everything(gallery):
    brightness = gallery_brightness(gallery)
    config = UserConfig(input_dir="", output_dir="", contrast=1.0)

    kept, kept_brightness = shrink_gallery(gallery, brightness, config)

    np.testing.assert_array_equal(kept, gallery)
    np.testing.assert_array_equal(kept_brightness, brightness)


def test_shrink_gallery_returns_matching_brightnesses(gallery):
    """The returned brightnesses must be the survivors' own, not a stale slice."""
    brightness = gallery_brightness(gallery)
    config = UserConfig(input_dir="", output_dir="", contrast=0.3)

    kept, kept_brightness = shrink_gallery(gallery, brightness, config)

    np.testing.assert_allclose(gallery_brightness(kept), kept_brightness)
