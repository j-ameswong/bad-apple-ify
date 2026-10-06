import cv2
import numpy as np
import pickle
from pathlib import Path
import pytest

from conftest import CELL
from main import (CifarGallery, UserConfig, VideoGallery, gallery_brightness,
                  GalleryTooLarge, load_gallery, probe_video, read_cifar_batch,
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


def test_cifar_pickle_cannot_load_arbitrary_globals(tmp_path):
    marker = tmp_path / "executed"

    class UnexpectedGlobal:
        def __reduce__(self):
            return (__import__("os").system, (f"touch {marker}",))

    path = tmp_path / "malicious"
    path.write_bytes(pickle.dumps({"data": UnexpectedGlobal()}))

    with pytest.raises(ValueError, match="unsupported global"):
        read_cifar_batch(path)
    assert not marker.exists()


@pytest.mark.parametrize("images", [
    np.empty((0, 3072), dtype=np.uint8),
    np.zeros((2, 3071), dtype=np.uint8),
    np.zeros((2, 3072), dtype=np.uint16),
    np.zeros((2, 32, 32, 3), dtype=np.uint8),
])
def test_cifar_array_must_have_the_expected_shape_and_dtype(tmp_path, images):
    path = tmp_path / "malformed"
    path.write_bytes(pickle.dumps({"data": images}))

    with pytest.raises(ValueError, match="non-empty uint8 array"):
        read_cifar_batch(path)


@pytest.mark.parametrize("contents", [b"", b"not a pickle"])
def test_malformed_cifar_pickle_is_a_value_error(tmp_path, contents):
    path = tmp_path / "malformed"
    path.write_bytes(contents)

    with pytest.raises(ValueError, match="Invalid CIFAR pickle"):
        read_cifar_batch(path)


def test_cifar_gallery_loads_at_cell_size(cifar_pickle):
    """1.2: `load()` returns tiles already at cell size, never full-resolution."""
    path, expected = cifar_pickle
    cell = (5, 3)

    tiles = CifarGallery(path).load(cell)

    assert tiles.shape == (len(expected), cell[1], cell[0], 3)
    assert tiles.dtype == np.uint8


def test_gallery_sources_expose_readonly_input_paths(cifar_pickle, video):
    cifar_path, _ = cifar_pickle
    video_path, _ = video
    assert CifarGallery(cifar_path).input_paths == (cifar_path.resolve(),)
    assert VideoGallery(video_path).input_paths == (video_path.resolve(),)


def test_cifar_gallery_matches_raw_load_then_resize(cifar_pickle):
    """The abstraction must not change the pixels, only where the resize lives."""
    path, expected = cifar_pickle

    tiles = CifarGallery(path).load(CELL)

    np.testing.assert_array_equal(tiles, resize_gallery_to_cells(expected, CELL))


def test_cifar_load_checks_decoded_count_before_resizing(cifar_pickle, monkeypatch):
    path, _ = cifar_pickle

    class UnderestimatingCifar(CifarGallery):
        def estimate_count(self):
            return 1

    def unexpected_resize(*args, **kwargs):
        raise AssertionError("oversized CIFAR rows reached the resize")

    monkeypatch.setattr("main.resize_gallery_to_cells", unexpected_resize)
    with pytest.raises(GalleryTooLarge):
        UnderestimatingCifar(path).load(CELL, budget=100)


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


def test_video_metadata_rejects_nonfinite_values_and_releases_capture(
        tmp_path, monkeypatch):
    path = tmp_path / "metadata.mkv"
    path.touch()

    class FakeCapture:
        released = False

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float("nan")
            return 48.0

        def isOpened(self):
            return True

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    assert VideoGallery(path).native_aspect is None
    assert capture.released


def test_video_frame_count_rejects_infinity_and_releases_capture(
        tmp_path, monkeypatch):
    path = tmp_path / "metadata.mkv"
    path.touch()

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, _):
            return float("inf")

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    assert VideoGallery(path).estimate_count() is None
    assert capture.released


def test_unknown_video_count_does_not_skip_later_file_validation(tmp_path, monkeypatch):
    unknown = tmp_path / "a-unknown.mkv"
    unreadable = tmp_path / "b-unreadable.mkv"
    unknown.touch()
    unreadable.touch()
    captures = []

    class FakeCapture:
        def __init__(self, path):
            self.path = Path(path)
            self.released = False
            captures.append(self)

        def isOpened(self):
            return self.path == unknown

        def get(self, _):
            return float("inf")

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    with pytest.raises(ValueError, match="could not be opened"):
        VideoGallery(tmp_path).estimate_count()
    assert [capture.path.name for capture in captures] == [
        "a-unknown.mkv", "b-unreadable.mkv"]
    assert all(capture.released for capture in captures)


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
