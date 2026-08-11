"""Shared fixtures.

Everything the suite runs on is generated at test time: a synthetic video
instead of `assets/source.mp4`, a 100-image array instead of the 155 MB CIFAR
pickle. A clean checkout with no assets is enough to run `uv run pytest`.
"""

import pickle

import cv2
import numpy as np
import pytest

from main import resize_gallery_to_cells

# Lossless, so a written fixture video decodes back to exactly the frames that
# went in and tests can assert on pixels rather than tolerances.
FOURCC = cv2.VideoWriter_fourcc(*"FFV1")
VIDEO_SUFFIX = ".mkv"

# The (width, height) cell the tile-level tests work at.
CELL = (4, 4)


def make_frames(count: int, width: int, height: int, seed: int = 0) -> np.ndarray:
    """(count, height, width, 3) BGR frames with a moving bright block.

    Deterministic, and varied enough across frames that a per-frame result
    cannot pass by accident on frame 0 alone.
    """
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, (count, height, width, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        top = (i * height // count) % height
        frame[top:top + max(height // count, 1), :, :] = 255
    return frames


def write_video(path, frames: np.ndarray, fps: int = 30) -> None:
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), FOURCC, fps, (width, height))
    if not writer.isOpened():
        pytest.skip(f"no encoder available for {FOURCC}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def read_video(path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


@pytest.fixture
def video_factory(tmp_path):
    """Write a synthetic video and return (path, frames as written)."""
    counter = iter(range(1000))

    def factory(count: int = 10, width: int = 64, height: int = 48,
                fps: int = 30, seed: int = 0):
        path = tmp_path / f"src{next(counter)}{VIDEO_SUFFIX}"
        frames = make_frames(count, width, height, seed)
        write_video(path, frames, fps)
        return path, frames

    return factory


@pytest.fixture
def video(video_factory):
    """A 10-frame 64x48 (4:3) video — the default source under test."""
    return video_factory()


@pytest.fixture
def gallery() -> np.ndarray:
    """(100, 32, 32, 3) BGR fake gallery.

    Image i is a textured patch centred on brightness level i/99, so the
    brightnesses span the full 0-1 range with a known, strictly increasing
    order — which is what the brightness matching is indexed by.
    """
    rng = np.random.default_rng(1)
    n = 100
    # Inset from 0/255 by the texture amplitude so clipping never pulls an
    # image's mean off its assigned level.
    base = np.linspace(8, 247, n)
    # Texture is zero-mean per image, so it perturbs appearance without moving
    # the image's mean brightness off its assigned level.
    texture = rng.integers(-8, 9, (n, 32, 32, 3)).astype(np.float64)
    texture -= texture.mean(axis=(1, 2, 3), keepdims=True)
    imgs = base[:, None, None, None] + texture
    return np.clip(np.rint(imgs), 0, 255).astype(np.uint8)


@pytest.fixture
def cell_gallery(gallery) -> np.ndarray:
    """The gallery as a `GallerySource` hands it over: already at cell size.

    Everything downstream of the load — brightness, matching, mosaicking — only
    ever sees tiles at this resolution, so that is what the fixtures feed it.
    """
    return resize_gallery_to_cells(gallery, CELL)


@pytest.fixture
def cifar_pickle(tmp_path, gallery):
    """A CIFAR-format pickle built from the fake gallery.

    Same layout as the real `train` batch: 'data' is (N, 3072) uint8, planar
    RGB. Returns (path, expected BGR array).
    """
    rgb = gallery[..., ::-1]
    flat = rgb.transpose(0, 3, 1, 2).reshape(len(rgb), -1)
    path = tmp_path / "train"
    with open(path, "wb") as fo:
        pickle.dump({"data": flat, "labels": list(range(len(flat)))}, fo)
    return path, gallery
