from __future__ import annotations

from typing import Callable, Iterator, Protocol, cast
import glob
import hashlib
import importlib
import os
import pickle
import tempfile
import tokenize
import warnings
from fractions import Fraction
from math import isfinite
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import tqdm

from .config import DerivedConfig, HARD_BUDGET, SOFT_BUDGET, UserConfig
from .types import Fit, Image

class GallerySource(Protocol):
    """A source of tiles, handed over already at cell size.

    Never at full resolution, which is what forces the probe-before-load
    ordering in `main()`. See docs/gallery-sources.md.
    """

    @property
    def fingerprint(self) -> str:
        """Identifies what `load()` will return, ignoring cell size.

        The cache keys on this, so under-reporting here serves stale tiles.
        """
        ...

    @property
    def native_aspect(self) -> tuple[int, int] | None:
        """The ratio the source's own images are shaped to. None if they vary.

        `tile_fit="native"` shapes the cell to this, so tiles never distort.
        """
        ...

    def estimate_count(self) -> int | None:
        """Roughly how many tiles `load()` will return, cheaply. None if unknowable.

        Erring high is free; erring low defeats the point.
        """
        ...

    def load(self, cell_size: tuple[int, int], fit: Fit = "native",
             budget: int = HARD_BUDGET) -> Image:
        """(N, cell_h, cell_w, 3) BGR tiles at the given (width, height) cell size.

        `budget` is the same ceiling `check_gallery_budget()` priced the estimate
        against, for a source that can only find out the real count as it goes.
        """
        ...

def crop_to_aspect(image: Image, cell_size: tuple[int, int]) -> Image:
    """The largest centred rectangle of `image` with the cell's aspect ratio."""
    cell_w, cell_h = cell_size
    h, w = image.shape[:2]
    too_wide = w * cell_h > h * cell_w
    if too_wide:
        crop_w, crop_h = max(round(h * cell_w / cell_h), 1), h
    else:
        crop_w, crop_h = w, max(round(w * cell_h / cell_w), 1)
    x, y = (w - crop_w) // 2, (h - crop_h) // 2
    return image[y:y + crop_h, x:x + crop_w]

def fit_to_cell(image: Image, cell_size: tuple[int, int], fit: Fit = "native",
                dst: Image | None = None) -> Image:
    """Resize one image to cell size, cropping first unless asked to stretch.

    Under `native` the cell already carries the tiles' ratio, so the crop is a
    no-op — except in single-frame mode, where the cell is the source frame.
    """
    source = image if fit == "stretch" else crop_to_aspect(image, cell_size)
    src_h, src_w = source.shape[:2]
    # Shrinking, bilinear samples a 2x2 patch where AREA averages; growing, AREA
    # is nearest-neighbour. See docs/tile-shape.md.
    shrinking = cell_size[0] <= src_w and cell_size[1] <= src_h
    # dst= needs an exact shape and dtype match or cv2 quietly drops the write.
    return cast(Image, cv2.resize(
        source, cell_size, dst=dst,
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR))

def resize_gallery_to_cells(gallery: Image, cell_size: tuple[int, int],
                            fit: Fit = "native") -> Image:
    """Resize every gallery image to cell size once, at load time.

    Filled in place: stacking a list comprehension would hold every tile twice
    while `np.array` copies it, doubling peak RAM.
    """
    cell_w, cell_h = cell_size
    tiles = np.empty((len(gallery), cell_h, cell_w, 3), dtype=gallery.dtype)
    for tile, img in zip(tiles, gallery):
        fit_to_cell(img, cell_size, fit, dst=tile)
    return tiles

CIFAR_IMAGE_BYTES = 32 * 32 * 3

VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"})

FALLBACK_CAPACITY = 1024

def read_cifar_batch(path: Path) -> Image:
    """Read a CIFAR pickle batch as an (N, 32, 32, 3) BGR array."""
    class CifarUnpickler(pickle.Unpickler):
        """Only construct NumPy arrays used by the legacy CIFAR files."""

        _numpy_globals = {
            (module, name): value
            for module in ("numpy.core.multiarray", "numpy._core.multiarray")
            for name, value in (("_reconstruct", getattr(
                importlib.import_module("numpy._core.multiarray"), "_reconstruct")),
                                ("scalar", getattr(
                importlib.import_module("numpy._core.multiarray"), "scalar")))
        }
        _numpy_globals.update({
            (module, "_frombuffer"): getattr(
                importlib.import_module("numpy._core.numeric"), "_frombuffer")
            for module in ("numpy.core.numeric", "numpy._core.numeric")
        })
        _numpy_globals.update({
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        })

        def find_class(self, module: str, name: str) -> object:
            try:
                return self._numpy_globals[(module, name)]
            except KeyError as error:
                raise pickle.UnpicklingError(
                    f"CIFAR pickle requested unsupported global {module}.{name}") from error

    with open(path, 'rb') as fo:
        with warnings.catch_warnings():
            # Ancient NumPy pickled the dtype with `align=0`, which 2.4
            # deprecates. Not fixable short of re-serialising the file.
            warnings.filterwarnings("ignore", message=".*align=0.*")
            try:
                data = CifarUnpickler(fo, encoding='latin1').load()
            except (pickle.UnpicklingError, EOFError, ValueError, TypeError,
                    AttributeError, IndexError, ImportError, OverflowError) as error:
                raise ValueError(f"Invalid CIFAR pickle at {path}: {error}") from error
        if not isinstance(data, dict) or "data" not in data:
            raise ValueError("CIFAR pickle must contain a 'data' array")
        images = data["data"]
        if (not isinstance(images, np.ndarray) or images.dtype != np.uint8
                or images.ndim != 2 or images.shape[0] == 0
                or images.shape[1] != CIFAR_IMAGE_BYTES):
            raise ValueError("CIFAR 'data' must be a non-empty uint8 array of shape (N, 3072)")

        # reminder to self, transpose works by putting in the old positions
        rgb = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        # Faster to resize from, not required by cv2. See docs/gallery-sources.md.
        return np.ascontiguousarray(rgb[..., ::-1])

class CifarGallery:
    """Tiles from a CIFAR-100 pickle batch. 32x32 images, resized to cell size."""

    def __init__(self, path: Path):
        self._path = Path(path)

    @property
    def input_paths(self) -> tuple[Path, ...]:
        """Files this gallery reads, for protecting them from output paths."""
        return (self._path.resolve(),)

    @property
    def fingerprint(self) -> str:
        stat = self._path.stat()
        return f"cifar:{self._path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"

    @property
    def native_aspect(self) -> tuple[int, int] | None:
        return (1, 1)

    def estimate_count(self) -> int | None:
        # Labels and filenames pad the pickle, so this reads 1.1% high on the
        # real train batch (50537 against 50000). Fine for a budget.
        size = self._path.stat().st_size
        if size == 0:
            raise ValueError(f"CIFAR gallery file is empty: {self._path}")
        if size < CIFAR_IMAGE_BYTES:
            raise ValueError(f"CIFAR gallery file is too small to contain an image: {self._path}")
        return size // CIFAR_IMAGE_BYTES

    def load(self, cell_size: tuple[int, int], fit: Fit = "native",
             budget: int = HARD_BUDGET) -> Image:
        images = read_cifar_batch(self._path)
        enforce_gallery_budget(len(images), cell_size, budget)
        return resize_gallery_to_cells(images, cell_size, fit)

class TileBuffer:
    """Tiles written in place, doubling when full but never past `budget`.

    `_allocate()` prices every backing-buffer allocation, and growth waits for
    `keep()` so a duplicate never costs a slot. See docs/video-gallery.md.
    """

    def __init__(self, capacity: int, cell_size: tuple[int, int], budget: int):
        cell_w, cell_h = cell_size
        self._cell_size = cell_size
        self._budget = budget
        # The most tiles `enforce_gallery_budget()` lets through.
        self._ceiling = (budget - 1) // (cell_h * cell_w * 3)
        self._tiles = self._allocate(capacity, needed=1)
        self._scratch: Image = np.empty((cell_h, cell_w, 3), dtype=np.uint8)
        self.count = 0

    def _allocate(self, wanted: int, needed: int) -> Image:
        """Room for `wanted` tiles, capped at the budget. Raises if `needed` won't fit."""
        enforce_gallery_budget(needed, self._cell_size, self._budget)
        cell_w, cell_h = self._cell_size
        return np.empty((min(wanted, self._ceiling), cell_h, cell_w, 3),
                        dtype=np.uint8)

    def next_slot(self) -> Image:
        """Where the next tile goes: its slot, or the scratch tile when full."""
        if self.count == len(self._tiles):
            return self._scratch
        slot: Image = self._tiles[self.count]
        return slot

    def keep(self) -> None:
        """Keep whatever was just written to `next_slot()`, growing to fit it."""
        if self.count == len(self._tiles):
            # Both buffers are held over the copy, so half again, briefly.
            grown = self._allocate(2 * self.count, needed=self.count + 1)
            grown[:self.count] = self._tiles
            grown[self.count] = self._scratch
            self._tiles = grown
        self.count += 1

    def trimmed(self) -> Image:
        """The kept tiles, copied down to size if a slice would pin much slack."""
        kept: Image = self._tiles[:self.count]
        return kept.copy() if self.count < 0.9 * len(self._tiles) else kept

class VideoGallery:
    """Tiles decoded from a video (or a directory of them), keeping every
    `stride`-th frame. See docs/video-gallery.md."""

    def __init__(self, path: Path, stride: int = 10):
        if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
            raise ValueError("video gallery stride must be a positive integer")
        self._path = Path(path)
        self._stride = stride
        self._count: int | None = None
        self._counted = False

    def _files(self) -> list[Path]:
        """Return existing video files in stable order, rejecting empty inputs."""
        if self._path.is_dir():
            # Dotfiles are skipped for the AppleDouble `._name.mkv` stubs a rip
            # off a Mac leaves behind: right suffix, 4 KB of resource fork.
            files = sorted(p for p in self._path.iterdir()
                           if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
                           and not p.name.startswith("."))
        elif glob.has_magic(str(self._path)):
            matches = (Path(match) for match in glob.glob(str(self._path), recursive=True))
            files = sorted(path for path in matches
                           if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
                           and not path.name.startswith("."))
        elif self._path.is_file():
            files = [self._path]
        else:
            raise ValueError(f"Video gallery path not found: {self._path}")
        if not files:
            raise ValueError(f"No videos found at {self._path}")
        return files

    @property
    def input_paths(self) -> tuple[Path, ...]:
        """Video files this gallery reads, in decode order."""
        return tuple(path.resolve() for path in self._files())

    @property
    def fingerprint(self) -> str:
        files = ",".join(
            f"{p.resolve()}:{p.stat().st_mtime_ns}:{p.stat().st_size}"
            for p in self._files())
        return f"video:{files}:stride={self._stride}"

    @property
    def native_aspect(self) -> tuple[int, int] | None:
        """The first file's frame shape, assumed to hold for the rest.

        A season is one rip at one resolution; a mixed bag would need the cell
        to fit them all, which no single ratio does.
        """
        files = self._files()
        cap = cv2.VideoCapture(str(files[0]))
        try:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        finally:
            cap.release()
        if not isfinite(width) or not isfinite(height):
            return None
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            return None
        ratio = Fraction(width, height)
        return (ratio.numerator, ratio.denominator)

    def estimate_count(self) -> int | None:
        """Frame count over stride, summed over the files. Upper bound.

        Memoised. Container metadata, so it can lie, but `TileBuffer` re-prices
        anything past it. See docs/video-gallery.md.
        """
        if not self._counted:
            self._count = self._scan_count()
            self._counted = True
        return self._count

    def _scan_count(self) -> int | None:
        total = 0
        unknown_count = False
        for path in self._files():
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    raise ValueError(f"Video at {path} could not be opened!")
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if not isfinite(frames):
                    unknown_count = True
                    continue
                if frames <= 0:
                    # Nothing in the container; seek to the end for a duration instead.
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if not isfinite(fps) or not isfinite(ms):
                        unknown_count = True
                        continue
                    frames = fps * ms / 1000.0 if fps > 0 and ms > 0 else 0
                if not isfinite(frames):
                    unknown_count = True
                    continue
            finally:
                cap.release()
            if frames <= 0:
                unknown_count = True
                continue
            total += -(-int(frames) // self._stride)
        return None if unknown_count else total

    def load(self, cell_size: tuple[int, int], fit: Fit = "native",
             budget: int = HARD_BUDGET) -> Image:
        """Decode every file in order, keeping one frame in `stride`, deduped.

        Straight through, no seeking: a frame-accurate seek on long-GOP video
        costs more than decoding past what we skip. See docs/video-gallery.md.
        """
        files = self._files()
        buffer = TileBuffer(self.estimate_count() or FALLBACK_CAPACITY,
                            cell_size, budget)
        seen: set[bytes] = set()
        sampled = 0

        with tqdm.tqdm(desc="Decoding gallery...", unit="frame",
                       total=self._frame_total()) as bar:
            for path in files:
                cap = cv2.VideoCapture(str(path))
                try:
                    if not cap.isOpened():
                        raise ValueError(f"Video at {path} could not be opened!")
                    for frame in self._sampled_frames(cap, bar.update):
                        slot = buffer.next_slot()
                        fit_to_cell(frame, cell_size, fit, dst=slot)
                        sampled += 1
                        digest = hashlib.blake2b(slot.tobytes(),
                                                 digest_size=8).digest()
                        if digest not in seen:
                            seen.add(digest)
                            buffer.keep()
                finally:
                    cap.release()

        print(f"Sampled {sampled} frames from {len(files)} file(s) -> "
              f"{buffer.count} tiles ({sampled - buffer.count} duplicates dropped)")
        return buffer.trimmed()

    def _sampled_frames(self, cap: cv2.VideoCapture,
                        tick: Callable[[], object]) -> Iterator[Image]:
        """Every `stride`-th frame of an open capture, ticking once per frame
        decoded. The caller releases the capture."""
        index = 0
        while cap.grab():
            if index % self._stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    # cv2's stubs won't commit to a dtype; decode is uint8.
                    yield cast(Image, frame)
            index += 1
            tick()

    def _frame_total(self) -> int | None:
        """Frames to be decoded, for the progress bar. Metadata, so a hint only."""
        count = self.estimate_count()
        return count * self._stride if count else None

DEFAULT_CACHE_DIR = Path(".cache/gallery")

TILE_VERSION = 2

def cache_key(source: GallerySource, cell_size: tuple[int, int],
              fit: Fit = "native") -> str:
    """Filename-safe digest of everything that determines the loaded tiles."""
    material = (f"{source.fingerprint}|cell={cell_size[0]}x{cell_size[1]}"
                f"|fit={fit}|v={TILE_VERSION}")
    return hashlib.sha256(material.encode()).hexdigest()[:16]

class GalleryTooLarge(RuntimeError):
    """The tile array would blow the budget, so nothing was loaded."""

def format_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.3g} {unit}"
        size /= 1024
    return f"{size:.3g} TB"

def enforce_gallery_budget(count: int, cell_size: tuple[int, int],
                           budget: int = HARD_BUDGET) -> int:
    """Price a tile array of `count` tiles, raising if it's over budget."""
    cell_w, cell_h = cell_size
    size = count * cell_h * cell_w * 3
    if size >= budget:
        raise GalleryTooLarge(
            f"Gallery needs ~{count} tiles x {cell_w}x{cell_h}x3 B = "
            f"{format_bytes(size)}, over the {format_bytes(budget)} budget. "
            f"Raise grid_size for a smaller cell, or stride for fewer tiles — "
            f"or raise gallery_budget if you really do have the RAM.")
    return size

def check_gallery_budget(source: GallerySource, cell_size: tuple[int, int],
                         budget: int = HARD_BUDGET) -> None:
    """Price the tile array before anything is decoded, and say so.

    Count and cell size multiply, so a character's difference blows the array
    up — hence a refusal at the top end, not a warning. See docs/gallery-size.md.
    """
    count = source.estimate_count()
    cell_w, cell_h = cell_size
    if count is None:
        print(f"Gallery size unknown: this source can't estimate its tile count. "
              f"Each tile is {cell_w}x{cell_h}x3 B.")
        return

    size = enforce_gallery_budget(count, cell_size, budget)
    arithmetic = (f"~{count} tiles x {cell_w}x{cell_h}x3 B = "
                  f"{format_bytes(size)}")
    if size >= SOFT_BUDGET:
        print(f"WARNING: gallery is {arithmetic}, all of it held in RAM. "
              f"Raise grid_size for a smaller cell, or stride for fewer tiles.")
    else:
        print(f"Gallery estimate: {arithmetic}")

def read_cached_tiles(path: Path, cell_size: tuple[int, int],
                      budget: int = HARD_BUDGET) -> Image | None:
    """The tiles cached at `path`, or None if the file is malformed.

    One handle throughout, and the count is priced only once the file size
    backs it up. See docs/gallery-cache.md.
    """
    readers = {(1, 0): np.lib.format.read_array_header_1_0,
               (2, 0): np.lib.format.read_array_header_2_0}
    cell_w, cell_h = cell_size
    with open(path, "rb") as f:
        try:
            reader = readers.get(np.lib.format.read_magic(f))
            if reader is None:
                return None
            shape, fortran_order, dtype = reader(f)
        except (ValueError, SyntaxError, TypeError, tokenize.TokenError):
            # numpy's parser throws all four on a mangled header, not just ValueError.
            return None

        if (dtype != np.uint8 or fortran_order or len(shape) != 4 or shape[0] <= 0
                or shape[1:] != (cell_h, cell_w, 3)):
            return None
        # A truncated file has an honest header and too few bytes behind it.
        count = shape[0]
        if os.fstat(f.fileno()).st_size != f.tell() + count * cell_h * cell_w * 3:
            return None

        # A cache written under a bigger budget is refused, not re-decoded:
        # the decode would come to the same count.
        enforce_gallery_budget(count, cell_size, budget)
        tiles: Image = np.empty(shape, dtype=np.uint8)
        # Short only if something truncated the file since the size check.
        if f.readinto(tiles) != tiles.nbytes:
            return None
    return tiles

def load_gallery(source: GallerySource, derived: DerivedConfig, *,
                 fit: Fit = "native",
                 cache_dir: Path = DEFAULT_CACHE_DIR,
                 use_cache: bool = True,
                 budget: int = HARD_BUDGET) -> Image:
    """Load tiles at the derived cell size, going through the on-disk cache.

    A load costs a full decode pass over the source, so it's worth keeping.
    See docs/gallery-cache.md.
    """
    cell_size = derived.cell_size
    path = cache_dir / f"tiles-{cache_key(source, cell_size, fit)}.npy"
    if use_cache and path.exists():
        cached = read_cached_tiles(path, cell_size, budget)
        if cached is not None:
            print(f"Gallery cache hit: {path} ({len(cached)} tiles, "
                  f"{format_bytes(cached.nbytes)})")
            return cached
        print(f"Gallery cache at {path} is malformed, reloading")

    # A hit prices the real count above; only a decode needs the estimate.
    check_gallery_budget(source, cell_size, budget)
    tiles = source.load(cell_size, fit, budget)
    expected_shape = (tiles.shape[0], cell_size[1], cell_size[0], 3) \
        if isinstance(tiles, np.ndarray) and tiles.ndim == 4 else None
    if (expected_shape is None or tiles.shape[0] <= 0
            or tiles.shape != expected_shape or tiles.dtype != np.uint8):
        raise ValueError(
            "Gallery source must return a non-empty uint8 array shaped "
            f"(N, {cell_size[1]}, {cell_size[0]}, 3)")
    # Estimates can be low or stale. Check the actual returned tile bytes too.
    enforce_gallery_budget(len(tiles), cell_size, budget)
    if not use_cache:
        return tiles

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Write-then-rename, so a run killed mid-write leaves the old cache intact.
    # Keeps the .npy suffix, which np.save would otherwise append itself.
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp.npy",
                                     dir=cache_dir)
    os.close(fd)
    temp = Path(temp_name)
    try:
        np.save(temp, tiles, allow_pickle=False)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return tiles
