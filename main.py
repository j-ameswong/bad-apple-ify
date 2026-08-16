from typing import Iterator, Literal, Protocol, cast
import numpy as np
import numpy.typing as npt
import cv2
import tqdm
import hashlib
import os
import pickle
import warnings
from fractions import Fraction
from pathlib import Path
from dataclasses import dataclass
import subprocess

# A bare np.ndarray is ndarray[Any, dtype[Any]], which says nothing.
type Image = npt.NDArray[np.uint8]
type Brightness = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.int64]

# How a gallery image fills a cell it doesn't share a shape with.
type Fit = Literal["native", "crop", "stretch"]

# Rec.601 luma weights in BGR order — what cv2.COLOR_BGR2GRAY applies.
LUMA_BGR = np.array([0.114, 0.587, 0.299])

# Tile-array sizes to warn at and to stop at. See docs/gallery-size.md.
SOFT_BUDGET = 1 << 30  # 1 GB
HARD_BUDGET = 8 << 30  # 8 GB


@dataclass(frozen=True)
class UserConfig:
    """Parameters the user supplies. Nothing here depends on the source video."""

    input_dir: str
    output_dir: str
    contrast: float = 0.1
    grid_size: int = 16  # multiplier for aspect ratio
    tile_fit: Fit = "native"  # cell takes the tiles' own ratio, or crop/stretch into it
    candidates: int = 256  # tiles to sample from per brightness level
    epsilon: float = 0.005  # max brightness error (0-1) a candidate may have
    seed: int = 0
    gallery_budget: int = HARD_BUDGET  # bytes of tiles to refuse past


@dataclass(frozen=True)
class DerivedConfig:
    """Everything computed from a `UserConfig` once the source has been probed.

    Built once by `probe_video()` and never mutated, so the grid and the frame
    size can't drift apart mid-run. See docs/grid-and-sizing.md.
    """

    src_fps: int
    src_dimensions: tuple[int, int]
    src_frame_count: int
    # Smallest integer pair approximating the source's own aspect ratio.
    aspect_ratio: tuple[int, int]
    grid: tuple[int, int]
    cell_size: tuple[int, int]

    @classmethod
    def from_source(cls, config: UserConfig, *, fps: int,
                    dimensions: tuple[int, int], frame_count: int,
                    tile_aspect: tuple[int, int] | None = None) -> "DerivedConfig":
        # Simplest integer pair near the source ratio, so 2.39:1 stays 2.39:1.
        ratio = Fraction(*dimensions).limit_denominator(16)
        aspect_ratio = (ratio.numerator, ratio.denominator)
        # Single-frame mode wants a literal 1x1, not the aspect pair.
        grid = ((1, 1) if config.grid_size == 1
                else (aspect_ratio[0] * config.grid_size,
                      aspect_ratio[1] * config.grid_size))
        # Nearest whole multiple of the grid, at least 1px per cell.
        cell_size = (max(round(dimensions[0] / grid[0]), 1),
                     max(round(dimensions[1] / grid[1]), 1))

        if tile_aspect is not None and config.grid_size > 1:
            # Rows stay put, the cell takes the tiles' ratio, and the columns are
            # however many then fit across. Across the *snapped* width, mind —
            # the raw source width stopped sharing a scale with the cell the
            # moment the rows rounded. See docs/grid-and-sizing.md.
            cell_h = cell_size[1]
            snapped_w = grid[0] * cell_size[0]
            cell_w = max(round(cell_h * tile_aspect[0] / tile_aspect[1]), 1)
            grid = (max(round(snapped_w / cell_w), 1), grid[1])
            cell_size = (cell_w, cell_h)

        return cls(src_fps=fps, src_dimensions=dimensions,
                   src_frame_count=frame_count, aspect_ratio=aspect_ratio,
                   grid=grid, cell_size=cell_size)

    @property
    def grid_x(self) -> int:
        return self.grid[0]

    @property
    def grid_y(self) -> int:
        return self.grid[1]

    @property
    def output_fps(self) -> int:
        return self.src_fps

    @property
    def target_dimensions(self) -> tuple[int, int]:
        return (self.grid[0] * self.cell_size[0],
                self.grid[1] * self.cell_size[1])


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
    if w * cell_h > h * cell_w:  # too wide — trim the sides
        crop_w, crop_h = max(round(h * cell_w / cell_h), 1), h
    else:  # too tall — trim top and bottom
        crop_w, crop_h = w, max(round(w * cell_h / cell_w), 1)
    x, y = (w - crop_w) // 2, (h - crop_h) // 2
    return image[y:y + crop_h, x:x + crop_w]


def fit_to_cell(image: Image, cell_size: tuple[int, int], fit: Fit = "native",
                dst: Image | None = None) -> Image:
    """Shrink one image to cell size, cropping first unless asked to stretch.

    Under `native` the cell already carries the tiles' ratio, so the crop is a
    no-op — except in single-frame mode, where the cell is the source frame.
    """
    source = image if fit == "stretch" else crop_to_aspect(image, cell_size)
    src_h, src_w = source.shape[:2]
    # Bilinear reads a 2x2 neighbourhood, so 1080p into a 14x8 tile samples the
    # frame rather than averaging it and the brightness lands nowhere near the
    # truth. AREA averages, but degenerates to nearest going up (single-frame
    # mode blows 32x32 CIFAR up to the whole frame), hence the switch.
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


# One CIFAR image on disk: 32*32*3 planar uint8.
CIFAR_IMAGE_BYTES = 3072

# What a directory gallery counts as a video.
VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"})

# Tiles to allocate room for when a video source can't say how many there'll be.
FALLBACK_CAPACITY = 1024


def read_cifar_batch(path: Path) -> Image:
    """Read a CIFAR pickle batch as an (N, 32, 32, 3) BGR array."""
    with open(path, 'rb') as fo:
        with warnings.catch_warnings():
            # Ancient NumPy pickled the dtype with `align=0`, which 2.4
            # deprecates. Not fixable short of re-serialising the file.
            warnings.filterwarnings("ignore", message=".*align=0.*")
            data = pickle.load(fo, encoding='latin1')
        images = data['data']

        # reminder to self, transpose works by putting in the old positions
        temp = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        # Contiguous because cv2 will not take a reverse-strided view later.
        return np.ascontiguousarray(temp[..., ::-1])  # RGB -> BGR


class CifarGallery:
    """Tiles from a CIFAR-100 pickle batch. 32x32 images, resized to cell size."""

    def __init__(self, path: Path):
        self._path = Path(path)

    @property
    def fingerprint(self) -> str:
        return f"cifar:{self._path.resolve()}:{self._path.stat().st_mtime_ns}"

    @property
    def native_aspect(self) -> tuple[int, int] | None:
        return (1, 1)

    def estimate_count(self) -> int | None:
        # Labels and filenames pad the pickle, so this reads 1.1% high on the
        # real train batch (50537 against 50000). Fine for a budget.
        return self._path.stat().st_size // CIFAR_IMAGE_BYTES

    def load(self, cell_size: tuple[int, int], fit: Fit = "native",
             budget: int = HARD_BUDGET) -> Image:
        # The pickle's count is known before any resizing, so nothing can grow
        # past what the estimate already priced.
        return resize_gallery_to_cells(read_cifar_batch(self._path), cell_size, fit)


class VideoGallery:
    """Tiles decoded from a video (or a directory of them), keeping every
    `stride`-th frame. See docs/video-gallery.md."""

    def __init__(self, path: Path, stride: int = 10):
        self._path = Path(path)
        self._stride = stride
        self._count: int | None = None
        self._counted = False

    def _files(self) -> list[Path]:
        """The videos this gallery reads, in a stable order."""
        if self._path.is_dir():
            # Dotfiles are skipped for the AppleDouble `._name.mkv` stubs a rip
            # off a Mac leaves behind: right suffix, 4 KB of resource fork.
            return sorted(p for p in self._path.iterdir()
                          if p.suffix.lower() in VIDEO_SUFFIXES
                          and not p.name.startswith("."))
        return [self._path]

    @property
    def fingerprint(self) -> str:
        files = ",".join(f"{p.resolve()}:{p.stat().st_mtime_ns}" for p in self._files())
        return f"video:{files}:stride={self._stride}"

    @property
    def native_aspect(self) -> tuple[int, int] | None:
        """The first file's frame shape, assumed to hold for the rest.

        A season is one rip at one resolution; a mixed bag would need the cell
        to fit them all, which no single ratio does.
        """
        files = self._files()
        if not files:
            return None
        cap = cv2.VideoCapture(str(files[0]))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width <= 0 or height <= 0:
            return None
        ratio = Fraction(width, height)
        return (ratio.numerator, ratio.denominator)

    def estimate_count(self) -> int | None:
        """Frame count over stride, summed over the files. Upper bound.

        Container metadata, so it can lie — fine for a warning, never for
        sizing an allocation. Measured exact on the three rips to hand.

        Memoised: three call sites per load, and each one opens every file in
        the directory. A directory that changes mid-run gets a stale count,
        which costs a resize at worst.
        """
        if not self._counted:
            self._count = self._scan_count()
            self._counted = True
        return self._count

    def _scan_count(self) -> int | None:
        total = 0
        for path in self._files():
            cap = cv2.VideoCapture(str(path))
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if frames <= 0:
                # Nothing in the container; seek to the end for a duration instead.
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                frames = fps * ms / 1000.0 if fps > 0 and ms > 0 else 0
            cap.release()
            if frames <= 0:
                return None
            total += -(-int(frames) // self._stride)
        return total

    def load(self, cell_size: tuple[int, int], fit: Fit = "native",
             budget: int = HARD_BUDGET) -> Image:
        """Decode every file in order, keeping one frame in `stride`, deduped.

        Straight through, no seeking: frame-accurate seek on long-GOP video
        costs more than decoding past the frames we don't want. `grab()` skips
        the colour conversion for those, which is most of them.
        """
        cell_w, cell_h = cell_size
        files = self._files()
        if not files:
            raise ValueError(f"No videos found at {self._path}")

        tiles = np.empty((self.estimate_count() or FALLBACK_CAPACITY,
                          cell_h, cell_w, 3), dtype=np.uint8)
        seen: set[bytes] = set()
        kept = sampled = 0

        with tqdm.tqdm(desc="Decoding gallery...", unit="frame",
                       total=self._frame_total()) as bar:
            for path in files:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    raise ValueError(f"Video at {path} could not be opened!")

                index = 0
                while cap.grab():
                    if index % self._stride == 0:
                        ok, frame = cap.retrieve()
                        if ok:
                            if kept == len(tiles):
                                # The estimate was low, so re-price before
                                # doubling — nothing else stands between a
                                # lying container and an OOM. np.resize holds
                                # both buffers over the copy, so the moment
                                # costs half again on top.
                                grown = 2 * len(tiles)
                                enforce_gallery_budget(grown, cell_size, budget)
                                tiles = np.resize(tiles, (grown, cell_h,
                                                          cell_w, 3))
                            # Written straight into its slot, then kept or not:
                            # a scratch tile per frame would be the only copy.
                            # cv2's stubs won't commit to a dtype; decode is uint8.
                            fit_to_cell(cast(Image, frame), cell_size, fit,
                                        dst=tiles[kept])
                            digest = hashlib.blake2b(tiles[kept].tobytes(),
                                                     digest_size=8).digest()
                            sampled += 1
                            if digest not in seen:
                                seen.add(digest)
                                kept += 1
                    index += 1
                    bar.update()
                cap.release()

        print(f"Sampled {sampled} frames from {len(files)} file(s) -> "
              f"{kept} tiles ({sampled - kept} duplicates dropped)")
        # A slice would pin the whole buffer, and dedupe can leave most of it
        # empty. Copying costs a moment's double-hold to give the rest back.
        return tiles[:kept].copy() if kept < 0.9 * len(tiles) else tiles[:kept]

    def _frame_total(self) -> int | None:
        """Frames to be decoded, for the progress bar. Metadata, so a hint only."""
        count = self.estimate_count()
        return count * self._stride if count else None


DEFAULT_CACHE_DIR = Path(".cache/gallery")


# Bump when the pixels a given (source, cell, fit) produces change — v2 is the
# INTER_AREA switch in `fit_to_cell()`. Nothing in the key covers resampling,
# so without this a stale cache serves the old tiles forever.
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
    """Price a tile array of `count` tiles, raising if it's over budget.

    Called on the estimate up front, and again by a source that outgrows its
    estimate mid-decode — the estimate is container metadata and metadata lies.
    """
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
        tiles: Image = np.load(path)
        # A truncated or hand-edited file isn't worth trusting over a re-decode.
        if tiles.shape[1:] == (cell_size[1], cell_size[0], 3):
            print(f"Gallery cache hit: {path}")
            return tiles
        print(f"Gallery cache at {path} is malformed, reloading")

    # Only worth estimating on the path that decodes; a cache hit knows the truth.
    check_gallery_budget(source, cell_size, budget)
    tiles = source.load(cell_size, fit, budget)
    if not use_cache:
        return tiles

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Write-then-rename, so a run killed mid-write leaves the old cache intact.
    # Keeps the .npy suffix, which np.save would otherwise append itself.
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temp, tiles, allow_pickle=False)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return tiles


def gallery_brightness(gallery: Image) -> Brightness:
    """Average brightness (0-1) for each gallery image.

    Luma is linear, so the mean of the luma equals the luma of the channel
    means: one (N, 3) reduction instead of a per-image cvtColor.
    """
    brightness: Brightness = gallery.mean(axis=(1, 2)) @ LUMA_BGR / 255.0
    return brightness


def probe_video(config: UserConfig,
                tile_aspect: tuple[int, int] | None = None) -> DerivedConfig:
    """Read source video metadata and derive the grid, cell and target size."""
    cap = cv2.VideoCapture(config.input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {config.input_dir} not found!")

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # Container metadata, used only as a tqdm display hint — may be inaccurate.
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    derived = DerivedConfig.from_source(config, fps=fps, dimensions=dimensions,
                                        frame_count=frame_count,
                                        tile_aspect=tile_aspect)

    print(f"Source: {derived.src_dimensions}, Target: {derived.target_dimensions}, "
          f"Grid: {derived.grid_x}x{derived.grid_y}, Cell: {derived.cell_size}")
    return derived


def stream_frames(config: UserConfig, derived: DerivedConfig) -> Iterator[Image]:
    """Decode and yield source frames one at a time, resized to target dimensions."""
    cap = cv2.VideoCapture(config.input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {config.input_dir} not found!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2's stubs won't commit to a dtype, but resize keeps the input's.
        yield cast(Image, cv2.resize(frame, derived.target_dimensions))

    cap.release()


class BrightnessMetric:
    """Match each cell to a gallery image of near-identical average brightness.

    A cell's brightness rounds to one of 256 levels, so precompute resolves
    every possible match once: each level gets a bucket of the `candidates`
    nearest images (none further than `epsilon`) and match() samples one.
    See docs/brightness-matching.md.
    """

    def __init__(self, candidates: int = 1, epsilon: float = 0.0, seed: int = 0):
        self._candidates = candidates
        self._epsilon = epsilon
        self._rng = np.random.default_rng(seed)

    def precompute(self, gallery: Image, cell_size: tuple[int, int],
                   brightness: Brightness | None = None) -> None:
        # Tiles arrive at cell size; unchecked, a mismatch surfaces much later
        # as a misshapen mosaic.
        cell_w, cell_h = cell_size
        if gallery.shape[1:3] != (cell_h, cell_w):
            raise ValueError(f"gallery tiles are {gallery.shape[2]}x{gallery.shape[1]}, "
                             f"expected cell size {cell_w}x{cell_h}")

        bright = gallery_brightness(gallery) if brightness is None else brightness
        order = np.argsort(bright)
        sorted_bright = bright[order]
        n = len(sorted_bright)
        levels = np.arange(256) / 255.0
        k = int(np.clip(self._candidates, 1, n))

        # The k nearest images to a level are contiguous in the sorted gallery,
        # so a bucket is just an offset and a count.
        insert = np.searchsorted(sorted_bright, levels)
        eps_lo = np.searchsorted(sorted_bright, levels - self._epsilon, side="left")
        eps_hi = np.searchsorted(sorted_bright, levels + self._epsilon, side="right")

        lo = np.empty(256, dtype=np.int64)
        count = np.empty(256, dtype=np.int64)
        for i, level in enumerate(levels):
            # Centre a k-wide window on the level, then slide it onto the true
            # k nearest — brightness isn't spread uniformly.
            start = min(max(insert[i] - k // 2, 0), n - k)
            while start > 0 and level - sorted_bright[start - 1] < sorted_bright[start + k - 1] - level:
                start -= 1
            while start + k < n and sorted_bright[start + k] - level < level - sorted_bright[start]:
                start += 1

            begin, end = max(start, eps_lo[i]), min(start + k, eps_hi[i])
            if end <= begin:
                # Nothing within epsilon: take whichever neighbour is closer.
                left, right = max(insert[i] - 1, 0), min(insert[i], n - 1)
                begin = (left if abs(level - sorted_bright[left])
                         <= abs(sorted_bright[right] - level) else right)
                end = begin + 1
            lo[i], count[i] = begin, end - begin

        # Compact the gallery to the images some bucket can actually reach,
        # counting span coverage with a difference array.
        spans = np.zeros(n + 1, dtype=np.int64)
        np.add.at(spans, lo, 1)
        np.add.at(spans, lo + count, -1)
        reachable = np.flatnonzero(np.cumsum(spans)[:n] > 0)
        self._remap = np.zeros(n, dtype=np.int64)
        self._remap[reachable] = np.arange(len(reachable))

        self._lo, self._count = lo, count
        self._tiles = gallery[order[reachable]]
        self._cell_w, self._cell_h = cell_size

    def match(self, frame: Image) -> Indices:
        """(H, W, 3) frame -> (grid_y, grid_x) array of tile indices."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grid_y = grey.shape[0] // self._cell_h
        grid_x = grey.shape[1] // self._cell_w

        cell_means = grey.reshape(grid_y, self._cell_h,
                                  grid_x, self._cell_w).mean(axis=(1, 3))
        levels = np.rint(cell_means).astype(np.uint8)

        lo = self._lo[levels]
        picks = lo + self._rng.integers(self._count[levels])
        indices: Indices = self._remap[picks]
        return indices

    @property
    def bucket_size(self) -> float:
        """Median candidates per level, ignoring levels outside the gallery's range."""
        real = self._count[self._count > 1]
        return float(np.median(real)) if len(real) else 1.0

    @property
    def tiles(self) -> Image:
        """(U, cell_h, cell_w, 3) pre-resized tiles, indexed by match()."""
        return self._tiles


def mosaic_frame(frame: Image, metric: BrightnessMetric) -> Image:
    """Build a mosaic for a single frame by matching each grid cell."""
    tiles = metric.tiles[metric.match(frame)]
    grid_y, grid_x, cell_h, cell_w, _ = tiles.shape
    return tiles.transpose(0, 2, 1, 3, 4).reshape(grid_y * cell_h, grid_x * cell_w, 3)


def shrink_gallery(gallery: Image, brightness: Brightness,
                   config: UserConfig) -> tuple[Image, Brightness]:
    """Trim the gallery to a percentile band of brightness around the midpoint.

    Hands back the surviving brightnesses too, so the caller needs no second
    pass — and the very arrays it was given when nothing is trimmed.
    """
    percentiles = (50 - (50 * config.contrast), 50 + (50 * config.contrast))
    low = np.percentile(brightness, percentiles[0])
    high = np.percentile(brightness, percentiles[1])
    mask = (brightness >= low) & (brightness <= high)
    if mask.all():
        # Boolean indexing would copy the lot anyway: a second full-size array
        # beside the first, the peak of the whole run. See docs/gallery-size.md.
        return gallery, brightness

    return gallery[mask], brightness[mask]


def build_metric(gallery: Image, config: UserConfig,
                 derived: DerivedConfig) -> BrightnessMetric:
    """Trim the gallery and precompute the matcher over what survives."""
    brightness = gallery_brightness(gallery)
    # Rebinding drops the last reference to the loaded tiles, so precompute's
    # copy replaces them rather than joining them. See docs/gallery-size.md.
    gallery, brightness = shrink_gallery(gallery, brightness, config)

    metric = BrightnessMetric(candidates=config.candidates,
                              epsilon=config.epsilon, seed=config.seed)
    metric.precompute(gallery, derived.cell_size, brightness)
    print(f"Gallery: {len(gallery)} images -> {len(metric.tiles)} usable tiles, "
          f"{metric.bucket_size:.0f} candidates per cell (median)")
    return metric


def build_mosaics(frames: Iterator[Image], metric: BrightnessMetric,
                  derived: DerivedConfig) -> Iterator[Image]:
    """Turn a stream of source frames into a stream of mosaics.

    Lazy end to end: one frame in, one mosaic out, so peak memory holds a couple
    of frames however long the source is.
    """
    for frame in tqdm.tqdm(frames, desc="Building mosaics...",
                           total=derived.src_frame_count or None):
        yield mosaic_frame(frame, metric)


def encode_video(mosaics: Iterator[Image], derived: DerivedConfig,
                 output_path: Path) -> Path:
    """Pipe raw mosaic frames into ffmpeg and return the encoded file's path."""
    width, height = derived.target_dimensions
    proc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-framerate", str(derived.output_fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-y", str(output_path)
    ], stdin=subprocess.PIPE)
    # Only ever None when stdin=PIPE wasn't asked for, which it was.
    assert proc.stdin is not None

    try:
        for mosaic in mosaics:
            proc.stdin.write(mosaic.tobytes())
    except BrokenPipeError:
        # ffmpeg died early; its exit code below is the useful error.
        pass
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed with exit code {proc.returncode}")
    return output_path


def combine_videos(source_path: Path, mosaic_path: Path, output_path: Path,
                   dimensions: tuple[int, int],
                   total_frames: int = 0) -> Path:
    """Stack the source and its mosaic side by side into one video.

    `hstack` wants both inputs at the same height, and target dimensions are
    snapped to a grid multiple, so the source gets scaled rather than trusted
    to match.

    `-progress pipe:1` makes ffmpeg emit `key=value` lines on stdout as it goes;
    the `frame=` ones drive the bar.
    """
    width, height = dimensions
    proc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-progress", "pipe:1",
        "-i", str(source_path),
        "-i", str(mosaic_path),
        "-filter_complex",
        f"[0:v]scale={width}:{height},setsar=1[src];[src][1:v]hstack=inputs=2",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-y", str(output_path)
    ], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    # Only ever None when stdout=PIPE wasn't asked for, which it was.
    assert proc.stdout is not None

    with tqdm.tqdm(desc="Combining videos...", total=total_frames or None,
                   unit="frame") as bar:
        for line in proc.stdout:
            key, _, value = line.partition("=")
            if key == "frame":
                # It's a running total, not a delta.
                bar.update(int(value) - bar.n)
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg combine failed with exit code {proc.returncode}")
    return output_path


def main(gallery_source: GallerySource, config: UserConfig) -> Path:
    """Run the pipeline end to end, returning the combined video's path."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe first: the gallery can't load until the cell size is known. Under
    # `native` the cell shape is the gallery's, which is cheap metadata.
    tile_aspect = (gallery_source.native_aspect
                   if config.tile_fit == "native" else None)
    derived = probe_video(config, tile_aspect)
    # Handed straight over: a local pinning the tiles would keep the whole array
    # alive beside the metric's own copy for the rest of the run.
    metric = build_metric(
        load_gallery(gallery_source, derived, fit=config.tile_fit,
                     budget=config.gallery_budget),
        config, derived)

    frames = stream_frames(config, derived)
    mosaics = build_mosaics(frames, metric, derived)
    mosaic_path = encode_video(mosaics, derived, output_dir / "output.mp4")

    combined = combine_videos(Path(config.input_dir), mosaic_path,
                              output_dir / "combined.mp4",
                              derived.target_dimensions,
                              derived.src_frame_count)
    print(f"Done. Output written to ./{output_dir}/")
    return combined


if __name__ == "__main__":
    # Swap for VideoGallery(Path("./assets/videos"), stride=10) to build the
    # tiles out of your own videos instead. CIFAR is what docs/ tells you to
    # download, so it's what a clean checkout can actually run.
    main(CifarGallery(Path("./assets/gallery/train")),
         UserConfig(input_dir="./assets/source.mp4",
                    output_dir="./output/",
                    contrast=1.0,
                    grid_size=8))
