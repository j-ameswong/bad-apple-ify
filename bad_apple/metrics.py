from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import cv2

from .config import DerivedConfig, UserConfig
from .types import Brightness, Image, Indices

LUMA_BGR = np.array([0.114, 0.587, 0.299])

def gallery_brightness(gallery: Image) -> Brightness:
    """Average brightness (0-1) for each gallery image.

    Luma is linear, so the mean of the luma equals the luma of the channel
    means: one (N, 3) reduction instead of a per-image cvtColor.
    """
    brightness: Brightness = gallery.mean(axis=(1, 2)) @ LUMA_BGR / 255.0
    return brightness

class Metric(Protocol):
    """How a grid cell picks its tile.

    Grid-wise, not cell-wise: a per-cell `score()` would put back the Python
    loop that vectorising `mosaic_frame()` took out.
    """

    def precompute(self, gallery: Image, cell_size: tuple[int, int],
                   brightness: Brightness | None = None) -> None:
        """Build the lookup and keep the tiles it can reach.

        `brightness` is the caller's leftover from `shrink_gallery()`, offered
        so nothing recomputes it; a metric with no use for it ignores it.
        """
        ...

    def keys(self, frame: Image) -> Indices:
        """(H, W, 3) frame -> (grid_y, grid_x) array of bucket keys.

        A key says which bucket a cell landed in, not which tile it got. Two
        frames agreeing here agree about the picture; see docs/colour-matching.md.
        """
        ...

    def sample(self, keys: Indices) -> Indices:
        """Bucket keys -> one tile index each, drawn from the bucket."""
        ...

    def match(self, frame: Image) -> Indices:
        """(H, W, 3) frame -> (grid_y, grid_x) array of indices into `tiles`."""
        ...

    @property
    def tiles(self) -> Image:
        """(U, cell_h, cell_w, 3) tiles at cell size, indexed by `match()`."""
        ...

    @property
    def bucket_size(self) -> float:
        """Median tiles a cell picks between — what `candidates` actually bought."""
        ...

def check_cell_size(gallery: Image, cell_size: tuple[int, int]) -> None:
    """Tiles arrive at cell size; unchecked, a mismatch surfaces much later as a
    misshapen mosaic."""
    cell_w, cell_h = cell_size
    if gallery.shape[1:3] != (cell_h, cell_w):
        raise ValueError(f"gallery tiles are {gallery.shape[2]}x{gallery.shape[1]}, "
                         f"expected cell size {cell_w}x{cell_h}")
    if len(gallery) == 0:
        raise ValueError("gallery is empty, nothing to match against")

def cell_means(image: Image, cell_size: tuple[int, int]) -> npt.NDArray[np.float64]:
    """Per-cell mean of every channel: (H, W, ...) -> (grid_y, grid_x, ...).

    Exact, and taken one axis at a time: a single `.mean(axis=(1, 3))` over the
    strided uint8 block costs 10x as much. See docs/colour-matching.md.
    """
    cell_w, cell_h = cell_size
    h, w = image.shape[:2]
    block = image.reshape(h // cell_h, cell_h, w // cell_w, cell_w,
                          *image.shape[2:])
    totals = block.sum(axis=1, dtype=np.uint32).sum(axis=2)
    means: npt.NDArray[np.float64] = totals / (cell_h * cell_w)
    return means

def compact_buckets(lo: Indices, count: Indices, n: int) -> tuple[Indices, Indices]:
    """Which of `n` sorted tiles some bucket reaches, and where each one lands.

    Returns (reachable positions, sorted position -> compacted index), counting
    coverage of the `[lo, lo + count)` spans with a difference array.
    """
    spans = np.zeros(n + 1, dtype=np.int64)
    np.add.at(spans, lo, 1)
    np.add.at(spans, lo + count, -1)
    reachable = np.flatnonzero(np.cumsum(spans)[:n] > 0)
    remap = np.zeros(n, dtype=np.int64)
    remap[reachable] = np.arange(len(reachable))
    return reachable, remap

def draw_from_buckets(lo: Indices, count: Indices, remap: Indices,
                      keys: Indices, rng: np.random.Generator) -> Indices:
    """Pick one tile per key, uniformly among its bucket's candidates.

    Both metrics reduce to this once they've turned a frame into keys — a
    brightness level and a lattice index index the same three arrays.
    """
    picks = lo[keys] + rng.integers(count[keys])
    indices: Indices = remap[picks]
    return indices

class BrightnessMetric:
    """Match each cell to a gallery image of near-identical average brightness.

    Each of the 256 levels a cell can round to gets a bucket of the `candidates`
    nearest images, none past `epsilon`. See docs/brightness-matching.md.
    """

    def __init__(self, candidates: int = 1, epsilon: float = 0.0, seed: int = 0):
        self._candidates = candidates
        self._epsilon = epsilon
        self._rng = np.random.default_rng(seed)

    def precompute(self, gallery: Image, cell_size: tuple[int, int],
                   brightness: Brightness | None = None) -> None:
        check_cell_size(gallery, cell_size)
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

        reachable, self._remap = compact_buckets(lo, count, n)
        self._lo, self._count = lo, count
        self._tiles = gallery[order[reachable]]
        self._cell_w, self._cell_h = cell_size

    def keys(self, frame: Image) -> Indices:
        """(H, W, 3) frame -> (grid_y, grid_x) array of brightness levels."""
        grey = cast(Image, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        means = cell_means(grey, (self._cell_w, self._cell_h))
        return np.rint(means).astype(np.int64)

    def sample(self, keys: Indices) -> Indices:
        return draw_from_buckets(self._lo, self._count, self._remap,
                                 keys, self._rng)

    def match(self, frame: Image) -> Indices:
        return self.sample(self.keys(frame))

    @property
    def bucket_size(self) -> float:
        """Median candidates per level, ignoring levels outside the gallery's range."""
        real = self._count[self._count > 1]
        return float(np.median(real)) if len(real) else 1.0

    @property
    def tiles(self) -> Image:
        return self._tiles

def nearest_occupied(occupied: npt.NDArray[np.bool_], bins: int) -> Indices:
    """For every cell of a bins^3 lattice, the index of the nearest occupied one.

    A BFS wave over the six face neighbours, so "nearest" is Manhattan rather
    than Euclidean. See docs/colour-matching.md.
    """
    if not occupied.any():
        raise ValueError("gallery has no tiles to match against")

    donor = np.where(occupied, np.arange(bins ** 3), -1).reshape(bins, bins, bins)
    while (empty := donor < 0).any():
        # Donors from the previous wave only, so every cell takes the closest
        # one rather than whichever direction happened to be checked last.
        source = donor.copy()
        for axis in range(3):
            for shift in (1, -1):
                near = np.roll(source, shift, axis=axis)
                # roll wraps the lattice around; colour space doesn't.
                edge = [slice(None)] * 3
                edge[axis] = slice(0, 1) if shift == 1 else slice(bins - 1, bins)
                near[tuple(edge)] = -1
                fill = empty & (near >= 0)
                donor[fill] = near[fill]
                empty &= ~fill
    indices: Indices = donor.reshape(-1)
    return indices

class ColourMetric:
    """Match each cell to a gallery image of near-identical average colour.

    Mean BGR quantised onto a `bins`^3 lattice, where an empty lattice cell
    borrows the nearest occupied one. See docs/colour-matching.md.
    """

    def __init__(self, bins: int = 32, candidates: int = 1, seed: int = 0):
        # The lattice is bins^3 cells and the fill walks it wave by wave, so the
        # ceiling is about keeping precompute in milliseconds.
        if not 2 <= bins <= 64:
            raise ValueError(f"colour_bins must be between 2 and 64, got {bins}")
        self._bins = bins
        self._candidates = candidates
        self._rng = np.random.default_rng(seed)

    def precompute(self, gallery: Image, cell_size: tuple[int, int],
                   brightness: Brightness | None = None) -> None:
        check_cell_size(gallery, cell_size)
        bins = self._bins
        keys = self._quantise(gallery.mean(axis=(1, 2)))
        # Tiles sharing a lattice cell land contiguously, so a bucket is an
        # offset and a count, same as a brightness bucket.
        order = np.argsort(keys, kind="stable")
        lo = np.searchsorted(keys[order], np.arange(bins ** 3))
        # Everything in a lattice cell is equally acceptable by construction —
        # the bin width *is* the tolerance — so the cap takes the first few.
        held = np.minimum(np.bincount(keys, minlength=bins ** 3),
                          max(self._candidates, 1))

        donor = nearest_occupied(held > 0, bins)
        self._lo, self._count = lo[donor], held[donor]
        reachable, self._remap = compact_buckets(self._lo, self._count, len(gallery))
        self._held = held
        self._tiles = gallery[order[reachable]]
        self._cell_w, self._cell_h = cell_size

    def _quantise(self, colours: npt.NDArray[np.float64]) -> Indices:
        """(..., 3) BGR in 0-255 -> (...) flat index into the lattice."""
        q = np.clip((colours * self._bins / 256).astype(np.int64), 0, self._bins - 1)
        flat: Indices = (q[..., 0] * self._bins + q[..., 1]) * self._bins + q[..., 2]
        return flat

    def keys(self, frame: Image) -> Indices:
        """(H, W, 3) frame -> (grid_y, grid_x) array of lattice indices."""
        return self._quantise(cell_means(frame, (self._cell_w, self._cell_h)))

    def sample(self, keys: Indices) -> Indices:
        return draw_from_buckets(self._lo, self._count, self._remap,
                                 keys, self._rng)

    def match(self, frame: Image) -> Indices:
        return self.sample(self.keys(frame))

    @property
    def bucket_size(self) -> float:
        """Median candidates per occupied lattice cell.

        Empty cells are left out: they're copies of a neighbour's bucket, and
        counting them would just weight whichever colours are widespread.
        """
        real = self._held[self._held > 0]
        return float(np.median(real)) if len(real) else 1.0

    @property
    def tiles(self) -> Image:
        return self._tiles

class SteadyMetric:
    """Hold a cell's tile for as long as its bucket key doesn't change.

    Without it every cell re-rolls its bucket every frame, so regions that
    aren't moving still shimmer. See docs/colour-matching.md.
    """

    def __init__(self, inner: Metric):
        self._inner = inner
        # Last frame's keys beside what they drew, or None on the first frame.
        self._previous: tuple[Indices, Indices] | None = None

    def precompute(self, gallery: Image, cell_size: tuple[int, int],
                   brightness: Brightness | None = None) -> None:
        self._inner.precompute(gallery, cell_size, brightness)
        self._previous = None

    def keys(self, frame: Image) -> Indices:
        return self._inner.keys(frame)

    def sample(self, keys: Indices) -> Indices:
        # Draws for every cell and throws most of it away — cheaper than the
        # bookkeeping a partial draw needs.
        picks = self._inner.sample(keys)
        if self._previous is not None and self._previous[0].shape == keys.shape:
            last_keys, last_picks = self._previous
            picks = np.where(keys == last_keys, last_picks, picks)
        self._previous = (keys, picks)
        return picks

    def match(self, frame: Image) -> Indices:
        return self.sample(self.keys(frame))

    @property
    def bucket_size(self) -> float:
        return self._inner.bucket_size

    @property
    def tiles(self) -> Image:
        return self._inner.tiles

def mosaic_frame(frame: Image, metric: Metric) -> Image:
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
                 derived: DerivedConfig) -> Metric:
    """Trim the gallery and precompute the matcher over what survives."""
    brightness = gallery_brightness(gallery)
    # Rebinding drops the last reference to the loaded tiles, so precompute's
    # copy replaces them rather than joining them. See docs/gallery-size.md.
    gallery, brightness = shrink_gallery(gallery, brightness, config)

    metric: Metric
    if config.metric == "colour":
        metric = ColourMetric(bins=config.colour_bins,
                              candidates=config.candidates, seed=config.seed)
    else:
        metric = BrightnessMetric(candidates=config.candidates,
                                  epsilon=config.epsilon, seed=config.seed)
    if config.hold_tiles:
        metric = SteadyMetric(metric)
    metric.precompute(gallery, derived.cell_size, brightness)
    print(f"Gallery: {len(gallery)} images -> {len(metric.tiles)} usable tiles, "
          f"{metric.bucket_size:.0f} candidates per cell (median)")
    return metric
