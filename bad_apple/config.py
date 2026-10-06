from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, isfinite

from .types import Fit, Image, MetricName

SOFT_BUDGET = 1 << 30

HARD_BUDGET = 8 << 30

@dataclass(frozen=True)
class UserConfig:
    """Parameters the user supplies. Nothing here depends on the source video."""

    input_dir: str
    output_dir: str
    contrast: float = 0.1
    grid_size: int = 16  # multiplier for aspect ratio
    cell_size: int | None = None  # output cell height; native tiles set width from their ratio
    grid: tuple[int, int] | None = None  # explicit columns, rows
    tile_fit: Fit = "native"  # cell takes the tiles' own ratio, or crop/stretch into it
    metric: MetricName = "colour"  # what a cell matches on
    candidates: int = 256  # tiles to sample from per bucket
    epsilon: float = 0.005  # brightness only: max error (0-1) a candidate may have
    colour_bins: int = 32  # colour only: lattice edge, so bins^3 buckets
    seed: int = 0
    hold_tiles: bool = True  # keep a cell's tile while its bucket is unchanged
    gallery_budget: int = HARD_BUDGET  # bytes of tiles to refuse past
    use_cache: bool = True
    segment_frames: int = 0
    start: float = 0.0  # source seconds, inclusive
    duration: float | None = None  # source seconds; None runs to EOF

    def __post_init__(self) -> None:
        if isinstance(self.contrast, bool) or not isinstance(self.contrast, (int, float)) \
                or not isfinite(self.contrast) \
                or not 0 < self.contrast <= 1:
            raise ValueError("contrast must be greater than 0 and at most 1")
        if not _positive_int(self.grid_size):
            raise ValueError("grid_size must be a positive integer")
        if self.cell_size is not None and not _positive_int(self.cell_size):
            raise ValueError("cell_size must be a positive integer")
        if self.grid is not None and (not isinstance(self.grid, tuple) or len(self.grid) != 2
                                      or not all(_positive_int(value) for value in self.grid)):
            raise ValueError("grid must be a pair of positive integers (columns, rows)")
        if self.tile_fit not in ("native", "crop", "stretch"):
            raise ValueError("tile_fit must be 'native', 'crop' or 'stretch'")
        if self.metric not in ("colour", "brightness"):
            raise ValueError("metric must be 'colour' or 'brightness'")
        if not _positive_int(self.candidates):
            raise ValueError("candidates must be a positive integer")
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)) \
                or not isfinite(self.epsilon) \
                or self.epsilon < 0:
            raise ValueError("epsilon must be a finite, non-negative number")
        if not _positive_int(self.colour_bins) or not 2 <= self.colour_bins <= 64:
            raise ValueError("colour_bins must be an integer between 2 and 64")
        if not _nonnegative_int(self.seed):
            raise ValueError("seed must be a non-negative integer")
        if not _positive_int(self.gallery_budget):
            raise ValueError("gallery_budget must be a positive integer")
        if not _nonnegative_int(self.segment_frames):
            raise ValueError("segment_frames must be a non-negative integer")
        if not isinstance(self.hold_tiles, bool) or not isinstance(self.use_cache, bool):
            raise ValueError("hold_tiles and use_cache must be booleans")
        if isinstance(self.start, bool) or not isinstance(self.start, (int, float)) \
                or not isfinite(self.start) or self.start < 0:
            raise ValueError("start must be a finite, non-negative number of seconds")
        if self.duration is not None and (isinstance(self.duration, bool)
                                          or not isinstance(self.duration, (int, float))
                                          or not isfinite(self.duration)
                                          or self.duration <= 0):
            raise ValueError("duration must be a finite, positive number of seconds")

def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0

def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0

def even_span(count: int, cell: int, ideal: float) -> tuple[int, int]:
    """(count, cell) with an even product, one step off if need be, nearest `ideal`.

    The count takes the step so the cell keeps its shape, unless the count is
    the single frame's 1. See docs/grid-and-sizing.md.
    """
    if count * cell % 2 == 0:
        return count, cell
    options = ([(count, c) for c in (cell - 1, cell + 1) if c > 0] if count == 1
               else [(count - 1, cell), (count + 1, cell)])
    return min(options, key=lambda o: (abs(o[0] * o[1] - ideal), o[0] * o[1]))

@dataclass(frozen=True)
class DerivedConfig:
    """Everything computed from a `UserConfig` once the source has been probed.

    Built once by `probe_video()` and never mutated, so the grid and the frame
    size can't drift apart mid-run. See docs/grid-and-sizing.md.
    """

    # Exact, never rounded: 29.97 as 30 drifts a frame every 33s.
    src_fps: Fraction
    src_dimensions: tuple[int, int]
    src_frame_count: int
    aspect_ratio: tuple[int, int]
    grid: tuple[int, int]
    cell_size: tuple[int, int]
    start_frame: int = 0
    stop_frame: int | None = None  # exclusive; never clipped to source metadata

    @classmethod
    def from_source(cls, config: UserConfig, *, fps: Fraction,
                    dimensions: tuple[int, int], frame_count: int,
                    tile_aspect: tuple[int, int] | None = None) -> "DerivedConfig":
        if not isfinite(fps) or fps <= 0:
            raise ValueError("source frame rate must be positive")
        if len(dimensions) != 2 or not all(_positive_int(value) for value in dimensions):
            raise ValueError("source dimensions must be a pair of positive integers")
        if tile_aspect is not None and (len(tile_aspect) != 2
                                        or not all(_positive_int(value)
                                                   for value in tile_aspect)):
            raise ValueError("tile aspect must be a pair of positive integers")
        # Keep frame timestamps in [start, start + duration), including at NTSC rates.
        start = Fraction(str(config.start))
        start_frame = ceil(start * fps)
        stop_frame = (None if config.duration is None else
                      ceil((start + Fraction(str(config.duration))) * fps))
        if stop_frame is not None and stop_frame <= start_frame:
            raise ValueError("the requested source range contains no frames")
        # Simplest integer pair near the source ratio, so 2.39:1 stays 2.39:1.
        ratio = Fraction(*dimensions).limit_denominator(16)
        aspect_ratio = (ratio.numerator, ratio.denominator)
        if config.cell_size is not None or config.grid is not None:
            grid = (config.grid if config.grid is not None else
                    ((1, 1) if config.grid_size == 1 else
                     (aspect_ratio[0] * config.grid_size,
                      aspect_ratio[1] * config.grid_size)))
            if config.cell_size is not None:
                cell_h = config.cell_size
                # A single native tile fills the source-shaped frame. Else native
                # tiles set the width from their own ratio; crop/stretch stay square.
                native_single = (config.grid_size == 1 and config.grid is None
                                 and config.tile_fit == "native")
                if native_single:
                    cell_w = max(round(cell_h * dimensions[0] / dimensions[1]), 1)
                elif tile_aspect is not None and config.tile_fit == "native":
                    cell_w = max(round(cell_h * tile_aspect[0] / tile_aspect[1]), 1)
                else:
                    cell_w = cell_h
            else:
                cell_w = max(round(dimensions[0] / grid[0]), 1)
                cell_h = max(round(dimensions[1] / grid[1]), 1)
                if tile_aspect is not None and config.tile_fit == "native" \
                        and grid != (1, 1):
                    cell_w = max(round(cell_h * tile_aspect[0] / tile_aspect[1]), 1)
            if grid[0] * cell_w % 2 or grid[1] * cell_h % 2:
                raise ValueError("explicit grid and cell size must produce even output dimensions for yuv420p")
            return cls(src_fps=fps, src_dimensions=dimensions,
                       src_frame_count=frame_count, aspect_ratio=aspect_ratio,
                       grid=grid, cell_size=(cell_w, cell_h),
                       start_frame=start_frame, stop_frame=stop_frame)

        # Single-frame mode wants a literal 1x1, not the aspect pair.
        grid = ((1, 1) if config.grid_size == 1
                else (aspect_ratio[0] * config.grid_size,
                      aspect_ratio[1] * config.grid_size))
        cell_size = (max(round(dimensions[0] / grid[0]), 1),
                     max(round(dimensions[1] / grid[1]), 1))

        if tile_aspect is not None and config.grid_size > 1:
            # Columns fit the *snapped* width: the raw one stopped sharing a
            # scale with the cell once the rows rounded. See docs/tile-shape.md.
            cell_h = cell_size[1]
            snapped_w = grid[0] * cell_size[0]
            cell_w = max(round(cell_h * tile_aspect[0] / tile_aspect[1]), 1)
            grid = (max(round(snapped_w / cell_w), 1), grid[1])
            cell_size = (cell_w, cell_h)

        # yuv420p won't take an odd side. Rows first, then the width they imply.
        rows, cell_h = even_span(grid[1], cell_size[1], dimensions[1])
        ideal_w = rows * cell_h * dimensions[0] / dimensions[1]
        # A row step moves the height a whole cell, so recount the columns too.
        cols = (grid[0] if rows == grid[1]
                else max(round(ideal_w / cell_size[0]), 1))
        cols, cell_w = even_span(cols, cell_size[0], ideal_w)
        grid, cell_size = (cols, rows), (cell_w, cell_h)

        return cls(src_fps=fps, src_dimensions=dimensions,
                   src_frame_count=frame_count, aspect_ratio=aspect_ratio,
                   grid=grid, cell_size=cell_size,
                   start_frame=start_frame, stop_frame=stop_frame)

    @property
    def grid_x(self) -> int:
        return self.grid[0]

    @property
    def grid_y(self) -> int:
        return self.grid[1]

    @property
    def output_fps(self) -> Fraction:
        return self.src_fps

    @property
    def output_frame_count(self) -> int | None:
        """Progress hint for the slice, still subject to bad source metadata."""
        remaining = (max(self.src_frame_count - self.start_frame, 0)
                     if self.src_frame_count > 0 else None)
        if self.stop_frame is None:
            return remaining
        requested = self.stop_frame - self.start_frame
        return requested if remaining is None else min(requested, remaining)

    @property
    def target_dimensions(self) -> tuple[int, int]:
        return (self.grid[0] * self.cell_size[0],
                self.grid[1] * self.cell_size[1])
