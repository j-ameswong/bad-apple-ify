from typing import Iterator
import numpy as np
import cv2
import tqdm
import pickle
import warnings
from fractions import Fraction
from pathlib import Path
from dataclasses import dataclass
import subprocess

# Rec.601 luma weights in BGR order — what cv2.COLOR_BGR2GRAY applies.
LUMA_BGR = np.array([0.114, 0.587, 0.299])


@dataclass(frozen=True)
class UserConfig:
    """Parameters the user supplies. Nothing here depends on the source video."""

    input_dir: str
    output_dir: str
    contrast: float = 0.1
    grid_size: int = 16  # multiplier for aspect ratio
    candidates: int = 256  # tiles to sample from per brightness level
    epsilon: float = 0.005  # max brightness error (0-1) a candidate may have
    seed: int = 0


@dataclass(frozen=True)
class DerivedConfig:
    """Everything computed from a `UserConfig` once the source has been probed.

    Built once by `probe_video()` and never mutated afterwards, so the grid and
    the frame size a run was set up with cannot drift apart mid-pipeline.
    """

    src_fps: int
    src_dimensions: tuple[int, int]
    src_frame_count: int
    # Smallest integer pair approximating the source's aspect ratio. The grid
    # only needs *some* integer pair, so any source keeps its own shape instead
    # of being snapped to an allowlisted ratio.
    aspect_ratio: tuple[int, int]
    grid: tuple[int, int]
    cell_size: tuple[int, int]

    @classmethod
    def from_source(cls, config: UserConfig, *, fps: int,
                    dimensions: tuple[int, int], frame_count: int) -> "DerivedConfig":
        # Any integer pair close to the source ratio will do, so take the
        # simplest one rather than snapping to a fixed list — a 2.39:1 source
        # should stay 2.39:1, not get stretched to 16:9. Capping the denominator
        # keeps the pair small: an exact reduction of e.g. 2048x858 is 1024:429,
        # which is useless as a grid multiplier.
        ratio = Fraction(*dimensions).limit_denominator(16)
        aspect_ratio = (ratio.numerator, ratio.denominator)
        grid = (aspect_ratio[0] * config.grid_size,
                aspect_ratio[1] * config.grid_size)
        # Snap the frame to the nearest whole multiple of the grid, at least
        # 1px per cell however fine the grid is.
        cell_size = (max(round(dimensions[0] / grid[0]), 1),
                     max(round(dimensions[1] / grid[1]), 1))
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


def get_gallery(input_dir: str) -> np.ndarray:
    """Gets CIFAR gallery and converts to BGR."""
    with open(Path(input_dir), 'rb') as fo:
        with warnings.catch_warnings():
            # The CIFAR pickle carries a dtype pickled by an ancient NumPy with
            # `align=0`; NumPy 2.4 deprecates the int form. Nothing we can fix
            # from this side short of re-serialising the file.
            warnings.filterwarnings("ignore", message=".*align=0.*")
            data = pickle.load(fo, encoding='latin1')
        images = data['data']

        # reminder to self, transpose works by putting in the old positions
        temp = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        # Contiguous because cv2 will not take a reverse-strided view later.
        return np.ascontiguousarray(temp[..., ::-1])  # RGB -> BGR


def gallery_brightness(gallery: np.ndarray) -> np.ndarray:
    """Average brightness (0-1) for each gallery image.

    Luma is linear, so the mean of the luma equals the luma of the channel
    means — one (N, 3) reduction instead of a per-image cvtColor.
    """
    return gallery.mean(axis=(1, 2)) @ LUMA_BGR / 255.0


def probe_video(config: UserConfig) -> DerivedConfig:
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
                                        frame_count=frame_count)

    print(f"Source: {derived.src_dimensions}, Target: {derived.target_dimensions}, "
          f"Grid: {derived.grid_x}x{derived.grid_y}, Cell: {derived.cell_size}")
    return derived


def stream_frames(config: UserConfig, derived: DerivedConfig) -> Iterator[np.ndarray]:
    """Decode and yield source frames one at a time, resized to target dimensions."""
    cap = cv2.VideoCapture(config.input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {config.input_dir} not found!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.resize(frame, derived.target_dimensions)

    cap.release()


def resize_gallery_to_cells(gallery: np.ndarray, cell_size: tuple) -> np.ndarray:
    """Pre-resize every gallery image to cell size once, ahead of the per-frame loop."""
    cell_w, cell_h = cell_size
    return np.array([cv2.resize(img, (cell_w, cell_h)) for img in gallery])


class BrightnessMetric:
    """Match each cell to a gallery image of near-identical average brightness.

    A cell's brightness rounds to one of 256 levels, so every possible match is
    resolved once at precompute time. Rather than pinning each level to a single
    best image — which collapses a 40k gallery down to ~100 tiles and makes the
    mosaic visibly repetitive — precompute gives each level a bucket of the
    `candidates` images nearest it in brightness, and match() samples one
    uniformly. Matching stays O(1) per cell and the whole gallery gets used.

    `candidates` is the variety knob and means the same thing whatever the
    gallery: a fixed brightness radius would over-sample a dense gallery and
    starve a sparse one. `epsilon` is the accuracy ceiling — no candidate may
    differ from the level by more than that, so a sparse gallery yields a
    smaller bucket rather than a tonally wrong one. A level with nothing inside
    epsilon (including epsilon=0, or candidates=1) falls back to the single
    nearest image, i.e. the argmin answer.
    """

    def __init__(self, candidates: int = 1, epsilon: float = 0.0, seed: int = 0):
        self._candidates = candidates
        self._epsilon = epsilon
        self._rng = np.random.default_rng(seed)

    def precompute(self, gallery: np.ndarray, cell_size: tuple[int, int],
                   brightness: np.ndarray | None = None) -> None:
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
            # Centre a k-wide window on the level, then slide it onto the true k
            # nearest — brightness is not uniformly spread, so the midpoint of
            # the window is not the midpoint of the gallery around it.
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

        # Compact the gallery down to the images some bucket can actually reach.
        spans = np.zeros(n + 1, dtype=np.int64)
        np.add.at(spans, lo, 1)
        np.add.at(spans, lo + count, -1)
        reachable = np.flatnonzero(np.cumsum(spans)[:n] > 0)
        self._remap = np.zeros(n, dtype=np.int64)
        self._remap[reachable] = np.arange(len(reachable))

        self._lo, self._count = lo, count
        self._tiles = resize_gallery_to_cells(gallery[order[reachable]], cell_size)
        self._cell_w, self._cell_h = cell_size

    def match(self, frame: np.ndarray) -> np.ndarray:
        """(H, W, 3) frame -> (grid_y, grid_x) array of tile indices."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grid_y = grey.shape[0] // self._cell_h
        grid_x = grey.shape[1] // self._cell_w

        cell_means = grey.reshape(grid_y, self._cell_h,
                                  grid_x, self._cell_w).mean(axis=(1, 3))
        levels = np.rint(cell_means).astype(np.uint8)

        lo = self._lo[levels]
        picks = lo + self._rng.integers(self._count[levels])
        return self._remap[picks]

    @property
    def bucket_size(self) -> float:
        """Median candidates per level, ignoring levels outside the gallery's range."""
        real = self._count[self._count > 1]
        return float(np.median(real)) if len(real) else 1.0

    @property
    def tiles(self) -> np.ndarray:
        """(U, cell_h, cell_w, 3) pre-resized tiles, indexed by match()."""
        return self._tiles


def mosaic_frame(frame: np.ndarray, metric: BrightnessMetric) -> np.ndarray:
    """Build a mosaic for a single frame by matching each grid cell."""
    tiles = metric.tiles[metric.match(frame)]
    grid_y, grid_x, cell_h, cell_w, _ = tiles.shape
    return tiles.transpose(0, 2, 1, 3, 4).reshape(grid_y * cell_h, grid_x * cell_w, 3)

def shrink_gallery(gallery: np.ndarray, brightness: np.ndarray,
                   config: UserConfig) -> tuple[np.ndarray, np.ndarray]:
    """Trim the gallery to a percentile band of brightness around the midpoint.

    Takes the brightnesses rather than recomputing them, and returns the
    surviving ones alongside the images so the caller never needs a second pass.
    """
    percentiles = (50 - (50 * config.contrast), 50 + (50 * config.contrast))
    low = np.percentile(brightness, percentiles[0])
    high = np.percentile(brightness, percentiles[1])
    mask = (brightness >= low) & (brightness <= high)

    return gallery[mask], brightness[mask]

def main():
    config = UserConfig(input_dir="./assets/source.mp4",
                        output_dir="./output/",
                        contrast=1.0,
                        grid_size=8
                        )

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    gallery = get_gallery(input_dir="./assets/gallery/train")
    brightness = gallery_brightness(gallery)
    gallery_shrunk, brightness_shrunk = shrink_gallery(gallery, brightness, config)

    derived = probe_video(config)
    metric = BrightnessMetric(candidates=config.candidates,
                              epsilon=config.epsilon, seed=config.seed)
    metric.precompute(gallery_shrunk, derived.cell_size, brightness_shrunk)
    print(f"Gallery: {len(gallery_shrunk)} images -> {len(metric.tiles)} usable tiles, "
          f"{metric.bucket_size:.0f} candidates per cell (median)")

    tw, th = derived.target_dimensions
    output_path = f"{config.output_dir}/output.mp4"
    proc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{tw}x{th}", "-framerate", str(derived.output_fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-y", output_path
    ], stdin=subprocess.PIPE)

    for frame in tqdm.tqdm(stream_frames(config, derived), desc="Building mosaics...",
                           total=derived.src_frame_count or None):
        mosaic = mosaic_frame(frame, metric)
        proc.stdin.write(mosaic.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed with exit code {proc.returncode}")

    # Combine source and output side by side
    print("Combining source and mosaic side by side...")
    # hstack requires both inputs to be the same height, and the mosaic is only
    # incidentally the source's size — compute_target_dimensions() snaps to a
    # grid multiple. Scale the source to match rather than relying on that.
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", config.input_dir,
        "-i", f"{config.output_dir}/output.mp4",
        "-filter_complex", f"[0:v]scale={tw}:{th},setsar=1[src];[src][1:v]hstack=inputs=2",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-y", f"{config.output_dir}/combined.mp4"
    ], check=True, stdin=subprocess.DEVNULL)
    print(f"Done. Output written to {config.output_dir}")

if __name__ == "__main__":
    main()
