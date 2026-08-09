from typing import Iterator
import numpy as np
import cv2
import tqdm
import pickle
from pathlib import Path
from dataclasses import dataclass
import subprocess


@dataclass
class Config:
    ASPECT_RATIOS = (
        (4, 3),
        (16, 9),
        (16, 10)
    )

    input_dir: str
    output_dir: str
    src_fps: int = 30
    src_dimensions: tuple = (512, 384)
    target_dimensions: tuple = (512, 384)
    output_fps: int = 30
    src_frame_count: int = 0
    aspect_ratio: tuple = ASPECT_RATIOS[0]
    contrast: float = 0.1
    grid_size: int = 16  # multiplier for aspect ratio

    @property
    def grid_x(self) -> int:
        return self.aspect_ratio[0] * self.grid_size

    @property
    def grid_y(self) -> int:
        return self.aspect_ratio[1] * self.grid_size

    def src_ratio(self) -> float:
        return self.src_dimensions[0] / float(self.src_dimensions[1])

    def cell_size(self) -> tuple:
        return (self.target_dimensions[0] // self.grid_x,
                self.target_dimensions[1] // self.grid_y)

    def compute_target_dimensions(self):
        """Snap frame dimensions to nearest multiple of grid."""
        cell_w = round(self.src_dimensions[0] / self.grid_x)
        cell_h = round(self.src_dimensions[1] / self.grid_y)
        # Ensure at least 1px per cell
        cell_w = max(cell_w, 1)
        cell_h = max(cell_h, 1)
        self.target_dimensions = (self.grid_x * cell_w, self.grid_y * cell_h)

    def __post_init__(self):
        if self.aspect_ratio not in self.ASPECT_RATIOS:
            raise ValueError(f"{self.aspect_ratio} is not a valid aspect ratio, "
                             f" please select from {self.ASPECT_RATIOS}")


def get_gallery(input_dir: str) -> np.ndarray:
    """Gets CIFAR gallery and converts to BGR."""
    with open(Path(input_dir), 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
        images = data['data']

        # reminder to self, transpose works by putting in the old positions
        temp = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in temp])


def gallery_brightness(gallery: np.ndarray) -> np.ndarray:
    """Precompute average brightness (0-1) for each gallery image."""
    return np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0
                     for img in gallery])


def probe_video(config: Config) -> None:
    """Read source video metadata and derive target dimensions/grid. Mutates config."""
    cap = cv2.VideoCapture(config.input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {config.input_dir} not found!")

    config.src_fps = round(cap.get(cv2.CAP_PROP_FPS))
    config.output_fps = config.src_fps
    config.src_dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # Container metadata, used only as a tqdm display hint — may be inaccurate.
    config.src_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    source_ratio = config.src_ratio()
    diffs = [abs(source_ratio - (r[0] / float(r[1]))) for r in config.ASPECT_RATIOS]
    config.aspect_ratio = config.ASPECT_RATIOS[np.argmin(diffs)]
    config.compute_target_dimensions()

    print(f"Source: {config.src_dimensions}, Target: {config.target_dimensions}, "
          f"Grid: {config.grid_x}x{config.grid_y}, Cell: {config.cell_size()}")


def stream_frames(config: Config) -> Iterator[np.ndarray]:
    """Decode and yield source frames one at a time, resized to target dimensions."""
    cap = cv2.VideoCapture(config.input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {config.input_dir} not found!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.resize(frame, config.target_dimensions)

    cap.release()


def resize_gallery_to_cells(gallery: np.ndarray, cell_size: tuple) -> np.ndarray:
    """Pre-resize every gallery image to cell size once, ahead of the per-frame loop."""
    cell_w, cell_h = cell_size
    return np.array([cv2.resize(img, (cell_w, cell_h)) for img in gallery])


class BrightnessMetric:
    """Match each cell to the gallery image of closest average brightness.

    A cell's brightness rounds to one of 256 levels, so every possible match is
    resolved once at precompute time into a 256-entry lookup table. Matching is
    then an O(1) table read per cell rather than an argmin over the gallery, and
    only the handful of images the table actually names need resizing to tiles.
    """

    def precompute(self, gallery: np.ndarray, cell_size: tuple) -> None:
        bright = gallery_brightness(gallery)
        levels = np.arange(256) / 255.0
        nearest = np.argmin((levels[:, None] - bright[None, :]) ** 2, axis=-1)

        # Compact the gallery down to the images the LUT can actually reach.
        used, self._lut = np.unique(nearest, return_inverse=True)
        self._tiles = resize_gallery_to_cells(gallery[used], cell_size)
        self._cell_w, self._cell_h = cell_size

    def match(self, frame: np.ndarray) -> np.ndarray:
        """(H, W, 3) frame -> (grid_y, grid_x) array of tile indices."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grid_y = grey.shape[0] // self._cell_h
        grid_x = grey.shape[1] // self._cell_w

        cell_means = grey.reshape(grid_y, self._cell_h,
                                  grid_x, self._cell_w).mean(axis=(1, 3))
        return self._lut[np.rint(cell_means).astype(np.uint8)]

    @property
    def tiles(self) -> np.ndarray:
        """(U, cell_h, cell_w, 3) pre-resized tiles, indexed by match()."""
        return self._tiles


def mosaic_frame(frame: np.ndarray, metric: BrightnessMetric) -> np.ndarray:
    """Build a mosaic for a single frame by matching each grid cell."""
    tiles = metric.tiles[metric.match(frame)]
    grid_y, grid_x, cell_h, cell_w, _ = tiles.shape
    return tiles.transpose(0, 2, 1, 3, 4).reshape(grid_y * cell_h, grid_x * cell_w, 3)

def shrink_gallery(gallery: np.ndarray, config: Config):
    """Cut off point for lower and upper bound brightnesses"""
    percentiles = (50 - (50 * config.contrast), 50 + (50 * config.contrast))
    temp = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in gallery])
    brightnesses = temp.mean(axis=(1,2)) / 255.0 # normalise
    low = np.percentile(brightnesses, percentiles[0])
    high = np.percentile(brightnesses, percentiles[1])
    mask = (brightnesses >= low) & (brightnesses <= high)

    return gallery[mask]

def main():
    config = Config(input_dir="./assets/source.mp4",
                    output_dir="./output/",
                    contrast=0.8,
                    grid_size=8
                    )

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    gallery = get_gallery(input_dir="./assets/gallery/train")
    gallery_shrunk = shrink_gallery(gallery, config)

    probe_video(config)
    metric = BrightnessMetric()
    metric.precompute(gallery_shrunk, config.cell_size())
    print(f"Gallery: {len(gallery_shrunk)} images -> {len(metric.tiles)} distinct tiles")

    tw, th = config.target_dimensions
    output_path = f"{config.output_dir}/output.mp4"
    proc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{tw}x{th}", "-framerate", str(config.output_fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-y", output_path
    ], stdin=subprocess.PIPE)

    for frame in tqdm.tqdm(stream_frames(config), desc="Building mosaics...",
                           total=config.src_frame_count or None):
        mosaic = mosaic_frame(frame, metric)
        proc.stdin.write(mosaic.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed with exit code {proc.returncode}")

    # Combine source and output side by side
    print("Combining source and mosaic side by side...")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", config.input_dir,
        "-i", f"{config.output_dir}/output.mp4",
        "-filter_complex", "hstack=inputs=2",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-y", f"{config.output_dir}/combined.mp4"
    ], check=True, stdin=subprocess.DEVNULL)
    print(f"Done. Output written to {config.output_dir}")

if __name__ == "__main__":
    main()
