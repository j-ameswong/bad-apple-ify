from typing import Literal
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
    output_fps: int = 30
    aspect_ratio: tuple = ASPECT_RATIOS[0]
    img_format: Literal["png", "jpg"] = "png"
    contrast: float = 0.1 # 0 - 1
    grid_x: int = 64  # tiles across
    grid_y: int = 48   # tiles down

    def src_ratio(self) -> float:
        return self.src_dimensions[0] / float(self.src_dimensions[1])

    def cell_size(self) -> tuple:
        """Returns (cell_width, cell_height) in pixels."""
        return (self.src_dimensions[0] // self.grid_x,
                self.src_dimensions[1] // self.grid_y)

    def __post_init__(self):
        if self.aspect_ratio not in self.ASPECT_RATIOS:
            raise ValueError(f"{self.aspect_ratio} is not a valid aspect ratio, "
                             f" please select from {self.ASPECT_RATIOS}")


def get_gallery(input_dir: str) -> np.ndarray:
    """Gets CIFAR gallery and converts to BGR."""
    with open(Path(input_dir), 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
        images = data['data']
        temp = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in temp])


def gallery_brightness(gallery: np.ndarray) -> np.ndarray:
    """Precompute average brightness (0-1) for each gallery image."""
    return np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0
                     for img in gallery])


def extract_video_frames(config: Config) -> np.ndarray:
    """Extract all frames as BGR uint8."""
    cap = cv2.VideoCapture(config.input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {config.input_dir} not found!")

    config.src_fps = round(cap.get(cv2.CAP_PROP_FPS))
    config.output_fps = config.src_fps
    config.src_dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    source_ratio = config.src_ratio()
    diffs = [abs(source_ratio - (r[0] / float(r[1]))) for r in config.ASPECT_RATIOS]
    config.aspect_ratio = config.ASPECT_RATIOS[np.argmin(diffs)]

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_data = np.empty((num_frames, config.src_dimensions[1],
                            config.src_dimensions[0], 3), dtype=np.uint8)

    for f in tqdm.tqdm(range(num_frames), desc="Extracting frames..."):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {f}")
        output_data[f] = frame

    cap.release()
    return output_data


def mosaic_frame(frame: np.ndarray, gallery: np.ndarray,
                 bright: np.ndarray, config: Config) -> np.ndarray:
    """Build a mosaic for a single frame by matching each grid cell."""
    cell_w, cell_h = config.cell_size()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = np.empty_like(frame)

    for row in range(config.grid_y):
        for col in range(config.grid_x):
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w

            cell_brightness = grey[y0:y1, x0:x1].mean() / 255.0
            idx = np.argmin((bright - cell_brightness) ** 2)

            tile = cv2.resize(gallery[idx], (cell_w, cell_h))
            output[y0:y1, x0:x1] = tile

    return output

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
                    contrast=1,
                    grid_x=32,
                    grid_y=24)

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    gallery = get_gallery(input_dir="./assets/gallery/train")
    gallery_shrunk = shrink_gallery(gallery, config)

    bright = gallery_brightness(gallery_shrunk)
    frames = extract_video_frames(config)

    for f in tqdm.tqdm(range(len(frames)), desc="Building mosaics..."):
        mosaic = mosaic_frame(frames[f], gallery_shrunk, bright, config)
        cv2.imwrite(f"{output}/frame_{f:05d}.{config.img_format}", mosaic)

    # Run ffmpeg and stitch the generated images together
    subprocess.run([
        "ffmpeg", "-framerate", str(config.output_fps),
        "-i", f"{config.output_dir}/frame_%05d.{config.img_format}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-y", f"{config.output_dir}/output.mp4"
    ], check=True)

    # Delete now unneeded frames
    for f in Path(config.output_dir).glob(f"*.{config.img_format}"):
        f.unlink()

    subprocess.run([
        "ffmpeg",
        "-i", config.input_dir,
        "-i", f"{config.output_dir}/output.mp4",
        "-filter_complex", "hstack=inputs=2",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-y", "combined.mp4"
    ], check=True)

if __name__ == "__main__":
    main()
