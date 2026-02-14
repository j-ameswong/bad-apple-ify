from typing import Literal
import numpy as np
import cv2
import tqdm
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    ASPECT_RATIOS = (
            (4, 3),
            (16, 9),
            (16, 10)
    )

    source_fps: int = 30
    output_fps: int = 30
    aspect_ratio: tuple = ASPECT_RATIOS[0]
    save_source: bool = False
    img_format: Literal["png", "jpg"] = "png"
    grid_size: int = 16

    def __post_init__(self):
        if self.aspect_ratio not in self.ASPECT_RATIOS:
            raise ValueError(f"{self.aspect_ratio} is not a valid aspect ratio, "
                " please select from {self.ASPECT_RATIOS}")

def extract_frames(config: Config, input_dir: str, output_dir: str) -> np.ndarray:
    """Extract and convert all frames to grayscale"""
    if config.save_source:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {input_dir} not found!")

    # Save frame rate
    config.source_fps = round(cap.get(cv2.CAP_PROP_FPS))
    config.output_fps = config.source_fps

    # Calculate nearest Config.ASPECT_RATIOS
    source_ratio = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    diffs = [abs((source_ratio[0] / source_ratio[1])
                 - (ratio[0] / float(ratio[1])))
             for ratio in config.ASPECT_RATIOS] 

    config.aspect_ratio = config.ASPECT_RATIOS[diffs.index(min(diffs))]

    # grid here is the new pixelised array, higher res by increasing grid_size
    grid = tuple(x * config.grid_size for x in config.aspect_ratio)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # initialize output array
    output_data = np.empty((num_frames, grid[1], grid[0]), dtype=np.float32)

    # Extract frames with progress bar
    for f in tqdm.tqdm(range(num_frames), desc="Extracting frames..."):
        ret, frame = cap.read()

        output_data[f] = fingerprint(frame, grid)
        if config.save_source:
            to_write = cv2.resize((output_data[f] * 255.0).astype(np.uint8),
                                  (source_ratio[0], source_ratio[1]),
                                  interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(f"{output}/frame_{f:05d}.{config.img_format}", to_write)

    cap.release()
    return output_data

def fingerprint(img: np.ndarray, grid: tuple) -> np.ndarray:
    """Returns brightness fingerprint by averaging grid cells"""

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey, grid, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

def get_stretched_res(img: np.ndarray, config: Config) -> tuple:
    y = len(img)
    x = len(img[0])

    if (x / float(y)) == config.aspect_ratio:
        return (y, x)

    if x > y:
        y = int(x * config.aspect_ratio[1] / config.aspect_ratio[0])
    else:
        x = int(y * config.aspect_ratio[0] / config.aspect_ratio[1])

    return (y, x)

def main():
    config = Config(save_source=True)

    frames = extract_frames(config, "./assets/bad_apple.mp4", "./output")
    print(f"Success! Images extracted, {len(frames)} frames processed")

    found = False

if __name__ == "__main__":
    main()
