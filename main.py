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
    img_format: Literal["png", "jpg"] = "png"
    grid_size: int = 16

    def __post_init__(self):
        if self.aspect_ratio not in self.ASPECT_RATIOS:
            raise ValueError(f"{self.aspect_ratio} is not a valid aspect ratio, "
                " please select from {self.ASPECT_RATIOS}")

def extract_frames(config: Config, input_dir: str, output_dir: str):
    """Extract and convert all frames to grayscale"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {input_dir} not found!")

    # Save frame rate
    config.source_fps = round(cap.get(cv2.CAP_PROP_FPS))
    config.output_fps = config.source_fps

    # Calculate nearest Config.ASPECT_RATIOS
    source_ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diffs = [abs(source_ratio - (ratio[0] / float(ratio[1]))) for ratio in config.ASPECT_RATIOS] 
    config.aspect_ratio = config.ASPECT_RATIOS[diffs.index(min(diffs))]

    # Extract frames with progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in tqdm.tqdm(range(total_frames), desc="Extracting frames..."):
        ret, frame = cap.read()

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{output}/frame_{f:05d}.{config.img_format}", grey)

    cap.release()


def main():
    config = Config(source_fps=30)

    total_frames = extract_frames(config, "./assets/bad_apple.mp4", "./output")
    print(f"Success! Images extracted, {total_frames} frames processed")


if __name__ == "__main__":
    main()
