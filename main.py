from typing import Literal
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    source_fps: int
    output_fps: int = 30
    img_format: Literal["png", "jpg"] = "png"
    grid_size: int = 16


def extract_frames(config: Config, input_dir: str, output_dir: str) -> int:
    """Extract and convert all frames to grayscale"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {input_dir} not found!")

    # Save frame rate
    config.source_fps = round(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{output}/frame_{frame_count:05d}.{config.img_format}", grey)
        frame_count += 1

    cap.release()
    return frame_count


def main():
    config = Config(source_fps=30)

    total_frames = extract_frames(config, "./assets/bad_apple.mp4", "./output")
    print(f"Success! Images extracted, {total_frames} frames processed")


if __name__ == "__main__":
    main()
