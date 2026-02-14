from os import close
from typing import Literal
import numpy as np
import cv2
import tqdm
import pickle
from pathlib import Path
from dataclasses import dataclass

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
    grid_size: int = 16
    batch_size: int = 100

    # grid here is the new pixelated image vector, higher res by increasing grid_size
    def grid(self) -> tuple:
        return tuple(x * self.grid_size for x in self.aspect_ratio)

    def src_ratio(self) -> float:
        return self.src_dimensions[0] / float(self.src_dimensions[1])

    def __post_init__(self):
        if self.aspect_ratio not in self.ASPECT_RATIOS:
            raise ValueError(f"{self.aspect_ratio} is not a valid aspect ratio, "
                " please select from {self.ASPECT_RATIOS}")

def get_gallery(input_dir: str) -> np.ndarray:
    """Gets gallery and converts to BGR"""
    with open(Path(f"{input_dir}"), 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
        images = data['data']

        # reshape and transpose from RGB,Y,X to become YxXxRGB
        # note for self, transpose = make old pos into new args pos
        temp = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in temp])

def write_gallery(gallery: np.ndarray, config: Config, output_dir: str = "./output/gallery", img_format: str = "png"):
    """Write raw CIFAR gallery images"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for i in tqdm.tqdm(range(len(gallery)), desc=f"Writing gallery to {output}..."):
        img = cv2.resize(gallery[i], config.src_dimensions, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f"{output}/gallery_{i:05d}.{img_format}", img)
        break

    print(f"Success! {len(gallery)} gallery images written")

def write_frames(frames: np.ndarray, config: Config):
    """Stretch to original dimenions and write frames to output dir"""
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for f in tqdm.tqdm(range(len(frames)),
            desc = f"Writing frames to {output}..."):

        to_write = cv2.resize((frames[f] * 255.0).astype(np.uint8),
                              (config.src_dimensions[0], config.src_dimensions[1]),
                              interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(f"{output}/frame_{f:05d}.{config.img_format}", to_write)

    print(f"Success! Images extracted, {len(frames)} frames processed")

def extract_video_frames(config: Config) -> np.ndarray:
    """Extract and convert all frames to grayscale"""
    cap = cv2.VideoCapture(config.input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {config.input_dir} not found!")

    # Save frame rate
    config.src_fps = round(cap.get(cv2.CAP_PROP_FPS))
    config.output_fps = config.src_fps

    # Calculate nearest Config.ASPECT_RATIOS
    config.src_dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    diffs = [abs(config.src_ratio() - (ratio[0] / float(ratio[1])))
             for ratio in config.ASPECT_RATIOS] 

    config.aspect_ratio = config.ASPECT_RATIOS[np.argmin(diffs)]

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # initialize output array
    output_data = np.empty((num_frames, config.src_dimensions[1],
                            config.src_dimensions[0], 3), dtype=np.float32)

    # Extract frames with progress bar
    for f in tqdm.tqdm(range(num_frames), desc="Extracting frames..."):
        ret, frame = cap.read()
        output_data[f] = frame

    cap.release()
    print(f"Success! Images extracted, {num_frames} frames processed")
    return output_data

def fingerprint(img: np.ndarray, grid: tuple) -> np.ndarray:
    """Returns brightness fingerprint by averaging grid cells"""

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey, grid, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

def find_closest(frame: np.ndarray, gallery: np.ndarray, config) -> int:
    """Find closest match from gallery"""

    # mean square error
    diffs = (gallery - frame) ** 2
    mses = diffs.mean(axis=(1,2))
    return np.argmin(mses).astype(int)


def main():
    config = Config(input_dir="./assets/bad_apple.mp4",
                    output_dir="./output/source/",
                    batch_size=10)
    gallery = get_gallery(input_dir="./assets/gallery/train")

    frames = extract_video_frames(config)

    # fingerprint all the images, then calculate mse
    frames_fp = np.array([fingerprint(img, config.grid()) for img in frames])
    gallery_fp = np.array([fingerprint(img, config.grid()) for img in gallery])

    # pre-flatten once
    gallery_flat = gallery_fp.reshape(len(gallery_fp), -1)  # (num_gallery, grid_h * grid_w)
    frames_flat = frames_fp.reshape(len(frames), -1)            # (num_frames, grid_h * grid_w)

    closest_indices = np.empty(len(frames), dtype=np.int32)

    for i in tqdm.tqdm(range(0, len(frames), config.batch_size), desc="Matching frames..."):
        batch = frames_flat[i:i + config.batch_size]
        diffs = batch[:, np.newaxis, :] - gallery_flat[np.newaxis, :, :]
        mses = (diffs ** 2).mean(axis=2)
        closest_indices[i:i + config.batch_size] = np.argmin(mses, axis=1)

    print(closest_indices)
    # write_frames(frames, config)

if __name__ == "__main__":
    main()
