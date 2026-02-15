# bad-apple-ify

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?logo=OpenCV&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](https://numpy.org/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-171717?logo=ffmpeg&logoColor=5cb85c)](https://ffmpeg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

<p align="center">
    <img align="center" src="./docs/images/preview.gif">
</p>

<p align="center"><i>Make your own Bad Apple!</i></p>

## What is this?

A fun side project that takes a video and reconstructs each frame as a **photo mosaic**, where every tile is a real image pulled from a gallery based on brightness matching. Originally inspired by Bad Apple, but it works on any video!

The output is a side-by-side video of the original and the mosaic version, stitched together with the original audio.

## How it works

Each frame of the source video gets divided into a grid of cells. For every cell, the script calculates the average brightness and finds the closest-matching image from a gallery (currently using CIFAR-100). The matched images are tiled together to recreate the frame as a mosaic, and all the mosaic frames are stitched into a video using ffmpeg.

The pipeline looks like this:

1. **Load gallery** — read CIFAR-100 images and optionally trim the brightness extremes for better contrast
2. **Extract frames** — pull every frame from the source video
3. **Build mosaics** — for each frame, split into a grid, match each cell's brightness against the gallery, and composite the best matches into a mosaic frame
4. **Stitch** — ffmpeg combines the mosaic frames into a video, then places it side-by-side with the original (including audio)

## Gallery

The brightness matching is super simple: it's just comparing scalar averages, but it works surprisingly well, especially on high-contrast videos like Bad Apple. The `contrast` parameter controls how much of the gallery's brightness range to use: lower values trim the extremes and give "cooler" results, while `1.0` uses the full gallery.

## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) for package management (recommended but you can use something else)
- [ffmpeg](https://ffmpeg.org/) installed and on your PATH
- A CIFAR-100 dataset file (the `train` batch in pickle format) placed in `./assets/gallery/`

## Usage

1. Download dependencies with `uv sync`
2. Place your source video at `./assets/source.mp4` (or update the path in `main.py`)
3. Drop the CIFAR-10 train batch into `./assets/gallery/train`
4. `uv run main.py`

The script will output the mosaic video and a combined side-by-side version in `./output/`.

## Configuration

All the tweakable parameters live in the `Config` dataclass at the top of `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_x` | 32 | Number of tiles across each frame |
| `grid_y` | 24 | Number of tiles down each frame |
| `contrast` | 1.0 | Gallery brightness range (0–1). Lower values trim dark/bright extremes |
| `img_format` | `png` | Format for intermediate frames (`png` or `jpg`) |

Higher grid values = more tiles = finer detail but slower processing. The grid is independent of the source resolution, the cell size is calculated automatically.

## Dependencies

- `opencv-python` — video/image processing
- `numpy` — all the number crunching
- `tqdm` — progress bars (essential for sanity)

## Limitations

This is a weekend project, so there are some known rough edges:

- CIFAR-100 images are only 32×32, so tiles look blurry up close. A higher-res gallery would look much better
- The brightness matching is a single scalar per tile — no colour or texture matching (yet?)
- The whole video gets loaded into memory, so very long videos will eat your RAM
- ~~Frame dimensions need to be evenly divisible by `grid_x` and `grid_y`~~ Frame dimensions now snap to the nearest `config.ASPECT_RATIO`!

## Acknowledgements

- [Bad Apple!!](https://www.nicovideo.jp/watch/sm8628149) — the classic
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) — for the gallery images

## License

MIT
