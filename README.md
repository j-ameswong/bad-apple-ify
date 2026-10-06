# bad-apple-ify

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?logo=OpenCV&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](https://numpy.org/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-171717?logo=ffmpeg&logoColor=5cb85c)](https://ffmpeg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

<p align="center"><img src="docs/images/preview.gif" alt="Photo mosaic video preview"></p>

<p align="center"><i>Make your own Bad Apple!</i></p>

bad-apple-ify rebuilds each frame of a video as a mosaic. It matches every
video cell to an image from a gallery, then writes both the mosaic and a
side-by-side video with the original audio.

## Browser app

[Mosaic studio](https://bad-apple-mosaic-studio.wongchengan.chatgpt.site) runs
the pipeline locally in a browser, with file selection, previews, downloads
and exported resume checkpoints. Selected media is never uploaded. It supports
the matching and geometry controls below, with browser-dependent codecs and a
256 MiB default tile budget. Use a desktop browser with WebCodecs support.

## Requirements and setup

- Python 3.14 or later
- [uv](https://docs.astral.sh/uv/) for installing and running the project
- [FFmpeg](https://ffmpeg.org/) available on `PATH`
- NumPy, OpenCV and tqdm (installed by `uv sync`)
- A gallery: a CIFAR-100 Python `train` batch, a video, a directory of videos,
  or a video glob

Install the dependencies and place a source video and CIFAR batch as follows:

```sh
uv sync
mkdir -p assets/gallery
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzf cifar-100-python.tar.gz
mv cifar-100-python/train assets/gallery/
uv run cli.py --source assets/source.mp4 --gallery assets/gallery/train
```

The gallery type is inferred from the path when possible. Pass
`--gallery-type cifar` or `--gallery-type video` to choose explicitly. A video
gallery may be one video, a directory, or a glob such as `'./clips/*.mkv'`;
`--stride` selects every Nth decoded frame (default 10).

The output directory contains `output.mp4` (the mosaic) and `combined.mp4`
(source beside mosaic, with audio). The gallery cache is stored in
`.cache/gallery/` by default. Run `uv run cli.py --help` for the current flags.

## Configuration

The CLI reads `config.toml` in the current directory when it exists. Choose a
different file with `--config path/to/file.toml`. Paths in TOML are relative to
that file; paths in command-line flags are relative to the current directory.
Command-line values override TOML values. The repository includes a complete
[config.example.toml](config.example.toml).

| TOML key / flag | Default | Effect |
|---|---:|---|
| `source` / `--source` | required | Input video path |
| `gallery` / `--gallery` | required | CIFAR pickle, video, directory or glob |
| `gallery_type` / `--gallery-type` | inferred | `cifar` or `video` |
| `output_dir` / `--output-dir` | `./output` | Output and resume files |
| `grid_size` / `--grid-size` | `8` | Grid density multiplier |
| `grid` / `--grid` | unset | Fixed `COLSxROWS` grid |
| `cell_size` / `--cell-size` | unset | Cell height in pixels |
| `tile_fit` / `--tile-fit` | `native` | `native`, `crop` or `stretch` |
| `contrast` / `--contrast` | `0.8` | Central fraction of images to retain by brightness percentile (ties may retain more) |
| `metric` / `--metric` | `colour` | Match on colour or brightness |
| `candidates` / `--candidates` | `256` | Choices sampled from each match bucket |
| `epsilon` / `--epsilon` | `0.005` | Brightness-match error ceiling |
| `colour_bins` / `--colour-bins` | `32` | Colour lattice edge, from 2 to 64 |
| `stochastic` / `--stochastic`, `--no-stochastic` | `true` | Sample candidates; false pins one candidate |
| `seed` / `--seed` | `0` | Reproducible random choices |
| `hold_tiles` / `--hold-tiles`, `--no-hold-tiles` | `true` | Keep a cell's tile while its match bucket stays the same |
| `start` / `--start` | `0` seconds | Start time in the source |
| `duration` / `--duration` | unset | Slice length; unset means to EOF |
| `stride` / `--stride` | `10` | Frames between samples in a video gallery |
| `segment` / `--segment` | `5000` frames | Checkpointed encode length; `0` disables resume |
| `use_cache` / `--cache`, `--no-cache` | `true` | Read/write or bypass the tile cache |
| `gallery_budget` / `--gallery-budget` | `8G` | Maximum tile-array bytes; accepts bytes or K/M/G/T |
| `dry_run` / `--dry-run` | `false` | Probe and check the estimated gallery size without processing |

For example, process a ten-second slice, or estimate a video gallery before
decoding it:

```sh
uv run cli.py --source assets/source.mp4 --gallery assets/gallery/train \
  --start 60 --duration 10 --grid-size 8

uv run cli.py --source assets/source.mp4 --gallery './clips/*.mkv' \
  --gallery-type video --stride 10 --gallery-budget 2G --dry-run
```

See [CLI and TOML configuration](docs/cli.md) for precedence, validation and
more examples. See [segmented encoding and resume](docs/resume.md) to continue
an interrupted run with the same settings.

## Matching and memory

The default colour metric puts each cell's mean BGR colour into a 3D lattice
and finds gallery tiles from that bucket. The brightness metric matches luma
alone. `contrast` trims the darkest and brightest gallery images before either
metric is built. With `hold_tiles`, a cell keeps its selected image until its
match bucket changes, which reduces flicker. Details and trade-offs are in
[matching metrics](docs/colour-matching.md) and
[brightness matching](docs/brightness-matching.md).

Frames are decoded and processed as a stream, so memory use does not grow with
source-video duration. The resized gallery tiles do stay in memory, and metric
precomputation can temporarily hold additional arrays. `gallery_budget` guards
the tile array and warns or refuses based on the estimate; it does not cap all
process memory. A larger cell or gallery may need substantially more RAM. See
[gallery sizing](docs/gallery-size.md).

## Development

```sh
uv run pytest
uv run mypy
```

The tests make their own small videos and galleries; source media and the full
CIFAR download are not needed. The code includes POSIX and Windows checkpoint
locking paths, but verification has been run on Linux.

## Acknowledgements

- [Bad Apple!!](https://www.nicovideo.jp/watch/sm8628149) for the inspiration
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) for its image batch

## License

MIT. See [LICENSE](LICENSE).
