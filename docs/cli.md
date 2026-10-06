# Command line

Run `python cli.py --source video.mp4 --gallery ./assets/gallery/train`. The
gallery can be a CIFAR pickle, a video, a directory of videos, or a video glob.
Video suffixes and directories are detected automatically; use
`--gallery-type cifar|video` to choose explicitly. Video galleries sample every
`--stride` frame (default 10).

The CLI reads `config.toml` from the current directory when it exists. Use
`--config path/to/file.toml` to select another file; paths in that file are
relative to the TOML file. Paths passed as flags are relative to the current
directory. Flags take precedence over file values. TOML uses flat snake_case
keys, such as `source`, `gallery`, `grid_size`, `use_cache`, and `segment`.
Unknown keys, malformed TOML and values with the wrong type are reported as
errors. See [config.example.toml](../config.example.toml) for a starting point.

`--grid-size` controls the grid multiplier (default 8). `--grid COLSxROWS` can
specify a fixed grid, and `--cell-size` sets a tile edge in pixels. The default
tile fit is `native`, matching the current pipeline behaviour; `crop` and
`stretch` are also available. Matching defaults to colour, contrast 0.8, and
256 candidates. Use `--no-stochastic` to pin matching to one candidate, or
`--stochastic` to enable candidate selection. `--no-cache` bypasses the gallery
cache. `--segment` sets frames per encoded segment (default 5000).

`--gallery-budget` accepts bytes or a binary size such as `2G` or `1.5M`; the
default is 8 GiB. `--dry-run` probes the source and reports the gallery size
estimate, including the budget check. It stops before loading gallery tiles or
creating output, cache or segment files.

For example:

```sh
python cli.py --source movie.mkv --gallery './season/*.mkv' \
  --gallery-type video --stride 10 --grid-size 8 --gallery-budget 2G \
  --dry-run
```
