"""Command line and TOML configuration for the mosaic pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from glob import has_magic
import math
from pathlib import Path
import sys
import tomllib
from typing import Sequence, cast

import main as pipeline


@dataclass(frozen=True)
class RunOptions:
    config: pipeline.UserConfig
    gallery: pipeline.GallerySource
    dry_run: bool = False


_DEFAULTS: dict[str, object] = {
    "stride": 10,
    "grid_size": 8,
    "cell_size": None,
    "grid": None,
    "tile_fit": "native",
    "contrast": 0.8,
    "metric": "colour",
    "candidates": 256,
    "epsilon": 0.005,
    "colour_bins": 32,
    "stochastic": True,
    "seed": 0,
    "hold_tiles": True,
    "start": 0.0,
    "duration": None,
    "segment": 5000,
    "use_cache": True,
    "gallery_budget": 8 << 30,
    "output_dir": "./output",
    "gallery_type": None,
    "dry_run": False,
}

_KEYS = frozenset({"source", "gallery", *_DEFAULTS})


def _finite_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def _grid(value: str) -> tuple[int, int]:
    try:
        columns, rows = value.lower().split("x", 1)
        result = (int(columns), int(rows))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("must be COLSxROWS, for example 32x24") from error
    if min(result) < 1:
        raise argparse.ArgumentTypeError("grid dimensions must be positive")
    return result


def _budget(value: str) -> int:
    """Parse byte counts with optional binary K/M/G/T suffixes."""
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("must be a byte count, optionally with K/M/G/T")
    suffix = text[-1:].upper()
    multiplier = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30,
                  "T": 1 << 40}.get(suffix, 1)
    numeric = text[:-1] if suffix in "KMGT" else text
    try:
        amount = float(numeric)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a byte count, optionally with K/M/G/T") from error
    if not math.isfinite(amount) or amount <= 0:
        raise argparse.ArgumentTypeError("must be a finite, positive size")
    result = int(amount * multiplier)
    if result <= 0:
        raise argparse.ArgumentTypeError("size must be at least one byte")
    return result


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild a video as a photo mosaic.")
    parser.add_argument("--config", type=Path, help="TOML configuration file")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--gallery", type=Path)
    parser.add_argument("--gallery-type", choices=("cifar", "video"))
    parser.add_argument("--stride", type=int)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--grid", type=_grid, metavar="COLSxROWS")
    parser.add_argument("--cell-size", type=int)
    parser.add_argument("--tile-fit", choices=("native", "crop", "stretch"))
    parser.add_argument("--contrast", type=_finite_float)
    parser.add_argument("--metric", choices=("brightness", "colour"))
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--epsilon", type=_finite_float)
    parser.add_argument("--colour-bins", type=int)
    stochastic = parser.add_mutually_exclusive_group()
    stochastic.add_argument("--stochastic", dest="stochastic", action="store_true")
    stochastic.add_argument("--no-stochastic", dest="stochastic", action="store_false")
    parser.add_argument("--seed", type=int)
    hold = parser.add_mutually_exclusive_group()
    hold.add_argument("--hold-tiles", dest="hold_tiles", action="store_true")
    hold.add_argument("--no-hold-tiles", dest="hold_tiles", action="store_false")
    parser.add_argument("--start", type=_finite_float)
    parser.add_argument("--duration", type=_finite_float)
    parser.add_argument("--segment", type=int)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache", dest="use_cache", action="store_true")
    cache.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--gallery-budget", type=_budget)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(stochastic=None, hold_tiles=None, use_cache=None, dry_run=None)
    return parser


def _validate_toml(data: object, config_path: Path) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: expected a flat TOML table")
    values = cast(dict[str, object], data)
    unknown = sorted(set(values) - _KEYS)
    if unknown:
        raise ValueError(f"{config_path}: unknown configuration key(s): {', '.join(unknown)}")

    expected: dict[str, type | tuple[type, ...]] = {
        "source": str, "gallery": str, "gallery_type": str, "stride": int,
        "grid_size": int, "cell_size": int, "grid": str, "tile_fit": str,
        "contrast": (int, float), "metric": str, "candidates": int,
        "epsilon": (int, float), "colour_bins": int, "stochastic": bool,
        "seed": int, "hold_tiles": bool, "start": (int, float),
        "duration": (int, float), "segment": int, "use_cache": bool,
        "gallery_budget": (int, str), "output_dir": str, "dry_run": bool,
    }
    for key, value in values.items():
        allowed = expected[key]
        # bool is a subclass of int in Python, but is not a valid number here.
        valid = isinstance(value, allowed)
        if isinstance(value, bool) and (allowed is int or allowed == (int, float)):
            valid = False
        if key == "gallery_budget" and isinstance(value, bool):
            valid = False
        if not valid:
            label = " or ".join(t.__name__ for t in allowed) if isinstance(allowed, tuple) else allowed.__name__
            raise ValueError(f"{config_path}: {key} must be {label}")
    return values


def _relative_path(value: object, base: Path) -> Path:
    path = Path(cast(str | Path, value)).expanduser()
    return path if path.is_absolute() else base / path


def _path_config_values(values: dict[str, object], base: Path) -> dict[str, object]:
    result = dict(values)
    for name in ("source", "gallery", "output_dir"):
        value = result.get(name)
        if value is not None:
            result[name] = _relative_path(value, base)
    return result


def _build_gallery(path: Path, gallery_type: str | None, stride: int) -> pipeline.GallerySource:
    kind = gallery_type
    if kind is None:
        kind = "video" if path.is_dir() or has_magic(str(path)) or path.suffix.lower() in pipeline.VIDEO_SUFFIXES else "cifar"
    if kind == "cifar":
        return pipeline.CifarGallery(path)
    return pipeline.VideoGallery(path, stride=stride)


def parse_args(argv: Sequence[str] | None = None) -> RunOptions:
    """Parse CLI/TOML layers and construct an immutable pipeline run."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in raw_argv or "-h" in raw_argv:
        _make_parser().parse_args(raw_argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_arg, _ = config_parser.parse_known_args(raw_argv)
    explicit_config = config_arg.config is not None
    config_path = config_arg.config or Path("config.toml")
    base = Path.cwd()
    merged: dict[str, object] = {}
    if config_path.exists():
        try:
            with config_path.open("rb") as stream:
                loaded = tomllib.load(stream)
            merged = _path_config_values(_validate_toml(loaded, config_path), config_path.resolve().parent)
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            raise ValueError(str(error)) from error
    elif explicit_config:
        raise ValueError(f"config file not found: {config_path}")

    parser = _make_parser()
    cli = vars(parser.parse_args(raw_argv))
    cli.pop("config", None)
    cli = {key: value for key, value in cli.items() if value is not None}
    cli_paths: dict[str, object] = {}
    for key, value in cli.items():
        if key in ("source", "gallery", "output_dir") and value is not None:
            cli_paths[key] = _relative_path(value, base)
        else:
            cli_paths[key] = value
    merged.update(cli_paths)

    values = dict(_DEFAULTS)
    values.update(merged)
    if isinstance(values.get("grid"), str):
        try:
            values["grid"] = _grid(cast(str, values["grid"]))
        except argparse.ArgumentTypeError as error:
            raise ValueError(f"grid {error}") from error
    if values["gallery_type"] not in (None, "cifar", "video"):
        raise ValueError("gallery_type must be 'cifar' or 'video'")
    if values["tile_fit"] not in ("native", "crop", "stretch"):
        raise ValueError("tile_fit must be 'native', 'crop' or 'stretch'")
    if values["metric"] not in ("brightness", "colour"):
        raise ValueError("metric must be 'brightness' or 'colour'")
    source = values.get("source")
    gallery_path = values.get("gallery")
    if source is None or gallery_path is None:
        missing = [f"--{name}" for name, value in (("source", source), ("gallery", gallery_path)) if value is None]
        raise ValueError(f"required value missing: {', '.join(missing)}")

    for key in ("stride", "grid_size", "cell_size", "candidates", "colour_bins"):
        item = values.get(key)
        if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 1):
            raise ValueError(f"{key.replace('_', '-')} must be a positive integer")
    if isinstance(values["segment"], bool) or not isinstance(values["segment"], int) or values["segment"] < 0:
        raise ValueError("segment must be a non-negative integer")
    for key in ("contrast", "epsilon", "start", "duration"):
        value = values[key]
        if value is not None and not math.isfinite(float(cast(float, value))):
            raise ValueError(f"{key.replace('_', '-')} must be finite")
    if values["duration"] is not None and float(cast(float, values["duration"])) <= 0:
        raise ValueError("duration must be positive")
    if float(cast(float, values["start"])) < 0:
        raise ValueError("start must be non-negative")
    if float(cast(float, values["contrast"])) < 0 or float(cast(float, values["contrast"])) > 1:
        raise ValueError("contrast must be between 0 and 1")
    if float(cast(float, values["epsilon"])) < 0 or float(cast(float, values["epsilon"])) > 1:
        raise ValueError("epsilon must be between 0 and 1")
    if int(cast(int, values["colour_bins"])) not in range(2, 65):
        raise ValueError("colour-bins must be between 2 and 64")
    if not values["stochastic"]:
        values["candidates"] = 1
    budget_value = values["gallery_budget"]
    budget = _budget(str(budget_value)) if isinstance(budget_value, str) else int(cast(int, budget_value))
    if budget <= 0:
        raise ValueError("gallery_budget must be positive")
    values["gallery_budget"] = budget

    source_path = cast(Path, source)
    gallery = _build_gallery(cast(Path, gallery_path), values["gallery_type"], int(cast(int, values["stride"])))
    config = pipeline.UserConfig(
        input_dir=str(source_path),
        output_dir=str(cast(Path, values["output_dir"])),
        contrast=cast(float, values["contrast"]),
        grid_size=cast(int, values["grid_size"]),
        cell_size=cast(int | None, values["cell_size"]),
        grid=cast(tuple[int, int] | None, values["grid"]),
        tile_fit=values["tile_fit"],
        metric=values["metric"],
        candidates=cast(int, values["candidates"]),
        epsilon=cast(float, values["epsilon"]),
        colour_bins=cast(int, values["colour_bins"]),
        seed=cast(int, values["seed"]),
        hold_tiles=cast(bool, values["hold_tiles"]),
        gallery_budget=budget,
        start=cast(float, values["start"]),
        duration=cast(float | None, values["duration"]),
        use_cache=cast(bool, values["use_cache"]),
        segment_frames=values["segment"],
    )
    return RunOptions(config=config, gallery=gallery,
                      dry_run=bool(values["dry_run"]))


def cli_main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning 2 for concise user-facing errors."""
    try:
        options = parse_args(argv)
        if options.dry_run:
            tile_aspect = (options.gallery.native_aspect
                           if options.config.tile_fit == "native" else None)
            derived = pipeline.probe_video(options.config, tile_aspect)
            pipeline.check_gallery_budget(options.gallery, derived.cell_size,
                                          options.config.gallery_budget)
            return 0
        pipeline.main(options.gallery, options.config)
        return 0
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli_main())
