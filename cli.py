"""Compatibility entrypoint for the package CLI."""

from bad_apple import gallery, pipeline, video
from bad_apple.cli import (RunOptions, _budget, _build_gallery, _finite_float,
                           _grid, _make_parser, _path_config_values,
                           _relative_path, _validate_toml, cli_main, parse_args)

__all__ = ["RunOptions", "cli_main", "parse_args"]

if __name__ == "__main__":
    raise SystemExit(cli_main())
