"""Compatibility facade for the bad_apple package.

New code should import from the module that owns each part of the pipeline.
The facade keeps the original public names working for scripts and older tests.
"""

from __future__ import annotations

import argparse

from bad_apple.types import Brightness, Fit, Image, Indices, MetricName
from bad_apple.config import (DerivedConfig, HARD_BUDGET, SOFT_BUDGET, UserConfig,
                              even_span)
from bad_apple.gallery import (CIFAR_IMAGE_BYTES, DEFAULT_CACHE_DIR,
                               FALLBACK_CAPACITY, TILE_VERSION, VIDEO_SUFFIXES,
                               CifarGallery, GallerySource, GalleryTooLarge,
                               TileBuffer, VideoGallery, cache_key,
                               check_gallery_budget, crop_to_aspect,
                               enforce_gallery_budget, fit_to_cell,
                               format_bytes, load_gallery, read_cached_tiles,
                               read_cifar_batch, resize_gallery_to_cells)
from bad_apple.metrics import (LUMA_BGR, BrightnessMetric, ColourMetric, Metric,
                               SteadyMetric, build_metric, cell_means,
                               check_cell_size, compact_buckets,
                               draw_from_buckets, gallery_brightness,
                               mosaic_frame, nearest_occupied, shrink_gallery)
from bad_apple.video import combine_videos, encode_video, probe_video, stream_frames
from bad_apple.pipeline import build_mosaics, main

__all__ = [
    "Brightness", "BrightnessMetric", "CIFAR_IMAGE_BYTES", "CifarGallery",
    "ColourMetric", "DEFAULT_CACHE_DIR", "DerivedConfig", "FALLBACK_CAPACITY",
    "Fit", "GallerySource", "GalleryTooLarge", "HARD_BUDGET", "Image",
    "Indices", "LUMA_BGR", "Metric", "MetricName", "SOFT_BUDGET",
    "SteadyMetric", "TILE_VERSION", "TileBuffer", "UserConfig",
    "VIDEO_SUFFIXES", "VideoGallery", "build_metric", "build_mosaics",
    "cache_key", "cell_means", "check_cell_size", "check_gallery_budget",
    "combine_videos", "compact_buckets", "crop_to_aspect", "draw_from_buckets",
    "encode_video", "enforce_gallery_budget", "even_span", "fit_to_cell",
    "format_bytes", "gallery_brightness", "load_gallery", "main",
    "mosaic_frame", "nearest_occupied", "parse_config", "probe_video",
    "read_cached_tiles", "read_cifar_batch", "resize_gallery_to_cells",
    "shrink_gallery", "stream_frames",
]


def parse_config(argv: list[str] | None = None) -> UserConfig:
    """Deprecated slice-only compatibility helper; use ``cli.parse_args``."""
    parser = argparse.ArgumentParser(description="Rebuild a source video as a photo mosaic.")
    parser.add_argument("--start", type=float, default=0.0, metavar="SECONDS",
                        help="source start time in seconds (default: 0)")
    parser.add_argument("--duration", type=float, metavar="SECONDS",
                        help="seconds of source to process (default: to the end)")
    args = parser.parse_args(argv)
    try:
        return UserConfig(input_dir="./assets/source.mp4", output_dir="./output/",
                          contrast=1.0, grid_size=8,
                          start=args.start, duration=args.duration)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    from bad_apple.cli import cli_main

    raise SystemExit(cli_main())
