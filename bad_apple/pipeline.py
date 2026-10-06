from __future__ import annotations

from typing import Iterator
import tqdm
from itertools import islice
from pathlib import Path

from .config import DerivedConfig, UserConfig
from .gallery import GallerySource, check_gallery_budget, load_gallery
from .metrics import Metric, build_metric, mosaic_frame
from .segments import encode_segmented
from . import video
from .types import Image

def build_mosaics(frames: Iterator[Image], metric: Metric,
                  derived: DerivedConfig, *, warmup_frames: int = 0) -> Iterator[Image]:
    """Turn a stream of source frames into a stream of mosaics.

    Lazy end to end: one frame in, one mosaic out, so peak memory holds a couple
    of frames however long the source is.
    """
    if warmup_frames:
        for frame in tqdm.tqdm(islice(frames, warmup_frames),
                               desc="Replaying tile choices...", total=warmup_frames):
            metric.match(frame)
    for frame in tqdm.tqdm(frames, desc="Building mosaics...",
                           total=derived.output_frame_count):
        yield mosaic_frame(frame, metric)

def main(gallery_source: GallerySource, config: UserConfig) -> Path:
    """Run the pipeline end to end, returning the combined video's path."""
    output_dir = Path(config.output_dir)
    inputs = (Path(config.input_dir), *getattr(gallery_source, "input_paths", ()))
    for output in (output_dir / "output.mp4", output_dir / "combined.mp4"):
        for source in inputs:
            if (output.resolve() == source.resolve()
                    or (output.exists() and source.exists() and output.samefile(source))):
                raise ValueError(f"output would overwrite an input: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe first: the gallery can't load until the cell size is known. Under
    # `native` the cell shape is the gallery's, which is cheap metadata.
    tile_aspect = (gallery_source.native_aspect
                   if config.tile_fit == "native" else None)
    derived = video.probe_video(config, tile_aspect)
    # Handed straight over: a local pinning the tiles would keep the whole array
    # alive beside the metric's own copy for the rest of the run.
    metric = build_metric(
        load_gallery(gallery_source, derived, fit=config.tile_fit,
                     budget=config.gallery_budget, use_cache=config.use_cache),
        config, derived)

    # Reproduce the full run's RNG and held tiles, without assembling its prefix.
    if config.segment_frames:
        mosaic_path = encode_segmented(gallery_source, config, derived, metric,
                                       output_dir / "output.mp4")
    else:
        warmup = derived.start_frame if config.candidates > 1 else 0
        frames = video.stream_frames(config, derived, include_prefix=bool(warmup))
        mosaics = build_mosaics(frames, metric, derived, warmup_frames=warmup)
        mosaic_path = video.encode_video(mosaics, derived, output_dir / "output.mp4")

    combined = video.combine_videos(Path(config.input_dir), mosaic_path,
                                    output_dir / "combined.mp4",
                                    derived.target_dimensions,
                                    derived.output_frame_count or 0,
                                    start_frame=derived.start_frame)
    print(f"Done. Output written to ./{output_dir}/")
    return combined
