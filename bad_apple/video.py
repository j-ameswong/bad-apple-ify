from __future__ import annotations

from typing import Iterator, cast

import cv2
import tqdm
import os
from fractions import Fraction
from itertools import chain
from math import isfinite
from pathlib import Path
import subprocess
import tempfile

from .config import DerivedConfig, UserConfig
from .types import Image

def probe_video(config: UserConfig,
                tile_aspect: tuple[int, int] | None = None) -> DerivedConfig:
    """Read source video metadata and derive the grid, cell and target size."""
    cap = cv2.VideoCapture(config.input_dir)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Video at {config.input_dir} not found!")

    try:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        if not isfinite(raw_fps) or raw_fps <= 0:
            raise ValueError(f"source video has invalid frame rate: {raw_fps}")
        # A double of n/1001 snaps straight back. See docs/streaming-and-encoding.md.
        fps = Fraction(raw_fps).limit_denominator(1001)
        raw_dimensions = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not all(isfinite(value) for value in raw_dimensions):
            raise ValueError(f"source video has invalid dimensions: {raw_dimensions}")
        dimensions = (int(raw_dimensions[0]), int(raw_dimensions[1]))
        # Container metadata, used only as a tqdm display hint — may be inaccurate.
        raw_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = (int(raw_frame_count) if isfinite(raw_frame_count)
                       and raw_frame_count > 0 else 0)
    finally:
        cap.release()

    derived = DerivedConfig.from_source(config, fps=fps, dimensions=dimensions,
                                        frame_count=frame_count,
                                        tile_aspect=tile_aspect)

    print(f"Source: {derived.src_dimensions}, Target: {derived.target_dimensions}, "
          f"Grid: {derived.grid_x}x{derived.grid_y}, Cell: {derived.cell_size}")
    return derived

def stream_frames(config: UserConfig, derived: DerivedConfig, *,
                  include_prefix: bool = False) -> Iterator[Image]:
    """Yield the selected source frames, or their prefix too for metric warm-up.

    Count decoded frames rather than trusting frame seeks or container totals.
    See docs/streaming-and-encoding.md.
    """
    cap = cv2.VideoCapture(config.input_dir)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Video at {config.input_dir} not found!")

    try:
        first = 0 if include_prefix else derived.start_frame
        for _ in range(first):
            if not cap.grab():
                return
        index = first
        while derived.stop_frame is None or index < derived.stop_frame:
            ret, frame = cap.read()
            if not ret:
                break
            # cv2's stubs won't commit to a dtype, but resize keeps the input's.
            yield cast(Image, cv2.resize(frame, derived.target_dimensions))
            index += 1
    finally:
        cap.release()

def encode_video(mosaics: Iterator[Image], derived: DerivedConfig,
                 output_path: Path) -> Path:
    """Pipe raw mosaic frames into ffmpeg and publish only a complete encode."""
    source_mosaics = mosaics
    try:
        first = next(source_mosaics, None)
    except BaseException:
        close = getattr(source_mosaics, "close", None)
        if close is not None:
            close()
        raise
    if first is None:
        close = getattr(source_mosaics, "close", None)
        if close is not None:
            close()
        raise ValueError("the requested source range contains no frames")
    mosaics = chain((first,), source_mosaics)
    del first
    width, height = derived.target_dimensions
    # str(Fraction) is "30000/1001", which ffmpeg takes as the exact rate.
    temporary: Path | None = None
    proc: subprocess.Popen[bytes] | None = None
    iterator_error: BaseException | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{output_path.stem}.",
                                         suffix=output_path.suffix,
                                         dir=output_path.parent)
        temporary = Path(temp_name)
        os.close(fd)
        proc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-framerate", str(derived.output_fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-y", str(temporary)
        ], stdin=subprocess.PIPE)
        assert proc.stdin is not None

        try:
            for mosaic in mosaics:
                proc.stdin.write(mosaic.tobytes())
        except BrokenPipeError:
            # ffmpeg died early; its exit code below is the useful error.
            pass
        except BaseException as error:
            iterator_error = error
        finally:
            try:
                proc.stdin.close()
            except OSError:
                # Closing a failed pipe can raise too. Always reap ffmpeg below.
                pass
            finally:
                proc.wait()

        if iterator_error is not None:
            raise iterator_error
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed with exit code {proc.returncode}")
        assert temporary is not None
        with temporary.open("rb") as encoded:
            os.fsync(encoded.fileno())
        os.replace(temporary, output_path)
        _fsync_parent(output_path)
        return output_path
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        close = getattr(source_mosaics, "close", None)
        try:
            if close is not None:
                close()
        except BaseException:
            pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

def combine_videos(source_path: Path, mosaic_path: Path, output_path: Path,
                   dimensions: tuple[int, int],
                   total_frames: int = 0, *, start_frame: int = 0) -> Path:
    """Stack the source and its mosaic side by side into one video.

    The source is scaled to the mosaic's size: `hstack` needs equal heights, and
    the two only match by coincidence. See docs/streaming-and-encoding.md.
    """
    resolved_output = output_path.resolve()
    for input_path in (source_path, mosaic_path):
        if resolved_output == input_path.resolve():
            raise ValueError(f"combined output would overwrite an input: {input_path}")
        try:
            if output_path.exists() and input_path.exists() \
                    and output_path.samefile(input_path):
                raise ValueError(f"combined output would overwrite an input: {input_path}")
        except OSError:
            pass
    width, height = dimensions
    # This is our completed encode, so its count is the actual slice length,
    # even when the source ended before the requested duration.
    cap = cv2.VideoCapture(str(mosaic_path))
    try:
        raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(raw_count) if isfinite(raw_count) and raw_count > 0 else 0
        fps = (Fraction(raw_fps).limit_denominator(1001)
               if isfinite(raw_fps) and raw_fps > 0 else Fraction(0))
    finally:
        cap.release()
    if count <= 0 or fps <= 0:
        raise ValueError(f"No mosaic frames found at {mosaic_path}")
    start_seconds = f"{float(start_frame / fps):.9f}"
    duration_seconds = f"{float(count / fps):.9f}"
    # Both panes use the same frame clock; mkv timestamps round to milliseconds.
    clock = f"settb=expr=1/({fps}),setpts=N"
    temporary: Path | None = None
    proc: subprocess.Popen[str] | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{output_path.stem}.",
                                         suffix=output_path.suffix,
                                         dir=output_path.parent)
        temporary = Path(temp_name)
        os.close(fd)
        proc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-progress", "pipe:1",
        "-i", str(source_path),
        "-i", str(mosaic_path),
        "-filter_complex",
        f"[0:v:0]trim=start_frame={start_frame}:end_frame={start_frame + count},"
        f"scale={width}:{height},setsar=1,{clock}[src];"
        f"[1:v:0]{clock}[mosaic];[src][mosaic]hstack=inputs=2:shortest=1[out]",
        "-map", "[out]", "-map", "0:a:0?",
        "-af", f"atrim=start={start_seconds}:duration={duration_seconds},asetpts=PTS-STARTPTS",
        "-t", duration_seconds,
        "-r", str(fps),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-y", str(temporary)
        ], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        with tqdm.tqdm(desc="Combining videos...", total=total_frames or None,
                       unit="frame") as bar:
            # -progress prints key=value lines, and frame= is a running total.
            for line in proc.stdout:
                key, _, value = line.partition("=")
                if key == "frame":
                    bar.update(int(value) - bar.n)
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg combine failed with exit code {proc.returncode}")
        assert temporary is not None
        with temporary.open("rb") as encoded:
            os.fsync(encoded.fileno())
        os.replace(temporary, output_path)
        _fsync_parent(output_path)
        return output_path
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def _fsync_parent(path: Path) -> None:
    """Persist a rename where directory handles can be synced (POSIX)."""
    if os.name == "nt":
        return
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
