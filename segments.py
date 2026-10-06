"""Fixed-size, resumable mosaic encoding.

The manifest is the commit record. A segment is only considered complete once
ffmpeg has closed it, it has been renamed into place, its hash has been
recorded, and the matching metric state has been published.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
import fcntl
import hashlib
from itertools import chain
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Generator, Iterator, cast

import cv2
import numpy as np

from main import (DerivedConfig, GallerySource, Image, Metric, UserConfig,
                  encode_video, mosaic_frame, stream_frames)

CHECKPOINT_VERSION = 1
CODE_CHECKPOINT_VERSION = "segmented-encode-v1"
MANIFEST_NAME = "checkpoint.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_value(value: Any) -> Any:
    """Convert dataclass/config values to canonical JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(part) for part in value]
    if isinstance(value, dict):
        return {str(key): _json_value(part) for key, part in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot include {type(value).__name__} in run identity")


def _identity(source: GallerySource, config: UserConfig,
              derived: DerivedConfig) -> dict[str, Any]:
    source_path = Path(config.input_dir).expanduser().resolve(strict=True)
    stat = source_path.stat()
    ignored = {"input_dir", "output_dir", "use_cache"}
    config_values = {
        item.name: _json_value(getattr(config, item.name))
        for item in fields(config) if item.name not in ignored
    }
    data = {
        "checkpoint_version": CODE_CHECKPOINT_VERSION,
        "config": config_values,
        "source": {
            "path": str(source_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "gallery_fingerprint": source.fingerprint,
        },
        "derived": {
            "fps": str(derived.src_fps),
            "dimensions": list(derived.src_dimensions),
            "frame_count": derived.src_frame_count,
            "start_frame": derived.start_frame,
            "stop_frame": derived.stop_frame,
            "target_dimensions": list(derived.target_dimensions),
            "grid": list(derived.grid),
            "cell_size": list(derived.cell_size),
        },
    }
    return data


def _identity_digest(identity: dict[str, Any]) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _rng_owner(metric: Metric) -> Any:
    current: Any = metric
    while not hasattr(current, "_rng"):
        if not hasattr(current, "_inner"):
            raise TypeError("metric does not expose a NumPy random generator")
        current = current._inner
    return current


def _snapshot(metric: Metric, state_path: Path) -> None:
    owner = _rng_owner(metric)
    rng_json = json.dumps(owner._rng.bit_generator.state, sort_keys=True)
    previous = getattr(metric, "_previous", None)
    if previous is None:
        has_previous = np.array(False)
        previous_keys = np.empty((0,), dtype=np.int64)
        previous_picks = np.empty((0,), dtype=np.int64)
    else:
        has_previous = np.array(True)
        previous_keys = np.asarray(previous[0], dtype=np.int64)
        previous_picks = np.asarray(previous[1], dtype=np.int64)

    temporary = state_path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(temporary, rng_json=np.array(rng_json),
                            has_previous=has_previous,
                            previous_keys=previous_keys,
                            previous_picks=previous_picks)
        _fsync_file(temporary)
        os.replace(temporary, state_path)
        _fsync_directory(state_path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _restore(metric: Metric, state_path: Path) -> None:
    with np.load(state_path, allow_pickle=False) as state:
        rng_json = str(state["rng_json"].item())
        has_previous = bool(state["has_previous"].item())
        keys = np.array(state["previous_keys"], dtype=np.int64, copy=True)
        picks = np.array(state["previous_picks"], dtype=np.int64, copy=True)
    owner = _rng_owner(metric)
    owner._rng.bit_generator.state = json.loads(rng_json)
    if hasattr(metric, "_previous"):
        metric._previous = ((keys, picks) if has_previous else None)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.",
                                          dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _job_lock(path: Path) -> Iterator[None]:
    """Hold an OS lock for the whole run; the lock file itself can persist."""
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another segmented encode is active ({path})") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_manifest(manifest_path: Path, expected: dict[str, Any],
                   expected_digest: str, segment_dir: Path
                   ) -> tuple[list[dict[str, Any]], int, Path | None]:
    if not manifest_path.exists():
        return [], 0, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid resume checkpoint at {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("resume checkpoint must contain a JSON object")
    if manifest.get("version") != CHECKPOINT_VERSION:
        raise ValueError("resume checkpoint version is unsupported")
    if manifest.get("identity_digest") != expected_digest or manifest.get("identity") != expected:
        raise ValueError("existing segmented output belongs to a different source or configuration")
    records = manifest.get("segments")
    if not isinstance(records, list):
        raise ValueError("resume checkpoint has an invalid segment list")
    completed = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("resume checkpoint has an invalid segment record")
        expected_name = f"part_{index:04d}.mp4"
        if record.get("file") != expected_name:
            raise ValueError("resume checkpoint segment order is invalid")
        segment_path = segment_dir / expected_name
        if not segment_path.is_file() or _sha256(segment_path) != record.get("sha256"):
            raise ValueError(f"completed segment is missing or corrupt: {segment_path}")
        frame_count = record.get("frames")
        if not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError(f"completed segment has an invalid frame count: {segment_path}")
        completed += frame_count
    if manifest.get("completed_frames") != completed:
        raise ValueError("resume checkpoint frame total does not match its segments")
    state_name = manifest.get("state_file")
    state_path = None
    if records:
        if not isinstance(state_name, str) or Path(state_name).name != state_name:
            raise ValueError("resume checkpoint has an invalid metric-state path")
        state_path = segment_dir / state_name
        if not state_path.is_file():
            raise ValueError(f"metric state is missing: {state_path}")
        if _sha256(state_path) != manifest.get("state_sha256"):
            raise ValueError(f"metric state is corrupt: {state_path}")
    return records, completed, state_path


def _seek_frames(config: UserConfig, derived: DerivedConfig,
                 start: int) -> Generator[Image, None, None]:
    """Seek directly to an absolute source frame and verify OpenCV's position."""
    cap = cv2.VideoCapture(config.input_dir)
    if not cap.isOpened():
        raise ValueError(f"Video at {config.input_dir} not found!")
    try:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise RuntimeError(f"video decoder refused seek to frame {start}")
        reported = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if int(round(reported)) != start:
            raise RuntimeError(f"video decoder sought to frame {reported}, expected {start}")
        index = start
        while derived.stop_frame is None or index < derived.stop_frame:
            ret, frame = cap.read()
            if not ret:
                break
            after_read = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if int(round(after_read)) != index + 1:
                raise RuntimeError(
                    f"video decoder advanced to frame {after_read}, expected {index + 1}")
            yield cast(Image, cv2.resize(frame, derived.target_dimensions))
            index += 1
    finally:
        cap.release()


def _take_mosaics(frames: Iterator[Image], metric: Metric,
                  limit: int) -> Iterator[Image]:
    for _ in range(limit):
        frame = next(frames, None)
        if frame is None:
            return
        yield mosaic_frame(frame, metric)


def encode_segmented(source: GallerySource, config: UserConfig,
                     derived: DerivedConfig, metric: Metric,
                     output_path: Path) -> Path:
    """Encode fixed-size pieces and resume from a committed metric snapshot."""
    segment_frames = config.segment_frames
    if segment_frames <= 0:
        raise ValueError("segment_frames must be positive")
    segment_dir = output_path.parent / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = segment_dir / MANIFEST_NAME
    identity = _identity(source, config, derived)
    identity_digest = _identity_digest(identity)

    with _job_lock(segment_dir / ".encode.lock"):
        records, completed, state_path = _load_manifest(
            manifest_path, identity, identity_digest, segment_dir)
        if state_path is None:
            warmup = derived.start_frame if config.candidates > 1 else 0
            frames = cast(Generator[Image, None, None], stream_frames(
                config, derived, include_prefix=bool(warmup)))
            try:
                for _ in range(warmup):
                    warmup_frame = next(frames, None)
                    if warmup_frame is None:
                        raise ValueError("the requested source range contains no frames")
                    metric.match(warmup_frame)
            except BaseException:
                frames.close()
                raise
        else:
            _restore(metric, state_path)
            frames = _seek_frames(config, derived, derived.start_frame + completed)

        try:
            while True:
                first_frame = next(frames, None)
                if first_frame is None:
                    if records:
                        break
                    raise ValueError("the requested source range contains no frames")

                segment_index = len(records)
                segment_path = segment_dir / f"part_{segment_index:04d}.mp4"
                temp_path = segment_dir / f".part_{segment_index:04d}.tmp.mp4"
                temp_path.unlink(missing_ok=True)
                consumed = 0
                first_mosaic = mosaic_frame(first_frame, metric)

                def counted() -> Iterator[Image]:
                    nonlocal consumed
                    for mosaic in chain((first_mosaic,), _take_mosaics(
                            frames, metric, segment_frames - 1)):
                        consumed += 1
                        yield mosaic

                try:
                    encode_video(counted(), derived, temp_path)
                except BaseException:
                    temp_path.unlink(missing_ok=True)
                    raise

                # encode_video succeeds only after ffmpeg closes the file.
                _fsync_file(temp_path)
                os.replace(temp_path, segment_path)
                _fsync_directory(segment_dir)
                state_name = f"state_{segment_index + 1:08d}.npz"
                state_path = segment_dir / state_name
                _snapshot(metric, state_path)
                record = {"file": segment_path.name, "frames": consumed,
                          "sha256": _sha256(segment_path)}
                records.append(record)
                _atomic_json(manifest_path, {
                    "version": CHECKPOINT_VERSION,
                    "identity": identity,
                    "identity_digest": identity_digest,
                    "segments": records,
                    "completed_frames": completed + consumed,
                    "state_file": state_name,
                    "state_sha256": _sha256(state_path),
                })
                completed += consumed
                if consumed < segment_frames:
                    break

            if not records:
                raise ValueError("the requested source range contains no frames")

            concat_path = segment_dir / ".concat.txt"
            try:
                with concat_path.open("w", encoding="utf-8") as concat:
                    for record in records:
                        path = (segment_dir / str(record["file"])).resolve()
                        escaped = str(path).replace("'", "'\\''")
                        concat.write(f"file '{escaped}'\n")
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{output_path.stem}.", suffix=output_path.suffix,
                    dir=output_path.parent)
                os.close(fd)
                temp_output = Path(temp_name)
                try:
                    result = subprocess.run([
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", str(concat_path),
                        "-c", "copy", str(temp_output),
                    ], check=False, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"ffmpeg concat failed with exit code {result.returncode}: "
                            f"{result.stderr.strip()}")
                    _fsync_file(temp_output)
                    os.replace(temp_output, output_path)
                    _fsync_directory(output_path.parent)
                finally:
                    temp_output.unlink(missing_ok=True)
            finally:
                concat_path.unlink(missing_ok=True)
        finally:
            frames.close()
    return output_path
