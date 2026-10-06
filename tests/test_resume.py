"""Segment checkpoints preserve matcher state and source position."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import bad_apple.segments as segments
import bad_apple.gallery as gallery_io
from conftest import read_video
from main import UserConfig, build_metric, probe_video


class FakeGallery:
    def __init__(self, fingerprint: str = "test-gallery"):
        self.fingerprint = fingerprint
        self.native_aspect = None


def setup_run(tmp_path: Path, video_factory, gallery, count: int = 6,
              segment_frames: int = 2):
    source_path, _ = video_factory(count=count)
    output = tmp_path / "out" / "mosaic.mp4"
    config = UserConfig(input_dir=str(source_path), output_dir=str(output.parent),
                        grid_size=2, candidates=8, colour_bins=8, seed=17,
                        segment_frames=segment_frames)
    derived = probe_video(config)
    tiles = gallery_io.resize_gallery_to_cells(gallery, derived.cell_size)
    metric = build_metric(tiles, config, derived)
    return source_path, output, config, derived, metric


def test_interrupted_resume_restores_metric_and_seeks_after_checkpoint(
        tmp_path, video_factory, gallery, monkeypatch):
    source_path, interrupted_output, config, derived, metric = setup_run(
        tmp_path, video_factory, gallery, count=7)
    actual_encode = segments.encode_video
    calls = 0

    def fail_after_one(mosaics, current_derived, path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return actual_encode(mosaics, current_derived, path)

    monkeypatch.setattr(segments, "encode_video", fail_after_one)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        segments.encode_segmented(FakeGallery(), config, derived, metric,
                                  interrupted_output)
    assert (interrupted_output.parent / "segments" / "part_0000.mp4").is_file()

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    resumed_derived = probe_video(resumed_config)
    resumed_metric = build_metric(gallery_io.resize_gallery_to_cells(
        gallery, resumed_derived.cell_size), resumed_config, resumed_derived)
    resumed_output = interrupted_output
    resumed_starts: list[int] = []
    real_seek = segments._seek_frames
    real_stream_frames = segments.stream_frames

    def observed_seek(current_config, current_derived, start):
        resumed_starts.append(start)
        return real_seek(current_config, current_derived, start)

    def no_prefix_replay(*args, **kwargs):
        raise AssertionError("resume replayed the source prefix")

    monkeypatch.setattr(segments, "_seek_frames", observed_seek)
    monkeypatch.setattr(segments, "stream_frames", no_prefix_replay)
    monkeypatch.setattr(segments, "encode_video", actual_encode)
    segments.encode_segmented(FakeGallery(), resumed_config, resumed_derived,
                              resumed_metric, resumed_output)
    assert resumed_starts == [2]
    monkeypatch.setattr(segments, "stream_frames", real_stream_frames)

    full_root = tmp_path / "full"
    full_root.mkdir()
    full_config = replace(config, output_dir=str(full_root))
    full_derived = probe_video(full_config)
    full_metric = build_metric(gallery_io.resize_gallery_to_cells(gallery,
                                                            full_derived.cell_size),
                               full_config, full_derived)
    full_output = full_root / "mosaic.mp4"
    segments.encode_segmented(FakeGallery(), full_config, full_derived,
                              full_metric, full_output)
    np.testing.assert_array_equal(read_video(resumed_output), read_video(full_output))


@pytest.mark.parametrize("count, expected_parts", [(5, [2, 2, 1]), (6, [2, 2, 2])])
def test_final_short_segment_and_exact_multiple(tmp_path, video_factory, gallery,
                                                count, expected_parts):
    _, output, config, derived, metric = setup_run(
        tmp_path, video_factory, gallery, count=count, segment_frames=2)
    segments.encode_segmented(FakeGallery(), config, derived, metric, output)
    manifest = __import__("json").loads(
        (output.parent / "segments" / "checkpoint.json").read_text())
    assert [part["frames"] for part in manifest["segments"]] == expected_parts
    assert len(read_video(output)) == count


def test_changed_configuration_and_source_are_refused(tmp_path, video_factory,
                                                       gallery):
    _, output, config, derived, metric = setup_run(tmp_path, video_factory, gallery)
    segments.encode_segmented(FakeGallery(), config, derived, metric, output)

    changed_config = replace(config, seed=config.seed + 1)
    changed_derived = probe_video(changed_config)
    changed_metric = build_metric(gallery_io.resize_gallery_to_cells(
        gallery, changed_derived.cell_size), changed_config, changed_derived)
    with pytest.raises(ValueError, match="different source or configuration"):
        segments.encode_segmented(FakeGallery(), changed_config, changed_derived,
                                 changed_metric, output)

    changed_gallery = FakeGallery("different-gallery")
    same_metric = build_metric(gallery_io.resize_gallery_to_cells(gallery,
                                                            derived.cell_size),
                               config, derived)
    with pytest.raises(ValueError, match="different source or configuration"):
        segments.encode_segmented(changed_gallery, config, derived, same_metric,
                                  output)

    source_path = Path(config.input_dir)
    stat = source_path.stat()
    import os
    os.utime(source_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    touched_metric = build_metric(gallery_io.resize_gallery_to_cells(gallery,
                                                               derived.cell_size),
                                  config, derived)
    with pytest.raises(ValueError, match="different source or configuration"):
        segments.encode_segmented(FakeGallery(), config, derived, touched_metric,
                                  output)


def test_corrupt_segment_is_rejected(tmp_path, video_factory, gallery):
    _, output, config, derived, metric = setup_run(tmp_path, video_factory, gallery)
    segments.encode_segmented(FakeGallery(), config, derived, metric, output)
    part = output.parent / "segments" / "part_0000.mp4"
    part.write_bytes(part.read_bytes() + b"corruption")
    new_metric = build_metric(gallery_io.resize_gallery_to_cells(gallery,
                                                           derived.cell_size),
                              config, derived)
    with pytest.raises(ValueError, match="missing or corrupt"):
        segments.encode_segmented(FakeGallery(), config, derived, new_metric,
                                  output)


def test_manifest_rejects_boolean_frame_count(tmp_path, video_factory, gallery):
    import json

    _, output, config, derived, metric = setup_run(
        tmp_path, video_factory, gallery, count=4, segment_frames=2)
    segments.encode_segmented(FakeGallery(), config, derived, metric, output)
    manifest_path = output.parent / "segments" / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["segments"][0]["frames"] = True
    manifest_path.write_text(json.dumps(manifest))

    new_metric = build_metric(gallery_io.resize_gallery_to_cells(
        gallery, derived.cell_size), config, derived)
    with pytest.raises(ValueError, match="invalid frame count"):
        segments.encode_segmented(FakeGallery(), config, derived, new_metric,
                                  output)
