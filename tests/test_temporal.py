"""2.3: holding a cell's tile while its bucket key doesn't change.

The frame-cache half of 2.3 is dead (see PLAN.md); what survives is temporal
stability. A cell whose colour hasn't moved should keep the tile it had instead
of re-rolling its bucket every frame.
"""

import numpy as np
import pytest

from conftest import CELL, make_frames
from main import (BrightnessMetric, ColourMetric, SteadyMetric, UserConfig,
                  DerivedConfig, build_metric)

GRID = (3, 2)  # cells across, cells down


def flat_frame(level: int) -> np.ndarray:
    grid_x, grid_y = GRID
    return np.full((grid_y * CELL[1], grid_x * CELL[0], 3), level, dtype=np.uint8)


def frame_with_one_cell(level: int, other: int) -> np.ndarray:
    """`level` everywhere except the top-left cell, which is `other`."""
    frame = flat_frame(level)
    frame[:CELL[1], :CELL[0]] = other
    return frame


@pytest.fixture(params=["brightness", "colour"])
def make_metric(request, cell_gallery):
    """Builds a precomputed metric of one kind, with buckets big enough to re-roll.

    A factory rather than an instance because the sampling is stateful once
    `SteadyMetric` is involved, and a couple of tests want two of the same.
    """
    def make():
        if request.param == "brightness":
            made = BrightnessMetric(candidates=16, epsilon=0.5, seed=3)
        else:
            made = ColourMetric(bins=8, candidates=16, seed=3)
        made.precompute(cell_gallery, CELL)
        return made
    return make


@pytest.fixture
def metric(make_metric):
    return make_metric()


def test_match_is_sample_of_keys(make_metric):
    """The split is a pure refactor: match() is still sample(keys())."""
    frame = make_frames(1, GRID[0] * CELL[0], GRID[1] * CELL[1])[0]
    metric, twin = make_metric(), make_metric()
    assert np.array_equal(metric.match(frame), twin.sample(twin.keys(frame)))


def test_unwrapped_metric_rerolls(metric):
    """The behaviour being fixed: same frame twice, different tiles."""
    frame = flat_frame(128)
    assert not np.array_equal(metric.match(frame), metric.match(frame))


def test_steady_holds_a_static_frame(metric):
    steady = SteadyMetric(metric)
    frame = flat_frame(128)
    first = steady.match(frame)
    for _ in range(5):
        assert np.array_equal(steady.match(frame), first)


def test_steady_resamples_when_every_key_moves(metric):
    steady = SteadyMetric(metric)
    first = steady.match(flat_frame(40))
    second = steady.match(flat_frame(200))
    assert (first != second).all()


def test_steady_holds_the_cells_that_did_not_move(metric):
    steady = SteadyMetric(metric)
    first = steady.match(flat_frame(128))
    second = steady.match(frame_with_one_cell(128, 220))
    assert second[0, 0] != first[0, 0]
    assert np.array_equal(second[0, 1:], first[0, 1:])
    assert np.array_equal(second[1:], first[1:])


def test_steady_passes_through_tiles_and_buckets(metric):
    steady = SteadyMetric(metric)
    assert steady.tiles is metric.tiles
    assert steady.bucket_size == metric.bucket_size


def test_build_metric_honours_hold_tiles(cell_gallery):
    derived = DerivedConfig(src_fps=30, src_dimensions=(12, 8),
                            src_frame_count=1, aspect_ratio=(3, 2),
                            grid=GRID, cell_size=CELL)
    config = UserConfig(input_dir="x", output_dir="y", metric="colour",
                        colour_bins=8, candidates=16, contrast=1.0)
    assert isinstance(build_metric(cell_gallery, config, derived), SteadyMetric)

    off = UserConfig(input_dir="x", output_dir="y", metric="colour",
                     colour_bins=8, candidates=16, contrast=1.0,
                     hold_tiles=False)
    assert isinstance(build_metric(cell_gallery, off, derived), ColourMetric)


def test_reprecompute_discards_choices_from_the_previous_gallery():
    steady = SteadyMetric(ColourMetric(bins=8, candidates=2, seed=0))
    gallery = np.full((2, 1, 1, 3), 128, dtype=np.uint8)
    frame = gallery[0]
    steady.precompute(gallery, (1, 1))
    assert steady.match(frame).item() == 1
    steady.precompute(gallery[:1], (1, 1))
    assert steady.match(frame).item() == 0


def test_changing_frame_geometry_starts_a_new_held_grid(metric):
    steady = SteadyMetric(metric)
    steady.match(flat_frame(128))
    larger = np.full((CELL[1] * 4, CELL[0] * 5, 3), 128, dtype=np.uint8)
    assert steady.match(larger).shape == (4, 5)
