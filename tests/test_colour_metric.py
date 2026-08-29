"""2.2: colour matching, and that a metric is swappable.

The gallery here is a lattice of solid colours rather than the grey ramp in
`conftest`, because brightness-only matching cannot tell red from blue and this
is the whole point of the metric.
"""

import numpy as np
import pytest

from conftest import CELL, make_frames
from main import (BrightnessMetric, ColourMetric, DerivedConfig, UserConfig,
                  build_metric, build_mosaics, gallery_brightness,
                  nearest_occupied)

BIN_WIDTH = 256 / 8  # one lattice step at bins=8


def colour_gallery(levels: int = 8) -> np.ndarray:
    """A levels^3 lattice of solid BGR tiles, already at cell size.

    Every bin of an 8-bin lattice holds exactly one tile, so a query colour's
    match is decided by the lattice alone with nothing to sample between.
    """
    steps = np.linspace(8, 247, levels)
    grid = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1)
    colours = np.rint(grid.reshape(-1, 3)).astype(np.uint8)
    return np.broadcast_to(colours[:, None, None, :],
                           (len(colours), CELL[1], CELL[0], 3)).copy()


def flat_frame(colour, grid: tuple[int, int] = (3, 2)) -> np.ndarray:
    grid_x, grid_y = grid
    frame = np.empty((grid_y * CELL[1], grid_x * CELL[0], 3), dtype=np.uint8)
    frame[:] = colour
    return frame


@pytest.fixture
def tiles() -> np.ndarray:
    return colour_gallery()


def make_metric(tiles, **kwargs) -> ColourMetric:
    metric = ColourMetric(**{"bins": 8, "candidates": 1, **kwargs})
    metric.precompute(tiles, CELL)
    return metric


def test_match_lands_in_the_query_colour_s_bin(tiles):
    """Every bin is occupied here, so a match never leaves the query's own bin."""
    metric = make_metric(tiles)
    rng = np.random.default_rng(0)

    for colour in rng.integers(0, 256, (32, 3)):
        picked = metric.tiles[metric.match(flat_frame(colour))]
        error = np.abs(picked.reshape(-1, 3).astype(float) - colour)
        assert error.max() < BIN_WIDTH


def test_colour_beats_brightness_on_equally_bright_colours(tiles):
    """Two colours of the same luma are one tile to brightness, two to colour."""
    # Rec.601 weights blue at 0.114 and red at 0.299, so full blue and a dim red
    # land within 0.1/255 of each other in luma and nowhere near in BGR.
    blue, red = (255, 0, 0), (0, 0, 97)

    colour = make_metric(tiles)
    bright = BrightnessMetric(candidates=1)
    bright.precompute(tiles, CELL, gallery_brightness(tiles))

    frame = np.zeros((CELL[1], CELL[0] * 2, 3), dtype=np.uint8)
    frame[:, :CELL[0]] = blue
    frame[:, CELL[0]:] = red

    left, right = colour.match(frame)[0]
    assert not np.array_equal(colour.tiles[left], colour.tiles[right])
    # And each pane picks something recognisably that colour.
    assert colour.tiles[left][0, 0].argmax() == 0
    assert colour.tiles[right][0, 0].argmax() == 2

    # Brightness sees one level for both, so both panes get the same tile.
    left, right = bright.match(frame)[0]
    np.testing.assert_array_equal(bright.tiles[left], bright.tiles[right])


def test_metrics_disagree_on_a_colourful_frame(tiles):
    """Per-pixel noise averages to grey, so the frame is blocks of real colour."""
    rng = np.random.default_rng(4)
    cells = rng.integers(0, 256, (6, 4, 3), dtype=np.uint8)
    frame = np.repeat(np.repeat(cells, CELL[1], axis=0), CELL[0], axis=1)

    colour = make_metric(tiles)
    bright = BrightnessMetric(candidates=1)
    bright.precompute(tiles, CELL, gallery_brightness(tiles))

    assert not np.array_equal(colour.tiles[colour.match(frame)],
                              bright.tiles[bright.match(frame)])


def test_empty_bins_borrow_the_nearest_occupied_one():
    """A gallery covering part of the space still answers for all of it."""
    reds = np.zeros((4, CELL[1], CELL[0], 3), dtype=np.uint8)
    reds[:, :, :, 2] = np.array([64, 96, 128, 160])[:, None, None]
    metric = make_metric(reds)

    # Nothing blue exists; the answer is the closest red, not a crash.
    picked = metric.tiles[metric.match(flat_frame((255, 0, 0)))]
    assert picked.reshape(-1, 3)[:, 2].max() == 64  # darkest red is nearest


def test_nearest_occupied_is_identity_on_a_full_lattice():
    full = np.ones(2 ** 3, dtype=bool)
    np.testing.assert_array_equal(nearest_occupied(full, 2), np.arange(8))


def test_nearest_occupied_rejects_an_empty_lattice():
    with pytest.raises(ValueError):
        nearest_occupied(np.zeros(2 ** 3, dtype=bool), 2)


def test_candidates_one_is_deterministic(tiles):
    metric = make_metric(tiles)
    frame = flat_frame((90, 140, 200), grid=(8, 8))
    np.testing.assert_array_equal(metric.match(frame), metric.match(frame))


def test_candidates_sample_among_tiles_sharing_a_bin():
    """Duplicate colours in one bin are what `candidates` picks between."""
    shades = np.zeros((16, CELL[1], CELL[0], 3), dtype=np.uint8)
    shades[:, :, :, 1] = np.arange(120, 136)[:, None, None]  # all one bin at bins=8

    def distinct(candidates: int) -> int:
        metric = make_metric(shades, candidates=candidates)
        return len(np.unique(metric.match(flat_frame((0, 128, 0), grid=(8, 8)))))

    assert distinct(1) == 1
    assert distinct(16) > 1


def test_seed_makes_runs_reproducible(tiles):
    shades = np.repeat(tiles, 4, axis=0)
    frame = flat_frame((90, 140, 200), grid=(8, 8))

    def run(seed: int) -> np.ndarray:
        return make_metric(shades, candidates=8, seed=seed).match(frame)

    np.testing.assert_array_equal(run(0), run(0))
    assert not np.array_equal(run(0), run(1))


def test_tiles_are_compacted_to_reachable_images(tiles):
    """With one candidate per bin, duplicates of a colour are unreachable."""
    dense = np.repeat(tiles, 4, axis=0)
    metric = make_metric(dense)

    assert len(metric.tiles) == len(tiles)
    assert len(metric.tiles) < len(dense)


def test_bucket_size_grows_with_candidates(tiles):
    dense = np.repeat(tiles, 8, axis=0)

    def size(candidates: int) -> float:
        return make_metric(dense, candidates=candidates).bucket_size

    assert size(8) > size(2)


def test_tiles_are_at_cell_size(tiles):
    metric = make_metric(tiles)
    assert metric.tiles.shape[1:] == (CELL[1], CELL[0], 3)
    assert metric.tiles.dtype == np.uint8


def test_precompute_rejects_a_gallery_that_is_not_at_cell_size(gallery):
    with pytest.raises(ValueError):
        ColourMetric(bins=8).precompute(gallery, CELL)


@pytest.mark.parametrize("bins", [1, 0, 128])
def test_bins_outside_the_supported_range_is_rejected(bins):
    with pytest.raises(ValueError):
        ColourMetric(bins=bins)


def test_build_metric_picks_the_metric_the_config_names(cell_gallery):
    derived = DerivedConfig(src_fps=30, src_dimensions=(16, 16), src_frame_count=0,
                            aspect_ratio=(4, 4), grid=(4, 4), cell_size=CELL)

    def built(name):
        config = UserConfig(input_dir="", output_dir="", metric=name,
                            contrast=1.0, candidates=4)
        return build_metric(cell_gallery, config, derived)

    assert isinstance(built("colour"), ColourMetric)
    assert isinstance(built("brightness"), BrightnessMetric)


def test_build_mosaics_takes_either_metric(tiles, cell_gallery):
    """Swapping the metric costs nothing downstream — same call, same shapes."""
    derived = DerivedConfig(src_fps=30, src_dimensions=(16, 12), src_frame_count=0,
                            aspect_ratio=(4, 3), grid=(4, 3), cell_size=CELL)
    frames = list(make_frames(3, CELL[0] * 4, CELL[1] * 3, seed=2))

    bright = BrightnessMetric(candidates=4, epsilon=0.1, seed=0)
    bright.precompute(cell_gallery, CELL, gallery_brightness(cell_gallery))

    shapes = {name: [m.shape for m in build_mosaics(iter(frames), metric, derived)]
              for name, metric in (("colour", make_metric(tiles)),
                                   ("brightness", bright))}

    assert shapes["colour"] == shapes["brightness"]
    assert shapes["colour"] == [(CELL[1] * 3, CELL[0] * 4, 3)] * 3
