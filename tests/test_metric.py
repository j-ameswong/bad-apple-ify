import cv2
import numpy as np
import pytest

from main import BrightnessMetric, gallery_brightness

CELL = (4, 4)


def argmin_match(brightness: np.ndarray, level: float) -> int:
    return int(np.argmin(np.abs(brightness - level)))


def flat_frame(value: int, grid: tuple[int, int] = (3, 2)) -> np.ndarray:
    """A frame of `grid` cells, every pixel `value` — one known brightness level."""
    grid_x, grid_y = grid
    return np.full((grid_y * CELL[1], grid_x * CELL[0], 3), value, dtype=np.uint8)


@pytest.fixture
def brightness(gallery):
    return gallery_brightness(gallery)


def test_lut_matches_argmin_for_all_levels(gallery, brightness):
    """0.3: with one candidate, the LUT must reproduce the argmin answer exactly."""
    metric = BrightnessMetric(candidates=1)
    metric.precompute(gallery, CELL, brightness)

    for level in range(256):
        picked = metric.tiles[metric.match(flat_frame(level))]
        expected = gallery[argmin_match(brightness, level / 255.0)]
        expected = cv2.resize(expected, CELL)
        # Every cell of a flat frame resolves to the same tile.
        for row in picked.reshape(-1, CELL[1], CELL[0], 3):
            np.testing.assert_array_equal(row, expected)


def test_match_returns_grid_of_indices(gallery, brightness):
    metric = BrightnessMetric(candidates=8, epsilon=0.05)
    metric.precompute(gallery, CELL, brightness)

    indices = metric.match(flat_frame(128, grid=(5, 3)))

    assert indices.shape == (3, 5)
    assert indices.dtype.kind == "i"
    assert np.all((indices >= 0) & (indices < len(metric.tiles)))


def test_match_uses_per_cell_brightness(gallery, brightness):
    """Cells differing in brightness must resolve to different tiles."""
    metric = BrightnessMetric(candidates=1)
    metric.precompute(gallery, CELL, brightness)

    frame = np.zeros((CELL[1], CELL[0] * 2, 3), dtype=np.uint8)
    frame[:, CELL[0]:] = 255

    left, right = metric.match(frame)[0]
    assert left != right


def test_stochastic_selection_stays_within_epsilon(gallery, brightness):
    """0.4: variety must not cost tonal accuracy."""
    epsilon = 0.05
    metric = BrightnessMetric(candidates=32, epsilon=epsilon, seed=0)
    metric.precompute(gallery, CELL, brightness)
    tile_brightness = gallery_brightness(metric.tiles)

    for level in range(0, 256, 8):
        target = level / 255.0
        picked = metric.match(flat_frame(level, grid=(8, 8)))
        errors = np.abs(tile_brightness[picked] - target)
        # Either inside epsilon, or the gallery has nothing there and we fell
        # back to the nearest image — which is exactly the argmin error.
        floor = np.abs(brightness - target).min()
        assert errors.max() <= max(epsilon, floor) + 1 / 255


def test_stochastic_selection_increases_variety(gallery, brightness):
    """More candidates must draw on more of the gallery for the same input."""
    frame = flat_frame(128, grid=(16, 16))

    def distinct(candidates: int) -> int:
        metric = BrightnessMetric(candidates=candidates, epsilon=0.5, seed=0)
        metric.precompute(gallery, CELL, brightness)
        return len(np.unique(metric.match(frame)))

    assert distinct(1) == 1
    assert distinct(16) > 1


def test_candidates_one_is_deterministic(gallery, brightness):
    metric = BrightnessMetric(candidates=1)
    metric.precompute(gallery, CELL, brightness)
    frame = flat_frame(200, grid=(8, 8))

    np.testing.assert_array_equal(metric.match(frame), metric.match(frame))


def test_seed_makes_runs_reproducible(gallery, brightness):
    frame = flat_frame(140, grid=(8, 8))

    def run(seed: int) -> np.ndarray:
        metric = BrightnessMetric(candidates=16, epsilon=0.1, seed=seed)
        metric.precompute(gallery, CELL, brightness)
        return metric.match(frame)

    np.testing.assert_array_equal(run(0), run(0))
    assert not np.array_equal(run(0), run(1))


def test_epsilon_zero_falls_back_to_nearest(gallery, brightness):
    """No candidate can sit inside a zero radius, so every level takes argmin."""
    metric = BrightnessMetric(candidates=64, epsilon=0.0, seed=0)
    metric.precompute(gallery, CELL, brightness)
    exact = BrightnessMetric(candidates=1)
    exact.precompute(gallery, CELL, brightness)

    for level in range(0, 256, 4):
        frame = flat_frame(level, grid=(4, 4))
        np.testing.assert_array_equal(metric.tiles[metric.match(frame)],
                                      exact.tiles[exact.match(frame)])


def test_candidates_exceeding_gallery_is_clamped(gallery, brightness):
    metric = BrightnessMetric(candidates=10 * len(gallery), epsilon=1.0, seed=0)
    metric.precompute(gallery, CELL, brightness)

    indices = metric.match(flat_frame(128, grid=(8, 8)))
    assert np.all(indices < len(metric.tiles))


def test_tiles_are_at_cell_size(gallery, brightness):
    """The per-cell resize is gone: tiles come out of precompute ready to place."""
    metric = BrightnessMetric(candidates=4, epsilon=0.05)
    metric.precompute(gallery, CELL, brightness)

    assert metric.tiles.shape[1:] == (CELL[1], CELL[0], 3)
    assert metric.tiles.dtype == np.uint8


def test_tiles_are_compacted_to_reachable_images(gallery):
    """Unreachable gallery images are never resized or stored.

    With one candidate there are only 256 possible answers, so a gallery
    denser than that must come out smaller — this is the 0.3 observation that
    40k CIFAR images collapse to ~97 tiles.
    """
    dense = np.repeat(gallery, 4, axis=0)
    metric = BrightnessMetric(candidates=1)
    metric.precompute(dense, CELL, gallery_brightness(dense))

    assert len(metric.tiles) <= 256
    assert len(metric.tiles) < len(dense)


def test_bucket_size_grows_with_candidates(gallery, brightness):
    def size(candidates: int) -> float:
        metric = BrightnessMetric(candidates=candidates, epsilon=0.5)
        metric.precompute(gallery, CELL, brightness)
        return metric.bucket_size

    assert size(16) > size(4)
