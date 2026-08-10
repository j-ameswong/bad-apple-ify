import numpy as np

from main import (BrightnessMetric, Config, gallery_brightness, mosaic_frame,
                  probe_video, stream_frames)

CELL = (4, 4)


def assemble_by_loop(tiles: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """The pre-0.1 per-cell assembly, kept as the reference to match."""
    grid_y, grid_x = indices.shape
    cell_h, cell_w = tiles.shape[1:3]
    out = np.zeros((grid_y * cell_h, grid_x * cell_w, 3), dtype=tiles.dtype)
    for y in range(grid_y):
        for x in range(grid_x):
            out[y * cell_h:(y + 1) * cell_h,
                x * cell_w:(x + 1) * cell_w] = tiles[indices[y, x]]
    return out


def test_vectorised_assembly_is_byte_identical(gallery):
    """0.1: fancy indexing + transpose must equal the nested-loop placement."""
    metric = BrightnessMetric(candidates=8, epsilon=0.1, seed=0)
    metric.precompute(gallery, CELL, gallery_brightness(gallery))

    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, (CELL[1] * 6, CELL[0] * 9, 3), dtype=np.uint8)

    indices = metric.match(frame)
    fast = metric.tiles[indices]
    grid_y, grid_x, cell_h, cell_w, _ = fast.shape
    fast = fast.transpose(0, 2, 1, 3, 4).reshape(grid_y * cell_h, grid_x * cell_w, 3)

    np.testing.assert_array_equal(fast, assemble_by_loop(metric.tiles, indices))


def test_mosaic_frame_preserves_frame_size(gallery):
    metric = BrightnessMetric(candidates=4, epsilon=0.1, seed=0)
    metric.precompute(gallery, CELL, gallery_brightness(gallery))
    frame = np.full((CELL[1] * 5, CELL[0] * 7, 3), 90, dtype=np.uint8)

    mosaic = mosaic_frame(frame, metric)

    assert mosaic.shape == frame.shape
    assert mosaic.dtype == np.uint8


def test_every_block_of_the_mosaic_is_a_gallery_tile(gallery):
    """No blending or interpolation: each cell is one image, placed whole."""
    metric = BrightnessMetric(candidates=4, epsilon=0.1, seed=0)
    metric.precompute(gallery, CELL, gallery_brightness(gallery))
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (CELL[1] * 4, CELL[0] * 4, 3), dtype=np.uint8)

    mosaic = mosaic_frame(frame, metric)

    blocks = mosaic.reshape(4, CELL[1], 4, CELL[0], 3).transpose(0, 2, 1, 3, 4)
    for block in blocks.reshape(-1, CELL[1], CELL[0], 3):
        assert any(np.array_equal(block, tile) for tile in metric.tiles)


def test_mosaic_tracks_frame_brightness(gallery):
    """A brighter frame must produce a brighter mosaic."""
    metric = BrightnessMetric(candidates=4, epsilon=0.05, seed=0)
    metric.precompute(gallery, CELL, gallery_brightness(gallery))
    shape = (CELL[1] * 4, CELL[0] * 4, 3)

    dark = mosaic_frame(np.full(shape, 40, dtype=np.uint8), metric)
    light = mosaic_frame(np.full(shape, 210, dtype=np.uint8), metric)

    assert dark.mean() < light.mean()


def test_pipeline_over_a_video(video, gallery):
    """Every source frame yields one mosaic at the configured target size."""
    path, frames = video
    config = Config(input_dir=str(path), output_dir="", grid_size=2,
                    candidates=8, epsilon=0.1)
    probe_video(config)

    metric = BrightnessMetric(candidates=config.candidates,
                              epsilon=config.epsilon, seed=config.seed)
    metric.precompute(gallery, config.cell_size(), gallery_brightness(gallery))

    mosaics = [mosaic_frame(frame, metric) for frame in stream_frames(config)]

    assert len(mosaics) == len(frames)
    for mosaic in mosaics:
        assert mosaic.shape[1::-1] == config.target_dimensions
