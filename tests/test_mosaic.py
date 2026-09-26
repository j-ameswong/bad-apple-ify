import numpy as np

from main import mosaic_frame, resize_gallery_to_cells


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


class FixedMatch:
    """A metric that's already made up its mind, so the test knows exactly
    which tile `mosaic_frame()` was told to put in which cell."""

    def __init__(self, tiles: np.ndarray, indices: np.ndarray):
        self.tiles = tiles
        self._indices = indices

    def match(self, frame: np.ndarray) -> np.ndarray:
        return self._indices


def test_vectorised_assembly_is_byte_identical(gallery):
    """0.1: `mosaic_frame()` must place tiles exactly where the nested loop did.

    Cells and grid are both non-square and every cell gets a different tile, so
    no mix-up of axes can cancel out. At 4x4, swapping cell_h for cell_w would.
    """
    cell_w, cell_h = 5, 3
    tiles = resize_gallery_to_cells(gallery, (cell_w, cell_h))
    grid_y, grid_x = 6, 9
    rng = np.random.default_rng(2)
    indices = rng.permutation(len(tiles))[:grid_y * grid_x].reshape(grid_y, grid_x)
    frame = np.zeros((grid_y * cell_h, grid_x * cell_w, 3), dtype=np.uint8)

    mosaic = mosaic_frame(frame, FixedMatch(tiles, indices))

    np.testing.assert_array_equal(mosaic, assemble_by_loop(tiles, indices))
