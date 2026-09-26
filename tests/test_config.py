"""1.1: the split between user-supplied and derived configuration.

`UserConfig` is what the caller writes; `DerivedConfig` is what probing the
source produces. Both are frozen — a run's grid and frame size are fixed once
and cannot drift apart mid-pipeline.
"""

import dataclasses

import pytest

from main import DerivedConfig, UserConfig


def derive(width: int, height: int, tile_aspect=None, **kwargs) -> DerivedConfig:
    config = UserConfig(input_dir="", output_dir="", **kwargs)
    return DerivedConfig.from_source(config, fps=30, dimensions=(width, height),
                                     frame_count=10, tile_aspect=tile_aspect)


def test_user_config_is_frozen():
    config = UserConfig(input_dir="", output_dir="")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.grid_size = 4


def test_derived_config_is_frozen():
    derived = derive(64, 48)
    with pytest.raises(dataclasses.FrozenInstanceError):
        derived.cell_size = (1, 1)


def test_known_source_gives_known_grid_and_cell():
    """64x48 at grid_size=2 is 4:3 -> an 8x6 grid of 8x8 cells."""
    derived = derive(64, 48, grid_size=2)

    assert derived.aspect_ratio == (4, 3)
    assert (derived.grid_x, derived.grid_y) == (8, 6)
    assert derived.cell_size == (8, 8)
    assert derived.target_dimensions == (64, 48)


def test_cell_size_never_degenerate():
    """A grid finer than the source still gets at least one pixel per cell."""
    assert min(derive(64, 48, grid_size=64).cell_size) >= 1


def test_unusual_aspect_ratio_is_kept():
    """A 2.39:1-ish source must not be snapped to 16:9."""
    derived = derive(478, 200)

    num, den = derived.aspect_ratio
    assert num / den == pytest.approx(478 / 200, rel=0.02)
    assert den <= 16  # limit_denominator keeps the pair usable as a multiplier


def test_grid_size_scales_the_grid_not_the_ratio():
    coarse, fine = derive(64, 48, grid_size=2), derive(64, 48, grid_size=4)

    assert coarse.aspect_ratio == fine.aspect_ratio
    assert (fine.grid_x, fine.grid_y) == (coarse.grid_x * 2, coarse.grid_y * 2)


def test_grid_size_one_collapses_to_a_single_cell():
    """2.4: grid_size=1 is single-frame mode — one cell over the whole frame.

    Not 4x3: multiplying the aspect pair by 1 would give twelve tiles, which is
    not the single gallery image per frame the mode is for.
    """
    derived = derive(64, 48, grid_size=1)

    assert (derived.grid_x, derived.grid_y) == (1, 1)
    assert derived.cell_size == (64, 48)
    assert derived.target_dimensions == (64, 48)


def test_native_tiles_shape_the_cell():
    """2.1: 16:9 tiles get a 16:9-ish cell instead of being squashed square."""
    derived = derive(512, 384, grid_size=8, tile_aspect=(16, 9))

    cell_w, cell_h = derived.cell_size
    assert cell_w / cell_h == pytest.approx(16 / 9, rel=0.05)
    # Rows are untouched — it's the columns that give way.
    assert derived.grid_y == derive(512, 384, grid_size=8).grid_y
    assert derived.grid_x < derive(512, 384, grid_size=8).grid_x


def test_native_tiles_keep_the_frame_shape_when_the_rows_round():
    """1080p at grid_size=16 snaps 1080 to 1152, and the columns must follow.

    Deriving the columns off the raw 1920 instead put the mosaic at 1.64:1
    against a 1.78:1 source, an 8% vertical squash the 512x384 cases can't see
    (384/24 divides exactly, so there's nothing to round).
    """
    derived = derive(1920, 1080, grid_size=16, tile_aspect=(16, 9))

    width, height = derived.target_dimensions
    assert width / height == pytest.approx(1920 / 1080, rel=0.02)


def test_square_tiles_change_nothing():
    """CIFAR is 1:1, so `native` has to land exactly where the old derivation did.

    True where the source's own cells came out square, which 512x384 does.
    Elsewhere square tiles force a square cell the base grid never had — see
    `test_square_tiles_still_square_the_cell`.
    """
    assert derive(512, 384, grid_size=8, tile_aspect=(1, 1)) == derive(512, 384,
                                                                      grid_size=8)


def test_square_tiles_still_square_the_cell():
    """2.39:1 at grid_size=8 gives a 2x2 cell either way, so nothing moves.

    Pinned because `native` reshapes the cell whatever the tiles are, so a
    square gallery is only a no-op by arithmetic, not by construction.
    """
    native = derive(478, 200, grid_size=8, tile_aspect=(1, 1))

    assert native.cell_size[0] == native.cell_size[1]
    assert native.target_dimensions[0] / native.target_dimensions[1] == \
        pytest.approx(478 / 200, rel=0.05)


def test_native_tiles_do_not_touch_single_frame_mode():
    """The one cell is the whole frame; the tile fits into it, not the reverse."""
    derived = derive(512, 384, grid_size=1, tile_aspect=(16, 9))

    assert derived.cell_size == (512, 384)


def test_native_tiles_reach_the_smallest_real_grid():
    """grid_size=2 is the first grid with cells to reshape. The other native
    cases all sit at 8 or 16, so they can't see where it starts."""
    cell_w, cell_h = derive(64, 48, grid_size=2, tile_aspect=(16, 9)).cell_size

    assert cell_w > cell_h


def test_1080p_native_widescreen_tiles_come_out_even():
    """71 columns of 27px was 1917, which libx264 refuses under yuv420p. One
    more column (1944) lands nearer 1920 than one fewer (1890)."""
    derived = derive(1920, 1080, grid_size=8, tile_aspect=(16, 9))

    assert derived.cell_size == (27, 15)
    assert derived.target_dimensions == (1944, 1080)


def test_an_odd_row_count_gives_way_on_the_height():
    """720p at grid_size=3 was 27 rows of 27px, 729 high."""
    derived = derive(1280, 720, grid_size=3)

    assert derived.cell_size == (27, 27)
    assert derived.target_dimensions == (1296, 702)


def test_single_frame_mode_trims_the_cell_instead():
    """The 1x1 grid can't give way, so the one cell loses a pixel."""
    assert derive(853, 480, grid_size=1).target_dimensions == (852, 480)


@pytest.mark.parametrize("tile_aspect", [None, (16, 9), (4, 3), (1, 1), (21, 9)])
@pytest.mark.parametrize("grid_size", [1, 2, 3, 5, 8, 16])
@pytest.mark.parametrize("dimensions", [(1920, 1080), (1280, 720), (853, 480),
                                        (478, 200), (641, 359), (64, 48)])
def test_target_dimensions_are_always_even(dimensions, grid_size, tile_aspect):
    """No aspect check: at 3 columns a step is a third of the frame, so the
    shape is only as good as the grid is fine. The cases above pin that."""
    derived = derive(*dimensions, grid_size=grid_size, tile_aspect=tile_aspect)

    width, height = derived.target_dimensions
    assert width % 2 == 0 and height % 2 == 0
    assert (width, height) == (derived.grid_x * derived.cell_size[0],
                               derived.grid_y * derived.cell_size[1])
