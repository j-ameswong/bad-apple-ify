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


def test_derived_carries_source_metadata():
    derived = derive(64, 48)

    assert derived.src_fps == 30
    assert derived.output_fps == 30
    assert derived.src_dimensions == (64, 48)
    assert derived.src_frame_count == 10


def test_known_source_gives_known_grid_and_cell():
    """64x48 at grid_size=2 is 4:3 -> an 8x6 grid of 8x8 cells."""
    derived = derive(64, 48, grid_size=2)

    assert derived.aspect_ratio == (4, 3)
    assert (derived.grid_x, derived.grid_y) == (8, 6)
    assert derived.cell_size == (8, 8)
    assert derived.target_dimensions == (64, 48)


def test_target_dimensions_are_grid_multiples():
    """A source that isn't a whole multiple of the grid snaps to the nearest one."""
    derived = derive(70, 50, grid_size=2)

    cell_w, cell_h = derived.cell_size
    assert derived.target_dimensions == (derived.grid_x * cell_w,
                                         derived.grid_y * cell_h)


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


def test_single_frame_cell_keeps_the_source_shape():
    """The one cell is the frame, so it takes the source's ratio, not the pair's."""
    derived = derive(478, 200, grid_size=1)

    assert derived.cell_size == (478, 200)


def test_native_tiles_shape_the_cell():
    """2.1: 16:9 tiles get a 16:9-ish cell instead of being squashed square."""
    derived = derive(512, 384, grid_size=8, tile_aspect=(16, 9))

    cell_w, cell_h = derived.cell_size
    assert cell_w / cell_h == pytest.approx(16 / 9, rel=0.05)
    # Rows are untouched — it's the columns that give way.
    assert derived.grid_y == derive(512, 384, grid_size=8).grid_y
    assert derived.grid_x < derive(512, 384, grid_size=8).grid_x


def test_native_tiles_keep_the_frame_shape():
    """The tiles absorb the rounding, so the mosaic still looks like the source."""
    derived = derive(512, 384, grid_size=8, tile_aspect=(16, 9))

    width, height = derived.target_dimensions
    assert width / height == pytest.approx(512 / 384, rel=0.05)


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


def test_derivation_is_a_pure_function_of_its_inputs():
    """Same UserConfig and same source dimensions -> identical DerivedConfig."""
    assert derive(478, 200, grid_size=3) == derive(478, 200, grid_size=3)
