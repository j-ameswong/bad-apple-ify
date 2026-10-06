"""Explicit output sizing is independent of source resolution."""

import pytest

from main import DerivedConfig, UserConfig


def derive(width: int, height: int, **kwargs) -> DerivedConfig:
    tile_aspect = kwargs.pop("tile_aspect", None)
    config = UserConfig(input_dir="", output_dir="", **kwargs)
    return DerivedConfig.from_source(config, fps=30, dimensions=(width, height),
                                     frame_count=10, tile_aspect=tile_aspect)


def test_explicit_grid_and_cell_size_are_exact():
    derived = derive(1920, 1080, grid=(120, 68), cell_size=32)

    assert derived.grid == (120, 68)
    assert derived.cell_size == (32, 32)
    assert derived.target_dimensions == (3840, 2176)


def test_explicit_cell_size_uses_aspect_not_source_resolution():
    small = derive(640, 360, cell_size=32, grid_size=8)
    large = derive(1920, 1080, cell_size=32, grid_size=8)

    assert small.grid == large.grid == (128, 72)
    assert small.target_dimensions == large.target_dimensions == (4096, 2304)


def test_native_tile_width_comes_from_height_and_tile_aspect():
    derived = derive(640, 360, grid=(20, 12), cell_size=32,
                     tile_aspect=(16, 9))

    assert derived.cell_size == (57, 32)
    assert derived.target_dimensions == (1140, 384)


@pytest.mark.parametrize("tile_fit", ["crop", "stretch"])
def test_crop_and_stretch_keep_explicit_cells_square(tile_fit):
    derived = derive(640, 360, grid=(20, 12), cell_size=32,
                     tile_fit=tile_fit, tile_aspect=(16, 9))

    assert derived.cell_size == (32, 32)
    assert derived.target_dimensions == (640, 384)


def test_single_native_cell_keeps_source_aspect():
    derived = derive(640, 360, grid_size=1, cell_size=36,
                     tile_aspect=(16, 9))

    assert derived.grid == (1, 1)
    assert derived.cell_size == (64, 36)


@pytest.mark.parametrize("tile_fit", ["crop", "stretch"])
def test_single_explicit_cell_is_square_for_crop_and_stretch(tile_fit):
    derived = derive(640, 360, grid_size=1, cell_size=32, tile_fit=tile_fit,
                     tile_aspect=(16, 9))

    assert derived.cell_size == (32, 32)


@pytest.mark.parametrize("kwargs", [
    {"grid": (3, 2), "cell_size": 31},
    {"grid": (2, 3), "cell_size": 31},
])
def test_explicit_odd_output_dimensions_are_rejected(kwargs):
    with pytest.raises(ValueError, match="even output dimensions"):
        derive(640, 360, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"grid": (0, 2)}, {"grid": (2, -1)}, {"grid": (True, 2)},
    {"grid": [2, 2]}, {"cell_size": 0}, {"cell_size": True},
    {"grid_size": 0}, {"grid_size": True},
])
def test_sizing_inputs_must_be_positive_integers(kwargs):
    with pytest.raises(ValueError):
        UserConfig(input_dir="", output_dir="", **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"contrast": 0}, {"contrast": 1.1}, {"contrast": float("nan")},
    {"contrast": True},
    {"candidates": 0}, {"candidates": True}, {"epsilon": -0.1},
    {"epsilon": float("inf")}, {"colour_bins": 1}, {"colour_bins": 65},
    {"epsilon": True},
    {"colour_bins": False}, {"seed": -1}, {"seed": True},
    {"gallery_budget": 0}, {"gallery_budget": True},
    {"segment_frames": -1}, {"segment_frames": True},
    {"tile_fit": "bad"}, {"metric": "bad"},
    {"hold_tiles": 1}, {"use_cache": 1},
    {"start": True}, {"duration": True},
])
def test_config_rejects_invalid_algorithm_settings(kwargs):
    with pytest.raises(ValueError):
        UserConfig(input_dir="", output_dir="", **kwargs)
