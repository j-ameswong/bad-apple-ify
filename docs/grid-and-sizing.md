# Grid and sizing

`DerivedConfig.from_source()` turns a source video's dimensions into a grid, a
cell size, and a target frame size. Three decisions live in there.

## The aspect ratio comes from the source

The grid needs *some* integer pair to multiply by `grid_size`. Rather than
snapping the source to an allowlist of ratios (16:9, 4:3, ...), we take the
simplest integer pair close to its own ratio:

```python
ratio = Fraction(*dimensions).limit_denominator(16)
```

A 2.39:1 film stays 2.39:1 instead of being stretched to 16:9. The denominator
cap keeps the pair usable as a multiplier: an exact reduction of 2048x858 is
1024:429, which multiplied by `grid_size=8` would ask for eight thousand cells
across.

## `grid_size=1` is single-frame mode

At `grid_size=1` the grid collapses to 1x1, so each output frame is one whole
gallery image rather than a composite. There's no separate code path for it,
only a special case in the grid calculation: multiplying the aspect pair by 1
would give a 4x3 grid of 12 tiles, which is neither a mosaic worth the name nor
the single frame asked for.

The cell then takes the source's own aspect ratio instead of the pair's, which
is what a full-frame tile should do.

Watch the memory. Tiles are stored at cell size, so a full-frame cell means the
whole gallery is held at output resolution. CIFAR at 512x384 is about 29 GB.
Use a small gallery in this mode.

## Target dimensions snap to the grid

The cell size is the source dimension divided by the grid, rounded, floored at
1px per cell however fine the grid is. `target_dimensions` is then grid times
cell, which is usually a pixel or two off the source. That's why
`combine_videos()` rescales the source before stacking (see
[streaming and encoding](streaming-and-encoding.md)).
