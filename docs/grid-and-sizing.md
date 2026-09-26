# Grid and sizing

`DerivedConfig.from_source()` turns a source video's dimensions into a grid, a
cell size, and a target frame size. Four decisions live in there.

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

## Native tiles reshape the grid

`tile_fit="native"` hands `from_source()` the gallery's own aspect ratio, and
the cell is shaped to that instead of coming out square. Rows are untouched; the
columns are recounted around the new cell width. The full argument, including
which of the two roundings gives way, is in [tile shape](tile-shape.md).

## Target dimensions snap to the grid

The cell size is the source dimension divided by the grid, rounded, floored at
1px per cell however fine the grid is. `target_dimensions` is then grid times
cell, which is usually a pixel or two off the source. That's why
`combine_videos()` rescales the source before stacking (see
[streaming and encoding](streaming-and-encoding.md)).

## Both sides come out even

`encode_video()` asks libx264 for `yuv420p`, which halves the chroma both ways
and so won't open on an odd width or height. Grid times cell is odd whenever
both are, and that's not rare: 1080p at `grid_size=8` against 16:9 tiles was 71
columns of 27px, 1917 wide, and 720p at `grid_size=3` was 27 rows of 27px.

`even_span()` fixes a side by moving the count one step, never the cell, so a
native tile keeps its shape. It takes whichever step lands nearer the ideal:
1944 over 1890 for that 1080p case, since 1920 is 24 away from one and 30 from
the other. Single-frame mode is the exception. Its 1x1 grid can't move, so the
one cell gives up a pixel instead (853x480 becomes 852x480).

Rows go first, against the source height. A row step moves the height a whole
cell, so the columns are then recounted against the width the new height
implies, and only after that made even. Checking their parity alone isn't
enough: that 720p case drops to 26 rows, 702 high, and its 48 columns were
already even at 1296, 3.85% too wide for 16:9. Recounted, it's 46 columns and
1242, 0.5% off. The row count is the aspect pair's height times `grid_size`, so
it's only ever odd at an odd `grid_size`, which is why 8 and 16 never showed
this.

A step is a whole cell, so on a coarse grid the fix costs shape. At 3 columns a
step is a third of the frame. At the grid sizes worth running it's a few
percent at most.
