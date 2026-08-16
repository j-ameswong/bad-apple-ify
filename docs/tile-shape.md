# Tile shape

CIFAR is square and the grid follows the source's own aspect ratio, so cells
came out square and nothing ever had to be reshaped. A video gallery breaks
that: 16:9 frames into a square cell have to lose something.

`tile_fit` picks what.

## `native` (default)

The cell takes the tiles' ratio, so nothing is cropped or squashed — every tile
is a whole frame of the source video, whole. This is the only mode where the
mosaic is made of complete images.

The grid gives way instead. Rows stay where the source put them, the cell height
with them; the cell width comes from the tiles' ratio, and the number of columns
is however many then fit across. Bad Apple at `grid_size=8` against 16:9 tiles
goes from a 32x24 grid of 16x16 cells to 18x24 of 28x16 — 432 tiles instead of
768, each one nearly twice as wide.

Two roundings collide there, and only one of them can win:

- the cell has to be a whole number of pixels, so it can't hold the tiles' ratio
  exactly
- the target frame is grid times cell, so it can't hold the source's ratio
  exactly either

The source's shape wins. The cell width is the ideal one rounded, then `grid_x`
is however many of those fit across — the tiles absorb the leftover. On Bad
Apple that's a 28x16 cell (1.75 against 16:9's 1.778, 1.6% off) and a 504x384
frame beside a 512x384 source. A per-tile percent is invisible; a stretched
frame next to the original in the side-by-side is not.

"Across" means across the *snapped* width, `grid_x_base * cell_w_base`, not the
raw source width. Once the rows have rounded, the two no longer share a scale.
Measuring against the raw width put 1080p at `grid_size=16` at 1.64:1 against a
1.78:1 source, an 8% vertical squash — invisible at 512x384, where 384/24
divides exactly and there's nothing to round.

The source's `native_aspect` is what the cell is shaped to, and it's cheap
metadata — one video header — so asking for it before probing costs nothing.
A source that can't say (`None`) falls back to the square-ish grid.

## `crop`

Keep the grid as it was and take the largest centred rectangle of each gallery
image at the cell's ratio. Undistorted, but 16:9 into 1:1 throws away 44% of the
frame's width, and what it throws away is where animation puts a lot of its
composition.

## `stretch`

Keep the grid and squash the image into the cell. This is what the pipeline did
before 2.1.

With square tiles the three modes usually agree, but not by construction —
`native` reshapes the cell whatever the tiles are, and a 1:1 tile forces a
square cell the base grid may not have had. They coincide wherever the source's
own cells came out square, which 512x384 at `grid_size=8` does (16x16) and a
2.39:1 source at the same setting also does (2x2). A source that rounds to a
non-square cell will shift dimensions slightly under the new default.

## Single-frame mode

At `grid_size=1` the cell is the whole source frame, so it has the source's
ratio and there's nothing to reshape — `native` can't apply. The tile is fitted
into that cell by cropping, which is the better of the two things left.
