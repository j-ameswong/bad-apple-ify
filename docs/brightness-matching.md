# Brightness matching

`BrightnessMetric` matches each grid cell to a gallery image of near-identical
average brightness. A cell's brightness rounds to one of 256 levels, so every
possible match can be resolved once, at precompute time.

## Buckets, not argmins

Pinning each level to its single best image collapses a 40k gallery down to
~100 distinct tiles, and the mosaic looks visibly repetitive. Instead each level
gets a bucket of the `candidates` images nearest it in brightness, and `match()`
samples one uniformly. Matching stays O(1) per cell and the whole gallery gets
used.

`candidates` is the variety knob, and it means the same thing whatever the
gallery. A fixed brightness radius would over-sample a dense gallery and starve
a sparse one.

`epsilon` is the accuracy ceiling: no candidate may differ from its level by
more than that, so a sparse gallery yields a smaller bucket rather than a
tonally wrong one. A level with nothing inside epsilon (including `epsilon=0`,
or `candidates=1`) falls back to the single nearest image, which is the argmin
answer.

## How a bucket is built

Sort the gallery by brightness. The k nearest images to a level are then
contiguous, so a bucket is just an offset and a count.

`searchsorted` gives the insertion point, and the window starts centred there.
But brightness isn't uniformly spread, so the midpoint of the window isn't the
midpoint of the gallery around it: the two `while` loops slide the window until
it holds the true k nearest. Then it's clipped to the epsilon range, with the
nearest-neighbour fallback if that leaves it empty.

Finally the gallery is compacted to the images some bucket can actually reach,
via a difference array over the `[lo, lo + count)` spans. `_remap` translates a
sorted-gallery index into an index in the compacted `tiles` array, so
`match()` returns indices straight into `metric.tiles`.

## Brightness itself

`gallery_brightness()` uses Rec.601 luma weights in BGR order, the same ones
`cv2.COLOR_BGR2GRAY` applies. Luma is linear, so the mean of the luma equals the
luma of the channel means: one `(N, 3)` reduction instead of a per-image
`cvtColor`.

## Contrast

`shrink_gallery()` trims the gallery to a percentile band of brightness around
the midpoint, width set by `config.contrast`. At `1.0` the whole gallery
survives; lower values cut the dark and bright extremes for "cooler" output. It
takes the brightnesses as an argument and returns the survivors alongside the
images, so the caller never needs a second pass.

## Precompute rejects a misshapen gallery

Tiles arrive at cell size from their `GallerySource`, so `precompute()` resizes
nothing. It does check, because a mismatch would otherwise surface as a
silently misshapen mosaic much further down the pipeline.
