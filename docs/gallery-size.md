# Sizing the gallery before loading it

The tile array is `N x cell_h x cell_w x 3` bytes, held in RAM for the whole
run. Both factors are the user's to set and they multiply, so the same code
path spans five orders of magnitude:

| gallery | cell | tiles |
|---|---|---|
| CIFAR 50k | 16x16 | 37 MB |
| CIFAR 50k | 64x64 | 586 MB |
| CIFAR 50k | 512x384 (`grid_size=1`) | **27.8 GB** |
| Season, 41k tiles @ stride 10 | 16x16 | 30 MB |
| Season, 41k tiles @ stride 1 | 32x32 | **1.2 GB**, plus a ~2 hr decode |

Nothing tells you which side of that you're on until the allocation fails, and
on a video gallery it fails *after* the decode pass. Twelve minutes in is the
worst possible moment to learn the parameters were wrong, so `load_gallery()`
prices the array before calling `source.load()`.

## Estimating the count

The cell size is already known, so the tile count is the only unknown, and every
source can approximate it far more cheaply than a full load.

`CifarGallery` divides the file size by 3072, the on-disk size of one 32x32x3
planar image. The pickle also carries labels and filenames, so it reads a little
high: 50537 against a real 50000 on the CIFAR-100 train batch, 1.1% over. High
is the right direction to be wrong in.

`VideoGallery` sums `CAP_PROP_FRAME_COUNT / stride` over its files. That
metadata is container-level and can lie, which is why it never sizes an
allocation, only a warning — though it came back exact on all three rips here.
Where it's missing, seeking to the end and reading `POS_MSEC x fps` is the
backup. The held-cel dedupe in plan 2.1 only ever drops tiles, so the number
stays an upper bound.

`None` means "can't tell", which is not the same as "small". It's reported as
unknown rather than waved through as zero.

## Acting on it

Under 1 GB it's an ordinary line alongside the tile count and cell size already
logged. Over that, a warning with the arithmetic spelled out, because a bare
number doesn't tell you what to change. Over 8 GB (`gallery_budget`) it raises
`GalleryTooLarge` and nothing is decoded.

The refusal is the unusual choice, and it's deliberate: this is the one place
where trying anyway costs more than stopping. An OOM twelve minutes into a
decode destroys more work than a message does, and the override is one config
field away for anyone who really does have the RAM.

A cache hit skips the estimate entirely. The `.npy` on disk is the true answer,
and it already fit once.
