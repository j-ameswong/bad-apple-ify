# Gallery sources

`GallerySource` is the protocol every tile source implements: `load(cell_size)`
returns `(N, cell_h, cell_w, 3)` BGR tiles, plus a `fingerprint` string.

## Why `load()` takes the cell size

This is the point of the protocol, not a convenience. A video gallery is only
tractable because frames are downscaled *during* decode. 41k frames of anime at
1080p is 257 GB; the same frames at 16x16 are 32 MB. A protocol that promised
full-resolution images could not be implemented for video at all, so it never
promises them.

The cost is a load-order constraint: the video has to be probed and the cell
size derived before any gallery can load. That's why `main()` calls
`probe_video()` first.

## The fingerprint

`fingerprint` identifies what `load()` will return, ignoring cell size:
everything that changes the tiles (the files read, their mtimes, any sampling
parameters) and nothing that doesn't. The [tile cache](gallery-cache.md) keys on
it, so a source that under-reports here will happily serve stale tiles.

## Implementations

`CifarGallery` wraps a CIFAR-100 pickle batch: 32x32 images, resized to cell
size at load time.

Two quirks in `read_cifar_batch()`. The pickle carries a dtype serialised by an
ancient NumPy with `align=0`, which NumPy 2.4 deprecates in its int form;
nothing to fix from this side short of re-serialising the file, so the warning
is suppressed. And the RGB→BGR flip produces a reverse-strided view, which cv2
refuses later, hence the `ascontiguousarray`.

`VideoGallery` decodes a video (or a directory of them) keeping every
`stride`-th frame. Stubbed until plan phase 2.1.
