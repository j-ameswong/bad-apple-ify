# The tile cache

`load_gallery()` wraps `GallerySource.load()` in an on-disk cache of `.npy` tile
arrays under `.cache/gallery/`. Pass `use_cache=False` to bypass it.

## Why cache at all

A gallery load costs one decode pass over the whole source, roughly 12 minutes
for a season of anime. That's unusable to sit through on every run, and the
result is small enough to keep around: 41k tiles at 16x16 is 32 MB.

## The key

`cache_key()` is a sha256 of the source's fingerprint plus the cell size,
truncated to 16 hex characters. It's a plain digest rather than a structured
filename so a metric's own precompute could join the key later without changing
the layout. For `BrightnessMetric` that would mean caching a 10 ms computation
behind a 38 MB read, so only the tiles are cached today.

## Reads and writes

A cache hit skips the [size estimate](gallery-size.md) — the file on disk is the
real answer, so there's nothing left to guess about.

A cache hit still checks the array's shape against the expected cell size. A
truncated or hand-edited file isn't worth trusting over a re-decode, and the key
already guarantees the tiles are otherwise current.

Writes go to a temp file named with the pid and then `replace()` onto the real
path, so a run killed mid-write leaves the old cache intact rather than a half
file the next run would have to detect. The temp name keeps its `.npy`
extension, which `np.save` would otherwise append itself.
