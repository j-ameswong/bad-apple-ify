# The tile cache

`load_gallery()` wraps `GallerySource.load()` in an on-disk cache of `.npy` tile
arrays under `.cache/gallery/`. Pass `use_cache=False` to bypass it.

## Why cache at all

A gallery load costs one decode pass over the whole source, roughly 12 minutes
for a season of anime. That's unusable to sit through on every run, and the
result is small enough to keep around: 41k tiles at 16x16 is 32 MB.

## The key

`cache_key()` is a sha256 of the source's fingerprint plus the cell size and
the [fit](tile-shape.md),
truncated to 16 hex characters. It's a plain digest rather than a structured
filename so a metric's own precompute could join the key later without changing
the layout. For `BrightnessMetric` that would mean caching a 10 ms computation
behind a 38 MB read, so only the tiles are cached today.

`TILE_VERSION` rides along in the key for the things it can't otherwise see.
The fingerprint covers what went in and the cell size and fit cover the shape,
but neither says anything about *how* the resize was done, so changing the
interpolation (v1 to v2, bilinear to INTER_AREA) would have served the old
pixels forever. Bump it whenever the same inputs start producing different
tiles.

## Reads and writes

A cache hit skips the [size estimate](gallery-size.md), since the file on disk
is the real answer. It doesn't skip the budget. `cached_tile_count()` reads the
`.npy` header and stats the file, and the count it gets back is priced exactly
like an estimate before `np.load` touches the data. A cache written under a
bigger `gallery_budget` raises `GalleryTooLarge` rather than re-decoding, because
the decode would land on the same count.

The same header read vets the file: uint8, `(N, cell_h, cell_w, 3)`, and exactly
as many bytes behind the header as that shape needs. Anything else (a truncated
write, a hand-edited array, the wrong dtype) falls back to a re-decode. The key
already guarantees the tiles are otherwise current.

Writes go to a temp file named with the pid and then `replace()` onto the real
path, so a run killed mid-write leaves the old cache intact rather than a half
file the next run would have to detect. The temp name keeps its `.npy`
extension, which `np.save` would otherwise append itself.
