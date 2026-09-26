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
is the real answer. It doesn't skip the budget. `read_cached_tiles()` takes the
count from the `.npy` header and prices it exactly like an estimate before a
byte of tile data is read. A cache written under a bigger `gallery_budget`
raises `GalleryTooLarge` rather than re-decoding, because the decode would land
on the same count.

Writes go to a temp file named with the pid and then `replace()` onto the real
path, so a run killed mid-write leaves the old cache intact rather than a half
file the next run would have to detect. The temp name keeps its `.npy`
extension, which `np.save` would otherwise append itself.

## Malformed files

Anything short of a whole C-order uint8 `(N, cell_h, cell_w, 3)` array is a miss
and falls back to a re-decode: a truncated write, a hand-edited array, the wrong
dtype, a header that won't parse. The key already guarantees the tiles are
otherwise current, so there's nothing in a bad file worth salvaging.

The count has to square with the file size, exactly as many bytes behind the
header as the shape needs, before it gets priced. Price it first and a flipped
bit in the count comes back as a refusal telling you to raise `gallery_budget`,
when all the file needed was a re-decode. Exact rather than at-least catches a
count flipped downwards too, which would otherwise load a prefix of the tiles
and drop the rest without a word.

It all runs off one open handle, so the file that got vetted is the file that
gets read. The data goes straight into a preallocated array with `readinto()`,
and a read that still comes up short is a miss as well. That takes something
truncating the file in place after the size check, `cp` over it for one.
`readinto()` only fills a C-order array, which is why Fortran order is refused
rather than handled. `np.save` only writes it for a Fortran-ordered array, and
no source hands one over.

numpy's header parser means to raise `ValueError` on a bad header, and mostly
does. Flip each of the 1024 bits in a real 50000-tile header one at a time,
though, and 274 of them raise `tokenize.TokenError` instead. Most of those are
flipped spaces, which numpy pads the header with. A space is one bit from a null
byte, a double quote and an open bracket, any of which fails `ast.literal_eval`,
and numpy's fallback (a tokenizer-based filter for headers Python 2 wrote)
throws its own error. Stray bytes find two more. A comma in the dtype string
gets a `SyntaxError` out of `np.dtype`, and a `b'shape'` key gets a `TypeError`
out of the `sorted()` in numpy's own error message.

Those four are caught around the header read and nothing else is. `OSError`
goes up, since a permissions problem or a dying disk isn't a malformed cache and
a quiet re-decode would only hide it. So does the `MemoryError` the parser
throws on a header of 9000 minus signs, which no disk fault is going to write.

Bit rot in the tile data itself still loads. `.npy` has no checksum, and a
flipped bit there costs one wrong pixel, not a crash.
