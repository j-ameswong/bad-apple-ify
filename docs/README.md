# Docs

Design notes for the bits of `main.py` where the *why* doesn't fit in a comment.

- [Grid and sizing](grid-and-sizing.md) — aspect ratio derivation, cell size, single-frame mode
- [Gallery sources](gallery-sources.md) — the `GallerySource` protocol and why `load()` takes a cell size
- [The tile cache](gallery-cache.md) — what's cached, how it's keyed, how it's written
- [Brightness matching](brightness-matching.md) — buckets, `candidates`, `epsilon`, contrast
- [Streaming and encoding](streaming-and-encoding.md) — lazy frames, the ffmpeg pipe, side-by-side output
