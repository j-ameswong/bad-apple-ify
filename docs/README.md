# Docs

- [Python audit](audit.md) — confirmed fixes, performance decisions and validation limits

Guides and design notes for using and changing the Python pipeline.

- [CLI and TOML configuration](cli.md) — flags, defaults, path rules and examples
- [Segmented encoding and resume](resume.md) — checkpoints, validation and reruns
- [Grid and sizing](grid-and-sizing.md) — aspect ratio derivation, cell size, single-frame mode
- [Gallery sources](gallery-sources.md) — the `GallerySource` protocol, CIFAR and video sources
- [Video galleries](video-gallery.md) — decoding a season into tiles: stride, dedupe, the buffer
- [Tile shape](tile-shape.md) — `tile_fit`, how native-ratio tiles reshape the grid, and resampling
- [The tile cache](gallery-cache.md) — what's cached, how it's keyed, how it's written
- [Sizing the gallery](gallery-size.md) — estimating the tile array, and why it refuses
- [Brightness matching](brightness-matching.md) — buckets, `candidates`, `epsilon`, contrast
- [Matching metrics](colour-matching.md) — the `Metric` protocol, and colour on a 3D lattice
- [Streaming and encoding](streaming-and-encoding.md) — lazy frames, the ffmpeg pipe, side-by-side output
