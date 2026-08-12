# Streaming and encoding

## Everything stays lazy

`stream_frames()` yields one decoded frame at a time and `build_mosaics()` maps
`mosaic_frame()` over it, so peak memory holds a couple of frames however long
the source is. Holding a 130k-frame film at 512x384 would be 76 GB, so this
isn't a tidiness thing.

`DerivedConfig.src_frame_count` comes from container metadata and may be wrong
or absent. It's only ever used as a tqdm total hint.

## The encode pipe

`encode_video()` writes raw `bgr24` frames into an ffmpeg stdin pipe, so no
intermediate PNGs ever hit disk. If ffmpeg dies early the writes raise
`BrokenPipeError`; that's swallowed because ffmpeg's own exit code is the useful
error, not the write failure it caused.

## Stacking side by side

`hstack` requires both inputs to be the same height, and the mosaic is only
incidentally the source's size: target dimensions are snapped to a grid multiple
(see [grid and sizing](grid-and-sizing.md)). So `combine_videos()` scales the
source to the mosaic's dimensions rather than relying on the coincidence.
