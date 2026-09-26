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

## The frame rate stays a fraction

NTSC-family video runs at n x 1000/1001: 23.976, 29.97, 59.94. Rounding 29.97
to 30 gives a mosaic that's 0.1% fast. Its N frames play in N/30 seconds
against the source's N/29.97, so it slides a frame every 33 seconds against the
source picture and its audio, and a 24-minute episode ends 43 frames adrift.

`probe_video()` keeps the rate as a `Fraction`, and `encode_video()` hands
ffmpeg `str(fps)`, which is `30000/1001` and parses as exactly that. OpenCV only
exposes the rate as a double, but `limit_denominator(1001)` snaps a double of
n/1001 straight back to it and leaves 25 or 2997/100 alone.

The container can still be the limit. mkv and webm keep millisecond
timestamps, which can't hold 59.94 exactly, so libavformat guesses 19001/317
from them. ffprobe reports the same, so there's no truer number to go and get,
and it's 5 parts in 100 million off, a frame in 88 hours. mp4's timescale keeps
the rate exact, which is why the tests build their NTSC sources as mp4, with
ffmpeg rather than `cv2.VideoWriter`, which rounds 30000/1001 to 2997/100 on
the way in.

## Stacking side by side

`hstack` requires both inputs to be the same height, and the mosaic is only
incidentally the source's size: target dimensions are snapped to a grid multiple
(see [grid and sizing](grid-and-sizing.md)). So `combine_videos()` scales the
source to the mosaic's dimensions rather than relying on the coincidence.
