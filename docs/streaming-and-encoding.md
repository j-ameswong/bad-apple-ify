# Streaming and encoding

## Everything stays lazy

`stream_frames()` yields one decoded frame at a time and `build_mosaics()` maps
`mosaic_frame()` over it, so peak memory holds a couple of frames however long
the source is. Holding a 130k-frame film at 512x384 would be 76 GB, so this
isn't a tidiness thing.

`DerivedConfig.src_frame_count` comes from container metadata and may be wrong
or absent. It's only ever used as a tqdm total hint; `output_frame_count`
adjusts that hint for the requested slice.

## Processing a slice

`uv run main.py --start 60 --duration 10` processes source seconds 60–70.
Both flags accept fractional seconds. `start` defaults to zero; leaving out
`duration` runs to EOF. The same fields are available on `UserConfig`.
Negative or non-finite starts, and non-positive or non-finite durations, are
rejected before opening the source or gallery.

Frames are selected by their timestamps in `[start, start + duration)`. The
first index is `ceil(start × fps)` and the exclusive end is
`ceil((start + duration) × fps)`, calculated with fractions. At 30 fps, 60–70
seconds means exactly frames 1800–2099. At fractional rates or between frame
boundaries, the slice is rounded to whole frames and the audio follows those
boundaries. EOF can shorten a slice; source frame-count metadata never limits
the decode. An empty range raises a clear error before ffmpeg starts.

Decoding counts frames sequentially, so an inaccurate container index cannot
seek to the wrong picture. With multiple candidates, the prefix also goes
through `metric.match()` to replay its random draws and held tiles. Those
prefix frames are neither assembled nor encoded. This makes even a seeded,
stochastic slice byte-identical to the corresponding **raw mosaics** of a full
run. It does cost a decode and match of the prefix; `candidates=1` only grabs
the skipped frames and avoids resizing and matching them. Memory stays bounded.

The source pane is trimmed by the same frame indices. Both panes are put on
the mosaic's frame clock before stacking, avoiding extra frames from container
timestamp rounding. Audio is explicitly taken from the source's first audio
track, trimmed to the same start and the completed mosaic's actual duration,
and rebased to zero. Silent sources work too. This also keeps audio short when
the requested duration extends past EOF.

Only these two flags are exposed so far. The remaining CLI and TOML config
layering are still PLAN.md 2.8.

## The encode pipe

`encode_video()` writes raw `bgr24` frames into an ffmpeg stdin pipe, so no
intermediate PNGs ever hit disk. If ffmpeg dies early, the write or pipe close
can raise `BrokenPipeError`; ffmpeg is still waited for so its exit status is
collected. Both encode and combine write to a temporary file beside the final
path, then replace the final file only after ffmpeg succeeds. A failed encode or
an interrupted mosaic iterator leaves any previous output intact. On POSIX the
parent directory is synced after the rename; Windows doesn't allow this
directory sync.

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
