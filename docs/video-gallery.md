# Video galleries

`VideoGallery(path, stride)` decodes a video, or a whole directory of them, and
keeps every `stride`-th frame as a tile. A season is one argument rather than
twelve runs.

## Decoding straight through

No seeking. Frame-accurate seek on long-GOP video has to decode from the
previous keyframe anyway, so skipping to every tenth frame costs more than
reading them all. What the stride does buy is the colour conversion: `grab()`
decodes a frame without handing it over, `retrieve()` is what converts it to a
BGR array, so nine frames in ten never become numpy at all.

Measured on one Lucky Star episode (1080p HEVC-10bit, this machine): 41,647
frames in 73s, about 570 fps. A 27-file season is half an hour, once, behind the
[tile cache](gallery-cache.md).

Each kept frame is shrunk to cell size before it is stored — see
[gallery sources](gallery-sources.md) for why the protocol is shaped that way.

## The tile buffer

`load()` allocates `estimate_count()` tiles up front and fills them in place,
doubling if the estimate was low. Each frame is written straight into the next
free slot and only kept if it turns out to be new, so there is no scratch tile
and no list of arrays to stack afterwards.

The buffer is copied down to size at the end when more than a tenth of it is
unused. A bare slice would be free but would pin the whole allocation for the
rest of the run, and dedupe can leave a lot of it empty.

`CAP_PROP_FRAME_COUNT` read 42,590 against a real 41,647 on that episode — 2.3%
high, so the estimate stays an upper bound, which is what the
[budget check](gallery-size.md) needs it to be.

## Dedupe

Tiles are hashed after downscale (blake2b, 8 bytes) and duplicates dropped. This
is aimed at animation holding a cel for several ticks, and at the black frames
every fade and episode boundary contributes.

It earns much less than you'd hope at a stride of 10: 6 duplicates out of 4,165
tiles on that episode. Held cels last two or three frames, so a stride of 10
almost never lands on the same one twice — the dedupe is worth having at small
strides and for fades, not as a way to halve a sampled gallery.

## Directories

A directory gallery takes the video-suffixed files directly inside it, sorted,
skipping dotfiles. That last bit is for the AppleDouble `._episode.mkv` stubs a
rip made on a Mac leaves next to the real files: right suffix, 4 KB of resource
fork, and OpenCV will not open them.
