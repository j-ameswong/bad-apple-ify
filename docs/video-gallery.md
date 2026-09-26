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

`load()` hands `TileBuffer` the `estimate_count()` (or `FALLBACK_CAPACITY` when
there isn't one), and the buffer allocates that many tiles up front, fills them
in place and doubles if the estimate was low. Each frame is written straight
into the next free slot and only kept if it turns out to be new, so there is no
list of arrays to stack afterwards. The one exception is a full buffer, which
hands out a one-tile scratch instead (see below).

Every backing-buffer allocation, first and growth alike, goes through
`TileBuffer._allocate()`, which prices it against `gallery_budget` and caps it
at the most tiles the budget allows. So a lying container can't double its way
to an OOM, and a doubling that would overshoot grows only as far as the budget
instead of refusing a gallery that fits. It raises only when the next tile
genuinely won't fit. The fallback guess used to be allocated unpriced, which at
single-frame cell sizes on a 4K source was 25 GB of address space. The scratch
tile and the trimming copy at the end sit outside that check.

Growth waits for `keep()`, not `next_slot()`. That's what the scratch tile is
for: a frame gets fitted and hashed before it asks for room, so a duplicate at
the ceiling is dropped rather than refused.

The buffer is copied down to size at the end when more than a tenth of it is
unused. A bare slice would be free but would pin the whole allocation for the
rest of the run, and dedupe can leave a lot of it empty.

`CAP_PROP_FRAME_COUNT` read 42,590 against a real 41,647 on that episode — 2.3%
high, so the estimate stays an upper bound, which is what the
[budget check](gallery-size.md) needs it to be.

`estimate_count()` is memoised. A load asks for it three times (the budget
check, the buffer's first allocation, the progress bar's total), and every ask
opens each file in the directory. A directory that changes mid-run gets a stale
count, which costs a resize at worst.

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
