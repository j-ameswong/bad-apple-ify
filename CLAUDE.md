# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Takes any video and reconstructs each frame as a photo mosaic where every tile is a real image from a gallery (CIFAR-100) matched by brightness. Produces a side-by-side video of the original and mosaic version via ffmpeg.

## Running

```bash
uv sync          # install dependencies
uv run main.py   # run the pipeline
```

Requires `ffmpeg` on PATH and the CIFAR-100 `train` pickle file at `./assets/gallery/train`.

## Testing

```bash
uv run pytest
```

Runs on a clean checkout with no assets present — fixtures in `tests/conftest.py`
generate a synthetic FFV1/mkv video (lossless, so tests can assert on exact
pixels) and a 100-image fake gallery, plus a CIFAR-format pickle built from it.
Neither `assets/source.mp4` nor the 155 MB CIFAR pickle is needed.

## Type checking

```bash
uv run mypy
```

Config is in `pyproject.toml`: `strict = true` over `main.py` only, so the bare
command is the check. Arrays go through the `Image` / `Brightness` / `Indices`
aliases at the top of `main.py` rather than `np.ndarray`, which under strict is
`ndarray[Any, dtype[Any]]` and tells you nothing. The tests aren't checked.

## Configuration

User-supplied parameters live in the frozen `UserConfig` dataclass at the top of
`main.py`, and are constructed in the `__main__` block. Edit that to change them:

- `input_dir` / `output_dir` — source video and output folder paths (the gallery
  path is not a config field — it belongs to the `GallerySource` passed to `main()`)
- `grid_size` — multiplier for aspect ratio (higher = more tiles = finer detail,
  slower). `grid_size=1` is single-frame mode: the grid collapses to 1×1 and each
  source frame is replaced by one whole gallery image. No separate code path —
  the cell simply becomes the frame. Note the memory cost: tiles are stored at
  cell size, so a full-frame cell means the whole gallery is held at output
  resolution (CIFAR at 512×384 is ~29 GB). Use a small gallery for this mode.
- `contrast` — fraction of gallery brightness range to use (0–1); lower trims extremes for "cooler" results
- `candidates` — how many tiles each cell picks between; the variety knob, and gallery-independent (`1` pins every cell to its single closest tile)
- `epsilon` — accuracy ceiling: max brightness error (0–1) a candidate may have. Caps `candidates` on a sparse gallery rather than letting it reach for tonally wrong tiles
- `seed` — RNG seed for the sampling; fixed seed = reproducible output
- `gallery_budget` — bytes of tiles to refuse past (default 8 GB). `load_gallery()`
  estimates `N × cell_h × cell_w × 3` before decoding anything and raises
  `GalleryTooLarge` over this; over 1 GB it warns instead. Raise it if you really
  do have the RAM. See `docs/gallery-size.md`

Everything derived from the source video — FPS, source/target dimensions, aspect
ratio, grid and cell size — lives in `DerivedConfig`, built once by
`probe_video()` and immutable thereafter.

## Architecture

Design rationale that doesn't fit in a comment lives in `docs/` (see
`docs/README.md` for the index). Keep the comments in `main.py` short and put
the *why* there instead.

The entire pipeline is single-file (`main.py`):

1. **`probe_video()`** — reads source metadata (FPS, dimensions) and returns a `DerivedConfig`, which derives an integer aspect pair from the source's own ratio (`Fraction(...).limit_denominator(16)`) and snaps target dimensions to the nearest grid multiple (`grid_size=1` is the exception: the grid is 1×1 and the cell is the whole frame); **`stream_frames()`** then yields frames one at a time (never the whole video in RAM). Runs **first**, because the gallery cannot be loaded until the cell size is known
2. **`GallerySource`** — protocol for tile sources: `load(cell_size)` returns `(N, cell_h, cell_w, 3)` BGR tiles **already at cell size**, `fingerprint` identifies what `load()` will return (files read, their mtimes, sampling parameters), and `estimate_count()` guesses the tile count cheaply (`None` if it can't). Full-resolution gallery frames are never stored (a season of anime at 1080p is 257 GB; at 16×16 it is 32 MB), so the protocol never promises them. `CifarGallery(path)` wraps the CIFAR-100 pickle (`read_cifar_batch()` + `resize_gallery_to_cells()`); `VideoGallery(path, stride)` is stubbed until phase 2.1
3. **`load_gallery()`** — calls `source.load()` through an on-disk cache of `.npy` tile arrays under `.cache/gallery/`, keyed by `cache_key()` = sha256 of `(fingerprint, cell size)`. A hit skips the decode entirely (and the size estimate — it knows the real answer); writes go via a temp file and rename. `use_cache=False` bypasses the cache but not `check_gallery_budget()`. Metric precompute is *not* cached — it is ~10 ms against a 38 MB read
4. **`gallery_brightness()`** — precomputes per-tile brightness scalars (0–1)
5. **`shrink_gallery()`** — filters gallery to a percentile band around 50% brightness, controlled by `config.contrast`
6. **`BrightnessMetric`** — `precompute()` buckets the gallery by brightness into a 256-level table (one bucket per possible cell level, holding the `candidates` nearest images, none further than `epsilon`) and keeps only the reachable tiles; it resizes nothing, and rejects a gallery that is not already at cell size. `match()` turns a frame into a grid of tile indices by sampling uniformly from each cell's bucket
7. **`mosaic_frame()`** — assembles the matched tiles into a single mosaic frame; **`build_mosaics()`** maps it lazily over the frame stream (one frame in, one mosaic out, so peak memory never scales with video length)
8. **`build_metric()`** — steps 4–6 as one stage: brightness, shrink, `BrightnessMetric.precompute()`
9. **`encode_video()`** — owns the ffmpeg pipe, writing raw mosaic frames to its stdin; **`combine_videos()`** runs ffmpeg again for the side-by-side output, scaling the source to the mosaic's size (`hstack` demands equal heights and the two only match by coincidence)
10. **`main(gallery_source, config)`** — pure orchestration, no logic of its own: probe → load → build metric → stream → mosaic → encode → combine. It takes a `GallerySource` and never touches CIFAR-specific code; the `UserConfig` is built at the call site in `__main__`

## Dependencies

Managed with `uv` (see `pyproject.toml`). Runtime deps: `opencv-python`, `numpy` (via opencv), `tqdm`. External: `ffmpeg`.

## Voice

I like to write comments how I would talk to a coworker over coffee about it. I use British English and use a more casual tone.

Be direct. Have opinions. Use specific examples and names, not vague claims. State your point first, then support it. Trust the reader to recognise what matters without labelling it as "significant" or "important."

## Banned words

Never use these — they are the most flagged AI-writing markers:

delve, dive into, navigate (figurative), underscore, bolster, foster, harness, leverage, unpack, shed light on, pave the way, pivotal, groundbreaking, cutting-edge, transformative, game-changing, innovative, robust, comprehensive, seamless, intricate, nuanced (as empty praise), vibrant, multifaceted, holistic, testament, landscape (figurative), realm

Never use these phrases:

- "In today's [fast-paced/rapidly evolving/digital] world..."
- "It's important/worth noting that..."
- "One of the most [important/significant/crucial]..."
- "When it comes to..." / "At its core..." / "At the end of the day..."
- "This is where X comes in" / "Let's break it down"
- "Plays a crucial role in..." / "It cannot be overstated..."
- "...underscoring the importance of..." / "...highlighting the need for..."
- "...reflecting a broader trend toward..." / "...marking a significant shift in..."

Never use these structures:

- "It's not just X — it's Y"
- "Not only X, but Y"
- "This isn't about X. It's about Y."
- "No X. No Y. Just Z."

These mimic insight without providing any.

## Structure

- Vary paragraph and sentence length. Don't write uniform blocks.
- Never use the "Bold term: explanation sentence" list format. It's the single most recognisable AI pattern.
- Don't signpost ("Let's explore," "Now let's turn to"). Just make your point.
- Don't open with a sweeping contextual statement. Don't close with a summary or inspirational wrap-up. Start and end on substance.
- Don't restate the question back before answering it.

## Style

- Use contractions. "It's," "don't," "won't."
- Maximum one em dash per response. Use commas or parentheses instead.
- Don't over-format. Plain prose is often clearer than headers and bullet points.
- Drop preamble ("Great question!"), performative enthusiasm ("exciting," "incredible," "powerful"), and unsolicited caveats.
- Match tone to context. Casual question, casual answer.

## Before finishing, check:

1. Read it out loud. Does any sentence sound like a press release? Rewrite it.
2. Are you repeating the same point in different words? Say it once.
3. Does your opening sentence set the scene with a grand statement about the state of the world? Delete it, start with the second sentence.Copy
