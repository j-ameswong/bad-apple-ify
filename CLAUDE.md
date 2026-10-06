# Repository guide

## Purpose

This project turns a source video into a photo mosaic using images from a
CIFAR-100 pickle or a video gallery. It writes the mosaic video and a second,
side-by-side video with the source audio. Frames are processed as a stream;
the resized gallery and metric lookup tables stay in memory.

## Common commands

```sh
uv sync
uv run cli.py --source assets/source.mp4 --gallery assets/gallery/train
uv run pytest
uv run mypy
```

FFmpeg must be available on `PATH`. The gallery can be a CIFAR-100 `train`
pickle, a video, a directory of videos, or a video glob. Use
`--gallery-type cifar|video` to choose explicitly. The CLI reads `config.toml`
from the working directory when present; `--config` selects another TOML file
and command-line flags override its values. See [CLI configuration](docs/cli.md)
and [config.example.toml](config.example.toml).

## Tests and type checking

Tests build their own small FFV1 videos and CIFAR-format pickle, so the full
source video and CIFAR dataset are not needed. Run the test suite with
`uv run pytest`. The bare `uv run mypy` command checks the files listed under
`[tool.mypy]` in `pyproject.toml`, covering the package and root entrypoints.

The checkpoint code has POSIX and Windows locking implementations. The current
verification environment is Linux, so Windows execution still needs testing.

## Main behavior

- `--grid-size` chooses a grid multiplier. `--grid COLSxROWS` fixes the grid;
  `--cell-size` sets cell height. `tile_fit` is `native`, `crop` or `stretch`.
- The colour metric matches average BGR values on a 3D lattice. The brightness
  metric matches luma and uses `epsilon` as an error ceiling. `contrast` trims
  gallery brightness extremes. `candidates`, `stochastic` and `seed` control
  tile selection.
- `hold_tiles` keeps a cell's tile while its match bucket stays unchanged.
- `start` and `duration` select a source-time slice. The source pane, mosaic
  and audio share that slice. Seeded matching replays the skipped prefix so a
  slice agrees with the same frames from a full run.
- Video galleries use `stride` to sample frames and discard duplicate tiles.
  Their frames are resized as they decode.
- `gallery_budget` guards the resized tile array. Actual process memory may be
  higher while gallery and metric arrays overlap; it is not a process-wide RAM
  limit. See [gallery sizing](docs/gallery-size.md).
- `use_cache` controls the `.npy` gallery cache. `segment` sets checkpointed
  encode length; `0` disables segmentation. A rerun with the same inputs and
  settings resumes validated segments. See [resume](docs/resume.md).

The full option list and defaults are in [README](README.md) and
[docs/cli.md](docs/cli.md).

## Module layout

The code lives in the following package modules. Keep new code in
the module that owns its behavior; the root scripts remain thin entrypoints.

- `bad_apple/types.py` holds shared array and configuration types.
- `bad_apple/config.py` owns user and derived configuration, validation and
  geometry setup.
- `bad_apple/gallery.py` owns gallery sources, CIFAR loading, video-gallery
  decoding, cache keys and gallery budgets.
- `bad_apple/metrics.py` owns brightness and colour matching and held-tile
  behavior.
- `bad_apple/video.py` owns source probing, frame streaming, mosaic encoding
  and side-by-side output.
- `bad_apple/pipeline.py` coordinates gallery loading, matching and video work.
- `bad_apple/cli.py` owns CLI and TOML configuration.
- `bad_apple/segments.py` owns checkpointed encoding and resume.
- `main.py`, `cli.py` and `segments.py` provide the command and compatibility
  entrypoints.

Design decisions live in `docs/`, indexed by [docs/README.md](docs/README.md).
Keep source comments short and put longer rationale in the relevant note.

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
3. Does your opening sentence set the scene with a grand statement about the state of the world? Delete it, start with the second sentence.
