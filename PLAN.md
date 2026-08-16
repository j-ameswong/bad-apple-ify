# Refactor & Feature Plan

## Target scale

Bad Apple (6573 frames, 512×384) against CIFAR (50k × 32×32) is the *small*
case. The intended end state is feature-length source against feature-length
gallery — e.g. a mosaic of Shrek built from a season of anime. Sizing that:

| | |
|---|---|
| Shrek @ 90 min / 24 fps | ~130,000 source frames |
| 12-episode season @ 24 min | ~414,000 frames → ~41,000 tiles at stride 10 |
| Measured decode (this machine) | ~6100 fps @ 512×384, ~580 fps @ 1080p |

Three consequences fall straight out:

| | |
|---|---|
| Source video held in RAM (130k × 512×384×3) | **76 GB** — fatal |
| Gallery held at native 1080p (41k × 6.2 MB) | **257 GB** — fatal |
| Gallery held at 16×16 cell size (41k × 768 B) | **32 MB** — trivial |
| One-time gallery decode pass (414k frames) | **~12 min** — must be cached |

So: **0.2 (streaming) is a prerequisite, not an optimisation** — the current
`np.empty((num_frames, ...))` doesn't thrash at this size, it fails outright.
And **the gallery is only ever stored at cell resolution**, which drives the
load-order constraint in 1.2.

Note what the third and fourth rows have in common: the difference between
trivial and fatal is a parameter, not a hardware limit, and nothing in the
pipeline tells the user which side of it they are on until the allocation
fails — twelve minutes into a decode, in the video case. Hence 1.2c: estimate
the tile array before building it.

---

## Phase 0 — Performance & correctness

These come first. The pipeline currently spends ~191s of CPU on a 6573-frame
512×384 video doing work that measures at ~1.1s vectorised. Fixing that up front
means every later refactor step gets a full end-to-end verification run for free
instead of a three-minute wait. Phase 0.1 and 0.2 together also subsume the old
"streaming frames" item.

### 0.1 Vectorise `mosaic_frame()`

Measured on the real gallery at `grid_size=8`, `contrast=0.8`:

```
current mosaic_frame:  29.0  ms/frame  →  191s for the full video
vectorised:             0.16 ms/frame  →    1.1s
                                            ~180× speedup
```

Three separate wastes in the current `grid_y × grid_x` Python loop:

1. `cv2.resize(gallery[idx], (cell_w, cell_h))` runs per cell per frame —
   768 × 6573 ≈ 5M resizes producing only a handful of distinct outputs.
   Pre-resize the gallery to cell size **once**, at precompute time.
2. The per-cell `.mean()` is a reshape:
   `grey.reshape(grid_y, cell_h, grid_x, cell_w).mean(axis=(1, 3))`.
   (`cv2.resize(grey, (grid_x, grid_y), interpolation=cv2.INTER_AREA)` is the
   same operation — max error 0.5/255 vs. the exact mean. Use the reshape if
   exact equivalence matters, INTER_AREA if speed does.)
3. Tile assembly is fancy indexing:
   `tiles[idx].transpose(0, 2, 1, 3, 4).reshape(H, W, 3)`.

**Testable when:** the reshape-based version is byte-identical to the current
`mosaic_frame()` on the same input (verified — it is). The INTER_AREA variant
differs only where a cell mean sits within 0.5/255 of a decision boundary.

---

### 0.2 Pipe frames to ffmpeg instead of writing PNGs

Replace the write-6573-PNGs → ffmpeg → `unlink()` cycle with a raw pipe:

```
ffmpeg -f rawvideo -pix_fmt bgr24 -s {W}x{H} -framerate {fps} -i - ...
```

Write each mosaic to `proc.stdin` as it is produced.

This removes gigabytes of disk I/O, deletes the cleanup loop, deletes
`img_format` as a config knob (and its CLI flag in 2.4), and removes the
undocumented 99,999-frame cap baked into `frame_%05d`.

Combined with a generator-based `extract_video_frames()`, it bounds peak memory
at two frames regardless of video length — which is the whole of the old 1.5.

**Testable when:** peak RSS on a long video is bounded by frame size × 2, not
frame size × frame count. The decoded mosaic *frames* match the pre-pipe
version; do **not** assert on the mp4 file hash — libx264 threading is not
deterministic across differing input pacing.

---

### 0.3 Brightness matching is a 256-entry LUT

A scalar brightness match has at most 256 possible answers. After
`shrink_gallery` at `contrast=0.8`, **only 97 of the 40,000 surviving gallery
images are ever used** — the other 39,903 are decoration.

Build the LUT once (`levels = arange(256)/255` → gallery index) and matching
becomes `lut[cell_means_uint8]`, O(1) per cell instead of an `argmin` over
40,000 floats. Only the ~97 distinct tiles need pre-resizing (see 0.1).

This is `Metric.precompute()` from 2.2 doing real work, so land 0.3 in a shape
that 2.2 can adopt directly.

**Testable when:** LUT output matches `argmin` output for all 256 input levels.

---

### 0.4 Stochastic tile selection

Consequence of 0.3: the mosaic is far more repetitive than a 40k-image gallery
should produce. Instead of `argmin`, bucket the gallery by brightness at
precompute time and sample uniformly among all images within ε of the target.

Same tonal accuracy, dramatically more visual variety, still O(1) per cell, and
it finally justifies loading the full gallery.

**Testable when:** tonal error per cell stays within ε of the `argmin` result,
while the count of distinct tiles used across a frame rises from ~97 toward the
gallery size. Seeded RNG makes runs reproducible.

---

### 0.5 Correctness fixes

- **`hstack` works by luck.** The source is 512×384 / 4:3 with a 32×24 grid, so
  `compute_target_dimensions()` snaps back to exactly 512×384. Any source where
  it changes the size gives the two ffmpeg inputs different heights and
  `hstack` fails outright. Add an explicit `scale=` to the filter chain.
- **`ASPECT_RATIOS` distorts arbitrary input.** Snapping a 2.39:1 source to 16:9
  stretches it. The grid only needs *some* integer pair — derive `grid_x`/
  `grid_y` from the source's actual ratio and drop the allowlist. Removes the
  `__post_init__` validation, the ratio `argmin`, and a config field.
- **`CAP_PROP_FRAME_COUNT` is a container-metadata guess** and can over-report,
  making `extract_video_frames()` raise `RuntimeError` mid-extraction. The
  `while cap.read()` generator from 0.2 makes this a non-issue.
- **Duplicated brightness computation.** `shrink_gallery()` computes brightness,
  filters, discards it; `gallery_brightness()` recomputes it on the survivors.
  Compute once, mask both. The per-image `cvtColor` loop is a dot product:
  `gallery @ [0.114, 0.587, 0.299]`. Likewise `get_gallery()`'s per-image
  RGB→BGR loop is `[..., ::-1]`.
- **NumPy 2.4 deprecation warning** from the CIFAR pickle load in
  `get_gallery()`. Harmless now, will break eventually.
- **Unused `matplotlib` dependency** in `pyproject.toml` — 9.4 MB plus
  pillow/fonttools/kiwisolver for nothing.
- **`output.mp4` and `upload.mp4` are untracked at repo root.** `.gitignore`
  covers `output/*` and `*/output/` but not these.

---

### 0.6 Test infrastructure

Every item below has a "Testable when" clause and there is currently no test
runner. Add `pytest` plus fixtures before any refactor lands:

- a synthetic 10-frame generated video (no reliance on `assets/source.mp4`)
- a 100-image fake gallery array (no reliance on the 155 MB CIFAR pickle)

**Testable when:** `uv run pytest` passes on a clean checkout with no assets
present beyond the repo.

---

## Phase 1 — Cleanup

### 1.1 Config refactor

Split `Config` into two dataclasses:
- `UserConfig` — frozen, user-supplied parameters only (input path, output path,
  grid size, contrast, etc.)
- `DerivedConfig` — computed from `UserConfig` after the source video is probed
  (target dimensions, cell size, FPS, grid dimensions). Built once and immutable.

Remove the class-level `ASPECT_RATIOS` constant entirely (see 0.5). Remove all
mutation from methods.

**Testable when:** Given a fixed `UserConfig` and known source dimensions,
`DerivedConfig` produces the correct target dimensions and cell size. No
attribute is set after construction.

---

### 1.2 Gallery abstraction

Introduce a `GallerySource` protocol. Note `load()` **takes the cell size** and
returns tiles already at that resolution:

```python
class GallerySource(Protocol):
    def load(self, cell_size: tuple[int, int]) -> np.ndarray:
        ...  # returns (N, cell_h, cell_w, 3) BGR array
```

This is not cosmetic. A video gallery is only tractable because tiles are
downscaled *during decode* — 41k frames at native 1080p is 257 GB, at 16×16 it
is 32 MB (see Target scale). Storing full-res gallery frames is never viable, so
the protocol must never promise them.

**Load-order constraint:** the gallery therefore cannot be loaded until the
source has been probed and the cell size derived. `probe_video()` runs *before*
`load_gallery()` — the reverse of the ordering originally sketched in 1.3.
Cheap to get right now, expensive to unpick after 1.2 and 1.3 have landed.

This also absorbs the pre-resize step from 0.1: the gallery source hands back
tiles at cell size, so there is nothing left to pre-resize.

Implement:
- `CifarGallery(path: Path)` — wraps the current pickle logic from `get_gallery()`,
  resizing 32×32 → cell size on load
- `VideoGallery(path: Path, stride: int)` — new, stubs out as `raise NotImplementedError` until Phase 2

`main()` accepts a `GallerySource` and calls `.load(cell_size)` — it never
touches CIFAR-specific code directly.

**Testable when:** `CifarGallery.load(cell_size)` returns an array matching the
old `get_gallery()` put through the same resize. Swapping in a different
`GallerySource` requires zero changes to `main()`.

---

### 1.2b Gallery cache

A 12-minute decode before every run is unusable during iteration. After
`load()`, persist the tile array plus any metric precompute to `.npy`/`.npz`,
keyed by a hash of `(gallery path, mtime, stride, cell size, metric name)`.
Cache hit skips decode entirely.

**Testable when:** a second run with identical parameters performs no video
decode and produces byte-identical tiles. Changing any keyed parameter misses
the cache.

---

### 1.2c Gallery size estimate & warning

The tile array is `N × cell_h × cell_w × 3` bytes and **both factors are under
user control**, so the product is easy to blow up by accident:

| gallery | cell | tiles |
|---|---|---|
| CIFAR 50k | 16×16 | 38 MB — trivial |
| CIFAR 50k | 64×64 | 614 MB — noticeable |
| CIFAR 50k | 512×384 (`grid_size=1`, see 2.4) | **29 GB** — fatal |
| Season, 41k tiles @ stride 10 | 16×16 | 32 MB |
| Season, 41k tiles @ stride 1 | 32×32 | **1.3 GB**, plus a ~2 hr decode |

Today the only feedback is the OOM killer, and on a video gallery it arrives
*after* a twelve-minute decode pass — the worst possible time to learn the
parameters were wrong. Estimate before allocating.

**Add a size estimate to the `GallerySource` protocol.** The count is the only
unknown, and every source can approximate it far more cheaply than a full load:

```python
class GallerySource(Protocol):
    def estimate_count(self) -> int | None:
        """Rough tile count, cheaply. None if genuinely unknowable."""
```

- `CifarGallery` — **exact and free**: a CIFAR batch is `N × 3072` bytes of
  planar uint8, so `N ≈ path.stat().st_size // 3072` without unpickling. Worth
  confirming against the real file's header overhead before trusting the
  divisor.
- `VideoGallery` — `CAP_PROP_FRAME_COUNT / stride` per file, summed over the
  directory. That metadata is unreliable (see 0.5) and must not be used to size
  an allocation, but it is fine for an order-of-magnitude warning. Where it is
  absent or zero, `duration × fps` is a second guess; if both fail, return
  `None`. Dedupe (2.1) only ever makes the real count *lower*, so the estimate
  stays an upper bound — say so when reporting it.
- `None` means "cannot estimate", which is not the same as "small". Report it as
  unknown rather than silently proceeding as if it were zero.

**Then act on it, in `load_gallery()` — before `source.load()` is called:**

- Under a soft budget (default ~1 GB): print the estimate as an ordinary line,
  alongside the tile count and cell size already logged.
- Over it: warn loudly with the arithmetic spelled out — `41000 tiles ×
  64×64×3 B = 503 MB` — and name the two knobs that shrink it (`--cell-size`,
  `--stride`), because a bare number does not tell the user what to change.
- Over a hard budget (default ~8 GB, or some fraction of `psutil`-reported
  available RAM if that is worth the dependency): refuse, and require an
  explicit `--gallery-budget` override to proceed. This is the one place the
  pipeline should stop rather than try — an OOM twelve minutes in destroys more
  work than a refusal does.
- A cache hit (1.2b) knows the real array size from the `.npy` header and skips
  the estimate entirely; it has the true answer.

**Landed.** `estimate_count()` on the protocol, `check_gallery_budget()` in
`load_gallery()` ahead of `source.load()`, `gallery_budget` on `UserConfig`,
`GalleryTooLarge` on refusal. Notes in `docs/gallery-size.md`,
tests in `tests/test_budget.py`.

**Measured, before writing it:** the CIFAR divisor reads 50537 against a real
50000 (1.1% over — labels and filenames pad the pickle), and
`CAP_PROP_FRAME_COUNT` was exact on all three rips to hand. Both good enough to
refuse on, so neither got downgraded to a warning.

**Investigate first:** how accurate the two estimators actually are. Check the
CIFAR divisor against the real 155 MB `train` batch, and
`CAP_PROP_FRAME_COUNT` against a straight decode on a long-GOP file — the
warning is only worth having if it does not routinely cry wolf. If
`CAP_PROP_FRAME_COUNT` proves badly wrong on real anime rips, downgrade the
hard refusal to a warning for video sources and keep the refusal for sources
that can count exactly.

**Testable when:** `CifarGallery.estimate_count()` is within a few percent of
`len(load(...))` on the real batch, and exact on a synthetic one. A gallery
projected over the hard budget raises before any decode or allocation happens —
assert the load was never entered, not merely that it was slow. A source
returning `None` neither warns nor refuses, but says so. ✅ —
`tests/test_budget.py`, where the over-budget source's `load()` raises
`AssertionError` if it is ever reached.

---

### 1.3 Break up `main()`

Extract these pipeline stages as top-level functions:

```
probe_video(config: UserConfig) -> DerivedConfig          # must run first (see 1.2)
load_gallery(source: GallerySource, derived: DerivedConfig) -> np.ndarray
stream_frames(config, derived) -> Iterator[np.ndarray]
build_mosaics(frames_iter, metric, user_cfg, derived_cfg) -> Iterator[np.ndarray]
encode_video(mosaic_iter, derived_cfg, output_path) -> Path   # owns the ffmpeg pipe
combine_videos(source_path, mosaic_path, output_path)
```

`main()` becomes an orchestrator that calls these in order with no logic of its own.

**Testable when:** Each function can be called in isolation with known inputs and
produces a verifiable output without side effects on the others.

---

### 1.4 Type hints throughout

Add full PEP 484 annotations to every function signature. No `Any`, no bare
`tuple` — use `tuple[int, int]` etc.

**Landed.** The signatures were mostly there already; what `--strict` actually
caught was `np.ndarray` itself. A bare one is `ndarray[Any, dtype[Any]]`, so
every array in the pipeline was untyped in the way that matters — nothing
stopped a brightness array being passed where tiles were expected. Three
aliases now say which is which:

```python
type Image = npt.NDArray[np.uint8]        # frames, tiles, galleries
type Brightness = npt.NDArray[np.float64] # 0-1 scalars, one per tile
type Indices = npt.NDArray[np.int64]      # what match() returns
```

The rest was four `np.load`/`@`/fancy-index results coming back as `Any`
(annotated at the binding), one `cv2.resize` whose stub won't commit to a dtype
(cast), and `proc.stdin` being `IO[bytes] | None` when `stdin=PIPE` guarantees
it isn't (assert). `strict = true` and `files = ["main.py"]` live in
`pyproject.toml`, so `uv run mypy` bare is the check. Tests are out of scope —
97 of the 107 errors there are `-> None` on test functions.

**Testable when:** `mypy --strict main.py` passes with no errors. ✅ — `uv run mypy`.

---

## Phase 2 — New features

### 2.1 VideoGallery

Implement `VideoGallery(path: Path, stride: int)`:
- Opens the video with OpenCV
- Decodes sequentially, keeping every `stride`-th frame (no seeking — frame-accurate
  seek on long-GOP video is slower than a straight decode)
- **Downscales each kept frame to `cell_size` immediately**, before it is stored
- Returns `(N, cell_h, cell_w, 3)` BGR array

Accepts a directory or glob as well as a single file, so "a season" is one
argument rather than twelve runs.

Stride defaults to 10 (configurable via CLI/config).

**Non-square cells.** CIFAR is square so this never arose. 16:9 gallery frames
into a non-16:9 cell need an explicit choice; centre-crop to the cell's aspect
ratio then resize generally looks far better than squashing. Make it a flag
(`--tile-fit crop|stretch`, default `crop`).

**Held-cel dedupe.** Animation holds frames for multiple ticks, producing long
runs of byte-identical gallery frames. Hash tiles after downscale and drop
duplicates — on animation this plausibly halves the gallery for free, and it
directly improves 0.4 (a deduped gallery has more genuine variety per
brightness bucket).

**Testable when:** Given a known video, `VideoGallery.load(cell_size)` returns an
array whose length equals the number of frames actually kept at that stride,
each already at cell size. Each sampled frame matches the corresponding frame
extracted and resized manually. Do **not** assert against
`ceil(CAP_PROP_FRAME_COUNT / stride)` — that metadata is unreliable (see 0.5).
Peak RSS during load is bounded by one full-res frame plus the tile array.

---

### 2.2 Pluggable matching metric

Extract the match into a `Metric` protocol. Note the signature is **grid-wise,
not cell-wise** — a per-cell `score()` would re-introduce the Python loop that
0.1 removes:

```python
class Metric(Protocol):
    def precompute(self, gallery: np.ndarray, cell_size: tuple[int, int]) -> None:
        """Build the lookup structure and pre-resized tile array."""
    def match(self, frame: np.ndarray) -> np.ndarray:
        """(H, W, 3) frame -> (grid_y, grid_x) array of tile indices."""
    @property
    def tiles(self) -> np.ndarray:
        """(U, cell_h, cell_w, 3) pre-resized tiles, indexed by match()."""
```

Implement:
- `BrightnessMetric` — 256-entry LUT from 0.3, with optional stochastic
  selection from 0.4
- `ColourMetric` — quantised 3D BGR lattice lookup (e.g. 32³) built once, same
  O(1) per-cell cost

**Priority note.** At target scale `ColourMetric` stops being optional. 41k
anime frames cluster hard in the mid-tones, so brightness-only matching makes
the 97-distinct-tiles problem from 0.3 *worse*, not better — a huge gallery
collapses to a handful of usable tiles. Colour matching is what makes video
galleries look like anything at all. Consider pulling this ahead of 2.1 so the
first `VideoGallery` run is worth looking at.

`build_mosaics()` accepts a `Metric` instance and does nothing but
`metric.tiles[metric.match(frame)]` + reshape.

**Testable when:** `BrightnessMetric` with stochastic selection disabled
produces identical tile selections to the pre-refactor `mosaic_frame()` on the
same input. `ColourMetric` produces different selections on a colourful source
frame. Swapping metric requires no change to `build_mosaics()`.

---

### 2.3 Temporal frame caching

Bad Apple is ~95% temporally static — consecutive frames are mostly identical
black/white regions. Cache the previous frame's index grid and re-tile only the
cells whose index changed.

Roughly five lines on top of 0.1. Likely a large further win on this specific
source, much less useful on general video — gate it behind a flag, or measure
before committing. Note it interacts with 0.4: stochastic selection would make
every cell "change" unless the cache compares indices *before* sampling.

**Testable when:** output is identical to the uncached path (with stochastic
selection seeded or disabled), and measured frame time drops on a
low-motion source.

---

### 2.4 Grid-size-1 as single-frame mode

No separate code path. A `--grid-size 1` flag (or `grid_size: 1` in config)
collapses the grid to 1×1, replacing each input frame with its best single
gallery match.

**Landed.** It did *not* "just work": `grid_size` multiplies the aspect pair, so
`1` gave a 4×3 grid of twelve tiles — a coarse mosaic, not one image per frame.
One branch in `DerivedConfig.from_source` makes the grid literally `(1, 1)`;
everything downstream carries it unchanged.

**It is also the worst case for 1.2c.** A 1×1 grid means the cell *is* the
frame, and tiles are stored at cell size — so the whole gallery is held at
output resolution. CIFAR at 512×384 is 29 GB, i.e. the mode is unusable on the
default assets and fails only at the point of allocation. The 1.2c estimate is
what turns that into a message the user can act on before anything is decoded.

**The structural fix is out of scope for 2.4** and worth its own item if
single-frame mode is ever meant to run on a large gallery: compact the gallery
down to the tiles a metric can actually reach *before* resizing to cell size.
Selection depends only on brightness (or colour), which survives a downscale, so
the shape is: load small, precompute, then load again for the reachable subset
only. That needs a `GallerySource` that can load a subset — a protocol change,
and one that costs a second decode pass on video unless the cache (1.2b) covers
it. Not worth doing until something actually needs it.

**Testable when:** With `grid_size=1`, the output video has the same frame count
as the source, and each output frame is a single gallery image (not a tiled
composite). ✅ — `tests/test_single_frame.py`, asserting each mosaic is
byte-identical to one entry in `metric.tiles`.

---

### 2.5 Decouple output resolution from source

Cell size is currently derived as `src_dimensions / grid`, which caps tile
detail at whatever the source happens to be — a 512×384 source with a 32×24
grid gives 16px tiles and no way to ask for more.

Invert it: specify **grid dimensions and cell size** directly, and let output
resolution be `grid × cell`. A 1080p source with a 120×68 grid and 32px cells
produces a 3840×2176 mosaic. The source is then sampled down to the grid (which
0.1 already does via reshape/INTER_AREA) and its native resolution stops
mattering at all.

Note this makes the 0.5 `hstack` fix load-bearing rather than defensive — source
and mosaic will now routinely differ in size.

It also makes 1.2c load-bearing. Cell size stops being a derived quantity the
user can only nudge via `grid_size` and becomes a direct knob — and it enters
the tile-array size quadratically, so `--cell-size 64` instead of `16` is a 16×
memory jump from a single character's difference. Land 1.2c before this, not
after.

**Testable when:** output dimensions equal `grid × cell` for any source
resolution, and changing the source resolution alone does not change the output
resolution.

---

### 2.6 Segmented encode & resume

A 130k-frame run dying at frame 100k is expensive, and 0.2's ffmpeg pipe removes
the accidental checkpointing the intermediate PNGs provided.

Encode in fixed-size segments (say 5000 frames) to `part_%04d.mp4`, then
`ffmpeg -f concat` at the end. On restart, skip source frames already covered by
existing complete segments.

**Testable when:** killing a run mid-way and restarting produces the same output
as an uninterrupted run, and re-decodes only the unfinished segment.

---

### 2.7 `--start` / `--duration`

Process only a slice of the source. Needed to test on 10 seconds of Shrek before
committing to a 90-minute encode — this will get used constantly during
iteration on everything above.

**Testable when:** `--start 60 --duration 10` on a 30 fps source produces
exactly 300 frames, matching frames 1800–2099 of a full run.

---

### 2.8 CLI + config file

Use `argparse` with a TOML config file as the base layer:

- Default config loaded from `config.toml` if it exists in the working directory
- CLI flags override any config file value
- Required if neither config file nor CLI provides them: `--source`, `--gallery`

Exposed flags (all optional if in config):
```
--source PATH          source video
--gallery PATH         CIFAR pickle, video file, directory, or glob
--gallery-type         cifar | video  (auto-detected from extension if omitted)
--stride INT           gallery video frame stride (default: 10)
--grid-size INT        mosaic grid multiplier (default: 8)
--cell-size INT        tile edge in px; output = grid × cell (default: derive from source)
--tile-fit             crop | stretch  (default: crop)
--contrast FLOAT       gallery brightness band (default: 0.8)
--metric               brightness | colour  (default: colour)
--stochastic / --no-stochastic   vary tile choice among equal matches (default: on)
--seed INT             RNG seed for stochastic selection
--start FLOAT          source start time in seconds
--duration FLOAT       seconds of source to process
--segment INT          frames per encode segment (default: 5000)
--no-cache             bypass the gallery cache
--gallery-budget SIZE  max estimated tile-array size, e.g. 2G (default: 8G)
--output-dir PATH      (default: ./output)
```

`--gallery-budget` is the 1.2c override, and the estimate should also be
printed by a `--dry-run` that stops after probe + estimate — the cheapest way to
find out what a set of parameters is about to cost.

`--img-format` is gone — 0.2 removes intermediate frames entirely.

**Testable when:** Running with only CLI flags and no config file produces the
same output as running with an equivalent config file and no CLI flags. A CLI
flag always takes precedence over the config file value for the same key.

---

## Implementation order

Ordered so that each step is verifiable against the one before, and so that
nothing has to be unpicked once the target-scale constraints bite.

1. 0.6 (pytest + synthetic fixtures) — nothing below is verifiable without it
2. 0.1 (vectorise `mosaic_frame`) — 180×, byte-identical, independent
3. 0.2 (ffmpeg pipe + frame generator) — prerequisite for any long source
4. 0.3 (brightness LUT)
5. 0.5 (correctness fixes — cheap, do them while the code is still small)
6. 0.4 (stochastic selection) — first change that alters output on purpose
7. 1.1 (Config split) — unblocks 1.2 and 1.3
8. 1.2 + 1.2b + 1.2c (GallerySource taking cell size, CifarGallery, cache,
   size estimate) — 1.2c must precede 2.5, which turns cell size into a direct
   knob; 2.4 has already demonstrated what its absence costs
9. 1.3 (break up main — note probe-before-load ordering)
10. 1.4 (type hints + mypy)
11. 2.7 (`--start` / `--duration`) — pulled early; needed to iterate on anything
    feature-length without waiting out a full encode
12. 2.2 (pluggable metric, incl. `ColourMetric`) — **before** 2.1, so the first
    video-gallery run is actually worth looking at
13. 2.1 (VideoGallery + non-square fit + held-cel dedupe)
14. 2.5 (decouple output resolution)
15. 2.6 (segmented encode & resume)
16. 2.3 (temporal caching, measure first)
17. 2.4 (grid-size-1) — **done, out of order**; it needed a real change, and it
    is what motivated 1.2c
18. 2.8 (CLI + config file)

**First milestone worth targeting:** steps 1–13 are everything needed to run
Shrek against a season of anime end to end. 14–18 are quality and ergonomics on
top of a working pipeline.
