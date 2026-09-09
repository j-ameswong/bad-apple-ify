# Matching metrics

A `Metric` decides which tile each grid cell gets. Two ship: `BrightnessMetric`
(see [brightness-matching.md](brightness-matching.md)) and `ColourMetric`.
`build_metric()` picks between them off `config.metric`, and nothing downstream
knows which it got — `mosaic_frame()` is `metric.tiles[metric.match(frame)]`
either way.

## The protocol is grid-wise

```python
def match(self, frame: Image) -> Indices:  # (H, W, 3) -> (grid_y, grid_x)
```

A per-cell `score()` would read better and would put back the Python loop that
vectorising `mosaic_frame()` took out — 29 ms a frame against 0.16. So a match
takes the whole frame and answers for every cell at once.

`precompute()` also takes an optional `brightness`, which is the array
`shrink_gallery()` already computed. It's an offer, not a requirement:
`ColourMetric` ignores it.

`match()` is really two steps, and the protocol exposes both:

```python
def keys(self, frame: Image) -> Indices:   # (H, W, 3) -> bucket per cell
def sample(self, keys: Indices) -> Indices # bucket per cell -> tile per cell
```

`keys()` is where the metrics differ — a brightness level for one, a lattice
index for the other. `sample()` is the same three lines either way, so it lives
in `draw_from_buckets()` and both call it. Splitting them is what lets
`hold_tiles` compare frames by what they *look* like rather than by which tile
the RNG happened to hand out.

## Holding a tile still

A cell picks uniformly from its bucket, so with `candidates` above 1 it draws
again every frame even when nothing in front of it has moved. Static regions
boil. `hold_tiles` (default on) wraps the metric in `SteadyMetric`, which keeps
last frame's tile wherever the key is unchanged and only lets the RNG near the
cells that actually moved.

Measured over 600 frames of Bad Apple against CIFAR at `grid_size=8`, where
10.2% of cells change key frame to frame:

| | tile churn |
|---|---|
| `colour_bins=32` (median 2 candidates a bucket) | 11.4% → 10.2% |
| `colour_bins=8` (256 candidates a bucket) | 99.4% → 6.9% |

10.2% is the floor: it's the picture changing. The first row barely moves
because CIFAR spread over a 32-bin lattice leaves two tiles in a bucket, so
there's little to re-roll between. The second row is the one to look at, and
it's the shape a video gallery has — 41k anime frames cluster hard, buckets get
deep, and *every* cell was redrawing every frame.

It costs nothing. `SteadyMetric.sample()` draws for the whole grid and then
masks, because an RNG call over 768 cells is beneath measurement and a partial
draw needs bookkeeping that isn't. Match time is 0.31 ms/frame either way.

Turn it off with `hold_tiles=False` if you want the old behaviour; at
`candidates=1` it's a no-op, since a bucket of one has nothing to re-roll.

Note what this is *not*. PLAN.md's 2.3 asked for a frame cache — re-tile only
the cells whose index changed — and that's dead. You have to compute cell means
to know what changed, so `match()` runs regardless, and the only skippable work
is assembly at 0.08 ms of a ~1.8 ms frame. At 90% of cells held that's a 4%
ceiling, against a pipeline whose real cost is ffmpeg at 1.11 ms/frame. The
stability was the part worth having.

## Why colour at all

Brightness throws away two thirds of the signal, and it shows most on exactly
the galleries this project is aimed at. 41k anime frames cluster hard in the
mid-tones: to a brightness matcher a sunset and a forest are the same tile, so
a huge gallery collapses to a handful of usable ones. Against the real CIFAR
batch at a 16x16 cell, brightness reaches 33,151 of the 50,000 images and
colour reaches 49,954.

## The lattice

Mean BGR quantised onto a `colour_bins`³ lattice — 32³ by default, so 32,768
buckets over a 256³ space. Tiles sharing a bucket land contiguously once sorted
by lattice index, so a bucket is an offset and a count, the same shape as a
brightness bucket. `match()` is one `searchsorted`-free lookup per cell.

Everything in a bucket is equally acceptable by construction: the bin width
*is* the tolerance. So `candidates` only caps how many of a bucket's tiles get
sampled from, and it takes the first few in gallery order rather than sorting
within the bin. `colour_bins` is the accuracy knob, not `epsilon`.

## Empty buckets borrow

Most of the lattice is empty — CIFAR's 50k images touch maybe a tenth of it —
and a source frame is free to ask for a colour no tile has. `nearest_occupied()`
runs a BFS wave over the six face neighbours until every lattice cell points at
an occupied one, so "nearest" is Manhattan rather than Euclidean. Exact
Euclidean wants a KD-tree, which is a whole dependency for a tie-break nobody
can see in the output.

Each wave reads a snapshot of the previous one, so cells fill in true distance
order instead of taking whichever direction happened to be checked last.

## Picking `colour_bins`

Lower bins is a wider tolerance: more tiles per bucket, more variety, less
accuracy. The failure mode is not subtle. At `bins=8` a bucket spans 32 levels
per channel, so Bad Apple's black background draws tiles averaging (31, 31, 31)
and the silhouette washes out entirely. 32 holds the shape and is what the
default is for.

Going the other way costs variety, and on a gallery with nothing in the region
being asked for it costs all of it: CIFAR has about two near-white images, so
Bad Apple's white areas repeat the same tile no matter how high `candidates`
goes. That's the gallery, not the lattice — brightness matching does the same
thing on a flat white frame. A gallery of real video frames doesn't have that
hole.

The ceiling is 64, which is 262,144 buckets and a fill pass to walk. Above that
precompute stops being free and the lattice is finer than any gallery can fill.

## What it costs

Against 50k tiles at a 16x16 cell, 512x384 output:

| | precompute | match |
|---|---|---|
| `BrightnessMetric` | 0.16 s | 0.11 ms/frame |
| `ColourMetric`, 32 bins | 0.17 s | 0.28 ms/frame |

Both are dominated by `cell_means()`, which reduces one axis at a time rather
than in a single `.mean(axis=(1, 3))` — the strided uint8 reduction over both
axes at once costs 10x as much, and summing in uint32 gives the identical
answer.
