# Python audit, 6 October 2026

The audit covered configuration, both gallery sources, caching, both matching
metrics, streaming, encoding, the CLI and resume checkpoints. Changes were
implemented in bounded chunks and reviewed by a separate Sol agent.

## Correctness fixes

- CIFAR loading permits only the NumPy constructors needed by supported batch
  files and validates the resulting array. Empty, malformed and unsupported
  pickles fail with a readable error. This is not a general-purpose pickle
  reader or a memory sandbox.
- Gallery loads check actual tile count, dimensions and dtype, then enforce
  the byte budget even when a source estimate was wrong. Cache writes use
  unique temporary files; empty caches are misses. Fingerprints include file
  size. Video probes release their captures on errors and visit every input
  even when one file's count is unknown.
- Encodes and side-by-side output publish only after FFmpeg succeeds. Failed
  pipes and interrupted iterators release resources and preserve any previous
  final output. Output paths cannot alias built-in gallery inputs or the
  source video. Custom galleries may expose `input_paths` for this check.
- Rebuilding a held-tile matcher clears its old choices. A changed frame grid
  also begins a new held state.
- Invalid TOML budget strings and numeric overflow produce concise CLI errors.
  Resume validates segment sizes, total frames and state-file ordering.
- NumPy is a direct dependency. Checkpoint locks have POSIX and Windows paths;
  directory syncing is conditional on platform. Windows has not been tested.

## Performance decisions

Source frames remain streamed; the gallery remains stored at cell resolution.
The budget guards the tile array, not the peak of all simultaneous allocations.
The existing vectorised matchers and tile assembly are retained.

A local FFmpeg pipe experiment compared `frame.tobytes()` with
`memoryview(frame)` over 180 repeated 512×384 frames, three runs each. Median
wall time was 0.168 seconds for bytes and 0.185 seconds for a view, with run
variation larger than the expected copy saving. This did not justify a change
or a speed claim. It is a small synthetic experiment, not a general benchmark.

## Verification

The defect-fix chunk passed 475 tests, strict mypy and `git diff --check`.
Regressions cover corrupt inputs, inaccurate budgets, interrupted encodes,
preserving previous outputs, input/output collisions, checkpoint validation
and held-state reuse. Tests generate their own media. The restricted CIFAR
reader was also exercised on the full local 50,000-image train batch.

Resuming remains dependent on unchanged inputs and configuration.
