# Session checkpoint, 6 October 2026

The user asked to stop at a reasonable point so they can restart with the
intended sandbox permissions. Work is stopped after the remaining PLAN features.
The broader audit/refactor and browser port are still outstanding.

## Requested work and accepted scope

1. Finish the remaining PLAN.md items.
2. Audit the entire Python project, fix defects, and improve performance,
   simplicity and modularity where justified.
3. Once the Python project is satisfactory, port its entire functionality to
   a webapp in another directory.
4. Use Luna agents for well-scoped implementation and Sol to review each
   feature. Make one-line conventional commits for each chunk. Work
   independently; the up-front questions have already been answered.

The user clarified the web scope:

- A public hosted app with no server persistence; file selection, processing
  and downloads should happen on the client if possible.
- Choose the simplest maintainable stack; Java is no longer a requirement.
- Browser file selection only, with progress, previews and downloads. No
  server-side path interface or accounts are requested.

The leading architecture is a static TypeScript app using WebCodecs and
Mediabunny for incremental media reading/encoding. This is a researched
direction, not an implemented or tested decision. No web project has been
created yet. Avoid replacing the media library with a large bespoke muxer
simply to work around this session's package-download restriction.

## Git and files

The original workspace is `/home/contessa/Documents/Projects/bad-apple-ify`.
It started clean at `965248d`. All source edits are present there, but its
`.git` became read-only during this session, so its HEAD is unchanged.

The same edits are committed on branch `session-checkpoint` in
`/tmp/bad-apple-ify-complete`, a local clone retaining the original history:

- `b9c8633 feat: support explicit mosaic grids and cell sizes`
- `4c9f860 feat: checkpoint segmented encoding for deterministic resume`
- `063810c feat: add layered CLI and TOML configuration`
- A final docs commit records this handoff.

A portable bundle of the checkpoint branch is saved at
`.cache/session/python-checkpoint.bundle` in the original workspace. It needs
the existing `965248d` history. Import the commits into the original repository
after permissions are restored and working-tree equality is verified. Preserve
any new user changes; do not blindly discard the currently uncommitted files.
Nothing was pushed.

The user's configured GPG signing failed because the sandbox made `.gnupg`
read-only. These temporary commits are unsigned, using a per-command
`commit.gpgsign=false` override. Global signing configuration was not changed.

## Implemented and reviewed

Luna implemented all three remaining features. Sol reviewed the CLI, sizing,
and resumable encoder, and confirmed the fixes to its findings. Root integrated
the new entrypoint and orchestration. PLAN.md sections 2.5, 2.6 and 2.8 now
record completion and link their documentation.

- Explicit `--grid COLSxROWS` and `--cell-size`, with existing sizing preserved
  when omitted. Cell size is height; native tiles derive width from their
  aspect ratio, crop/stretch use a square. Explicit odd output dimensions
  fail clearly rather than silently changing the requested sizes.
- Config validation for sizes, booleans, matching parameters, times, budgets
  and video geometry.
- `segments.py` writes atomic, synced segment and matcher checkpoints.
  Resume verifies identity and hashes, restores RNG and held-tile state,
  seeks to the unfinished source frame, and concatenates completed segments.
  A file lock prevents concurrent segmented writers. It uses `fcntl`, so
  portability is a remaining audit consideration.
- `cli.py` supports optional default TOML, explicit `--config`, file-relative
  config paths, CLI overrides, required source/gallery, dry-run estimates,
  all matching/sizing/slicing/budget/cache flags, and gallery auto-detection.
  Segments default to 5000 frames at the CLI; `--segment 0` disables them.
  The programmatic `UserConfig` default remains unsegmented for compatibility.
- Video glob expansion, stable file order, positive stride validation, and
  failure for missing, empty or unreadable galleries, including in dry runs.

Sol's fixes: globs previously selected a literal filename; dry run previously
accepted unreadable galleries; NPZ checkpoint data now fsyncs before rename.
There are no outstanding findings from these scoped reviews. This is not a
claim that the full-project audit is complete.

## Verification

- Initial baseline: 365 tests and strict mypy passed.
- Final combined run: `.venv/bin/pytest -q` -> **439 passed in 9.90 seconds**.
- `.venv/bin/mypy main.py cli.py segments.py` -> no issues in three files.
- After the final NPZ fsync correction: five resume tests and strict mypy
  for `segments.py` passed again.
- Sol separately exercised interruption after consuming part of segment 2,
  using a sliced 30000/1001 H.264 source. Resumed decoded frames exactly
  matched uninterrupted output for colour/brightness and held/unheld choices.
- Real source/CIFAR run succeeded:
  `.venv/bin/python main.py --source assets/source.mp4 --gallery assets/gallery/train --start 60 --duration 0.2 --cell-size 8 --segment 2 --output-dir /tmp/bad-apple-audit/real-output`
  Both outputs contain six frames at 30 fps; mosaic is 256x192, combined is
  512x192, and both video/audio duration is 0.2 seconds.
- `git diff --check` passed.

## Next work

First restore the checkpoint commits in the original repo as appropriate for
the restarted session's permissions. Then complete the requested full audit.
These are initial observations, not all confirmed defects:

- Split the large main.py into focused configuration, gallery/cache, matching,
  video I/O, pipeline, CLI and checkpoint modules. Keep the useful public
  interface; update tests that monkeypatch module-owned functions deliberately.
- Retire or adapt the legacy `parse_config()` slice-only interface, which is
  still present for existing callers/tests. main.py's actual executable
  entrypoint already uses the complete CLI.
- Review encoder failure/cancellation cleanup: a BrokenPipeError on stdin
  close can skip process waiting; encode/combine currently write final paths
  directly; the resumable path already publishes completed files atomically.
- Audit CIFAR deserialisation and array validation. `pickle.load` still accepts
  arbitrary globals. Check actual gallery size against budget as well as the
  estimate, and include file size in gallery fingerprints where useful.
- Review cache shape/count validation, same-process temporary-name collisions,
  metadata edge cases, and resetting held state when a metric is re-precomputed.
- Add NumPy as an explicit dependency instead of relying on OpenCV's transitive
  dependency. Expand mypy's default coverage to the extracted package/CLI.
- Rewrite the stale README and update CLAUDE.md and the docs index. README
  still claims the whole source video is held in RAM and shows old usage.
- Consider writing NumPy frame buffers via memoryview instead of `.tobytes()`.
  A small local measurement at 512x384 found 0.0122 ms for copying frame bytes
  versus 0.0002 ms for making a view; measure actual pipeline impact before
  claiming a performance improvement. Existing colour precompute is startup
  work (single-occupied-bin cases: ~24 ms at 32 bins, ~314 ms at 64 bins), so
  avoid complicating it without a practical gain.
- Add meaningful tests for fixes, document validation and limits, commit each
  chunk, then implement and review the full browser port. Browser checkpoint
  export/import and session-only caching must respect the no-persistence scope.

## Environment during the stopped session

Initial permissions were danger-full-access with network enabled. An explicit
live update changed them to workspace-write with network restricted and approval
policy never; only the original workspace and `/tmp` were writable. The original
`.git`, `.gnupg`, and sibling project directories were not writable.

`npm view mediabunny` failed with EAI_AGAIN. Mediabunny is not in the local npm
cache; Vite, TypeScript, Vitest and Playwright packages and browser binaries
are cached. Official web documentation was readable using the web tool. Do not
assume these restrictions continue after restart: inspect the new environment.

Installed tools include Python 3.14, uv, FFmpeg/ffprobe, Java, Maven, Node and npm.
Use `.venv/bin/pytest` and `.venv/bin/mypy` directly under restricted permissions,
or redirect uv's cache to `/tmp/bad-apple-uv-cache`. Existing source video and
CIFAR assets are available; automated tests generate their own fixtures.

The local `skills/clean-code/SKILL.md` was read as repository context. Root also
read the Sites skill to assess suitability but did not invoke a Sites workflow.
No hosting registration, remote writes or publication occurred.
