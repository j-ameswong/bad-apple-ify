# Session handoff, 6 October 2026

The requested Python audit/refactor and full browser port are complete. The
browser app is publicly deployed. No implementation task remains pending.

## Accepted scope

The user requested completion of PLAN.md, a full Python audit and refactor,
then a browser port in a separate directory. They authorised autonomous work,
Luna implementation agents, Sol review, and conventional one-line commits.

The browser must be publicly hosted, process selected local files, show
progress/previews and offer downloads, with no server persistence or accounts.
The simplest maintainable stack was preferred; Java was not required.

## Python repository

Workspace: `/home/contessa/Documents/Projects/bad-apple-ify`, branch `main`.
The previous checkpoint bundle was imported and the original workspace was
verified byte-for-byte before restoring its commits. Permissions and GPG
signing now work. Nothing was pushed to the Python GitHub remote.

Restored feature commits:

- `b9c8633 feat: support explicit mosaic grids and cell sizes`
- `4c9f860 feat: checkpoint segmented encoding for deterministic resume`
- `063810c feat: add layered CLI and TOML configuration`
- `37e6fa4 docs: record session checkpoint and remaining audit work`

Audit and refactor commits:

- `48b61c6 fix: validate gallery inputs and preserve outputs on encoder failure`
- `52bf8b4 refactor: split pipeline into focused package modules`
- `fa69e23 docs: describe current CLI architecture and audit results`
- The current documentation commit records completion and links the browser app.

Code now lives in `bad_apple/{types,config,gallery,metrics,video,pipeline,cli,segments}.py`.
Root entrypoints remain compatibility facades, including the deprecated
slice-only `main.parse_config` helper. Tests patch module owners directly.

The audit fixed restricted CIFAR deserialisation and validation, actual gallery
budgets and cache validation, unique cache writes, input/output aliasing,
atomic final outputs, encoder cleanup, metadata errors, held-state resets,
CLI overflow errors and checkpoint validation. Locking has POSIX and Windows
paths; directory syncing is conditional. NumPy is an explicit dependency.
README, CLAUDE.md and the docs index describe the current architecture.
See `docs/audit.md` for findings and the measured decision to retain the
existing FFmpeg byte writes.

Verification: **475 pytest tests passed**, strict mypy passed across 12 files,
and `git diff --check` passed. The refactor received a separate AST/behaviour
review. A real source/CIFAR slice produced six frames at 30 fps, mosaic 256×192
and combined 512×192 with audio. Windows execution remains untested.

## Browser repository and publication

Workspace: `/home/contessa/Documents/Projects/bad-apple-web`, its own Git
repository on `main`. The source is committed and pushed to the Sites source
repository; the original Python repository is separate.

- Commit: `c9fe537e3c80bffa2182f501753aad3cc989a6bd`
- Message: `feat: add local browser mosaic processing and resumable exports`
- Public URL: https://bad-apple-mosaic-studio.wongchengan.chatgpt.site
- Site project ID: `appgprj_6ac5359027f08191b9d3c313e0714edf`
- Saved version: `appgprj_6ac5359027f08191b9d3c313e0714edf~appgver_891d08f54f608191b7dec32b8fa197c4`
- Deployment: `appgdep_6ac53a63220481919367be388e81de0f`, succeeded, public

`.openai/hosting.json` identifies the existing Site. Reuse that project for
future edits; do not register a replacement. Follow the Sites skill to reopen
its source, build and publish. Credentials are short-lived and were kept in
session memory only. The archive is `/tmp/bad-apple-web-site.tar.gz`; recreating
it through the workflow is preferable to depending on temporary files.

The app uses vanilla TypeScript, Vite, WebCodecs and Mediabunny in a worker.
It supports colour/brightness matching, sizing/fitting, gallery contrast,
seeded candidates, held tiles, source slicing, video sampling/deduplication,
CIFAR and still-image galleries, session caching and tile budgets, estimates,
TOML/JSON settings, cancellation and validated ZIP checkpoint export/import.
It writes a mosaic and a combined source/mosaic video with trimmed source audio.
MP4/H.264/AAC is used when supported, otherwise WebM/VP9/Opus is tried.

Segments are encoded incrementally and joined with interleaved audio/video
packets. Checkpoints preserve RNG/held state and the next source timestamp;
resume seeks to that point and checks content identities. Input and output
bytes never go to the hosting server. No settings, media or checkpoints are
written to browser persistent storage. Users explicitly download their files.
The README explains module ownership, use, development and limits.

Validation on Linux:

- `npm test`: **19 passed**, one optional full-CIFAR fixture test skipped.
- `npm run build`: passed, including strict TypeScript checking.
- `npm run test:browser`: **9 passed** in real Chromium with FFmpeg/ffprobe.
- Coverage includes first-frame slice selection, native-ratio explicit sizing,
  delayed audio timestamps, both metrics, CIFAR opacity, settings round-trip,
  malformed/unsupported media, mobile layout and output URL cleanup.
- Cancellation, export/import, mismatched-input rejection and successful resume
  produce identical decoded frames to uninterrupted processing.
- A separate real-data browser run used the full 50,000-image CIFAR batch and
  actual source video, start 1 s / duration 0.2 s / cell height 8 / segment 2.
  Mosaic: six VP9 frames, 256×192, 0.200 s. Combined: six VP9 frames, 512×192,
  Opus audio, 0.220 s including codec padding. No browser page errors.
- Geometry matched Python across 108 representative cases.
- Full CIFAR parser output matched the Python data digest. Its optional test
  can be enabled with `BAD_APPLE_CIFAR_FIXTURE` pointing to the local train file.
- Optional WebMCP tools were tested with an injected registry, including invalid
  form reporting and repair; a native WebMCP runtime was unavailable.
- Dependency audit reported zero vulnerabilities. Final source diff checks pass.

Practical limits: codec availability varies by browser/device; source frame
order is encoded at detected nominal fps. Browser PRNG and Canvas resizing
differ from Python, so cross-implementation pixel equality is not promised.
The 256 MiB budget bounds RGBA tile storage, not all working/output memory.
Encoded outputs accumulate as browser-managed Blobs. CIFAR inputs are capped
at 512 MiB; ZIP checkpoints at 2 GiB. Geometry is capped before allocation,
with details in the browser README. A saved checkpoint is needed to survive
closing the tab. These limits are documented, not unfinished features.

## Local tools and assets

Use `.venv/bin/pytest -q` and `.venv/bin/mypy` in the Python repo. Use `npm test`,
`npm run build` and `npm run test:browser` in the browser repo; browser tests
need Chromium, Python 3, FFmpeg and ffprobe. They generate synthetic videos in
`/tmp`; small committed pickles are synthetic test fixtures. Original source
and CIFAR media remain in the Python repo's ignored `assets/` directory and
were not included in the Site source or deployed archive.

No recurring automation was requested or created. Local development servers
were stopped after completion. Both repositories are clean and committed.
