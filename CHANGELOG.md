# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The interface-layer tests failed in CI instead of skipping, because the test job
  installs only the `dev` extra and `viz` imports Streamlit at module level. They are now
  guarded, and `dev` pulls in Streamlit so that they actually run rather than being
  quietly skipped everywhere.
- **Annotation was unusable with streamlit-drawable-canvas 0.10 and later.** The 0.10
  rewrite moved to Fabric.js v6, which emits capitalised object types (`Rect` instead of
  `rect`). Parsing therefore found no box: "Validate the box" stayed disabled, automatic
  validation never fired, and nothing said why. It also removed `display_toolbar`
  (`TypeError` as soon as the canvas was drawn) and the `transform` drawing mode the
  eraser relies on. The supported range is now pinned to `>=0.9.3,<0.10`, and an
  unsupported version is refused at import with a message naming the pin, instead of
  producing an interface whose buttons quietly do nothing.
- An incompatible Streamlit / canvas pair took the whole application down at start-up:
  the component fails at import with `StreamlitAPIException` rather than `ImportError`,
  which the shim did not catch. The canvas is now reported as unavailable, the error
  explains why, and phase 2 remains usable.
- Canvas object types are matched case-insensitively, so parsing is not what breaks first
  when the Fabric.js major version changes.
- The `image_to_url` patch is no longer applied to canvas 0.10+, which never calls it.
- Dropped the deprecated `use_container_width` argument from `st.image`; its former
  behaviour is the current default.

## [0.1.0] — 2026-09-07

First public release.

### Added

- Two-phase annotation workflow: SAM 2 assisted instance annotation, then semantic
  grouping and structured description.
- Business layer (`segmentart.backend`) with no dependency on the user interface:
  mask operations, lossless compact RLE, SAM 2 engine, CLIP embeddings, clustering,
  parts, relations, orchestration, serialisation, evaluation and telemetry.
- Streamlit interface (`segmentart.frontend`) for both phases.
- Export to Visual Genome, COCO and a user-defined custom schema.
- Optional interaction telemetry, disabled by default, exportable as JSONL and CSV.
- Test suite covering RLE round-trips, IoU, export and import, rule-based attributes,
  and the backend/frontend separation rule.

### Notes

- SAM 2, OpenCV, CLIP and pycocotools are optional. The application starts without them
  and disables the corresponding features.
- Model weights are not distributed with this software.
