# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Annotation crashed with `TypeError: st_canvas() got an unexpected keyword argument
  'display_toolbar'` on streamlit-drawable-canvas 0.10 and later, where the keyword was
  removed. The compatibility shim now introspects the installed signature and drops
  keywords it does not accept, raising a clear error only when a keyword the annotation
  view genuinely depends on has disappeared.
- An incompatible Streamlit / canvas pair took the whole application down at start-up:
  the component now fails at import with `StreamlitAPIException` rather than
  `ImportError`, which the shim did not catch. The canvas is reported as unavailable
  instead, and phase 2 remains usable.
- The `image_to_url` patch is no longer applied to canvas 0.10+, which never calls it.

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
