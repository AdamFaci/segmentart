# SegmentART

Assisted image annotation combining **manual prompts** (box, positive and negative
points) with **automatic segmentation by SAM 2**, followed by **semantic grouping**
and structured, Visual Genome style description.

The aim is to reduce the **time** and the **number of interactions** dense annotation
requires, and to produce **richer descriptions** — attributes, relations, hierarchies —
rather than flat labels.

The code is organised in **two decoupled layers**: a business layer (`backend/`) with no
dependency on the interface, and an interface layer (`frontend/`) built on Streamlit. The
business layer is a reusable integration surface: another frontend (Label Studio ML
backend, CVAT, Gradio) can sit on top of it.

---

## Contents

- [Installation](#installation)
- [Running the app](#running-the-app)
- [Usage](#usage)
- [Export formats](#export-formats)
- [Telemetry](#telemetry)
- [Architecture](#architecture)
- [Development](#development)
- [Citing](#citing)
- [License](#license)

---

## Installation

Python 3.11 or later.

```bash
pip install -e ".[app]"
```

The `app` extra pulls in Streamlit and the drawing canvas. Optional extras:

| Extra | Enables |
|---|---|
| `app` | the Streamlit interface and the annotation canvas |
| `cv` | OpenCV post-processing and compressed COCO RLE |
| `semantic` | CLIP embeddings for semantic grouping (phase 2) |
| `dev` | test runner and linter |

Installing the package **without** an extra gives you the business layer alone — useful
if you are driving it from your own code or another frontend. Its only hard dependencies
are NumPy and Pillow.

### SAM 2

SAM 2 is optional and is not redistributed here:

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

Place the checkpoint where `segmentart/config.py` expects it (by default
`checkpoints/sam2_hiera_large.pt`), or edit `SAM2_CHECKPOINT` and `SAM2_MODEL_CFG`.
Install `torch` and `torchvision` for your platform.

Without SAM 2, OpenCV or CLIP, the app still starts and cleanly disables the
corresponding features. `streamlit-drawable-canvas` is required for the annotation phase.

---

## Running the app

```bash
./run.sh                # port 8501
PORT=8888 ./run.sh      # another port
```

or directly:

```bash
streamlit run run.py
```

### On a remote machine

Start the app on the server, then forward the port from your workstation:

```bash
ssh -L 8501:localhost:8501 <user>@<host>
```

and open <http://localhost:8501>. Use `tmux` or `screen` on the server if you want the
app to survive the SSH session closing.

### Behind a Jupyter or JupyterHub proxy

```bash
pip install jupyter-server-proxy
streamlit run run.py \
    --server.port 8501 \
    --server.baseUrlPath "/user/<username>/proxy/8501/" \
    --server.enableCORS false \
    --server.enableXsrfProtection false
```

### Data location

Images and annotations are read from and written to `data/` relative to the launch
directory. Override with the `SEGMENTART_DATA` environment variable.

---

## Usage

Two phases, selected in the sidebar.

**Phase 1 — Annotation.** Load images, pick a category, drag a box, validate, then place
positive points followed by negative ones (the switch can be automatic). Run SAM 2 — a
single prompt is usually enough — browse the candidate masks, clean them up with OpenCV
if needed, and validate. The list of validated annotations, across all images, lets you
correct or delete any item.

**Phase 2 — Grouping and description.** SAM 2 is released from GPU memory. Validated
objects are grouped by similarity (CLIP, with a lightweight fallback), then described:
group name and description, per-object attributes, region descriptions, and
subject → predicate → object relations.

Boxes are drawn in blue, positive points in green, negative points in red.

---

## Export formats

| Standard | Contents |
|---|---|
| **VisualGenome** | Objects (box, attributes, group), region descriptions, relations, groups |
| **COCO** | `images`, `annotations` (bbox, area, attributes), `categories` |
| **Custom** | Same as Visual Genome, with user-defined free fields |

Validated masks are stored as **lossless compact RLE**, roughly two to three orders of
magnitude smaller than dense arrays, both in memory and on export. A dense mask is kept
only for the annotation **in progress** (editing, re-entering SAM 2) and is rebuilt on
demand by `backend.masks.ensure_dense` for display, IoU and post-processing. The default
export embeds the RLE, which is fully reconstructible; the COCO export writes it as
`segmentation`, as is idiomatic there. A polygon-contour export is also available: it is
lighter still, but **lossy**.

---

## Telemetry

An optional, non-blocking action log — disabled by default — records, per
**(image × pipeline)**, the elapsed time and the interactions: clicks, prompts, model
invocations, corrections, validations. Enable it and export JSONL from the "Telemetry"
section of the sidebar.

It exists so that annotation strategies can be compared quantitatively. It records
interaction events only. It does not collect personal data and sends nothing anywhere:
the log stays in the session until you export it.

---

## Architecture

```
.
├── run.py                          # launcher: streamlit run run.py
├── pyproject.toml
└── segmentart/
    ├── config.py                   # constants and colors (no UI dependency)
    ├── backend/                    # ── BUSINESS LAYER (no Streamlit) ──
    │   ├── models.py               #   annotation schema and factory
    │   ├── sam2_engine.py          #   SAM 2 loading and inference
    │   ├── masks.py                #   compact representation, RLE, IoU
    │   ├── maskops.py              #   advanced mask post-processing
    │   ├── embeddings.py           #   CLIP, with a lightweight fallback
    │   ├── clustering.py           #   crops, clustering, rule-based attributes
    │   ├── hierarchical.py         #   part-conditioned hierarchical clustering
    │   ├── descriptor.py           #   multi-level semantic annotation
    │   ├── fewshot.py              #   few-shot / one-shot core
    │   ├── parts.py                #   decomposition into parts
    │   ├── relations.py            #   spatial relations and scene graph
    │   ├── direction.py            #   orientation (gaze, motion)
    │   ├── orchestrator.py         #   per-stage pipeline configuration
    │   ├── serialization.py        #   export and import
    │   ├── evaluation.py           #   masks against ground truth
    │   ├── measures.py             #   input-cost measures
    │   ├── accounting.py           #   group-level accounting
    │   ├── io_store.py             #   local images and annotations
    │   └── telemetry.py            #   action logging
    └── frontend/                   # ── INTERFACE LAYER (Streamlit) ──
        ├── app.py                  #   orchestration of the two phases
        ├── canvas_compat.py        #   canvas compatibility shim
        ├── state.py                #   session state and accessors
        ├── resources.py            #   cached heavy resources
        ├── viz.py                  #   rendering, canvas parsing, thumbnails
        ├── annotate_view.py        #   phase 1
        └── grouping_view.py        #   phase 2
```

**Separation rule.** No module in `backend/` imports Streamlit: functions receive their
data explicitly as NumPy arrays, annotation dictionaries and lists. All coupling to
`st.session_state` is concentrated in `frontend/state.py`, and model caching
(`st.cache_resource`) stays in `frontend/resources.py`. A test enforces this.

---

## Development

```bash
pip install -e ".[app,cv,dev]"
pytest
```

See `CONTRIBUTING.md`.

---

## Citing

If you use SegmentART in academic work, please cite it. `CITATION.cff` holds the
machine-readable metadata; GitLab and GitHub render it as a citation block, and tools
such as `cffconvert` turn it into BibTeX.

---

## Source

https://github.com/AdamFaci/segmentart

---

## License

Source code is licensed under the **Apache License, Version 2.0** — see `LICENSE`.

Documentation (this file, `CONTRIBUTING.md`, `CHANGELOG.md`) is licensed under
**Creative Commons Attribution 4.0 International** — see `LICENSES/CC-BY-4.0.txt`.

Creative Commons [recommends against](https://creativecommons.org/faq/) applying CC
licenses to software, because they address neither patent grants nor source
distribution. Apache 2.0 covers the code and carries the same attribution requirement,
while CC BY covers the prose, which is what it was designed for.

Third-party dependencies keep their own licenses and are not redistributed here — see
`NOTICE`. Model weights (SAM 2, CLIP) are downloaded by the user from their providers and
are governed by those providers' terms.
