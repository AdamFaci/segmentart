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

## Demo

https://sharedocs.huma-num.fr/wl/?id=RzIjXYMeor3VCv1SzPxNelgREKjphksX&fmode=open


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

Python 3.11 or later. The whole procedure, from nothing to a running annotation server:

```bash
git clone https://github.com/AdamFaci/segmentart.git
cd segmentart

conda create --name segmentapp python=3.11
conda activate segmentapp

pip install -e ".[app,cv,semantic]"

# SAM 2 — not redistributed here, installed from its own repository
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Model config, expected in the launch directory
curl -L https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2/sam2_hiera_l.yaml \
     -o sam2_hiera_l.yaml

# Model weights (~900 MB)
mkdir -p checkpoints
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
     -o checkpoints/sam2_hiera_large.pt

./run.sh
```

Then open <http://localhost:8501>, or forward the port first if the server is remote —
see [Running the app](#running-the-app).

`conda` is not required; any Python 3.11 environment works. It is used here because the
SAM 2 install pulls in `torch`, which is easier to keep isolated.

### Extras

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

SAM 2 needs **two** files at run time, and neither is redistributed here:

| File | Default location | Setting in `segmentart/config.py` |
|---|---|---|
| Hydra model config | `sam2_hiera_l.yaml` in the launch directory | `SAM2_MODEL_CFG` |
| Model weights | `checkpoints/sam2_hiera_large.pt` | `SAM2_CHECKPOINT` |

Three things have to agree, and none of them is checked for you:

**The config must match the weights.** `sam2_hiera_large.pt` only loads with
`sam2_hiera_l.yaml`. Pairing it with `sam2_hiera_s.yaml` or `_t.yaml` fails at load with a
tensor shape mismatch, because the config declares the architecture the weights are
supposed to fill. If you switch to another size, change both files *and* both settings.

**Both paths are relative to the directory you launch from**, not to the repository. This
is the usual trap on a server: `streamlit run /path/to/segmentart/run.py` from your home
directory looks for `./sam2_hiera_l.yaml` and `./checkpoints/` *in your home directory*
and finds nothing. Either `cd` into the clone before launching, or make the two settings
absolute paths.

**A missing file is quiet.** SAM 2 is optional by design, so when it cannot load, the app
starts anyway and simply disables automatic segmentation — you get an interface that
draws boxes but never produces a mask, rather than an error. Check the pair before your
first session:

```bash
python -c "
import os
from segmentart import config
for label, path in ('config', config.SAM2_MODEL_CFG), ('weights', config.SAM2_CHECKPOINT):
    ok = os.path.exists(path)
    print(f'{label:8s} {path:40s} {\"found\" if ok else \"MISSING from \" + os.getcwd()}')
"
```

Run it from the same directory you launch the app from — that is the whole point of the
check.

`torch` and `torchvision` come with the SAM 2 install. On a GPU machine, install them
yourself first, matching your CUDA version, or you may end up with a CPU-only build and
wonder why segmentation takes seconds per prompt.

Without SAM 2, OpenCV or CLIP, the app still starts and cleanly disables the
corresponding features. `streamlit-drawable-canvas` is required for the annotation phase,
and must be **below 0.10**: the 0.10 rewrite changed the canvas object format and removed
the drawing modes this application uses. The pin in `pyproject.toml` handles this; an
unsupported version is refused at start-up with an explicit message rather than leaving
the annotation buttons silently inert.

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

Forward the port from your workstation, then start the app inside that SSH session:

```bash
ssh -L 8501:localhost:8501 <user>@<host>
cd segmentart && ./run.sh
```

and open <http://localhost:8501> in your local browser. `run.sh` runs in the foreground,
so closing the session stops the app; start it under `tmux` or `screen` if you want it to
survive:

```bash
tmux new -s segmentart
./run.sh
# Ctrl-b then d to detach, tmux attach -t segmentart to come back
```

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

Images and annotations are read from and written to `data/` — **relative to the launch
directory**, the same rule as the SAM 2 files. Launching from somewhere else gives you an
empty, apparently working application that quietly writes to a new `data/` beside
wherever you started it. Set `SEGMENTART_DATA` to an absolute path to make the location
independent of where you launch:

```bash
export SEGMENTART_DATA=/absolute/path/to/data
```

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

If you use SegmentART in academic work, please cite it:

in APA:

`
Faci, A., & Maronet, L. (2026). SegmentART: Assisted image annotation with SAM 2 and semantic grouping (Version 0.1.0) [Computer software]. https://github.com/AdamFaci/segmentart
`

or Bibtex:

```bibtex
@software{faci_maronet_segmentart_2026,
  author  = {Faci, Adam and Maronet, L{\'e}a},
  title   = {{SegmentART}: assisted image annotation with {SAM~2} and semantic grouping},
  year    = {2026},
  version = {0.1.0},
  license = {Apache-2.0},
  url     = {https://github.com/AdamFaci/segmentart}
}
```

---

## Source

https://github.com/AdamFaci/segmentart

---

## License

Source code is licensed under the **Apache License, Version 2.0** — see `LICENSE`.

Documentation (this file, `CONTRIBUTING.md`, `CHANGELOG.md`) is licensed under
**Creative Commons Attribution 4.0 International** — see `LICENSES/CC-BY-4.0.txt`.

Third-party dependencies keep their own licenses and are not redistributed here — see
`NOTICE`. Model weights (SAM 2, CLIP) are downloaded by the user from their providers and
are governed by those providers' terms.
