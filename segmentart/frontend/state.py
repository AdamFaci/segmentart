# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Streamlit session state: initialisation and accessors.

Centralises all coupling to `st.session_state`. The backend layer never touches
it: data is passed to it explicitly.
"""

import streamlit as st

from segmentart import config
from segmentart.backend import models
from segmentart.backend.telemetry import Telemetry


def init_state():
    defaults = dict(
        images=[],                 # [{"name": str, "array": np.ndarray}]
        image_index=0,
        annotations=[],            # per image: list of annotations
        current_annotation=None,
        mode="box",                # "box" | "pos" | "neg"
        categories=config.DEFAULT_CATEGORIES.copy(),
        n_pos_auto=config.N_POS_AUTO,
        sam_loaded=False,
        _ann_counter=0,
        hi_res=False,
        show_validated=False,
        _bg=None,
        # ── Settings that reduce the number of interactions ────────────────────
        run_twice=True,         # re-run SAM2 on its own mask (better result)
        auto_postproc=True,     # auto chain: edge smoothing + hole filling
        auto_sam_neg=0,         # nb of − points auto-triggering SAM2 (0 = off)
        auto_validate_box=False,  # validate the box as soon as it is drawn (no button)
        poly_drop_last=True,    # ignore the last vertex of a polygon
        # ── Phase 2 ────────────────────────────────────────────────────────────
        phase="annot",             # "annot" | "group"
        standard="VisualGenome",   # "VisualGenome" | "COCO" | "Custom"
        custom_fields=["material", "condition"],
        group_meta={},
        relations=[],
        _emb_cache={},
        # ── Telemetry (optional) ───────────────────────────────────────────────
        telemetry_on=False,
        active_pipeline="P1",
        round_plan={},          # guided collection: {image: pipeline} for this round
        round_id=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "_telemetry" not in st.session_state:
        st.session_state["_telemetry"] = Telemetry(enabled=False)
    # Keep activation in sync with the UI toggle
    st.session_state["_telemetry"].enabled = st.session_state.get("telemetry_on", False)


# ── Image / annotation accessors ───────────────────────────────────────────────

def cur_anns():
    idx = st.session_state.image_index
    while len(st.session_state.annotations) <= idx:
        st.session_state.annotations.append([])
    return st.session_state.annotations[idx]


def cur_arr():
    imgs = st.session_state.images
    return imgs[st.session_state.image_index]["array"] if imgs else None


def cur_name():
    imgs = st.session_state.images
    return imgs[st.session_state.image_index]["name"] if imgs else ""


def all_annotations():
    """Return [(img_index, annotation)] across every image."""
    out = []
    for i in range(len(st.session_state.images)):
        anns = st.session_state.annotations[i] if i < len(st.session_state.annotations) else []
        for a in anns:
            out.append((i, a))
    return out


def arr_of(img_index):
    return st.session_state.images[img_index]["array"]


# ── Annotation factory (handles the id counter) ────────────────────────────────

def make_annotation(label):
    st.session_state._ann_counter += 1
    return models.new_annotation(st.session_state._ann_counter, label)


# ── Telemetry ──────────────────────────────────────────────────────────────────

def telemetry():
    return st.session_state["_telemetry"]


def log_event(event, **fields):
    """Shortcut: log an event against the current image and pipeline."""
    telemetry().log(event, image=cur_name(),
                    pipeline=st.session_state.get("active_pipeline"), **fields)
