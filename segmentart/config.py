# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Shared constants and color helpers (no UI dependency)."""

import os

# ── Project directories (local load / save) ────────────────────────────────────
# Overridable through environment variables; relative to the launch directory
# (the one that contains run.py).
DATA_DIR = os.environ.get("SEGMENTART_DATA", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")

# ── Default categories ─────────────────────────────────────────────────────────
# Example values only: replace them with the categories of your own dataset.
DEFAULT_CATEGORIES = ["elephant", "lotus", "lion"]

# ── Palette / colors ───────────────────────────────────────────────────────────
PALETTE = ["#1565C0", "#6A1B9A", "#F57F17", "#00695C",
           "#AD1457", "#4E342E", "#37474F", "#00838F"]
BOX_COLOR = "#1E88E5"   # blue  — boxes
POS_COLOR = "#43A047"   # green — positive points
NEG_COLOR = "#E53935"   # red   — negative points

# Point fill colors used by the canvas. Positive and negative points are then
# told apart by reading the dominant color of the fill the canvas returns.
POS_FILL = "rgba(67,160,71,0.85)"
NEG_FILL = "rgba(229,57,53,0.85)"
BOX_FILL = "rgba(30,136,229,0.12)"

MASK_ALPHA = 110        # 0-255, opacity of the mask overlay
POINT_R = 7             # display radius of the canvas points

# ── Display ────────────────────────────────────────────────────────────────────
DISPLAY_LIGHT = 640     # display width in light (fast) mode
DISPLAY_FULL = 1100     # display width in high resolution (occasional)

# ── Annotation behavior ────────────────────────────────────────────────────────
N_POS_AUTO = 3          # positive points before switching to negative mode

# ── SAM2 ───────────────────────────────────────────────────────────────────────
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CFG = "sam2_hiera_l.yaml"

# ── Mask storage ───────────────────────────────────────────────────────────────
# Validated masks are stored as compact RLE (lossless): ~100–600× lighter than
# dense masks (GB -> MB). Dense masks are kept only for the annotation IN
# PROGRESS (editing / feeding back into SAM); they are rebuilt on demand through
# `backend.masks.ensure_dense` for display, IoU and post-processing.
# The default export embeds the RLE, from which the mask can be rebuilt.
MASK_EXPORT_FORMAT = "rle"        # "rle" | "none" | "dense"
# Polygon contours: an even lighter export option, but LOSSY.
CONTOUR_EPS_FRAC = 0.005          # simplification tolerance (fraction of the perimeter)


def color_for_label(categories, label):
    """Stable color for a label, from its position in `categories`."""
    idx = categories.index(label) if label in categories else 0
    return PALETTE[idx % len(PALETTE)]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
