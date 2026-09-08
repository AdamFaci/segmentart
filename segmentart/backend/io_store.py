# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""Local input/output: images and annotations under the `data/` directories.

No UI dependency. Loads images from `data/images/`, loads and saves annotations
from and to `data/annotations/`, and persists images locally.
"""

import json
import os

import numpy as np
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def list_images(images_dir):
    """Names of the image files found in `images_dir` (sorted)."""
    if not os.path.isdir(images_dir):
        return []
    return sorted(f for f in os.listdir(images_dir)
                  if os.path.splitext(f)[1].lower() in IMG_EXT)


def load_image_array(path):
    """Load an image as an RGB array."""
    return np.array(Image.open(path).convert("RGB"))


def save_image_array(arr, path):
    """Write an image array to disk (creating the directory if needed)."""
    ensure_dir(os.path.dirname(path))
    Image.fromarray(np.asarray(arr).astype("uint8")).save(path)


def list_annotation_files(ann_dir):
    if not os.path.isdir(ann_dir):
        return []
    return sorted(f for f in os.listdir(ann_dir) if f.lower().endswith(".json"))


def save_text(text, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path):
    """Read a CSV into a list of dicts (DictReader)."""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def round_plan(path, round_n):
    """Return the ordered (image, pipeline) list for round `round_n`.

    Accepts either an `assignment.csv` (columns round1..N) or a
    `worksheet_round{n}.csv` (column `pipeline`)."""
    rows = read_csv_rows(path)
    if not rows:
        return []
    cols = set(rows[0].keys())
    rcol = f"round{round_n}"
    out = []
    for r in rows:
        if rcol in cols and r.get(rcol):
            out.append((r["image"], r[rcol]))
        elif "pipeline" in cols and r.get("pipeline"):
            out.append((r["image"], r["pipeline"]))
    return out
