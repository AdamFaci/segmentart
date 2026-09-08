# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Multi-level semantic annotation from a SINGLE descriptor space — no UI.

The single label is replaced by a structured description (category ↔
subcategory ↔ direction ↔ part attributes), obtained by querying *the same*
region-conditioned descriptor at several granularities and matching it few-shot
against banks of exemplars.

One mechanism (similarity to prototypes) yields several semantic levels,
depending on the granularity of the descriptor:
  - category / supercategory   ← global descriptor;
  - direction (gaze/motion)    ← left|right descriptor (asymmetry);
  - structural subcategory     ← quadrant-local descriptor;
  - part attribute             ← prototype of the part.

Pure NumPy logic, testable without a model.
"""

import numpy as np

from segmentart.backend import fewshot as fs
from segmentart.backend import hierarchical as H


def lr_descriptor(feat_map, mask):
    """Left|right descriptor: the prototypes of the left and the right half,
    concatenated (this discriminates direction: a feature at the front of the
    object switches side)."""
    reg = H.quadrant_regions(mask)
    h, w, d = feat_map.shape
    out = []
    for side in ("left", "right"):
        r = reg.get(side)
        if r is None:
            out.append(np.zeros(d, np.float32)); continue
        if r.shape[:2] != (h, w):
            r = fs.downsample_mask(r, (h, w))
        out.append(fs.masked_prototype(feat_map, r) if r.any() else np.zeros(d, np.float32))
    return np.concatenate(out).astype(np.float32)


# Descriptors available per level (granularity → function(feat_map, mask)).
DESCRIPTORS = {
    "whole": H.holistic_descriptor,
    "quadrants": H.part_local_descriptor,
    "lr": lr_descriptor,
}


def build_bank(exemplars, descriptor, feat_map_of):
    """Build a {label: prototype} bank for one level.
    `exemplars`: {label: [{'mask', …}]}; `feat_map_of(item) -> feat_map`."""
    bank = {}
    for label, items in exemplars.items():
        vecs = [fs._unit(descriptor(feat_map_of(it), it["mask"])) for it in items]
        if vecs:
            bank[label] = fs._unit(np.mean(vecs, axis=0))
    return bank


def semantic_annotate(feat_map, mask, banks, parts=None, part_attr_banks=None):
    """Multi-level annotation of one instance.

    `banks`: {level: {label: prototype}} with level ∈ {category, supercategory,
    subcategory, direction}. Each level uses its matching descriptor:
    category/supercategory→whole, subcategory→quadrants, direction→lr.
    `parts`: [(name, score, mask)]; `part_attr_banks`: {part_name: {label:
    prototype}} (prototype over the part). Returns a dict of (label, score)."""
    level_desc = {"category": "whole", "supercategory": "whole",
                  "subcategory": "quadrants", "direction": "lr"}
    out = {}
    for level, bank in banks.items():
        desc = DESCRIPTORS[level_desc.get(level, "whole")]
        out[level] = fs.classify(desc(feat_map, mask), bank)
    if parts and part_attr_banks:
        pa = {}
        for name, _score, pm in parts:
            if name in part_attr_banks:
                proto = fs.masked_prototype(feat_map, pm)
                pa[name] = fs.classify(proto, part_attr_banks[name])
        out["parts"] = pa
    return out


def to_phrase(annotation):
    """Render a multi-level annotation as a readable phrase (the "rich" output).
    Example: 'elephant (animal), facing right, head-up; leg: bent'."""
    bits = []
    cat = annotation.get("category")
    sup = annotation.get("supercategory")
    if cat:
        bits.append(f"{cat[0]}" + (f" ({sup[0]})" if sup else ""))
    if annotation.get("direction"):
        bits.append(annotation["direction"][0].replace("_", " "))
    if annotation.get("subcategory"):
        bits.append(annotation["subcategory"][0].replace("_", " "))
    head = ", ".join(bits)
    parts = annotation.get("parts") or {}
    tail = "; ".join(f"{k}: {v[0]}" for k, v in parts.items())
    return head + ("; " + tail if tail else "")
