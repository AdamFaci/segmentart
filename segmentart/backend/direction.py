# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Direction / orientation (gaze, motion, direction of growth) — no UI.

A **mirror-equivariant** representation of direction. Three pieces:
  - few-shot direction classification through the left|right descriptor;
  - a CONTINUOUS orientation score (−1 left … +1 right), by projecting the
    left/right asymmetry onto an axis anchored by exemplars;
  - **mirror consistency**: flipping the image must invert the predicted
    direction — a SELF-SUPERVISED signal (no labels) that serves both as a
    validation measure and as a *training loss* for a learned direction head
    (equivariance).

Pure NumPy logic, testable without a model.
"""

import numpy as np

from segmentart.backend import descriptor as D
from segmentart.backend import fewshot as fs

# Direction labels and their horizontal mirror.
MIRROR = {"facing_left": "facing_right", "facing_right": "facing_left",
          "left": "right", "right": "left"}


def classify_direction(feat_map, mask, dir_bank):
    """Few-shot direction from the left|right descriptor. Returns (label, score)."""
    return fs.classify(D.lr_descriptor(feat_map, mask), dir_bank)


def orientation_axis(dir_bank, left_label="facing_left", right_label="facing_right"):
    """Orientation axis = prototype(right) − prototype(left), normalized.
    An instance is projected onto it to obtain a continuous score."""
    if left_label not in dir_bank or right_label not in dir_bank:
        return None
    return fs._unit(dir_bank[right_label] - dir_bank[left_label])


def orientation_score(feat_map, mask, axis):
    """Continuous orientation score ∈ [−1, 1] (negative = left, positive = right),
    obtained by projecting the left|right descriptor onto the axis."""
    if axis is None:
        return 0.0
    v = fs._unit(D.lr_descriptor(feat_map, mask))
    return float(np.clip(v @ axis, -1.0, 1.0))


def _flip(feat_map, mask):
    """Horizontal flip of a feature map and of its mask (simulates the flipped
    image as seen by the extractor)."""
    return feat_map[:, ::-1, :].copy(), (np.asarray(mask)[:, ::-1]).copy()


def mirror_consistency(instances, dir_bank, flip_fn=None):
    """Fraction of the instances whose predicted direction flips correctly when
    the image is mirrored (mirror equivariance). `instances`:
    [{'feat_map','mask'}]. `flip_fn(feat_map, mask) -> (feat_map_flip,
    mask_flip)` (default: horizontal flip of the array; on real data, re-extract
    the features from the flipped image)."""
    flip_fn = flip_fn or _flip
    ok = 0
    n = 0
    for it in instances:
        lab, _ = classify_direction(it["feat_map"], it["mask"], dir_bank)
        ff, mf = flip_fn(it["feat_map"], it["mask"])
        lab_f, _ = classify_direction(ff, mf, dir_bank)
        if lab in MIRROR:
            n += 1
            ok += int(lab_f == MIRROR[lab])
    return round(ok / n, 4) if n else None
