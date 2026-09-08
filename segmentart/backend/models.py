# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""Annotation schema and the matching factory (no UI dependency).

An annotation is a dictionary. The public keys are stable and are serialized;
keys prefixed with "_" hold internal UI working state (never exported).

Public keys
-----------
    id            : int, unique identifier (across all images)
    label         : str, category
    box           : (x0, y0, w, h) in image coordinates, or None
    points_pos    : list of (x, y) — positive prompts
    points_neg    : list of (x, y) — negative prompts
    mask          : full-resolution boolean np.ndarray, or None
    attributes    : list of str (phase 2)
    region_desc   : str (phase 2, Visual Genome)
    group         : int, cluster identifier (phase 2)

Internal keys (UI)
------------------
    _epoch         : canvas remount counter
    _mode_locked   : mode picked by hand (disables the automatic switch)
    _mask_stack    : mask history (to undo SAM2 / OpenCV steps)
    _box_validated : the box has been validated
    _editing       : annotation currently being corrected
    _logits        : low-resolution logits from the last SAM2 run (re-entry)
    _candidates    : SAM2 candidate masks + scores + logits
    _cand_idx      : index of the selected candidate
"""


def new_annotation(ann_id, label):
    """Create an empty annotation with the given id and label."""
    return {
        "id": ann_id,
        "label": label,
        "box": None,
        "points_pos": [],
        "points_neg": [],
        "mask": None,
        "_epoch": 0,
        "_mode_locked": False,
        "_mask_stack": [],
        "_box_validated": False,
        "_editing": False,
        "_logits": None,
    }


def bump_epoch(ann):
    """Force a canvas remount (to re-inject initial_drawing)."""
    ann["_epoch"] = ann.get("_epoch", 0) + 1
