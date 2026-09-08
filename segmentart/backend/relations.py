# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Spatial relations and scene-graph construction (Visual Genome style) — no UI.

Geometric predicates (the baseline; learned relations are an extension).
Pure NumPy logic, testable.
"""

import numpy as np

from segmentart.backend import masks as M


def _dense(m):
    return M.ensure_dense(m)


def bbox_of(mask):
    m = _dense(mask)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def centroid(mask):
    m = _dense(mask)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def containment(inner, outer):
    """Fraction of the `inner` mask contained in `outer` (∈ [0, 1])."""
    a, b = _dense(inner), _dense(outer)
    ai = int(a.sum())
    return float((a & b).sum()) / ai if ai else 0.0


def adjacency(a, b, dilate=3):
    """True if the two masks touch (they overlap after dilation)."""
    try:
        import cv2
    except ImportError:
        return False
    A = _dense(a).astype(np.uint8)
    B = _dense(b).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    return bool((cv2.dilate(A, k) & B).any())


def relative_position(a, b):
    """Position of `a` relative to `b`: 'above' | 'below' | 'left of' |
    'right of', along the dominant axis between the centroids."""
    ca, cb = centroid(a), centroid(b)
    if ca is None or cb is None:
        return None
    dx, dy = ca[0] - cb[0], ca[1] - cb[1]
    if abs(dx) >= abs(dy):
        return "left of" if dx < 0 else "right of"
    return "above" if dy < 0 else "below"


def part_of_edges(instance_mask, part_named, contain_thr=0.6):
    """'part-of' relations: every part sufficiently contained in the instance.
    `part_named`: [(name, score, mask)]."""
    edges = []
    for name, _score, pm in part_named:
        if containment(pm, instance_mask) >= contain_thr:
            edges.append({"predicate": "part of", "part": name})
    return edges


def scene_graph(instances, with_relations=True, with_parts=True):
    """Build a Visual Genome style scene graph.

    `instances`: list of dicts {id, label, mask, parts?} where `parts` is
    [(name, score, mask)]. Returns {objects, relationships}: the objects carry
    their parts, and the relationships cover part-of (object↔part) as well as
    the spatial relations (object↔object)."""
    objects, rels = [], []
    for ins in instances:
        node = {"id": ins["id"], "label": ins["label"], "box": bbox_of(ins["mask"])}
        if with_parts and ins.get("parts"):
            node["parts"] = [{"name": n, "box": bbox_of(pm)} for n, _s, pm in ins["parts"]]
            for n, _s, pm in ins["parts"]:
                if containment(pm, ins["mask"]) >= 0.6:
                    rels.append({"subject": ins["id"], "predicate": "part of",
                                 "object": ins["label"], "part": n})
        objects.append(node)

    if with_relations:
        for i, a in enumerate(instances):
            for b in instances[i + 1:]:
                pred = relative_position(a["mask"], b["mask"])
                if pred:
                    rels.append({"subject": a["id"], "predicate": pred,
                                 "object": b["id"]})
                if adjacency(a["mask"], b["mask"]):
                    rels.append({"subject": a["id"], "predicate": "adjacent to",
                                 "object": b["id"]})
    return {"objects": objects, "relationships": rels}
