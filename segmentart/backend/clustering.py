# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Object grouping and rule-based characterization (no UI dependency)."""

import numpy as np
from PIL import Image

from segmentart.backend import embeddings as emb
from segmentart.backend import masks

# ── Object crops ───────────────────────────────────────────────────────────────

def object_crop(ann, base_array, masked=True, bg=(255, 255, 255)):
    """RGB crop of the object (over its box); the pixels outside the mask are
    optionally replaced by the color `bg` (e.g. (0, 0, 0) for a black
    background)."""
    if ann.get("box") is None:
        return None
    x0, y0, w, h = ann["box"]
    ih, iw = base_array.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(iw, x0 + w), min(ih, y0 + h)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = base_array[y0:y1, x0:x1].copy()
    if masked and ann.get("mask") is not None:
        dense = masks.ensure_dense(ann["mask"])
        m = (dense[y0:y1, x0:x1] > 0)
        crop[~m] = bg
    return Image.fromarray(crop).convert("RGB")


def compute_embedding(ann, base_array, clip, masked=True):
    """(img_emb, txt_emb) of one annotation. txt_emb is None in the light
    fallback."""
    crop = object_crop(ann, base_array, masked=masked) or Image.fromarray(base_array)
    if clip is not None:
        return emb.clip_embeddings(clip, crop, ann.get("label"))
    return emb.light_embedding(crop), None


# ── Clustering ─────────────────────────────────────────────────────────────────

def cluster_by_threshold(vecs, thr):
    """Group by connected components, linking pairs of cosine similarity ≥ thr."""
    n = len(vecs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    M = np.stack(vecs) if n else np.zeros((0, 1))
    sim = M @ M.T if n else None
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= thr:
                union(i, j)
    roots = {}
    labels = []
    for i in range(n):
        r = find(i)
        roots.setdefault(r, len(roots))
        labels.append(roots[r])
    return labels


def cluster_by_count(vecs, k):
    """Group into EXACTLY k clusters (agglomerative, average linkage over
    centroids). No external dependency; n is small (the number of
    annotations)."""
    n = len(vecs)
    if n == 0:
        return []
    k = max(1, min(int(k), n))
    M = np.stack(vecs)
    clusters = [[i] for i in range(n)]
    while len(clusters) > k:
        cents = []
        for c in clusters:
            v = M[c].mean(axis=0)
            cents.append(v / (np.linalg.norm(v) or 1.0))
        best_s, best_ij = -1e9, (0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = float(cents[i] @ cents[j])
                if s > best_s:
                    best_s, best_ij = s, (i, j)
        i, j = best_ij
        clusters[i] += clusters[j]
        clusters.pop(j)
    labels = [0] * n
    for ci, c in enumerate(clusters):
        for idx in c:
            labels[idx] = ci
    return labels


# ── Rule-based attributes (annotation aid) ─────────────────────────────────────

# Reference colors. The keys are the color names written to the exported schema.
_COLOR_REFS = {
    "red": (200, 40, 40), "green": (40, 160, 60), "blue": (40, 70, 200),
    "yellow": (220, 200, 40), "orange": (230, 140, 30), "purple": (140, 50, 180),
    "brown": (110, 70, 40), "white": (240, 240, 240), "black": (25, 25, 25),
    "gray": (128, 128, 128),
}


def _color_name(rgb):
    return min(_COLOR_REFS,
               key=lambda k: sum((a - b) ** 2 for a, b in zip(rgb, _COLOR_REFS[k])))


def rule_attributes(ann, base_array):
    """Automatic attributes: dominant color, relative size, position, shape.

    The returned keys and values are part of the exported annotation schema."""
    ih, iw = base_array.shape[:2]
    attrs = {}
    if ann.get("box"):
        x0, y0, w, h = ann["box"]
        x1, y1 = min(iw, x0 + w), min(ih, y0 + h)
        sub = base_array[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3)
        if ann.get("mask") is not None:
            dense = masks.ensure_dense(ann["mask"])
            m = dense[max(0, y0):y1, max(0, x0):x1].reshape(-1) > 0
            if m.any():
                sub = sub[m]
        if len(sub):
            attrs["color"] = _color_name(tuple(sub.mean(axis=0).astype(int)))
        frac = (w * h) / float(iw * ih)
        attrs["size"] = "small" if frac < 0.05 else "medium" if frac < 0.25 else "large"
        cx, cy = x0 + w / 2, y0 + h / 2
        vert = "top" if cy < ih / 3 else "bottom" if cy > 2 * ih / 3 else "center"
        horz = "left" if cx < iw / 3 else "right" if cx > 2 * iw / 3 else "center"
        attrs["position"] = vert if vert == horz else f"{vert}-{horz}"
        ar = w / float(h) if h else 1
        attrs["shape"] = "wide" if ar > 1.4 else "tall" if ar < 0.7 else "square"
    return attrs


def group_characterization(pairs):
    """(shared_attributes, differences) for one group.
    pairs = list of (annotation, image_array) — the members may come from
    different images."""
    attrs = [rule_attributes(a, arr) for a, arr in pairs]
    keys = ["color", "size", "position", "shape"]
    shared, diffs = {}, {}
    for k in keys:
        vals = [d.get(k) for d in attrs if d.get(k)]
        uniq = sorted(set(vals))
        if len(uniq) == 1:
            shared[k] = uniq[0]
        elif len(uniq) > 1:
            diffs[k] = uniq
    return shared, diffs
