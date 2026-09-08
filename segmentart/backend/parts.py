# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Decomposition of an instance into parts (legs, wings, petals…) — no UI.

Two routes, which can be combined:
  1. unsupervised parts: clustering of the dense features under the mask of the
     instance (the DINOv2 features of an object cluster into semantic parts)
     → part masks;
  2. few-shot naming: each part is named by similarity to part prototypes
     (reuses `segmentart.backend.fewshot`).

Pure NumPy logic, testable without a model.
"""

import numpy as np

from segmentart.backend import fewshot as fs


def _kmeans(X, k, iters=30, seed=0):
    """Minimal k-means (NumPy) → (labels, centroids)."""
    rng = np.random.default_rng(seed)
    if len(X) <= k:
        return np.arange(len(X)) % k, X[:k]
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab, C


def feature_parts(feat_map, mask, n_parts=3, seed=0):
    """Split the instance into `n_parts` parts by clustering the features under
    the mask. Returns a list of boolean masks (at the `feat_map` resolution).

    `feat_map`: (h, w, d). `mask`: at image resolution (reduced automatically)
    or already (h, w)."""
    h, w, _ = feat_map.shape
    m = mask
    if m.shape[:2] != (h, w):
        m = fs.downsample_mask(m, (h, w))
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return []
    feats = feat_map[ys, xs]
    # normalized coordinates are appended: parts = spatial compactness + semantics
    coords = np.stack([xs / w, ys / h], 1).astype(np.float32)
    X = np.concatenate([fs._unit(feats), 0.5 * coords], axis=1)
    k = min(n_parts, len(np.unique(X.round(3), axis=0)))
    lab, _ = _kmeans(X, max(1, k), seed=seed)
    parts = []
    for j in range(lab.max() + 1 if len(lab) else 0):
        pm = np.zeros((h, w), bool)
        sel = lab == j
        pm[ys[sel], xs[sel]] = True
        if pm.any():
            parts.append(pm)
    return parts


def part_prototypes(support_parts, feat_map_of):
    """{part_name: [{'feat_map','mask'}]} → {part_name: aggregated prototype}.
    `feat_map_of(item)` returns the feature map of the exemplar."""
    protos = {}
    for name, items in support_parts.items():
        ps = [fs.masked_prototype(feat_map_of(it), it["mask"]) for it in items]
        if ps:
            protos[name] = fs.aggregate_prototypes(ps)
    return protos


def name_parts(part_masks, feat_map, prototypes, min_score=0.0):
    """Name each part by cosine similarity to part prototypes.
    Returns [(name, score, mask)]. If `prototypes` is empty, the generic names
    part_0, part_1, … are returned."""
    out = []
    for i, pm in enumerate(part_masks):
        if not prototypes:
            out.append((f"part_{i}", None, pm))
            continue
        proto = fs.masked_prototype(feat_map, pm)
        name, score = fs.classify(proto, prototypes)
        out.append((name if score >= min_score else "unknown", round(float(score), 4), pm))
    return out


def upsample_part(part_mask, image_hw):
    """Upsample a part mask from the feature grid back to image resolution."""
    from PIL import Image
    ih, iw = image_hw
    m = (part_mask > 0).astype(np.uint8) * 255
    big = np.asarray(Image.fromarray(m).resize((iw, ih), Image.NEAREST))
    return big > 127
