# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Hierarchical clustering conditioned on local parts of the mask — no UI.

To discover subcategories, instances are grouped not on a global descriptor of
the object but on **local** descriptors (top/bottom/left/right, or semantic
parts). Two subtypes that differ only locally (the head, say) are separated by a
local descriptor, where a global one merges them.

NumPy/SciPy logic, testable without a model.
"""

import numpy as np

from segmentart.backend import fewshot as fs


def quadrant_regions(mask):
    """Split the bounding box of the mask into 4 regions (top/bottom/left/right),
    intersected with the mask. Returns {name: region_mask}."""
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) == 0:
        return {}
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h, w = mask.shape
    box = np.zeros((h, w), bool)
    box[y0:y1 + 1, x0:x1 + 1] = True
    m = (np.asarray(mask) > 0)
    reg = {}
    top = np.zeros_like(box); top[y0:cy + 1, x0:x1 + 1] = True
    bot = np.zeros_like(box); bot[cy:y1 + 1, x0:x1 + 1] = True
    lft = np.zeros_like(box); lft[y0:y1 + 1, x0:cx + 1] = True
    rgt = np.zeros_like(box); rgt[y0:y1 + 1, cx:x1 + 1] = True
    for name, r in [("top", top), ("bottom", bot), ("left", lft), ("right", rgt)]:
        reg[name] = r & m
    return reg


def holistic_descriptor(feat_map, mask):
    """Global descriptor: prototype over the whole mask, shape (d,)."""
    return fs.masked_prototype(feat_map, mask)


def part_local_descriptor(feat_map, mask, regions=None):
    """Local descriptor: the per-region prototypes concatenated, shape (4·d,).
    `regions`: dict {name: mask}; the quadrants by default."""
    if regions is None:
        regions = quadrant_regions(mask)
    h, w, d = feat_map.shape
    blocks = []
    for name in sorted(regions):
        r = regions[name]
        if r.shape[:2] != (h, w):
            r = fs.downsample_mask(r, (h, w))
        if r.any():
            blocks.append(fs.masked_prototype(feat_map, r))
        else:
            blocks.append(np.zeros(d, np.float32))
    return np.concatenate(blocks).astype(np.float32)


def hierarchical_clusters(descriptors, n_clusters=2, method="ward"):
    """Agglomerative clustering (SciPy) of (N, D) descriptors.
    Returns (labels, linkage_matrix). The tree is cut at `n_clusters`."""
    from scipy.cluster.hierarchy import fcluster, linkage
    X = np.asarray(descriptors, np.float32)
    if len(X) < 2:
        return np.zeros(len(X), int), None
    Z = linkage(X, method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    return labels, Z


def cluster_purity(labels, truth):
    """Mean cluster purity against the ground truth (∈ [0, 1])."""
    labels, truth = np.asarray(labels), np.asarray(truth)
    total = 0
    for c in np.unique(labels):
        members = truth[labels == c]
        if len(members):
            vals, counts = np.unique(members, return_counts=True)
            total += counts.max()
    return round(total / len(labels), 4) if len(labels) else 0.0
