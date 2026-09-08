# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""Mask toolbox (advanced post-processing) — no UI dependency.

Every function accepts a dense or RLE mask (through `masks.ensure_dense`) and
returns a dense boolean mask. Most of them require OpenCV.
"""

import numpy as np

from segmentart.backend import masks

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ── Descriptions and assessment of the "solid mask" methods ────────────────────
# (Shown in the UI; the assessment guides the choice when working on reliefs.)
OPENCV_METHODS = {
    "Fill holes": {
        "desc": "Fills the interior of every shape: any pixel enclosed by the "
                "outer contour is added. Best when SAM leaves internal holes "
                "(highlights, worn stone).",
        "best_for": "High — the safest option; it does not distort the contour.",
    },
    "Morphological closing": {
        "desc": "Dilation then erosion: closes small gaps and slightly smooths "
                "the edge, without filling large holes.",
        "best_for": "Medium — useful on noisy edges; may weld together nearby "
                   "parts that were meant to stay separate.",
    },
    "Convex hull": {
        "desc": "Replaces each component by its convex hull (the smallest convex "
                "polygon containing it).",
        "best_for": "Low on concave shapes (animals, lotus flowers) — it spills "
                   "onto the background; use only on near-convex objects.",
    },
    "Keep largest convex shape": {
        "desc": "Keeps only the largest component, filled by its convex hull; "
                "drops stray fragments.",
        "best_for": "Situational — handy to remove noise, but it loses "
                   "multi-part objects and flattens concavities.",
    },
}


def _dense_u8(mask):
    return np.ascontiguousarray(masks.ensure_dense(mask)).astype(np.uint8)


# ── Edge smoothing ─────────────────────────────────────────────────────────────

def smooth_edges(mask, strength=2):
    """Smooth the mask outline (no staircase edges) without changing its topology.

    Blurs the binary map then re-thresholds it: edges become regular while the
    overall area is preserved. `strength` ~ smoothing radius in pixels."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required.")
    m = _dense_u8(mask) * 255
    k = max(1, int(strength)) * 2 + 1
    blurred = cv2.GaussianBlur(m, (k, k), 0)
    return blurred > 127


# ── Controlled hole filling (size / +/- points) ────────────────────────────────

def fill_holes(mask, max_hole_frac=None, pos_points=None, neg_points=None):
    """Fill internal holes, with safeguards:

    - a hole whose area exceeds `max_hole_frac` x the mask area is NOT filled
      (the "fill the holes, but not the large ones" option);
    - a hole containing a negative point stays open (deliberately excluded area);
    - a hole containing a positive point is always filled.
    `pos_points` / `neg_points`: lists of (x, y) in image coordinates."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required.")
    m = _dense_u8(mask)
    h, w = m.shape
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    holes = (filled > 0) & (m == 0)                    # internal holes
    if not holes.any():
        return m > 0

    n_lab, lab = cv2.connectedComponents(holes.astype(np.uint8))
    total = max(1, int((m > 0).sum()))
    pos_points = pos_points or []
    neg_points = neg_points or []

    def _labels_at(points):
        out = set()
        for x, y in points:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < h and 0 <= xi < w and lab[yi, xi] > 0:
                out.add(int(lab[yi, xi]))
        return out

    neg_labels = _labels_at(neg_points)
    pos_labels = _labels_at(pos_points)

    out = (m > 0).copy()
    for li in range(1, n_lab):
        hole = lab == li
        if li in neg_labels:
            continue                                    # keep it open
        if li in pos_labels:
            out |= hole
            continue
        if max_hole_frac is not None and hole.sum() > max_hole_frac * total:
            continue                                    # too large -> leave it
        out |= hole
    return out


# ── Connect two masks ──────────────────────────────────────────────────────────

def connect_masks(mask_a, mask_b, thickness=None):
    """Merge two masks and, if they are disjoint, draw a bridge between their
    closest points (useful to reattach an object that SAM split in two)."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required.")
    a = masks.ensure_dense(mask_a)
    b = masks.ensure_dense(mask_b)
    union = (a | b)
    n_lab, _ = cv2.connectedComponents(union.astype(np.uint8))
    if n_lab <= 2:                                      # already connected (1 bg + 1 object)
        return union
    ys_a, xs_a = np.nonzero(a)
    ys_b, xs_b = np.nonzero(b)
    if len(xs_a) == 0 or len(xs_b) == 0:
        return union
    # closest pair of points (subsampled to stay fast)
    pa = np.stack([xs_a, ys_a], 1)
    pb = np.stack([xs_b, ys_b], 1)
    if len(pa) > 2000:
        pa = pa[np.random.default_rng(0).choice(len(pa), 2000, replace=False)]
    if len(pb) > 2000:
        pb = pb[np.random.default_rng(0).choice(len(pb), 2000, replace=False)]
    d2 = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(d2.argmin()), d2.shape)
    p1, p2 = tuple(pa[i]), tuple(pb[j])
    out = union.astype(np.uint8)
    th = thickness or max(2, int(0.01 * max(out.shape)))
    cv2.line(out, p1, p2, 1, th)
    return out > 0


# ── Prior mask (box + points) to seed SAM2 ─────────────────────────────────────

def basic_mask_from_prompts(shape, box=None, pos_points=None, neg_points=None,
                            radius_frac=0.06):
    """Build a coarse mask from the box and the points, meant to be passed to
    SAM2 as `mask_input` (through `masks.mask_to_logits`).

    Heuristic: the ellipse inscribed in the box (or disks around the positive
    points when there is no box), unioned with disks at the positive points;
    disks at the negative points are then subtracted. `shape` = (h, w)."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    pos_points = pos_points or []
    neg_points = neg_points or []
    r = max(3, int(radius_frac * min(h, w)))
    if box is not None and CV2_AVAILABLE:
        x0, y0, bw, bh = [int(round(v)) for v in box]
        cx, cy = int(x0 + bw / 2), int(y0 + bh / 2)
        cv2.ellipse(m, (cx, cy), (max(1, bw // 2), max(1, bh // 2)), 0, 0, 360, 1, -1)
    for x, y in pos_points:
        if CV2_AVAILABLE:
            cv2.circle(m, (int(x), int(y)), r, 1, -1)
        else:
            m[max(0, int(y) - r):int(y) + r, max(0, int(x) - r):int(x) + r] = 1
    for x, y in neg_points:
        if CV2_AVAILABLE:
            cv2.circle(m, (int(x), int(y)), r, 0, -1)
        else:
            m[max(0, int(y) - r):int(y) + r, max(0, int(x) - r):int(x) + r] = 0
    return m > 0


# ── Alternative mask: interior removed (ring) ──────────────────────────────────

def remove_central_part(mask, frac=0.5):
    """Return the mask with its central part removed (an edge ring).

    Erodes the mask to obtain a "core", then subtracts it. `frac` in ]0, 1[
    controls how much is removed (larger = wider core removed)."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required.")
    m = _dense_u8(mask)
    area = int((m > 0).sum())
    if area == 0:
        return m > 0
    # erosion radius ~ a fraction of the object's "size"
    eq_r = (area / np.pi) ** 0.5
    k = max(1, int(0.5 * frac * eq_r))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    core = cv2.erode(m, kernel)
    return (m > 0) & (core == 0)
