# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Measures supporting the claim that "placing points is easier than tracing a
contour" — no UI dependency.

Two families of measures, both computable automatically once the masks have been
validated:

1. Tolerance surface — the "error margin" each method offers:
   - contour: a thin band around the boundary (an exact path must be followed);
   - points +: the whole surface of the object (a point anywhere inside is
     valid);
   - points −: the negative of the object within the box (a point anywhere in the
     background of the box is valid).
   The larger the allowed surface relative to the action area, the more
   "tolerant", and therefore the easier, the method is.

2. Jitter robustness — the variation in IoU when point positions are perturbed
   slightly (this requires re-running the predictor; a generic harness is
   provided).
"""

import numpy as np

from segmentart.backend import masks

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def tolerance_surfaces(mask, box=None, band_px=None):
    """Tolerance surfaces of a mask (and of its box).

    Returns a dict:
      - area_mask           : area of the object (tolerance of a + point)
      - area_box            : area of the box
      - area_box_negative   : background area inside the box (tolerance of a
                              − point)
      - area_contour_band   : area of a band of width `band_px` around the
                              boundary (tolerance for tracing the contour)
      - band_px             : band width used
      - tol_point_pos       : area_mask / area_box        (∈ [0, 1])
      - tol_point_neg       : area_box_negative / area_box (∈ [0, 1])
      - tol_contour         : area_contour_band / area_box
      - ease_ratio_pos      : tol_point_pos / tol_contour  (how much more tolerant
                              a + point is than the contour)
      - ease_ratio_neg      : tol_point_neg / tol_contour
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required.")
    m = np.ascontiguousarray(masks.ensure_dense(mask)).astype(np.uint8)
    h, w = m.shape
    area_mask = int((m > 0).sum())

    if box is not None:
        x0, y0, bw, bh = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x0 + bw), min(h, y0 + bh)
        area_box = max(1, (x1 - x0) * (y1 - y0))
        in_box = np.zeros_like(m)
        in_box[y0:y1, x0:x1] = 1
        area_box_negative = int(((in_box > 0) & (m == 0)).sum())
    else:
        area_box = h * w
        area_box_negative = area_box - area_mask

    # default band width: ~1.5 % of the diagonal of the box
    if band_px is None:
        band_px = max(2, int(0.015 * (area_box ** 0.5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * band_px + 1, 2 * band_px + 1))
    dil = cv2.dilate(m, kernel)
    ero = cv2.erode(m, kernel)
    area_contour_band = int(((dil > 0) & (ero == 0)).sum())

    tol_pos = area_mask / area_box
    tol_neg = area_box_negative / area_box
    tol_contour = area_contour_band / area_box
    return {
        "area_mask": area_mask, "area_box": int(area_box),
        "area_box_negative": area_box_negative,
        "area_contour_band": area_contour_band, "band_px": int(band_px),
        "tol_point_pos": round(tol_pos, 4), "tol_point_neg": round(tol_neg, 4),
        "tol_contour": round(tol_contour, 4),
        "ease_ratio_pos": round(tol_pos / tol_contour, 2) if tol_contour else None,
        "ease_ratio_neg": round(tol_neg / tol_contour, 2) if tol_contour else None,
    }


def _jitter_iou(predict_fn, base_mask, perturb_fn, n_trials, sigma_px, seed):
    rng = np.random.default_rng(seed)
    base = masks.ensure_dense(base_mask)
    ious = []
    for _ in range(n_trials):
        pm = perturb_fn(rng)
        if pm is None:
            continue
        ious.append(masks.mask_iou(base, pm))
    if not ious:
        return {"iou_mean": None, "iou_std": None, "n_trials": 0,
                "sigma_px": sigma_px, "ious": []}
    arr = np.array(ious)
    return {"iou_mean": round(float(arr.mean()), 4),
            "iou_std": round(float(arr.std()), 4),
            "n_trials": len(ious), "sigma_px": sigma_px,
            "ious": [round(float(v), 4) for v in arr]}


def point_jitter_iou(predict_fn, base_mask, box, pos_points, neg_points,
                     sigma_px=8.0, n_trials=10, seed=0):
    """Robustness to POINT displacement: perturbs each point with Gaussian noise
    of standard deviation `sigma_px`, re-runs `predict_fn`, and measures the IoU
    against the reference mask. `predict_fn(box, pos, neg) -> mask` wraps SAM2."""
    def jit(rng, points):
        return [(float(x + rng.normal(0, sigma_px)),
                 float(y + rng.normal(0, sigma_px))) for (x, y) in points]

    def perturb(rng):
        return predict_fn(box, jit(rng, pos_points or []), jit(rng, neg_points or []))

    return _jitter_iou(predict_fn, base_mask, perturb, n_trials, sigma_px, seed)


def box_jitter_iou(predict_fn, base_mask, box, pos_points, neg_points,
                   sigma_px=8.0, n_trials=10, seed=0):
    """Robustness to BOX perturbation: because SAM2 is itself robust, small
    changes to the corners of the box barely change the mask. Perturbs
    (x0, y0, w, h) with Gaussian noise and measures the IoU against the reference
    mask."""
    def perturb(rng):
        if box is None:
            return None
        x0, y0, w, h = box
        jb = (x0 + rng.normal(0, sigma_px), y0 + rng.normal(0, sigma_px),
              max(1.0, w + rng.normal(0, sigma_px)), max(1.0, h + rng.normal(0, sigma_px)))
        return predict_fn(jb, pos_points or [], neg_points or [])

    return _jitter_iou(predict_fn, base_mask, perturb, n_trials, sigma_px, seed)
