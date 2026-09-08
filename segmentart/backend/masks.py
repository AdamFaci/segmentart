# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""Mask operations: compact representation, post-processing, IoU.

No UI dependency.

Representation
--------------
A mask is either:
  - a dense boolean NumPy array (full resolution), or
  - a compact RLE dictionary {"size": [h, w], "counts": ..., "fmt": ...}.

The RLE form is **lossless** and several hundred times lighter than the dense
one (a scene object: ~KB vs ~MB). It is the default representation for
validated masks and for export; the dense form is rebuilt on demand (display,
IoU, post-processing) through `ensure_dense`.

OpenCV is optional (morphological post-processing, polygon contours).
pycocotools is optional (compressed COCO RLE); without it an equivalent
lossless raw RLE is used instead.
"""

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from pycocotools import mask as _coco_mask
    PYCOCO_AVAILABLE = True
except ImportError:
    PYCOCO_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Compact representation (lossless RLE) and accessors
# ═══════════════════════════════════════════════════════════════════════════════

def is_rle(m):
    """True if `m` is a mask in the compact RLE format."""
    return isinstance(m, dict) and "counts" in m and "size" in m


def encode_rle(mask):
    """Encode a dense mask as a compact lossless RLE (COCO convention,
    column-major: `counts` starts with a run of zeros)."""
    m = (np.asarray(mask) > 0)
    h, w = m.shape[:2]
    if PYCOCO_AVAILABLE:
        r = _coco_mask.encode(np.asfortranarray(m.astype(np.uint8)))
        counts = r["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return {"size": [int(h), int(w)], "counts": counts, "fmt": "coco"}
    flat = np.asfortranarray(m).reshape(-1, order="F").astype(np.uint8)
    idx = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], idx, [flat.size]))
    runs = np.diff(bounds).astype(int).tolist()
    if flat.size and flat[0] == 1:
        runs = [0] + runs          # COCO: the first run counts zeros
    return {"size": [int(h), int(w)], "counts": runs, "fmt": "raw"}


def decode_rle(rle):
    """Rebuild a dense boolean mask from a compact RLE."""
    h, w = rle["size"]
    if rle.get("fmt") == "coco":
        if not PYCOCO_AVAILABLE:
            raise RuntimeError("pycocotools is required to decode a 'coco' RLE.")
        counts = rle["counts"]
        if isinstance(counts, str):
            counts = counts.encode("ascii")
        return _coco_mask.decode({"size": [h, w], "counts": counts}).astype(bool)
    flat = np.zeros(h * w, dtype=np.uint8)
    val, pos = 0, 0
    for c in rle["counts"]:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return np.ascontiguousarray(flat.reshape((h, w), order="F")).astype(bool)


def ensure_dense(m, shape=None):
    """Return a dense boolean mask, whether the input is dense, RLE or None."""
    if m is None:
        return None
    if is_rle(m):
        return decode_rle(m)
    return np.asarray(m) > 0


def to_compact(mask):
    """Convert to a compact RLE (lossless). Idempotent if already compact."""
    if mask is None or is_rle(mask):
        return mask
    return encode_rle(mask)


def mask_area(m):
    """Area (pixel count) of a mask, avoiding a full decode where possible."""
    if m is None:
        return 0
    if is_rle(m):
        if m.get("fmt") == "coco" and PYCOCO_AVAILABLE:
            counts = m["counts"]
            if isinstance(counts, str):
                counts = counts.encode("ascii")
            return int(_coco_mask.area({"size": m["size"], "counts": counts}))
        val, area = 0, 0
        for c in m["counts"]:
            if val:
                area += c
            val ^= 1
        return int(area)
    return int((np.asarray(m) > 0).sum())


def mask_shape(m):
    """Dimensions (h, w) of a dense or RLE mask."""
    if is_rle(m):
        return tuple(m["size"])
    return tuple(np.asarray(m).shape[:2])


# ═══════════════════════════════════════════════════════════════════════════════
# Morphological post-processing (OpenCV)
# ═══════════════════════════════════════════════════════════════════════════════

def fill_mask(mask, method):
    """
    OpenCV post-processing of a mask, to obtain a "solid" mask.
    Methods:
      - "Fill holes"                 : fills the outer contours
      - "Morphological closing"      : closes small gaps / smooths the edges
      - "Convex hull"                : convex hull of each component
      - "Keep largest convex shape"  : largest component only, filled by its
                                       convex hull
    Accepts a dense or RLE mask; returns a dense boolean mask.
    """
    m = np.ascontiguousarray(ensure_dense(mask)).astype(np.uint8) * 255
    if method == "Morphological closing":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        out = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    elif method == "Convex hull":
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(m)
        for c in cnts:
            cv2.drawContours(out, [cv2.convexHull(c)], -1, 255, cv2.FILLED)
    elif method == "Keep largest convex shape":
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(m)
        if cnts:
            biggest = max(cnts, key=cv2.contourArea)
            cv2.drawContours(out, [cv2.convexHull(biggest)], -1, 255, cv2.FILLED)
    else:  # "Fill holes"
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(m)
        cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Mask <-> SAM2 logits conversion
# ═══════════════════════════════════════════════════════════════════════════════

def mask_to_logits(mask):
    """
    Convert a mask (dense or RLE) into low-resolution logits (1, 256, 256) that
    can be fed to SAM2 as `mask_input` (useful when the original logits are not
    available, e.g. after OpenCV post-processing).
    """
    m = np.ascontiguousarray(ensure_dense(mask)).astype(np.uint8) * 255
    small = np.asarray(Image.fromarray(m).resize((256, 256), Image.BILINEAR)) / 255.0
    logits = (small - 0.5) * 16.0      # ~ +/-8 in logit units
    return logits[None, :, :].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Polygon contours (an even lighter export option, but lossy)
# ═══════════════════════════════════════════════════════════════════════════════

def mask_to_polygons(mask, eps_frac=0.0):
    """Mask (dense or RLE) -> polygons [[x, y, ...], ...]. `eps_frac` > 0 enables
    Douglas-Peucker simplification (lossy). Requires OpenCV."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required to convert a mask to contours.")
    m = np.ascontiguousarray(ensure_dense(mask)).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if eps_frac > 0:
            eps = eps_frac * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, eps, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1).astype(int).tolist())
    return polys


def polygons_to_mask(polygons, shape):
    """Rebuild a boolean mask (h, w) from polygons."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required to rebuild a mask from contours.")
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(out, [pts], 1)
    return out > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def mask_iou(a, b):
    """IoU between two masks (dense or RLE) of the same size."""
    a = ensure_dense(a)
    b = ensure_dense(b)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0
