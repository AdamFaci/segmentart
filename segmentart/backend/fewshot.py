# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Few-shot / one-shot core for annotation by example — with no dependency on the
UI, nor on a particular model (features may come from DINOv2, the SAM encoder,
and so on).

Implements the logic shared by PerSAM / Matcher / ProtoSAM:
  1. prototype of a concept = mean of the features under the reference mask;
  2. cosine similarity map between the prototype and the feature map of a query
     image;
  3. positive-negative location prior (points of max / min similarity) →
     prompts for SAM2.
Plus the k-shot protocol helpers and the score aggregation utilities (mIoU,
curve against the number of examples).

All the logic below is pure NumPy and testable without a model.
"""

import numpy as np


def _unit(v, axis=-1, eps=1e-8):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def downsample_mask(mask, hw):
    """Reduce an (H, W) mask to the (h, w) resolution of the feature map (nearest
    neighbor); returns an (h, w) boolean array."""
    from PIL import Image
    h, w = hw
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    small = np.asarray(Image.fromarray(m).resize((w, h), Image.NEAREST))
    return small > 127


def masked_prototype(feat_map, mask):
    """Prototype of a concept: L2-normalized mean of the (h, w, d) features under
    the mask. `mask` may be at image resolution (it is reduced automatically)."""
    h, w, _ = feat_map.shape
    m = mask
    if m.shape[:2] != (h, w):
        m = downsample_mask(m, (h, w))
    sel = feat_map[m]
    if len(sel) == 0:
        sel = feat_map.reshape(-1, feat_map.shape[-1])
    proto = sel.mean(axis=0)
    return _unit(proto)


def aggregate_prototypes(protos):
    """Aggregate several prototypes (k-shot) into one (L2-normalized mean)."""
    P = np.stack([_unit(p) for p in protos])
    return _unit(P.mean(axis=0))


def similarity_map(feat_map, prototype):
    """Cosine similarity map (h, w) between each feature pixel and the
    prototype."""
    f = _unit(feat_map, axis=-1)
    p = _unit(prototype)
    return f @ p


def classify(embedding, prototypes):
    """Classify an embedding by cosine similarity to {class: prototype}.
    Returns (class, score)."""
    e = _unit(np.asarray(embedding, dtype=np.float32))
    best, best_s = None, -2.0
    for cls, proto in prototypes.items():
        s = float(e @ _unit(proto))
        if s > best_s:
            best, best_s = cls, s
    return best, best_s


def location_prior(sim, n_pos=1, n_neg=1, min_dist=1):
    """PerSAM location prior: the points of highest similarity (positive) and of
    lowest similarity (negative), in feature-map coordinates (col=x, row=y).
    Rescaling them to image coordinates is left to the caller.

    `min_dist`: minimum Chebyshev distance between the positive points kept."""
    h, w = sim.shape
    order = np.argsort(sim, axis=None)[::-1]      # descending
    pos = []
    for idx in order:
        y, x = divmod(int(idx), w)
        if all(max(abs(x - px), abs(y - py)) >= min_dist for px, py in pos):
            pos.append((x, y))
        if len(pos) >= n_pos:
            break
    neg = []
    for idx in order[::-1]:                       # lowest first
        y, x = divmod(int(idx), w)
        neg.append((x, y))
        if len(neg) >= n_neg:
            break
    return pos, neg


def scale_points(points, feat_hw, image_hw):
    """Convert (x, y) points from the feature grid to the image grid (cell
    center)."""
    fh, fw = feat_hw
    ih, iw = image_hw
    sx, sy = iw / fw, ih / fh
    return [((x + 0.5) * sx, (y + 0.5) * sy) for (x, y) in points]


# ── k-shot protocol ────────────────────────────────────────────────────────────

def kshot_split(items_by_class, k, seed=0, stratify_key=None):
    """Split the items into a support and a query set, per class.

    `items_by_class`: {class: [item, …]} where item is any dict
    (e.g. {"image":…, "polygon":…}). `stratify_key`: optional key (e.g.
    "image") used to diversify the support set. Returns (support, query) in the
    same per-class dict format."""
    rng = np.random.default_rng(seed)
    support, query = {}, {}
    for cls, items in items_by_class.items():
        items = list(items)
        if stratify_key:
            # prefer support items that come from different images
            seen, ordered, rest = set(), [], []
            for it in rng.permutation(len(items)):
                v = items[it].get(stratify_key)
                (ordered if v not in seen else rest).append(items[it])
                seen.add(v)
            items = ordered + rest
        else:
            items = [items[i] for i in rng.permutation(len(items))]
        kk = min(k, max(0, len(items) - 1)) if len(items) > 1 else 0
        support[cls] = items[:kk]
        query[cls] = items[kk:]
    return support, query


# ── Result aggregation ─────────────────────────────────────────────────────────

def miou_by_class(rows, iou_key="iou", class_key="label"):
    """Per-class mIoU plus the macro average. `rows`: dicts holding IoU + class."""
    per = {}
    for r in rows:
        per.setdefault(r[class_key], []).append(float(r[iou_key]))
    out = {c: round(float(np.mean(v)), 4) for c, v in per.items()}
    macro = round(float(np.mean(list(out.values()))), 4) if out else None
    return out, macro


def shots_curve(macro_by_k):
    """Sort {k: macro_mIoU} into two lists (ks, values), to plot the
    "quality vs number of examples" curve."""
    ks = sorted(macro_by_k)
    return ks, [macro_by_k[k] for k in ks]


# ── Active learning selection ──────────────────────────────────────────────────

def prediction_margin(sims):
    """Top-1 − top-2 margin of a similarity vector (or dict). Low = uncertain."""
    vals = np.sort(np.array(list(sims.values()) if isinstance(sims, dict)
                            else sims, dtype=float))[::-1]
    return float(vals[0] - vals[1]) if len(vals) >= 2 else float(vals[0])


def select_uncertain(margin_by_id, k):
    """Return the `k` most uncertain ids (lowest margin), the ones a human should
    check first (active learning)."""
    return [i for i, _ in sorted(margin_by_id.items(), key=lambda kv: kv[1])[:k]]
