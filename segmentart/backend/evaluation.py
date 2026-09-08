# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Evaluation of masks against ground truth (no UI dependency).

Loads a ground truth (COCO or YOLO-seg), counts `n_shapes` per image, and matches
each predicted mask to the best ground-truth mask (greedy matching by IoU).

Common ground-truth structure: {image_basename: [ {"mask": dense|rle, "label": str} ]}.
"""

import glob
import json
import os

import numpy as np

from segmentart.backend import masks

# ── COCO segmentation to mask conversion ───────────────────────────────────────

def seg_to_mask(seg, h, w):
    """COCO segmentation → dense mask.
    `seg` may be: a list of polygons [[x,y,...], ...]; an RLE
    {"size","counts"} (counts as a list = raw, as a str = COCO-compressed)."""
    if isinstance(seg, dict) and "counts" in seg:
        rle = dict(seg)
        rle.setdefault("size", [h, w])
        if "fmt" not in rle:
            rle["fmt"] = "coco" if isinstance(rle.get("counts"), str) else "raw"
        return masks.decode_rle(rle)
    if isinstance(seg, list):
        polys = [list(map(round, p)) for p in seg]
        return masks.polygons_to_mask(polys, (h, w))
    raise ValueError("Unrecognized segmentation format.")


# ── Ground-truth loaders ───────────────────────────────────────────────────────

def load_coco_gt(path):
    """Load a ground truth in COCO format (RLE or polygon segmentation)."""
    with open(path, encoding="utf-8") as f:
        coco = json.load(f)
    imgs = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    out = {}
    for a in coco.get("annotations", []):
        im = imgs[a["image_id"]]
        name = os.path.basename(im["file_name"])
        h, w = im.get("height"), im.get("width")
        m = seg_to_mask(a["segmentation"], h, w)
        out.setdefault(name, []).append(
            {"mask": m, "label": cats.get(a.get("category_id"), "")})
    return out


def load_yolo_seg_gt(labels_dir, image_sizes, names=None):
    """Load a YOLO-seg ground truth: one .txt per image (class x1 y1 x2 y2 …,
    normalized coordinates). `image_sizes` = {basename: (h, w)}."""
    out = {}
    for txt in glob.glob(os.path.join(labels_dir, "*.txt")):
        stem = os.path.splitext(os.path.basename(txt))[0]
        match = [bn for bn in image_sizes if os.path.splitext(bn)[0] == stem]
        if not match:
            continue
        bn = match[0]
        h, w = image_sizes[bn]
        rows = []
        with open(txt, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7:
                    continue
                cls = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
                poly = []
                for i in range(0, len(coords) - 1, 2):
                    poly += [round(coords[i] * w), round(coords[i + 1] * h)]
                if len(poly) >= 6:
                    rows.append({"mask": masks.polygons_to_mask([poly], (h, w)),
                                 "label": (names or {}).get(cls, str(cls))})
        if rows:
            out[bn] = rows
    return out


def load_csv_gt(path, min_confidence=0.0, manual_only=False, label_col="motif",
                image_col="image_names", poly_col="polygon"):
    """Load a ground truth from a CSV holding one mask per row.

    Columns: `motif` (label), `polygon` ([[x, y], …] in absolute pixels),
    `image_names`, and optionally `confidence`, `worker_run_id`.
    The polygons are kept as they are; the dense mask is rasterized later at the
    resolution of the corresponding prediction (the image size is not stored in
    the CSV)."""
    import ast

    import pandas as pd
    df = pd.read_csv(path)
    if manual_only and "worker_run_id" in df.columns:
        df = df[df["worker_run_id"].astype(str) == "Manual"]
    if min_confidence > 0 and "confidence" in df.columns:
        df = df[df["confidence"].astype(float) >= min_confidence]
    out = {}
    for _, r in df.iterrows():
        name = os.path.basename(str(r[image_col]))
        poly = r[poly_col]
        if isinstance(poly, str):
            try:
                poly = ast.literal_eval(poly)
            except (ValueError, SyntaxError):
                continue
        if not poly or len(poly) < 3:
            continue
        out.setdefault(name, []).append({"polygon": poly, "label": str(r[label_col])})
    return out


def n_shapes_per_image(gt):
    """Number of ground-truth masks per image (the density covariate)."""
    return {name: len(v) for name, v in gt.items()}


# ── Prediction ↔ ground-truth matching ─────────────────────────────────────────

def match_and_iou(pred_masks, gt_masks, same_label=False,
                  pred_labels=None, gt_labels=None):
    """Greedy matching by decreasing IoU (at most one ground truth per prediction).

    Returns a list, aligned with `pred_masks`, of (gt_index | None, iou)."""
    n, m = len(pred_masks), len(gt_masks)
    iou = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (same_label and pred_labels and gt_labels
                    and pred_labels[i] != gt_labels[j]):
                continue
            iou[i, j] = masks.mask_iou(pred_masks[i], gt_masks[j])

    res = [(None, 0.0)] * n
    triples = sorted(((iou[i, j], i, j) for i in range(n) for j in range(m)),
                     reverse=True)
    used_pred, used_gt = set(), set()
    for val, i, j in triples:
        if val <= 0:
            break
        if i in used_pred or j in used_gt:
            continue
        res[i] = (j, float(val))
        used_pred.add(i)
        used_gt.add(j)
    return res


# ── Loading the predictions exported by SegmentART ─────────────────────────────

def load_predictions(export_path):
    """Load a SegmentART export (Visual Genome or COCO format) and return
    {image_name: [ {"object_id", "label", "mask": dense} ]}."""
    with open(export_path, encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    if isinstance(data, dict) and "images" in data and data.get("images") \
            and isinstance(data["images"][0], dict) and "objects" in data["images"][0]:
        # Visual Genome (export_standard "VisualGenome")
        for im in data["images"]:
            name = os.path.basename(im["image"])
            rows = []
            for o in im.get("objects", []):
                seg = o.get("segmentation")
                if seg is None:
                    continue
                rows.append({"object_id": o.get("object_id"),
                             "label": o.get("name", ""),
                             "mask": masks.ensure_dense(seg)})
            if rows:
                out[name] = rows
        return out

    if isinstance(data, dict) and "annotations" in data and "images" in data:
        # COCO (export_standard "COCO")
        id2name = {im["id"]: os.path.basename(im["file_name"]) for im in data["images"]}
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        for a in data["annotations"]:
            seg = a.get("segmentation")
            if seg is None:
                continue
            name = id2name.get(a["image_id"])
            out.setdefault(name, []).append({
                "object_id": a.get("id"),
                "label": cats.get(a.get("category_id"), ""),
                "mask": masks.ensure_dense(seg)})
        return out

    raise ValueError("Unrecognized export (expected Visual Genome or COCO).")


def _gt_masks(gts, shape):
    """Dense ground-truth masks: taken as they are when present, otherwise
    rasterized from the polygons at the image resolution `shape` (h, w), which
    comes from the predictions."""
    out = []
    for g in gts:
        if g.get("mask") is not None:
            out.append(g["mask"])
        elif g.get("polygon") is not None and shape is not None:
            flat = [c for xy in g["polygon"] for c in xy]
            out.append(masks.polygons_to_mask([flat], shape))
        else:
            out.append(None)
    return out


def evaluate(predictions, gt, same_label=False):
    """Compute the IoU of every predicted instance. Returns a list of dicts:
    image, object_id, label, iou, gt_label.

    If the ground truth is supplied as polygons (CSV loader), it is rasterized at
    the resolution of each image's predicted masks."""
    rows = []
    for name, preds in predictions.items():
        gts = gt.get(name, [])
        pm = [masks.ensure_dense(p["mask"]) for p in preds]
        shape = pm[0].shape if pm else None
        gm = _gt_masks(gts, shape)
        keep = [j for j, g in enumerate(gm) if g is not None]
        gm = [gm[j] for j in keep]
        gl = [gts[j]["label"] for j in keep]
        matches = match_and_iou(
            pm, gm, same_label=same_label,
            pred_labels=[p["label"] for p in preds], gt_labels=gl)
        for p, (gj, iou) in zip(preds, matches):
            rows.append({"image": name, "object_id": p["object_id"],
                         "label": p["label"], "iou": round(iou, 4),
                         "gt_label": gl[gj] if gj is not None else None})
    return rows
