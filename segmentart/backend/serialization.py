# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""Annotation serialization / import (no UI dependency).

The functions take the project state explicitly (images, annotations,
categories, and so on) instead of reading from a global state.
"""

import json

import numpy as np

from segmentart.backend import masks


def to_json(images, annotations, mask_format="rle"):
    """
    Simple per-image export. `images` = [{"name", "array"}],
    `annotations` = list (one entry per image) of annotation lists.

    `mask_format`:
      - "rle"   : compact RLE, lossless, fully reconstructible (default, light)
      - "none"  : shape and area only (lightest)
      - "dense" : full boolean array (very large — best avoided)
    """
    out = []
    for i, img_info in enumerate(images):
        anns = annotations[i] if i < len(annotations) else []
        serialised = []
        for a in anns:
            e = {k: a.get(k) for k in ("label", "box", "points_pos", "points_neg")}
            if a.get("mask") is not None:
                if mask_format == "dense":
                    e["mask"] = masks.ensure_dense(a["mask"]).tolist()
                elif mask_format == "none":
                    e["mask"] = {"shape": list(masks.mask_shape(a["mask"])),
                                 "area_px": masks.mask_area(a["mask"])}
                else:  # "rle"
                    e["mask"] = masks.to_compact(a["mask"])
            serialised.append(e)
        out.append({"image": img_info["name"], "annotations": serialised})
    return json.dumps(out, indent=2)


def export_standard(standard, images, annotations, categories,
                    custom_fields, relations, group_meta):
    """
    Serialize according to the chosen standard (VisualGenome / COCO / Custom).
    `annotations` is indexed by image; `relations` and `group_meta` are global
    (object ids are unique across all images).
    """
    def _anns(i):
        return annotations[i] if i < len(annotations) else []

    if standard == "COCO":
        out = {"images": [], "annotations": [], "categories": []}
        cats = {c: i + 1 for i, c in enumerate(categories)}
        out["categories"] = [{"id": i, "name": c} for c, i in cats.items()]
        ann_id = 1
        for img_i, info in enumerate(images):
            ih, iw = info["array"].shape[:2]
            out["images"].append({"id": img_i, "file_name": info["name"],
                                  "width": iw, "height": ih})
            for a in _anns(img_i):
                if a.get("box") is None:
                    continue
                x0, y0, w, h = a["box"]
                ann_rec = {
                    "id": ann_id, "image_id": img_i,
                    "category_id": cats.get(a["label"], 0),
                    "bbox": [x0, y0, w, h], "area": int(w * h), "iscrowd": 0,
                    "attributes": a.get("attributes", []),
                }
                if a.get("mask") is not None:
                    rle = masks.to_compact(a["mask"])
                    ann_rec["segmentation"] = rle
                    ann_rec["area"] = masks.mask_area(a["mask"])
                out["annotations"].append(ann_rec)
                ann_id += 1
        return json.dumps(out, indent=2, ensure_ascii=False)

    # VisualGenome & Custom: objects / relations / regions per image, plus a
    # global "groups" section (a group may span several images).
    images_out = []
    for img_i, info in enumerate(images):
        objs, regions = [], []
        for a in _anns(img_i):
            o = {"object_id": a["id"], "name": a["label"], "box": a.get("box"),
                 "attributes": a.get("attributes", []),
                 "group": a.get("group"),
                 "mask_area": (masks.mask_area(a["mask"])
                               if a.get("mask") is not None else None),
                 "segmentation": (masks.to_compact(a["mask"])
                                  if a.get("mask") is not None else None)}
            if standard == "Custom":
                o["fields"] = {f: a.get("custom_" + f, "") for f in custom_fields}
            objs.append(o)
            if a.get("region_desc"):
                regions.append({"region_id": a["id"], "object_id": a["id"],
                                "box": a.get("box"), "phrase": a["region_desc"]})
        images_out.append({"image": info["name"], "objects": objs,
                           "region_descriptions": regions})

    rels = [{"subject_id": r["subj"], "predicate": r["pred"], "object_id": r["obj"]}
            for r in relations]

    groups = {}
    for img_i, info in enumerate(images):
        for a in _anns(img_i):
            g = a.get("group")
            if g is None:
                continue
            meta = group_meta.get(g, {})
            groups.setdefault(g, {"group_id": g, "name": meta.get("name", f"group {g}"),
                                  "description": meta.get("description", ""), "members": []})
            groups[g]["members"].append({"object_id": a["id"],
                                         "image": info["name"], "label": a["label"]})
    out = {"images": images_out, "relationships": rels, "groups": list(groups.values())}
    return json.dumps(out, indent=2, ensure_ascii=False)


def import_annotations(data, images, annotations, make_annotation):
    """
    Import annotations exported by the tool into `annotations` (mutated).
    `make_annotation(label)` must build a fresh annotation with an id.
    Returns (n_added, n_skipped).
    """
    names = {info["name"]: i for i, info in enumerate(images)}
    added, skipped = 0, 0
    for entry in data:
        idx = names.get(entry.get("image"))
        if idx is None:
            skipped += len(entry.get("annotations", []))
            continue
        for a in entry.get("annotations", []):
            na = make_annotation(a.get("label", "object"))
            na["box"] = tuple(a["box"]) if a.get("box") else None
            na["points_pos"] = [tuple(p) for p in a.get("points_pos", [])]
            na["points_neg"] = [tuple(p) for p in a.get("points_neg", [])]
            na["_box_validated"] = na["box"] is not None
            m = a.get("mask")
            if masks.is_rle(m):                       # already a compact RLE
                na["mask"] = m
            elif isinstance(m, list):                 # full-resolution dense mask
                na["mask"] = masks.to_compact(np.array(m, dtype=bool))
            annotations[idx].append(na)
            added += 1
    return added, skipped
