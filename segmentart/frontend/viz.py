# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Display rendering and conversion of canvas objects.

Presentation layer (Streamlit). Builds on the config (colours) and on the
backend layer (object crops).
"""

import re

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from segmentart import config
from segmentart.backend import masks
from segmentart.backend.clustering import object_crop

MASK_ALPHA = config.MASK_ALPHA
POINT_R = config.POINT_R
BOX_COLOR = config.BOX_COLOR
POS_COLOR = config.POS_COLOR
NEG_COLOR = config.NEG_COLOR
POS_FILL = config.POS_FILL
NEG_FILL = config.NEG_FILL


def label_color(label):
    return config.color_for_label(st.session_state.categories, label)


def draw_circle(draw, cx, cy, color, r=POINT_R):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="white", width=2)


# ── Canvas background ──────────────────────────────────────────────────────────

def _display_base(base_array, dw, dh):
    """Base image scaled down to display size, cached per image."""
    cache = st.session_state.get("_disp_base")
    key = (id(base_array), dw, dh)
    if cache is not None and cache[0] == key:
        return cache[1].copy()
    pil = Image.fromarray(base_array).convert("RGBA").resize((dw, dh), Image.BILINEAR)
    st.session_state["_disp_base"] = (key, pil)
    return pil.copy()


def render_background(base_array, annotations, current_ann, dw, dh, scale,
                      show_validated=True):
    """Image + masks (and, if requested, boxes/points of validated annotations).

    The box and points of the annotation BEING EDITED are not drawn here: they
    are handled by the canvas (as editable objects)."""
    img = _display_base(base_array, dw, dh)
    draw = ImageDraw.Draw(img)

    def draw_mask(ann):
        nonlocal draw
        if ann.get("mask") is not None:
            bc = label_color(ann["label"])
            r, g, b = config.hex_to_rgb(bc)
            m = masks.ensure_dense(ann["mask"]).astype(np.uint8) * MASK_ALPHA
            alpha = Image.fromarray(m, "L").resize((dw, dh), Image.NEAREST)
            colored = Image.new("RGBA", img.size, (r, g, b, 0))
            colored.putalpha(alpha)
            img.paste(Image.alpha_composite(img, colored))
            draw = ImageDraw.Draw(img)

    def draw_full(ann):
        draw_mask(ann)
        bc = label_color(ann["label"])
        if ann.get("box"):
            x0, y0, w, h = (v * scale for v in ann["box"])
            draw.rectangle([x0, y0, x0 + w, y0 + h], outline=bc, width=4)
            draw.text((x0 + 3, max(0, y0 - 15)), ann["label"], fill=bc)
        for px, py in ann.get("points_pos", []):
            draw_circle(draw, px * scale, py * scale, POS_COLOR)
        for px, py in ann.get("points_neg", []):
            draw_circle(draw, px * scale, py * scale, NEG_COLOR)

    if show_validated:
        for ann in annotations:
            if ann is not current_ann:
                draw_full(ann)

    if current_ann is not None:
        draw_mask(current_ann)

    return img.convert("RGB")


def get_background(base_array, annotations, current_ann, dw, dh):
    """Cached background: rebuilt only when the image, the resolution, a mask or
    a validated annotation changes — not on every click."""
    show = st.session_state.get("show_validated", False)
    sig = (
        id(base_array), dw, show,
        tuple((a["id"],
               tuple(a["box"]) if a.get("box") else None,
               id(a["mask"]) if a.get("mask") is not None else 0)
              for a in annotations if a is not current_ann) if show else (),
        (current_ann["id"],
         id(current_ann["mask"]) if current_ann is not None
         and current_ann.get("mask") is not None else 0)
        if current_ann is not None else None,
    )
    cache = st.session_state.get("_bg")
    if cache is not None and cache[0] == sig:
        st.session_state["_bg_hit"] = True
        return cache[1]
    st.session_state["_bg_hit"] = False
    scale = dw / base_array.shape[1]
    pil = render_background(base_array, annotations, current_ann, dw, dh, scale,
                            show_validated=show)
    st.session_state["_bg"] = (sig, pil)
    return pil


# ── Canvas object <-> annotation conversion ────────────────────────────────────

def _fill_is_pos(fill):
    """True if the fill colour is predominantly green (positive point)."""
    nums = re.findall(r"[\d.]+", fill or "")
    if len(nums) >= 3:
        r, g, b = float(nums[0]), float(nums[1]), float(nums[2])
        return g >= r and g >= b
    return False


def parse_polygon_objects(objects, scale, drop_last=True, close_dist=0):
    """Canvas 'path' objects (polygon mode, manual P0) → (polygons, box).

    Cleaning up the stroke:
      - `drop_last`: ignore the last vertex (often the closing click);
      - `close_dist` > 0: if the last vertex is within `close_dist` pixels of the
        first one, it is dropped (closing by proximity).
    The full-resolution mask is built by the caller."""
    polys = []
    for o in objects:
        if o.get("type") != "path":
            continue
        pts = []
        for cmd in o.get("path", []):
            if not cmd:
                continue
            c = cmd[0]
            if c in ("M", "L") and len(cmd) >= 3:
                pts.append((round(float(cmd[1]) / scale), round(float(cmd[2]) / scale)))
        if len(pts) >= 4:
            near = (close_dist > 0
                    and abs(pts[-1][0] - pts[0][0]) <= close_dist
                    and abs(pts[-1][1] - pts[0][1]) <= close_dist)
            if near or drop_last:
                pts = pts[:-1]
        if len(pts) >= 3:
            polys.append([c for xy in pts for c in xy])
    if not polys:
        return [], None
    xs = [p[k] for p in polys for k in range(0, len(p), 2)]
    ys = [p[k] for p in polys for k in range(1, len(p), 2)]
    box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    return polys, box


def parse_canvas_objects(objects, scale):
    """Canvas objects (display coords) → box / points_pos / points_neg (image
    coords). Only one box is kept (the last one); points are classified by their
    fill colour."""
    box = None
    pos, neg = [], []
    for o in objects:
        t = o.get("type")
        if t == "rect":
            left = o.get("left", 0)
            top = o.get("top", 0)
            w = o.get("width", 0) * o.get("scaleX", 1)
            h = o.get("height", 0) * o.get("scaleY", 1)
            box = (round(left / scale), round(top / scale),
                   round(w / scale), round(h / scale))
        elif t == "circle":
            r = o.get("radius", POINT_R) * o.get("scaleX", 1)
            cx = (o.get("left", 0) + r) / scale
            cy = (o.get("top", 0) + r) / scale
            pt = (round(cx), round(cy))
            if _fill_is_pos(o.get("fill", "")):
                pos.append(pt)
            else:
                neg.append(pt)
    return box, pos, neg


def ann_to_initial_drawing(ann, scale):
    """Rebuild the fabric.js drawing (box + points) of an annotation so it can be
    corrected in the canvas. None if the annotation is empty."""
    objs = []
    if ann.get("box"):
        x0, y0, w, h = ann["box"]
        objs.append({
            "type": "rect", "originX": "left", "originY": "top",
            "left": x0 * scale, "top": y0 * scale,
            "width": w * scale, "height": h * scale,
            "fill": "rgba(0,0,0,0)", "stroke": BOX_COLOR, "strokeWidth": 4,
            "scaleX": 1, "scaleY": 1, "angle": 0,
        })

    def _circle(px, py, fill, stroke):
        return {
            "type": "circle", "originX": "left", "originY": "top",
            "left": px * scale - POINT_R, "top": py * scale - POINT_R,
            "radius": POINT_R, "fill": fill, "stroke": stroke, "strokeWidth": 1,
            "scaleX": 1, "scaleY": 1, "angle": 0,
        }

    for px, py in ann.get("points_pos", []):
        objs.append(_circle(px, py, POS_FILL, POS_COLOR))
    for px, py in ann.get("points_neg", []):
        objs.append(_circle(px, py, NEG_FILL, NEG_COLOR))

    if not objs:
        return None
    return {"version": "4.4.0", "objects": objs}


# ── Thumbnails / previews ──────────────────────────────────────────────────────

def full_with_mask(ann, base_array, max_w=360):
    """Full image + mask overlay, as a thumbnail. The box and points are drawn
    with a width proportional to the resolution, so they stay readable once the
    thumbnail is scaled down."""
    img = Image.fromarray(base_array).convert("RGBA")
    if ann.get("mask") is not None:
        bc = label_color(ann["label"])
        r, g, b = config.hex_to_rgb(bc)
        m = masks.ensure_dense(ann["mask"]).astype(np.uint8) * MASK_ALPHA
        colored = Image.new("RGBA", img.size, (r, g, b, 0))
        colored.putalpha(Image.fromarray(m, "L"))
        img = Image.alpha_composite(img, colored)
    h, w = base_array.shape[:2]
    line_w = max(6, int(0.006 * min(h, w)))
    pt_r = max(10, int(0.009 * min(h, w)))
    d = ImageDraw.Draw(img)
    if ann.get("box"):
        x0, y0, bw, bh = ann["box"]
        d.rectangle([x0, y0, x0 + bw, y0 + bh],
                    outline=label_color(ann["label"]), width=line_w)
    for px, py in ann.get("points_pos", []):
        d.ellipse([px - pt_r, py - pt_r, px + pt_r, py + pt_r],
                  fill=POS_COLOR, outline="white", width=max(2, line_w // 2))
    for px, py in ann.get("points_neg", []):
        d.ellipse([px - pt_r, py - pt_r, px + pt_r, py + pt_r],
                  fill=NEG_COLOR, outline="white", width=max(2, line_w // 2))
    img = img.convert("RGB")
    img.thumbnail((max_w, max_w))
    return img


def view_annotation(ann, base_array, mode, max_w=360):
    """Render an annotation according to the requested display mode."""
    if mode == "Image":
        img = Image.fromarray(base_array).convert("RGB")
        img.thumbnail((max_w, max_w))
        return img
    if mode == "Image + mask":
        return full_with_mask(ann, base_array, max_w)
    if mode == "Mask crop (black background)":
        c = object_crop(ann, base_array, masked=True, bg=(0, 0, 0))
        if c is not None:
            c.thumbnail((max_w, max_w))
        return c
    return None  # JSON is handled elsewhere


def annotation_preview(ann, base_array):
    """Zoomed crop around an annotation's box (with mask, box and points)."""
    if ann.get("box") is None:
        return None
    x0, y0, w, h = ann["box"]
    pad = 20
    ih, iw = base_array.shape[:2]
    x0c, y0c = max(0, x0 - pad), max(0, y0 - pad)
    x1c, y1c = min(iw, x0 + w + pad), min(ih, y0 + h + pad)
    if x1c <= x0c or y1c <= y0c:
        return None
    crop = base_array[y0c:y1c, x0c:x1c]
    img = Image.fromarray(crop).convert("RGBA")
    draw = ImageDraw.Draw(img)
    if ann.get("mask") is not None:
        bc = label_color(ann["label"])
        r, g, b = config.hex_to_rgb(bc)
        mask_crop = masks.ensure_dense(ann["mask"])[y0c:y1c, x0c:x1c]
        alpha_arr = (mask_crop.astype(np.uint8) * MASK_ALPHA)
        colored = Image.new("RGBA", img.size, (r, g, b, 0))
        colored.putalpha(Image.fromarray(alpha_arr, "L"))
        img = Image.alpha_composite(img, colored)
        draw = ImageDraw.Draw(img)
    rx0, ry0 = x0 - x0c, y0 - y0c
    draw.rectangle([rx0, ry0, rx0 + w, ry0 + h], outline=BOX_COLOR, width=2)
    for px, py in ann.get("points_pos", []):
        draw_circle(draw, px - x0c, py - y0c, POS_COLOR)
    for px, py in ann.get("points_neg", []):
        draw_circle(draw, px - x0c, py - y0c, NEG_COLOR)
    return img.convert("RGB")
