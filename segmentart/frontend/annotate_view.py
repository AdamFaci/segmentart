# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""Phase 1 — annotation interface (sidebar + canvas + validated list)."""

import json
import os
import time

import numpy as np
import streamlit as st

from segmentart import config
from segmentart.backend import io_store, masks, sam2_engine, serialization
from segmentart.backend.models import bump_epoch
from segmentart.backend.telemetry import PIPELINES
from segmentart.frontend import resources, state, viz
from segmentart.frontend.canvas_compat import CANVAS_AVAILABLE, st_canvas


def render_annotation_phase():
    if not CANVAS_AVAILABLE:
        st.error("The **streamlit-drawable-canvas** component is required.\n\n"
                 "Install it, then restart:\n\n"
                 "```\npip install streamlit-drawable-canvas\n```")
        st.stop()

    _apply_round_pipeline()
    _sidebar()
    _main_area()


def _apply_round_pipeline():
    """In guided collection, force the pipeline of the current image."""
    rp = st.session_state.get("round_plan") or {}
    if rp:
        nm = state.cur_name()
        if nm in rp:
            st.session_state.active_pipeline = rp[nm]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def _sidebar():
    with st.sidebar:
        st.title("🏷️ Annotation")

        # ── Images ─────────────────────────────────────────────────────────────
        st.subheader("📁 Images")
        uploaded = st.file_uploader("Load images",
            type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
        if uploaded:
            from PIL import Image
            existing = {img["name"] for img in st.session_state.images}
            for f in uploaded:
                if f.name not in existing:
                    arr = np.array(Image.open(f).convert("RGB"))
                    st.session_state.images.append({"name": f.name, "array": arr})
                    st.session_state.annotations.append([])

        if st.session_state.images:
            n = len(st.session_state.images)
            idx = st.session_state.image_index
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                if st.button("◀", disabled=(idx == 0)):
                    st.session_state.image_index -= 1
                    st.session_state.current_annotation = None
                    st.rerun()
            with c2:
                st.markdown(f"<div style='text-align:center;font-weight:bold;padding:4px'>{idx+1}/{n}</div>",
                            unsafe_allow_html=True)
            with c3:
                if st.button("▶", disabled=(idx == n - 1)):
                    st.session_state.image_index += 1
                    st.session_state.current_annotation = None
                    st.rerun()
            st.caption(f"📷 {state.cur_name()}")

        _sidebar_campaign()
        _sidebar_data_folder()

        # ── Annotation import ──────────────────────────────────────────────────
        ann_file = st.file_uploader(
            "Load annotations (JSON exported by this tool)",
            type=["json"], key="ann_loader")
        if ann_file is not None and st.button("Import these annotations"):
            try:
                data = json.load(ann_file)
                added, skipped = serialization.import_annotations(
                    data, st.session_state.images, st.session_state.annotations,
                    state.make_annotation)
                msg = f"{added} annotation(s) imported."
                if skipped:
                    msg += f" {skipped} skipped (image not loaded)."
                st.success(msg)
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

        st.divider()

        # ── Settings ───────────────────────────────────────────────────────────
        st.subheader("⚙️ Settings")
        st.session_state.n_pos_auto = st.number_input(
            "Positive points before switching automatically to negative mode",
            min_value=1, max_value=20, value=st.session_state.n_pos_auto, step=1)
        st.session_state.run_twice = st.toggle(
            "Run SAM2 twice (better mask, fewer corrections)",
            value=st.session_state.get("run_twice", True))
        st.session_state.auto_postproc = st.toggle(
            "Auto chain after SAM2: edge smoothing + hole filling",
            value=st.session_state.get("auto_postproc", True),
            help="Saves you from running OpenCV by hand → fewer interactions. "
                 "Uncheck to keep the raw mask.")
        st.session_state.auto_sam_neg = st.number_input(
            "− points that trigger SAM2 automatically (0 = disabled)",
            min_value=0, max_value=20, value=st.session_state.get("auto_sam_neg", 0),
            step=1)
        st.session_state.auto_validate_box = st.toggle(
            "Validate the box automatically (no button)",
            value=st.session_state.get("auto_validate_box", False),
            help="As soon as a box is drawn, go straight to the points.")
        st.session_state.poly_drop_last = st.toggle(
            "Polygon: ignore the last vertex",
            value=st.session_state.get("poly_drop_last", True))
        st.session_state.show_validated = st.toggle(
            "Show validated annotations on the image",
            value=st.session_state.show_validated,
            help="Off: the image stays “clean” so nothing gets in the way.")

        # ── Telemetry (optional) ───────────────────────────────────────────────
        with st.expander("📈 Telemetry (study)"):
            st.session_state.telemetry_on = st.toggle(
                "Enable action logging",
                value=st.session_state.get("telemetry_on", False),
                help="Non-blocking option: records time and interactions "
                     "per (image × pipeline) for analysis.")
            state.telemetry().enabled = st.session_state.telemetry_on
            if st.session_state.get("round_plan"):
                st.caption(f"Pipeline set by the guided collection: "
                           f"**{st.session_state.get('active_pipeline')}**")
            else:
                st.session_state.active_pipeline = st.selectbox(
                    "Current pipeline",
                    list(PIPELINES), index=list(PIPELINES).index(
                        st.session_state.get("active_pipeline", "P1")),
                    format_func=lambda p: f"{p} · {PIPELINES[p]}")
            tel = state.telemetry()
            st.caption(f"{len(tel.events)} event(s) recorded.")
            cD, cC, cR = st.columns(3)
            cD.download_button("JSONL (raw)", data=tel.to_jsonl(),
                               file_name="telemetry.jsonl", mime="application/json")
            from segmentart.backend.telemetry import instances_to_csv, reduce_to_instances
            csv_text = instances_to_csv(reduce_to_instances(tel.events))
            cC.download_button("CSV (instances)", data=csv_text,
                               file_name="instances.csv", mime="text/csv")
            if cR.button("Reset"):
                tel.reset()
                st.rerun()

        st.divider()

        # ── Categories ─────────────────────────────────────────────────────────
        st.subheader("🏷️ Categories")
        cats_str = st.text_area("One per line:",
            value="\n".join(st.session_state.categories), height=80)
        if st.button("Update categories"):
            st.session_state.categories = [c.strip() for c in cats_str.split("\n") if c.strip()]
            st.rerun()

        st.caption("Click to start a new annotation:")
        ann = st.session_state.current_annotation
        for cat in st.session_state.categories:
            active = (ann is not None and not ann.get("_editing")
                      and ann.get("label") == cat)
            label = f"{'● ' if active else ''}{cat}"
            if st.button(label, key=f"cat_{cat}"):
                a = state.make_annotation(cat)
                state.cur_anns().append(a)
                st.session_state.current_annotation = a
                st.session_state.mode = "box"
                state.log_event("category_start", ann_id=a["id"], label=cat)
                st.rerun()

        st.divider()

        # ── Drawing mode ───────────────────────────────────────────────────────
        st.subheader("✏️ Drawing mode")
        mode = st.session_state.mode

        def _set_mode(m):
            a = st.session_state.current_annotation
            # Moving to points with a NON-validated box → the box is dropped.
            if (m in ("pos", "neg") and a is not None
                    and a.get("box") is not None and not a.get("_box_validated")):
                a["box"] = None
                bump_epoch(a)
            st.session_state.mode = m
            if a is not None:
                a["_mode_locked"] = True
            st.rerun()

        c1, c2, c3 = st.columns(3)
        if c1.button(f"{'✅' if mode == 'box' else '📦'} Box"):
            _set_mode("box")
        if c2.button(f"{'✅' if mode == 'pos' else '🟢'} +Pt"):
            _set_mode("pos")
        if c3.button(f"{'✅' if mode == 'neg' else '🔴'} −Pt"):
            _set_mode("neg")
        if st.button(f"{'✅ ' if mode == 'poly' else '✏️ '}Polygon (P0 — manual)"):
            _set_mode("poly")
        if st.button(f"{'✅ ' if mode == 'erase' else '🧽 '}Eraser (select / delete)"):
            _set_mode("erase")
        if mode == "poly":
            st.caption("P0: draw a polygon (clicks = vertices, double-click to "
                       "close). SAM is disabled in this mode.")
        if ann is not None and ann.get("_mode_locked"):
            st.caption("🔒 Mode locked manually (no auto-switching).")

        st.divider()

        _sidebar_sam2()
        st.divider()
        _sidebar_opencv()
        st.divider()
        _sidebar_actions()
        st.divider()
        _sidebar_export()


def _apply_auto_chain(mask, a):
    """Optional cleanup chain applied after SAM2 (smoothing + hole filling, while
    respecting the +/− points) to save OpenCV clicks."""
    if not st.session_state.get("auto_postproc", True) or not masks.CV2_AVAILABLE:
        return mask
    from segmentart.backend import maskops
    out = maskops.smooth_edges(mask, strength=2)
    out = maskops.fill_holes(out, pos_points=a.get("points_pos"),
                             neg_points=a.get("points_neg"))
    return out


def _execute_sam2(a, run_twice, use_mask=False):
    """Run SAM2 (twice if requested) and apply the auto chain to the best mask.
    Updates the candidates / mask / logits of annotation `a`."""
    arr = state.cur_arr()
    predictor = resources.get_sam2_predictor()
    mask_input = None
    if use_mask and a.get("mask") is not None:
        if a.get("_logits") is not None:
            mi = np.asarray(a["_logits"])
            mask_input = mi[None] if mi.ndim == 2 else mi
        else:
            mask_input = masks.mask_to_logits(a["mask"])

    t0 = time.perf_counter()
    mlist, scores, logits = sam2_engine.predict(
        predictor, arr, box=a.get("box"), points_pos=a.get("points_pos"),
        points_neg=a.get("points_neg"), mask_input=mask_input)
    n_runs = 1
    if mlist and run_twice:
        mi = logits[0]
        mi = mi[None] if mi.ndim == 2 else mi
        m2, s2, l2 = sam2_engine.predict(
            predictor, arr, box=a.get("box"), points_pos=a.get("points_pos"),
            points_neg=a.get("points_neg"), mask_input=mi)
        if m2:
            mlist, scores, logits, n_runs = m2, s2, l2, 2

    if mlist:
        a.setdefault("_mask_stack", []).append(a.get("mask"))
        a["_candidates"] = {"masks": mlist, "scores": scores, "logits": logits}
        a["_cand_idx"] = 0
        a["mask"] = _apply_auto_chain(mlist[0], a)
        a["_logits"] = logits[0]
        state.log_event("sam2_run", ann_id=a["id"],
                        has_box=a.get("box") is not None,
                        n_pos=len(a.get("points_pos", [])),
                        n_neg=len(a.get("points_neg", [])),
                        n_candidates=len(mlist), best_score=round(scores[0], 4),
                        n_runs=n_runs,
                        auto_postproc=bool(st.session_state.get("auto_postproc", True)),
                        elapsed_s=round(time.perf_counter() - t0, 3))


def _sidebar_campaign():
    """Guided collection: load the images of one planning round and apply each
    image's pipeline automatically (telemetry then records the right condition
    for every image)."""
    with st.expander("🎯 Guided collection (session)"):
        path = st.text_input("Plan (assignment.csv or worksheet)",
                             value="campaign/assignment.csv")
        rnd = st.number_input("Round", min_value=1, max_value=8,
                              value=int(st.session_state.get("round_id") or 1), step=1)
        if st.button("Load this round"):
            if not os.path.exists(path):
                st.error("Plan file not found.")
            else:
                plan = io_store.round_plan(path, int(rnd))
                if not plan:
                    st.error("No image for this round (wrong format or round number?).")
                else:
                    avail = set(io_store.list_images(config.IMAGES_DIR))
                    existing = {im["name"] for im in st.session_state.images}
                    order, pipemap, missing = [], {}, []
                    for name, pipe in plan:
                        pipemap[name] = pipe
                        order.append(name)
                        if name in existing:
                            continue
                        if name in avail:
                            arr = io_store.load_image_array(
                                os.path.join(config.IMAGES_DIR, name))
                            st.session_state.images.append({"name": name, "array": arr})
                            st.session_state.annotations.append([])
                        else:
                            missing.append(name)
                    # jump to the first image of the round present in the session
                    present = [n for n in order if n in
                               {im["name"] for im in st.session_state.images}]
                    if present:
                        names = [im["name"] for im in st.session_state.images]
                        st.session_state.image_index = names.index(present[0])
                    st.session_state.round_plan = pipemap
                    st.session_state.round_id = int(rnd)
                    st.session_state.current_annotation = None
                    st.session_state.telemetry_on = True
                    state.telemetry().enabled = True
                    msg = f"Round {int(rnd)}: {len(present)} image(s) ready."
                    if missing:
                        msg += f" {len(missing)} missing from data/images."
                    st.success(msg)
                    st.rerun()

        if st.session_state.get("round_plan"):
            cur = state.cur_name()
            pipe = st.session_state.round_plan.get(cur, "—")
            st.info(f"Round {st.session_state.round_id} · image **{cur}** · "
                    f"pipeline in use: **{pipe}**")
            st.caption("Navigate with ◀ ▶; the pipeline follows the image. "
                       "Annotate every shape, then move on to the next image.")
            if st.button("Stop the guided collection"):
                st.session_state.round_plan = {}
                st.session_state.round_id = None
                st.rerun()


def _sidebar_data_folder():
    """Local loading / saving through the data/ folders."""
    with st.expander("📂 data/ folders"):
        st.caption(f"Images: `{config.IMAGES_DIR}` · Annotations: "
                   f"`{config.ANNOTATIONS_DIR}`")
        # ── Load the images from the folder ────────────────────────────────────
        files = io_store.list_images(config.IMAGES_DIR)
        if st.button(f"Load the images from data/images ({len(files)})",
                     disabled=not files):
            existing = {im["name"] for im in st.session_state.images}
            added = 0
            for fn in files:
                if fn in existing:
                    continue
                arr = io_store.load_image_array(os.path.join(config.IMAGES_DIR, fn))
                st.session_state.images.append({"name": fn, "array": arr})
                st.session_state.annotations.append([])
                added += 1
            st.success(f"{added} image(s) loaded.")
            st.rerun()

        # ── Load annotations from the folder ───────────────────────────────────
        ann_files = io_store.list_annotation_files(config.ANNOTATIONS_DIR)
        if ann_files:
            choice = st.selectbox("Annotations (data/annotations)", ann_files)
            if st.button("Import these annotations"):
                try:
                    data = io_store.load_json(os.path.join(config.ANNOTATIONS_DIR, choice))
                    added, skipped = serialization.import_annotations(
                        data, st.session_state.images, st.session_state.annotations,
                        state.make_annotation)
                    st.success(f"{added} imported" + (f", {skipped} skipped" if skipped else "") + ".")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

        # ── Save images + annotations to data/ ─────────────────────────────────
        if st.session_state.images and st.button("💾 Save to data/"):
            try:
                io_store.ensure_dir(config.IMAGES_DIR)
                on_disk = set(io_store.list_images(config.IMAGES_DIR))
                saved_img = 0
                for im in st.session_state.images:
                    if im["name"] not in on_disk:
                        io_store.save_image_array(
                            im["array"], os.path.join(config.IMAGES_DIR, im["name"]))
                        saved_img += 1
                txt = serialization.to_json(
                    st.session_state.images, st.session_state.annotations,
                    mask_format=config.MASK_EXPORT_FORMAT)
                io_store.save_text(txt, os.path.join(config.ANNOTATIONS_DIR,
                                                     "annotations.json"))
                st.success(f"{saved_img} image(s) + annotations.json saved.")
            except Exception as e:
                st.error(f"Save failed: {e}")


def _sidebar_sam2():
    st.subheader("🤖 SAM2")
    if st.session_state.mode == "poly":
        st.caption("Disabled in polygon mode (P0).")
        return
    if not sam2_engine.SAM2_AVAILABLE:
        st.warning("SAM2 not installed  \n`pip install sam2`")
        return

    if not st.session_state.sam_loaded:
        if st.button("Load the SAM2 model"):
            with st.spinner("Loading…"):
                resources.get_sam2_predictor()       # caches the predictor
                st.session_state.sam_loaded = True
            st.rerun()
        return

    st.success("SAM2 ready ✓")
    a = st.session_state.current_annotation
    has_input = a is not None and (
        a.get("box") is not None or a.get("points_pos") or a.get("points_neg"))
    has_mask = a is not None and a.get("mask") is not None

    mask_key = f"use_mask_{a['id']}" if a is not None else "use_mask_none"
    st.session_state.setdefault(mask_key, bool(has_mask))
    use_mask = st.checkbox(
        "Use the current mask as input",
        key=mask_key, disabled=not has_mask,
        help="Re-runs SAM2 from the current mask plus the points you added.")

    if st.button("▶ Run SAM2", type="primary", disabled=not has_input):
        _execute_sam2(a, st.session_state.get("run_twice", True), use_mask=use_mask)
        st.session_state.pop(mask_key, None)
        st.session_state.pop(f"cand_{a['id']}", None)
        st.rerun()

    # Browsing the proposed masks
    if a is not None and a.get("_candidates"):
        cand = a["_candidates"]
        nC = len(cand["masks"])
        idx = st.radio(
            "Proposed mask (browse the candidates)",
            list(range(nC)), index=min(a.get("_cand_idx", 0), nC - 1),
            format_func=lambda i: f"#{i+1} — score {cand['scores'][i]:.2f}",
            horizontal=True, key=f"cand_{a['id']}")
        a["_cand_idx"] = idx
        a["mask"] = cand["masks"][idx]
        a["_logits"] = cand["logits"][idx]

    if a is not None and not has_input:
        st.caption("Add at least one box or point to enable SAM2.")


def _sidebar_opencv():
    st.subheader("🩹 Solid mask (OpenCV)")
    if not masks.CV2_AVAILABLE:
        st.caption("OpenCV not installed  \n`pip install opencv-python-headless`")
        return
    from segmentart.backend.maskops import OPENCV_METHODS
    a = st.session_state.current_annotation
    can_fill = a is not None and a.get("mask") is not None
    method = st.selectbox("Method", list(OPENCV_METHODS), disabled=not can_fill)
    info = OPENCV_METHODS[method]
    st.caption(f"ℹ️ {info['desc']}  \n**Best for**: {info['best_for']}")
    if st.button("Get a solid mask", disabled=not can_fill):
        a.setdefault("_mask_stack", []).append(a.get("mask"))
        a["mask"] = masks.fill_mask(a["mask"], method)
        a["_logits"] = None
        a.pop("_candidates", None)
        st.session_state.pop(f"use_mask_{a['id']}", None)
        state.log_event("opencv_fill", ann_id=a["id"], method=method)
        st.rerun()
    if not can_fill:
        st.caption("Run SAM2 first to produce a mask to fill.")


def _sidebar_actions():
    st.subheader("⚙️ Actions")
    c1, c2 = st.columns(2)
    if c1.button("✅ Validate"):
        a = st.session_state.current_annotation
        if a is not None:
            a.pop("_candidates", None)
            state.log_event("validate", ann_id=a["id"], label=a["label"],
                            has_box=a.get("box") is not None,
                            has_mask=a.get("mask") is not None,
                            n_pos=len(a.get("points_pos", [])),
                            n_neg=len(a.get("points_neg", [])),
                            n_clicks=a.get("_clicks", 0))
            # Lossless RLE compression + release of the dense history: this is
            # where most of the memory is saved (GB -> MB).
            a["mask"] = masks.to_compact(a.get("mask"))
            a["_mask_stack"] = []
            a["_logits"] = None
        st.session_state.current_annotation = None
        st.session_state.mode = "box"
        st.rerun()
    if c2.button("🗑 Reset"):
        a = st.session_state.current_annotation
        anns_list = state.cur_anns()
        if a and a in anns_list:
            anns_list.remove(a)
        st.session_state.current_annotation = None
        st.session_state.mode = "box"
        st.rerun()

    # Unified undo: mask (SAM2 / OpenCV) > − point > + point > box
    if st.button("↩ Undo the last action"):
        a = st.session_state.current_annotation
        if a is not None:
            if a.get("mask") is not None or a.get("_mask_stack"):
                stack = a.get("_mask_stack") or []
                a["mask"] = stack.pop() if stack else None
            elif a.get("points_neg"):
                a["points_neg"].pop()
                bump_epoch(a)
            elif a.get("points_pos"):
                a["points_pos"].pop()
                bump_epoch(a)
            elif a.get("box") is not None:
                a["box"] = None
                st.session_state.mode = "box"
                bump_epoch(a)
        st.rerun()
    st.caption("The toolbar bin / arrows also undo boxes & points.")


def _sidebar_export():
    st.subheader("💾 Export")
    if not st.session_state.images:
        return
    fmt = st.selectbox(
        "Masks in the export", ["rle", "none", "dense"], index=0,
        format_func=lambda x: {"rle": "Compact RLE (light, lossless)",
                               "none": "area / shape only",
                               "dense": "full dense (very heavy)"}[x])
    if st.button("Prepare the JSON"):
        st.session_state["_export_json"] = serialization.to_json(
            st.session_state.images, st.session_state.annotations, mask_format=fmt)
    if st.session_state.get("_export_json"):
        st.download_button("Download annotations.json",
            data=st.session_state["_export_json"],
            file_name="annotations.json", mime="application/json")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

def _main_area():
    if not st.session_state.images:
        st.info("👈 Load images from the sidebar to get started.")
        st.markdown("""
### How to use
1. **Load** images (sidebar)
2. **Click a category** to start an annotation
3. **Box mode**: *click and drag* to stretch the rectangle (live preview),
   then **Validate the box** → automatic switch to +Point mode
4. **+Point mode**: click to add positive points; after N points
   (configurable) → automatic switch to −Point mode
5. **−Point mode**: negative points to exclude areas
6. **Run SAM2** (box and/or points: a single input is enough)
7. **Validate** → the annotation joins the list below
        """)
        return

    arr = state.cur_arr()
    if arr is None:
        return

    mode = st.session_state.mode
    ann = st.session_state.current_annotation
    anns = state.cur_anns()
    h, w = arr.shape[:2]

    # ── Status + resolution ────────────────────────────────────────────────────
    top_l, top_r = st.columns([4, 1])
    with top_r:
        st.session_state.hi_res = st.toggle(
            "🔍 High resolution", value=st.session_state.hi_res,
            help="Sharper display but slower. Turn it on only when needed.")
    with top_l:
        if ann:
            n_pos = len(ann.get("points_pos", []))
            n_neg = len(ann.get("points_neg", []))
            st.markdown(
                f"**Annotation in progress:** `{ann['label']}` &nbsp;|&nbsp; "
                f"Mode: `{mode}` &nbsp;|&nbsp; "
                f"Box: {'✓' if ann.get('box') else '–'} &nbsp;|&nbsp; "
                f"🟢 {n_pos} &nbsp;|&nbsp; 🔴 {n_neg} &nbsp;|&nbsp; "
                f"Mask: {'✓' if ann.get('mask') is not None else '–'}")
        else:
            st.markdown("_No annotation in progress — pick a category in the sidebar._")

    max_w = config.DISPLAY_FULL if st.session_state.hi_res else config.DISPLAY_LIGHT
    dw = min(max_w, w)
    dh = int(h * dw / w)
    scale = dw / w

    bg = viz.get_background(arr, anns, ann, dw, dh)

    if ann is None:
        st.image(bg, use_container_width=False)
    else:
        _canvas(ann, mode, bg, dw, dh, scale, arr.shape[:2])

    _validated_list(ann)


def _update_clicks(ann, n_points, n_rect, n_verts):
    """Exact count of canvas interactions, by state difference.

    Every object added OR removed (point, box, polygon vertex) counts as one
    action — so corrections / undos are counted too. Accumulates on
    `ann['_clicks']`."""
    prev = ann.get("_click_state") or {"points": 0, "rect": 0, "verts": 0}
    delta = (abs(n_points - prev["points"]) + abs(n_rect - prev["rect"])
             + abs(n_verts - prev["verts"]))
    if delta:
        ann["_clicks"] = ann.get("_clicks", 0) + delta
        ann["_click_state"] = {"points": n_points, "rect": n_rect, "verts": n_verts}


def _canvas(ann, mode, bg, dw, dh, scale, shape):
    poly_mode = (mode == "poly")
    if mode == "box":
        drawing_mode, stroke, fill = "rect", config.BOX_COLOR, config.BOX_FILL
        hint = "📦 Click and drag to stretch the box, then “Validate the box”."
    elif mode == "pos":
        drawing_mode, stroke, fill = "point", config.POS_COLOR, config.POS_FILL
        hint = f"🟢 Click to add positive points ({len(ann.get('points_pos',[]))}/{st.session_state.n_pos_auto})."
    elif mode == "neg":
        drawing_mode, stroke, fill = "point", config.NEG_COLOR, config.NEG_FILL
        hint = "🔴 Click to add negative points (areas to exclude)."
    elif mode == "erase":
        drawing_mode, stroke, fill = "transform", config.BOX_COLOR, config.BOX_FILL
        hint = ("✋ Eraser — click a point or the box to select it, then use the "
                "toolbar bin to delete it.")
    else:  # poly (P0, manual)
        drawing_mode, stroke, fill = "polygon", config.BOX_COLOR, "rgba(30,136,229,0.25)"
        hint = "✏️ P0 — click the vertices, double-click to close the polygon."
    st.caption(hint)

    canvas_key = f"cv_{st.session_state.image_index}_{ann['id']}_{ann.get('_epoch',0)}_{dw}"

    if poly_mode:
        init_draw = None      # the canvas keeps the polygon through its stable key
    else:
        # initial_drawing must stay stable while annotating: computed once per key,
        # then reused (otherwise the canvas resets and wipes the point being
        # placed).
        store = st.session_state.setdefault("_initdraw", {})
        if canvas_key not in store:
            store.clear()
            store[canvas_key] = viz.ann_to_initial_drawing(ann, scale)
        init_draw = store[canvas_key]

    result = st_canvas(
        background_image=bg,
        background_color="#f0f0f0",
        initial_drawing=init_draw,
        drawing_mode=drawing_mode,
        stroke_color=stroke,
        fill_color=fill,
        stroke_width=2,
        point_display_radius=config.POINT_R,
        update_streamlit=True,
        height=dh, width=dw,
        display_toolbar=True,
        key=canvas_key,
    )

    if result is not None and result.json_data is not None:
        objs = result.json_data.get("objects", [])
        if poly_mode:
            close_dist = max(8, int(0.01 * min(shape)))
            polys, box = viz.parse_polygon_objects(
                objs, scale, drop_last=st.session_state.get("poly_drop_last", True),
                close_dist=close_dist)
            n_verts = sum(len(p) // 2 for p in polys)
            _update_clicks(ann, 0, 0, n_verts)
            sig = tuple(tuple(p) for p in polys)
            if sig and ann.get("_poly_sig") != sig:
                # mask rebuilt only when the outline changes
                ann["_poly_sig"] = sig
                ann["mask"] = masks.polygons_to_mask(polys, shape)
                ann["box"] = box
                ann["points_pos"] = []
                ann["points_neg"] = []
                state.log_event("polygon", ann_id=ann["id"],
                                n_polygons=len(polys), n_vertices=n_verts)
        else:
            box, pos, neg = viz.parse_canvas_objects(objs, scale)
            ann["box"] = box
            ann["points_pos"] = pos
            ann["points_neg"] = neg
            _update_clicks(ann, len(pos) + len(neg), 1 if box else 0, 0)
            # automatic box validation (no button), when enabled
            if (mode == "box" and box is not None and not ann.get("_box_validated")
                    and st.session_state.get("auto_validate_box", False)):
                ann["_box_validated"] = True
                st.session_state.mode = "pos"
                bump_epoch(ann)
                state.log_event("box_validate", ann_id=ann["id"], auto=True)
                st.rerun()
            if (mode == "pos" and not ann.get("_mode_locked")
                    and len(pos) >= st.session_state.n_pos_auto):
                st.session_state.mode = "neg"
                st.rerun()
            # automatic SAM2 after n − points (cuts down the interaction count)
            if (st.session_state.get("auto_sam_neg", 0) > 0
                    and st.session_state.sam_loaded and mode == "neg"
                    and len(neg) >= st.session_state.auto_sam_neg
                    and ann.get("_auto_sam_at") != len(neg)):
                ann["_auto_sam_at"] = len(neg)
                _execute_sam2(ann, st.session_state.get("run_twice", True))
                st.rerun()

    if mode == "box" and not st.session_state.get("auto_validate_box", False):
        box_ok = ann.get("box") is not None
        if st.button("✅ Validate the box → + points", disabled=not box_ok):
            ann["_box_validated"] = True
            st.session_state.mode = "pos"
            bump_epoch(ann)
            state.log_event("box_validate", ann_id=ann["id"])
            st.rerun()
        if not box_ok:
            st.caption("Draw a box first by clicking and dragging.")
    elif mode == "poly":
        if ann.get("mask") is not None:
            st.caption("Polygon captured — click “✅ Validate” to finish (P0).")
        else:
            st.caption("Draw a closed polygon to generate the mask (P0).")


def _validated_list(current_ann):
    validated = [(i, a) for (i, a) in state.all_annotations() if a is not current_ann]
    if not validated:
        return
    st.divider()
    head_l, head_r = st.columns([2, 2])
    head_l.subheader(f"Validated annotations — {len(validated)} (all images)")
    view = head_r.radio(
        "Display",
        ["Image", "Image + mask", "Mask crop (black background)", "JSON"],
        horizontal=True, label_visibility="collapsed")

    if view == "JSON":
        def _ann_to_dict(i, a):
            d = {"image": st.session_state.images[i]["name"]}
            d.update({k: a.get(k) for k in ("label", "box", "points_pos", "points_neg")})
            if a.get("mask") is not None:
                d["mask"] = {"shape": list(masks.mask_shape(a["mask"])),
                             "area_px": masks.mask_area(a["mask"])}
            else:
                d["mask"] = None
            return d
        st.json([_ann_to_dict(i, a) for i, a in validated])
        return

    ncol = 4
    cols = st.columns(ncol)
    for k, (i, a) in enumerate(validated):
        arr_i = state.arr_of(i)
        with cols[k % ncol]:
            thumb = viz.view_annotation(a, arr_i, view, max_w=300)
            if thumb is not None:
                st.image(thumb)
            else:
                st.caption("_(no box)_")
            st.caption(f"**{a['label']}** · {st.session_state.images[i]['name']}")
            e1, e2 = st.columns(2)
            if e1.button("✏️", key=f"edit_{a['id']}", help="Edit"):
                st.session_state.image_index = i
                st.session_state.current_annotation = a
                a["_editing"] = True
                a["_box_validated"] = a.get("box") is not None
                a["mask"] = masks.ensure_dense(a.get("mask"))   # RLE -> editable dense
                # do not count restoring the geometry as clicks
                a["_click_state"] = {
                    "points": len(a.get("points_pos", [])) + len(a.get("points_neg", [])),
                    "rect": 1 if a.get("box") else 0, "verts": 0}
                bump_epoch(a)
                st.session_state.mode = "pos" if a.get("box") else "box"
                state.log_event("edit_start", ann_id=a["id"])
                st.rerun()
            if e2.button("🗑", key=f"del_{a['id']}", help="Delete"):
                st.session_state.annotations[i].remove(a)
                state.log_event("delete", ann_id=a["id"])
                st.rerun()
