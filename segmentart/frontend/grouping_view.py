# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Phase 2 — object grouping and Visual Genome style description."""

import json

import streamlit as st

from segmentart.backend import clustering, serialization
from segmentart.backend import embeddings as emb
from segmentart.backend.accounting import group_accounting
from segmentart.frontend import resources, state


def _embedding(a, arr, clip):
    """Fusable embedding of an annotation, with a session cache."""
    cache = st.session_state._emb_cache
    sig = (id(a.get("mask")) if a.get("mask") is not None else 0,
           a.get("label"), tuple(a.get("box") or ()))
    hit = cache.get(a["id"])
    if hit is not None and hit[0] == sig:
        return hit[1], hit[2]
    ie, te = clustering.compute_embedding(a, arr, clip, masked=True)
    cache[a["id"]] = (sig, ie, te)
    return ie, te


def render_grouping_phase():
    """SAM2 is stopped; validated annotations are grouped by similarity so they
    can be described Visual Genome style."""
    standard = st.session_state.standard

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🧩 Grouping & description")
        st.caption("SAM2 is stopped. Working on validated annotations.")
        st.divider()

        clip = resources.get_clip()
        if clip is None:
            st.warning("CLIP unavailable → falling back to lightweight visual "
                       "descriptors (colour/texture).  \n"
                       "`pip install open_clip_torch`  or  `pip install transformers`")
        else:
            st.success(f"CLIP ready ✓ ({clip[0]})")

        st.subheader("⚙️ Grouping")
        method = st.radio("Method", ["Similarity threshold", "Number of clusters"])
        n_total = len([1 for _, a in state.all_annotations() if a.get("box") is not None])
        if method == "Similarity threshold":
            thr = st.slider("Similarity threshold", 0.50, 0.99, 0.80, 0.01)
            k_clusters = None
        else:
            thr = None
            k_clusters = st.slider("Number of clusters", 1, max(1, n_total),
                                   min(3, max(1, n_total)), 1)
        w_visual = st.slider("Visual ↔ semantic weight", 0.0, 1.0, 0.6, 0.05,
                             help="1 = purely visual, 0 = purely textual (label)",
                             disabled=(clip is None))

        st.divider()
        st.subheader("💾 Export")
        st.download_button(f"Export ({standard})",
            data=serialization.export_standard(
                standard, st.session_state.images, st.session_state.annotations,
                st.session_state.categories, st.session_state.custom_fields,
                st.session_state.relations, st.session_state.group_meta),
            file_name=f"annotations_{standard.lower()}.json",
            mime="application/json")

    view = st.radio("View", ["Editing", "🗺️ Map view"], horizontal=True)

    if not st.session_state.images:
        st.info("No images. Go back to step 1 to annotate.")
        return

    pairs = [(i, a) for (i, a) in state.all_annotations() if a.get("box") is not None]
    if not pairs:
        st.warning("No validated annotation (with a box).")
        return

    # ── Embeddings + global clustering ─────────────────────────────────────────
    with st.spinner("Computing similarities (all images)…"):
        fused = []
        for i, a in pairs:
            ie, te = _embedding(a, state.arr_of(i), clip)
            fused.append(emb.fuse_embeddings(ie, te, w_visual))
        if k_clusters is not None:
            labels = clustering.cluster_by_count(fused, k_clusters)
        else:
            labels = clustering.cluster_by_threshold(fused, thr)
    for (i, a), lbl in zip(pairs, labels):
        a["group"] = int(lbl)

    groups = {}
    for (i, a), lbl in zip(pairs, labels):
        groups.setdefault(lbl, []).append((i, a))

    st.markdown(f"**{len(pairs)} objects ({len(st.session_state.images)} images) "
                f"→ {len(groups)} group(s)** — standard: `{standard}`")

    if view == "🗺️ Map view":
        _cartography(groups)
        return

    _edition(groups, standard, pairs)


def _p3_accounting(pairs):
    """The "P3 summary" panel: leverage of group-level descriptions + export."""
    objects = [{"image": st.session_state.images[i]["name"], "object_id": a["id"],
                "label": a.get("label", ""), "group": a.get("group"),
                "attributes": a.get("attributes") or [],
                "region_desc": a.get("region_desc") or ""}
               for i, a in pairs]
    summary, rows = group_accounting(objects, st.session_state.group_meta)
    with st.expander("📊 P3 summary — leverage of group-level descriptions",
                     expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Objects (N)", summary["n_objects"])
        c2.metric("Groups (G)", summary["n_groups"])
        c3.metric("Objects / group", summary["objects_per_group_mean"])
        c4.metric("Description units saved",
                  f"{summary['reduction_vs_per_object_pct']}%"
                  if summary["reduction_vs_per_object_pct"] is not None else "–")
        st.caption(
            f"{summary['n_groups_described']} group(s) described, covering "
            f"{summary['n_objects']} object(s) — that is {summary['descriptions_per_object']} "
            f"group description per object. Still described at the object level: "
            f"{summary['individual_region_descriptions']} regions, "
            f"{summary['individual_attribute_sets']} attribute sets.")
        import pandas as pd
        st.download_button(
            "Download group_accounting.csv",
            data=pd.DataFrame(rows).to_csv(index=False),
            file_name="group_accounting.csv", mime="text/csv")


def _cartography(groups):
    for g in sorted(groups):
        members = groups[g]
        meta = st.session_state.group_meta.get(g, {})
        name = meta.get("name") or f"Group {g}"
        st.markdown(f"### 🗂️ {name}  ·  {len(members)} object(s)")
        if meta.get("description"):
            st.caption(meta["description"])
        ncol = 6
        cols = st.columns(ncol)
        for k, (i, m) in enumerate(members):
            crop = clustering.object_crop(m, state.arr_of(i), masked=True, bg=(0, 0, 0))
            if crop is not None:
                crop.thumbnail((130, 130))
                cols[k % ncol].image(crop, caption=f"{m['label']}")
        st.divider()


def _edition(groups, standard, pairs):
    _p3_accounting(pairs)
    prev = st.session_state.setdefault("_grp_prev", {})
    for g in sorted(groups):
        members = groups[g]
        shared, diffs = clustering.group_characterization(
            [(a, state.arr_of(i)) for i, a in members])
        meta = st.session_state.group_meta.setdefault(g, {"name": "", "description": ""})

        with st.expander(f"🗂️ Group {g}  —  {len(members)} object(s): "
                         f"{', '.join(sorted({m['label'] for _, m in members}))}",
                         expanded=True):
            cols = st.columns(min(6, len(members)))
            for k, (i, m) in enumerate(members):
                crop = clustering.object_crop(m, state.arr_of(i), masked=False)
                if crop is not None:
                    crop.thumbnail((140, 140))
                    cols[k % len(cols)].image(
                        crop, caption=f"{m['label']} · {st.session_state.images[i]['name']}")

            ca, cb = st.columns(2)
            ca.markdown("**In common**: " +
                        (", ".join(f"{k}={v}" for k, v in shared.items()) or "_none_"))
            cb.markdown("**Differences**: " +
                        (", ".join(f"{k}∈{v}" for k, v in diffs.items()) or "_none_"))

            meta["name"] = st.text_input("Group name", value=meta["name"],
                                         key=f"gname_{g}")
            sugg = ""
            if shared:
                sugg = "Objects sharing " + ", ".join(f"{k} {v}" for k, v in shared.items())
                if diffs:
                    sugg += "; differing in " + ", ".join(diffs.keys())
            meta["description"] = st.text_area(
                "Group description (similarities / differences)",
                value=meta["description"] or sugg, key=f"gdesc_{g}", height=80)

            # phase 2 telemetry: group description (P3 event)
            cur = (meta.get("name", ""), meta.get("description", ""))
            if prev.get(g) != cur and (cur[0] or cur[1]):
                prev[g] = cur
                state.telemetry().log("group_describe", pipeline="P3", group_id=g,
                                      n_members=len(members),
                                      named=bool(cur[0]), desc_len=len(cur[1]))

            st.markdown("**Objects in this group:**")
            for i, m in members:
                rb = clustering.rule_attributes(m, state.arr_of(i))
                with st.container(border=True):
                    st.markdown(f"`{m['label']}` #{m['id']} · "
                                f"{st.session_state.images[i]['name']} — "
                                f"_{', '.join(f'{k}:{v}' for k, v in rb.items())}_")
                    cur_attr = m.get("attributes")
                    if cur_attr is None:
                        cur_attr = [f"{k}:{v}" for k, v in rb.items()]
                    txt = st.text_input("Attributes (comma-separated)",
                                        value=", ".join(cur_attr), key=f"attr_{m['id']}")
                    m["attributes"] = [s.strip() for s in txt.split(",") if s.strip()]
                    if standard == "VisualGenome":
                        m["region_desc"] = st.text_input(
                            "Region description (sentence)",
                            value=m.get("region_desc", ""), key=f"region_{m['id']}")
                    if standard == "Custom":
                        for f in st.session_state.custom_fields:
                            m["custom_" + f] = st.text_input(
                                f.capitalize(), value=m.get("custom_" + f, ""),
                                key=f"cf_{m['id']}_{f}")

    if standard == "VisualGenome":
        _relations(pairs)

    with st.expander("👁️ JSON preview (current standard)"):
        st.json(json.loads(serialization.export_standard(
            standard, st.session_state.images, st.session_state.annotations,
            st.session_state.categories, st.session_state.custom_fields,
            st.session_state.relations, st.session_state.group_meta)))


def _relations(pairs):
    st.divider()
    st.subheader("🔗 Relations (subject → predicate → object, all images)")
    rel_list = st.session_state.relations
    opts = {f"{a['label']} #{a['id']} ({st.session_state.images[i]['name']})": a["id"]
            for i, a in pairs}
    rc1, rc2, rc3, rc4 = st.columns([3, 3, 3, 1])
    subj = rc1.selectbox("Subject", list(opts), key="rel_subj")
    pred = rc2.text_input("Predicate", key="rel_pred",
                          placeholder="e.g. next to, holds, on")
    obj = rc3.selectbox("Object", list(opts), key="rel_obj")
    if rc4.button("➕", key="rel_add") and pred.strip():
        rel_list.append({"subj": opts[subj], "pred": pred.strip(), "obj": opts[obj]})
        state.telemetry().log("relation_add", pipeline="P3", predicate=pred.strip())
        st.rerun()
    id2label = {a["id"]: a["label"] for _, a in pairs}
    for k, r in enumerate(list(rel_list)):
        cc1, cc2 = st.columns([6, 1])
        cc1.markdown(f"**{id2label.get(r['subj'], r['subj'])}** "
                     f"→ _{r['pred']}_ → **{id2label.get(r['obj'], r['obj'])}**")
        if cc2.button("🗑", key=f"rel_del_{k}"):
            rel_list.pop(k)
            st.rerun()
