# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""SegmentART UI entry point (orchestrates the two phases)."""

import streamlit as st

# The image_to_url patch is applied when canvas_compat is imported, which happens
# through annotate_view below (before the canvas is ever used).
from segmentart.frontend import resources, state
from segmentart.frontend.annotate_view import render_annotation_phase
from segmentart.frontend.grouping_view import render_grouping_phase


def main():
    st.set_page_config(page_title="SegmentART", layout="wide", page_icon="🎨")
    state.init_state()

    st.markdown("""
    <style>
    /* offset so content is not hidden behind the Streamlit menu bar */
    .block-container { padding-top: 3.5rem; }
    .stButton > button { width:100%; border-radius:6px; }
    section[data-testid="stSidebar"] .stButton > button { font-size:0.82rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Step and standard selectors (shared by both phases) ────────────────────
    with st.sidebar:
        st.markdown("### 🎨 SegmentART")
        ph = st.radio("Step", ["1 · Annotation", "2 · Grouping & description"],
                      index=0 if st.session_state.phase == "annot" else 1)
        new_phase = "annot" if ph.startswith("1") else "group"
        if new_phase != st.session_state.phase:
            st.session_state.phase = new_phase
            if new_phase == "group":
                resources.free_sam2()              # release SAM2 when entering phase 2
                st.session_state.sam_loaded = False
            st.rerun()

        st.session_state.standard = st.selectbox(
            "Annotation standard", ["VisualGenome", "COCO", "Custom"],
            index=["VisualGenome", "COCO", "Custom"].index(st.session_state.standard))
        if st.session_state.standard == "Custom":
            cf = st.text_input("Custom fields (comma-separated)",
                               value=", ".join(st.session_state.custom_fields))
            st.session_state.custom_fields = [s.strip() for s in cf.split(",") if s.strip()]
        st.divider()

    if st.session_state.phase == "group":
        render_grouping_phase()
    else:
        render_annotation_phase()


if __name__ == "__main__":
    main()
