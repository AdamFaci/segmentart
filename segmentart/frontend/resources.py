# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Heavy resources cached on the interface side (SAM2, CLIP).

`st.cache_resource` is a Streamlit mechanism, so it stays in the interface
layer. The backend exposes the bare constructors.
"""

import streamlit as st

from segmentart import config
from segmentart.backend import embeddings, sam2_engine


@st.cache_resource
def get_sam2_predictor():
    return sam2_engine.build_predictor(config.SAM2_MODEL_CFG, config.SAM2_CHECKPOINT)


@st.cache_resource(show_spinner=False)
def get_clip():
    return embeddings.load_clip_model()


def free_sam2():
    """Release the SAM2 predictor and its GPU memory (on entering phase 2)."""
    try:
        get_sam2_predictor.clear()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
