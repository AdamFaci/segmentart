# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""SAM2 engine: building the predictor and running inference.

This module does not depend on Streamlit. Caching of the predictor (which is
expensive to load) is handled by the interface layer.
"""

import numpy as np

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


def pick_device():
    """Return the best available device (cuda > mps > cpu)."""
    if not SAM2_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_predictor(model_cfg, checkpoint, device=None):
    """Build a SAM2ImagePredictor. Raises if SAM2 is not installed."""
    if not SAM2_AVAILABLE:
        raise RuntimeError("SAM2 is not installed.")
    device = device or pick_device()
    model = build_sam2(model_cfg, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def predict(predictor, image_array, box=None, points_pos=None, points_neg=None,
            mask_input=None):
    """
    Run SAM2 with a box and/or points (at least one input is required) and,
    optionally, an input mask (logits of shape (1, 256, 256)).

    `box` is given as (x0, y0, w, h); it is converted to (x0, y0, x1, y1).
    Returns (masks, scores, logits) sorted by decreasing score.
    """
    predictor.set_image(image_array)

    input_box = None
    if box is not None:
        x0, y0, w, h = box
        input_box = np.array([x0, y0, x0 + w, y0 + h])

    points_pos = points_pos or []
    points_neg = points_neg or []
    if points_pos or points_neg:
        coords = np.array(points_pos + points_neg)
        labels = np.array([1] * len(points_pos) + [0] * len(points_neg))
    else:
        coords = labels = None

    masks, scores, low_res = predictor.predict(
        box=input_box, point_coords=coords, point_labels=labels,
        mask_input=mask_input, multimask_output=True)

    order = list(np.argsort(scores)[::-1])
    return ([masks[i] for i in order],
            [float(scores[i]) for i in order],
            [low_res[i] for i in order])
