# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Embeddings for semantic grouping (CLIP, with a lightweight fallback).

No UI dependency. Caching of the model is handled by the interface layer.
"""

import numpy as np


def load_clip_model():
    """
    Load a CLIP model (image + text). Tries open_clip first, then transformers.
    Returns a (backend, ...) tuple, or None if no backend is available.
    """
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        return ("open_clip", model, preprocess, tokenizer)
    except Exception:
        pass
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        return ("transformers", model, proc, None)
    except Exception:
        return None


def clip_embeddings(clip, pil_img, text):
    """Return normalized (img_emb, txt_emb) as np.float32, per the backend."""
    import torch
    backend = clip[0]
    with torch.no_grad():
        if backend == "open_clip":
            _, model, preprocess, tokenizer = clip
            img = preprocess(pil_img).unsqueeze(0)
            ie = model.encode_image(img)
            te = model.encode_text(tokenizer([text or "object"]))
        else:  # transformers
            _, model, proc, _ = clip
            ii = proc(images=pil_img, return_tensors="pt")
            ie = model.get_image_features(**ii)
            ti = proc(text=[text or "object"], return_tensors="pt",
                      padding=True, truncation=True)
            te = model.get_text_features(**ti)
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        return (ie[0].cpu().numpy().astype(np.float32),
                te[0].cpu().numpy().astype(np.float32))


def light_embedding(pil_img):
    """Fallback without CLIP: color histogram (normalized vector)."""
    arr = np.asarray(pil_img.resize((32, 32))).astype(np.float32) / 255.0
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=8, range=(0, 1), density=True)
        hist.append(h)
    feat = np.concatenate(hist)
    n = np.linalg.norm(feat) or 1.0
    return (feat / n).astype(np.float32)


def fuse_embeddings(ie, te, w_visual):
    """Fuse the image and text embeddings (visual weighting); returns a
    normalized vector."""
    if te is None:
        v = ie
    else:
        v = w_visual * ie + (1.0 - w_visual) * te
    n = np.linalg.norm(v) or 1.0
    return v / n
