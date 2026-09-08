# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Orchestrator: picks the configuration of every stage of the annotation
pipeline (extractor, prompts, post-processing, SAM passes), per motif.

The baseline is a search (random / grid) over the configuration space, driven by
a caller-supplied objective (predicted IoU, or human correction cost). A learned
policy is the natural extension. No UI, testable.
"""

import itertools

import numpy as np

# Default configuration space (extensible).
CONFIG_SPACE = {
    "extractor": ["dinov2", "sam_encoder", "siglip2"],
    "matcher": ["persam", "matcher", "sam2_memory"],
    "n_pos": [1, 2, 3],
    "n_neg": [0, 1, 2],
    "postproc": ["none", "smooth_fill"],
    "run_twice": [True, False],
    "finetune": ["none", "persam_f", "lora_decoder"],
}


def iter_configs(space=None):
    """Iterate over every configuration (Cartesian product)."""
    space = space or CONFIG_SPACE
    keys = list(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo))


def random_configs(n, space=None, seed=0):
    """Sample `n` random configurations."""
    space = space or CONFIG_SPACE
    rng = np.random.default_rng(seed)
    keys = list(space)
    for _ in range(n):
        yield {k: space[k][int(rng.integers(len(space[k])))] for k in keys}


def search(score_fn, space=None, budget=30, strategy="random", seed=0):
    """Search for the best configuration under `score_fn(config) -> float`
    (higher is better). `strategy` ∈ {"random", "grid"}. Returns
    (best_config, best_score, history)."""
    if strategy == "grid":
        candidates = list(iter_configs(space))
        candidates = candidates[:budget] if budget else candidates
    else:
        candidates = list(random_configs(budget, space, seed))
    history = []
    best, best_s = None, -np.inf
    for cfg in candidates:
        s = float(score_fn(cfg))
        history.append((cfg, s))
        if s > best_s:
            best, best_s = cfg, s
    return best, best_s, history


def per_motif_search(score_fn, motifs, space=None, budget=20, seed=0):
    """Best configuration PER motif: `score_fn(motif, config) -> float`.
    Returns {motif: (config, score)}."""
    out = {}
    for mtf in motifs:
        cfg, sc, _ = search(lambda c, _m=mtf: score_fn(_m, c), space=space,
                            budget=budget, seed=seed)
        out[mtf] = (cfg, sc)
    return out
