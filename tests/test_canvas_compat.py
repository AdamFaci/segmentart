# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""The canvas shim must absorb signature changes in streamlit-drawable-canvas.

`display_toolbar` existed up to 0.9.x and disappeared in the 0.10+ rewrite; passing it to
a recent canvas raises `TypeError: st_canvas() got an unexpected keyword argument`. These
tests pin the behaviour of the shim against both signatures without importing Streamlit.
"""

import pytest

from segmentart.frontend import canvas_compat as cc

# What annotate_view actually passes, as of this version.
CALL_KWARGS = dict(
    background_image="<image>",
    background_color="#f0f0f0",
    initial_drawing={},
    drawing_mode="rect",
    stroke_color="#1E88E5",
    fill_color="rgba(30,136,229,0.12)",
    stroke_width=2,
    point_display_radius=7,
    update_streamlit=True,
    height=480,
    width=640,
    display_toolbar=True,
    key="canvas-0",
)


def _old_signature(background_image=None, background_color="", initial_drawing=None,
                   drawing_mode="freedraw", stroke_color="black", fill_color=None,
                   stroke_width=20, point_display_radius=3, update_streamlit=True,
                   height=400, width=600, display_toolbar=True, key=None):
    """streamlit-drawable-canvas 0.9.x."""
    return dict(locals())


def _new_signature(background_image=None, background_color="", initial_drawing=None,
                   drawing_mode="freedraw", stroke_color="black", fill_color=None,
                   stroke_width=20, point_display_radius=3, update_streamlit=True,
                   height=400, width=600, key=None, label="", disabled=False):
    """streamlit-drawable-canvas 0.13.x — no display_toolbar."""
    return dict(locals())


def _incompatible_signature(width=600, height=400, key=None):
    """A hypothetical release that dropped keywords we depend on."""
    return dict(locals())


@pytest.fixture
def shim(monkeypatch):
    """Point the shim at a stand-in canvas and return a way to swap signatures."""
    def use(func):
        monkeypatch.setattr(cc, "_st_canvas", func)
        monkeypatch.setattr(cc, "CANVAS_AVAILABLE", True)
        monkeypatch.setattr(cc, "_SUPPORTED", cc._supported_kwargs(func))
    return use


def test_old_signature_receives_every_keyword(shim):
    shim(_old_signature)
    got = cc.st_canvas(**CALL_KWARGS)
    assert got["display_toolbar"] is True
    assert got["initial_drawing"] == {}
    assert (got["width"], got["height"]) == (640, 480)


def test_new_signature_drops_display_toolbar(shim):
    """The regression: 0.13 has no display_toolbar, and must not raise."""
    shim(_new_signature)
    got = cc.st_canvas(**CALL_KWARGS)
    assert "display_toolbar" not in got
    assert got["drawing_mode"] == "rect"
    assert got["key"] == "canvas-0"


def test_required_keywords_always_survive(shim):
    """Whatever is dropped, the keywords annotation depends on must get through."""
    for signature in (_old_signature, _new_signature):
        shim(signature)
        got = cc.st_canvas(**CALL_KWARGS)
        assert cc.REQUIRED_CANVAS_KWARGS <= set(got)


def test_missing_required_keyword_raises_a_useful_error(shim):
    shim(_incompatible_signature)
    with pytest.raises(RuntimeError, match="does not accept"):
        cc.st_canvas(**CALL_KWARGS)


def test_var_keyword_signature_is_passed_through_untouched(shim):
    def accepts_anything(**kwargs):
        return kwargs

    shim(accepts_anything)
    assert cc.st_canvas(**CALL_KWARGS) == CALL_KWARGS


def test_absent_canvas_raises_with_install_instructions(monkeypatch):
    monkeypatch.setattr(cc, "CANVAS_AVAILABLE", False)
    monkeypatch.setattr(cc, "CANVAS_IMPORT_ERROR", None)
    with pytest.raises(RuntimeError, match="pip install streamlit-drawable-canvas"):
        cc.st_canvas(**CALL_KWARGS)


def test_image_to_url_patch_is_skipped_on_rewritten_versions(monkeypatch):
    """The 0.10+ rewrite never calls image_to_url; we must not shadow the internal."""
    import streamlit.elements.image as img_mod

    monkeypatch.setattr(cc, "_canvas_version", lambda: (0, 13))
    monkeypatch.setattr(img_mod, "image_to_url", "sentinel", raising=False)
    cc._patch_image_to_url()
    assert img_mod.image_to_url == "sentinel"


def test_image_to_url_patch_applies_to_old_versions(monkeypatch):
    import streamlit.elements.image as img_mod

    monkeypatch.setattr(cc, "_canvas_version", lambda: (0, 9))
    monkeypatch.setattr(img_mod, "image_to_url", "sentinel", raising=False)
    cc._patch_image_to_url()
    assert img_mod.image_to_url is cc._shim_image_to_url
