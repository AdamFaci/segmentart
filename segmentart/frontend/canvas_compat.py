# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Compatibility shim between Streamlit and streamlit-drawable-canvas.

Two independent incompatibilities are handled here, so that the rest of the frontend can
call `st_canvas` without worrying about which versions are installed.

1. **Keyword arguments.** The canvas signature changes between releases: `display_toolbar`
   exists up to 0.9.x and was removed in the 0.10+ rewrite. We therefore introspect the
   installed function and drop the cosmetic keywords it does not accept, while failing
   loudly if a keyword we actually depend on has disappeared.

2. **`image_to_url`.** Canvas releases before the rewrite call
   `streamlit.elements.image.image_to_url()` with the old signature (2nd argument =
   integer width). That internal function moved and changed signature across Streamlit
   versions, raising either of:
       AttributeError: module 'streamlit.elements.image' has no attribute 'image_to_url'
       AttributeError: 'int' object has no attribute 'width'
   We substitute an implementation that returns a usable background URL — but only for
   the versions that need it, so that we never shadow a live Streamlit internal.

Importing this module applies both fixes and exposes `st_canvas` / `CANVAS_AVAILABLE`.
"""

import inspect

import numpy as np
from PIL import Image

# Keywords the annotation view relies on. If the installed canvas stops accepting one of
# these, silently dropping it would break annotation in ways that are hard to diagnose
# (drawings not restored, canvas not sized, state shared between images), so we raise.
REQUIRED_CANVAS_KWARGS = frozenset({
    "background_image",
    "drawing_mode",
    "initial_drawing",
    "height",
    "width",
    "key",
    "update_streamlit",
})


def _canvas_version():
    """(major, minor) of the installed canvas, or None if it cannot be determined."""
    try:
        from importlib.metadata import version
        parts = version("streamlit-drawable-canvas").split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def _shim_image_to_url(image, *args, **kwargs):
    import base64 as _b64
    import io as _io
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    # PNG encoding is done once (it is the main cost) and kept on the image
    # object. PNG rather than JPEG: fabric.js can render a black background
    # with some JPEG data URLs.
    data = getattr(image, "_dc_png", None)
    if data is None:
        buf = _io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        try:
            image._dc_png = data
        except Exception:
            pass

    # The image is registered with Streamlit's media file manager, which returns
    # a short URL (/media/<hash>.png) that the browser caches. The WebSocket
    # payload stays small, unlike a base64 data URL sent on every rerun.
    # Re-registering on each run is cheap and keeps the media manager from
    # cleaning the file up.
    try:
        from streamlit.runtime import get_instance
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        rt = get_instance()
        ctx = get_script_run_ctx()
        coords = f"{ctx.session_id}-dc-{abs(hash(data))}"
        return rt.media_file_mgr.add(data, "image/png", coords)
    except Exception:
        return "data:image/png;base64," + _b64.b64encode(data).decode()


def _patch_image_to_url():
    """Install the shim only for canvas versions that call `image_to_url`.

    The 0.10+ rewrite takes the background image directly and never calls it; patching
    there would replace a Streamlit internal that nothing in our stack uses, for no gain.
    """
    ver = _canvas_version()
    if ver is not None and ver >= (0, 10):
        return
    try:
        import streamlit.elements.image as _img_mod
        _img_mod.image_to_url = _shim_image_to_url
    except Exception:
        pass


_patch_image_to_url()

# The component must be imported AFTER the patch is applied.
#
# Catch every exception, not just ImportError: an incompatible Streamlit/canvas pair
# fails at import time with a StreamlitAPIException (the 0.10+ component registration
# rejecting the host Streamlit). Letting that escape takes the whole application down at
# start-up, when the honest outcome is "the canvas is unavailable, phase 2 still works".
try:
    from streamlit_drawable_canvas import st_canvas as _st_canvas
    CANVAS_AVAILABLE = True
    CANVAS_IMPORT_ERROR = None
except Exception as exc:                                  # noqa: BLE001
    _st_canvas = None
    CANVAS_AVAILABLE = False
    CANVAS_IMPORT_ERROR = exc


def _supported_kwargs(func):
    """Names accepted by `func`, or None when it takes arbitrary keywords."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return {name for name, p in params.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY)}


_SUPPORTED = _supported_kwargs(_st_canvas) if CANVAS_AVAILABLE else None


def st_canvas(**kwargs):
    """Call the installed canvas, dropping keywords this version does not accept.

    Cosmetic keywords that came and went across releases (`display_toolbar`,
    `point_display_radius`, …) are silently ignored when unsupported. A keyword listed in
    `REQUIRED_CANVAS_KWARGS` going missing means the installed version is genuinely
    incompatible, so it raises instead of degrading quietly.
    """
    if not CANVAS_AVAILABLE:
        detail = f"\n\nImport failed with: {CANVAS_IMPORT_ERROR!r}" if CANVAS_IMPORT_ERROR else ""
        raise RuntimeError(
            "streamlit-drawable-canvas is not usable. Install or repair it with:\n"
            "    pip install streamlit-drawable-canvas" + detail
        )
    if _SUPPORTED is None:
        return _st_canvas(**kwargs)

    missing_required = REQUIRED_CANVAS_KWARGS - _SUPPORTED
    if missing_required:
        from importlib.metadata import version
        try:
            installed = version("streamlit-drawable-canvas")
        except Exception:
            installed = "unknown"
        raise RuntimeError(
            f"streamlit-drawable-canvas {installed} does not accept "
            f"{sorted(missing_required)}, which SegmentART requires. "
            "Please report this at the project's issue tracker, or pin a known-good "
            "version:\n    pip install 'streamlit-drawable-canvas>=0.9.3,<0.14'"
        )

    return _st_canvas(**{k: v for k, v in kwargs.items() if k in _SUPPORTED})
