# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Compatibility shim between Streamlit and streamlit-drawable-canvas.

streamlit-drawable-canvas calls streamlit.elements.image.image_to_url() with the
old signature (2nd argument = integer width). That internal function has moved
and/or changed signature across Streamlit versions, which raises either:
    AttributeError: module 'streamlit.elements.image' has no attribute 'image_to_url'
    AttributeError: 'int' object has no attribute 'width'

To avoid depending on an unstable internal API, we substitute an implementation
that returns an image URL usable as the component background. Importing this
module applies the patch and exposes `st_canvas` / `CANVAS_AVAILABLE`.
"""

import numpy as np
from PIL import Image


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
        url = rt.media_file_mgr.add(data, "image/png", coords)
        return url
    except Exception:
        url = "data:image/png;base64," + _b64.b64encode(data).decode()
        return url


def _patch_image_to_url():
    try:
        import streamlit.elements.image as _img_mod
        _img_mod.image_to_url = _shim_image_to_url
    except Exception:
        pass


_patch_image_to_url()

# The component must be imported AFTER the patch is applied.
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    st_canvas = None
    CANVAS_AVAILABLE = False
