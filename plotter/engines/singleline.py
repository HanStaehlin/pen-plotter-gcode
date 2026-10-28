"""One line: the whole picture as a single unbroken stroke (see singleline.py).

The learned line network finds the contours, the most important are chained
into one path the way a continuous-line artist works, and the path is
fitted smooth on the GPU.  Without torch the chained path is used as it is.
"""

from __future__ import annotations

import cv2
import numpy as np

#: The fit runs at this long-edge resolution, whatever the canvas.
WORK_PX = 480


def generate(value, coverage, canvas, pen, cfg, features=None, form=None, image_rgb=None):
    from .. import learned_lines, singleline
    h, w = coverage.shape
    s = WORK_PX / max(h, w)
    W, H = max(8, int(round(w * s))), max(8, int(round(h * s)))
    rgb = image_rgb if image_rgb is not None else (
        np.dstack([value] * 3) * 255).astype(np.uint8)

    if learned_lines.available():
        dark = learned_lines.draw(rgb, (W, H), style="contour")
    else:
        # Without the network: dark, thin structure from the photo itself.
        small = cv2.resize(value, (W, H), interpolation=cv2.INTER_AREA)
        dark = np.clip(1.0 - small, 0.0, 1.0) * (cv2.Laplacian(small, cv2.CV_32F) > 0.02)
    from .. import subject
    near = subject.near(rgb, (W, H), cfg)
    face = singleline.face_weight(rgb, (W, H))
    if face is not None:
        # The face outranks everything else near the camera.
        base = near if near is not None else np.ones_like(face)
        near = np.maximum(0.3 * base, face)

    contours = singleline.select_contours(dark, near, detail=cfg.singleline_detail)
    if not contours:
        return []
    path = singleline.chain_contours(contours)
    try:
        import torch  # noqa: F401
        line = singleline.fit(singleline.contour_target(path, dark), init=path)
    except ImportError:
        line = path
    return [np.asarray(line, np.float32) / s]
