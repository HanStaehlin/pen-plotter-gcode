"""Which part of the photo is the subject: a soft 0..1 mask.

Every style spends ink in proportion to darkness, so a dark wall behind a
sitter costs as much plot time as the sitter and competes with them for the
eye.  Depth was the first answer and fails whenever the backdrop is also
near - a door frame beside a seated woman.  A salient-object model answers
the question directly.

The model is BiRefNet-lite (MIT), run through rembg.  Compared on the test
photos with u2net, IS-Net and the BiRefNet portrait model, it was the only
one that kept a whole jacket under a beard, the curls of a head of hair and
a dog's body, while finding almost nothing in a landscape.

That last case matters: a photo with no single subject should be drawn
whole, so a mask covering almost nothing or almost everything is treated as
"no subject" and returns None.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

MODEL = "birefnet-general-lite"
#: Below this share of the picture there is no real subject (a landscape's
#: one rock); above it the subject is the picture.
MIN_SHARE, MAX_SHARE = 0.04, 0.85

_SESSION = None
_CACHE: dict = {}


def available() -> bool:
    import importlib.util
    return importlib.util.find_spec("rembg") is not None


def _session():
    global _SESSION
    if _SESSION is None:
        from rembg import new_session
        _SESSION = new_session(MODEL)
    return _SESSION


def raw_mask(image_rgb: np.ndarray) -> np.ndarray:
    """The model's mask at the photo's size, 0..1.  Cached per photo."""
    key = hashlib.sha1(np.ascontiguousarray(image_rgb[::7, ::7]).tobytes()).hexdigest()
    if key not in _CACHE:
        from PIL import Image
        from rembg import remove
        h, w = image_rgb.shape[:2]
        k = min(1.0, 1024 / max(h, w))           # the model works at 1024 anyway
        small = cv2.resize(image_rgb, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        m = np.asarray(remove(Image.fromarray(small), session=_session(), only_mask=True))
        _CACHE.clear()                           # one photo at a time is enough
        _CACHE[key] = cv2.resize(m.astype(np.float32) / 255.0, (w, h),
                                 interpolation=cv2.INTER_LINEAR)
    return _CACHE[key]


def mask(image_rgb: np.ndarray, size: tuple[int, int] | None = None,
         feather: float = 0.012) -> np.ndarray | None:
    """Soft subject mask at ``size`` (w, h), or None when there is no subject.

    The edge is feathered by ``feather`` of the long side, so ink fades out
    over a few millimetres instead of stopping on a cut-out line.
    """
    m = raw_mask(image_rgb)
    share = float((m > 0.5).mean())
    if share < MIN_SHARE or share > MAX_SHARE:
        return None
    if size is not None:
        m = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    sigma = max(0.5, feather * max(m.shape))
    return np.clip(cv2.GaussianBlur(m, (0, 0), sigma), 0.0, 1.0).astype(np.float32)


def weight(subject: np.ndarray, background: float) -> np.ndarray:
    """Ink multiplier: 1 on the subject, ``background`` behind it."""
    b = float(np.clip(background, 0.0, 1.0))
    return (b + (1.0 - b) * subject).astype(np.float32)


#: A line counts as backdrop when, on average, it lies this little on the
#: subject.  The mask's feathered edge sits near 0.5, so the subject's own
#: silhouette stays.
LINE_ON_SUBJECT = 0.25


def keep_lines(lines, scores, near_map, focus: float):
    """Drop line strokes that lie on the backdrop, in proportion to ``focus``.

    Tone fades smoothly, but a line is either drawn or not, and a door frame
    or the furniture behind a sitter drawn at full weight pulls the eye just
    as hard as before.
    """
    if near_map is None or focus <= 0:
        return lines, scores
    from .engines.outline import _mean_along
    cut = LINE_ON_SUBJECT * float(np.clip(focus, 0.0, 1.0))
    keep = [i for i, p in enumerate(lines) if _mean_along(near_map, p) >= cut]
    return [lines[i] for i in keep], [scores[i] for i in keep]


def near(image_rgb: np.ndarray, size: tuple[int, int], cfg=None) -> np.ndarray | None:
    """0..1, how much each pixel belongs to the subject - what the styles use.

    The segmentation mask, or None when segmentation sees no single subject
    (a landscape, or a subject filling the frame): then the whole frame is
    drawn.  Only if segmentation cannot run at all does depth stand in.
    """
    if getattr(cfg, "subject_mask", True) and available():
        try:
            return mask(image_rgb, size)
        except Exception:           # model missing or its download failed
            pass
    from . import depth as depth_mod
    if not depth_mod.available():
        return None
    d = depth_mod.estimate(image_rgb, size=size,
                           model=getattr(cfg, "depth_model", "DepthAnythingV2_small"))
    n = np.clip((d - 0.25) / 0.45, 0.0, 1.0)
    return (n * n * (3.0 - 2.0 * n)).astype(np.float32)          # smoothstep
