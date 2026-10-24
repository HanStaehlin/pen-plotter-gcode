"""Monocular depth estimation, used to work out which way a surface turns.

Shading tells you where the light is, not where the surface is, which is why
an orientation field recovered from tone rings every bright patch instead of
wrapping around the form.  A depth estimate gives the geometry directly.

Torch is an optional dependency: everything else in the package works without
it, and this module reports itself unavailable rather than failing to import.
"""

from __future__ import annotations

import importlib.util
import warnings

import cv2
import numpy as np

_MODEL = None
_TRANSFORM = None
_DEVICE = None

#: Depth Anything V2 Small is the default: about 0.1-0.2 s on Metal, and it
#: resolves a nose, eye sockets and a muzzle where MiDaS_small returns one
#: smooth blob per subject - which is what the shadow band and the hatching
#: direction are built from.  MiDaS_small (torch.hub) is the fallback when
#: ``transformers`` is not installed.
MODELS = ("MiDaS_small", "DPT_Hybrid", "DPT_Large", "DepthAnythingV2_small")

#: Depth Anything V2 Small (Apache-2.0), official weights on Hugging Face.
#: Needs the optional ``transformers`` package.
_DA2 = "depth-anything/Depth-Anything-V2-Small-hf"


def available() -> bool:
    return importlib.util.find_spec("torch") is not None


def device_name() -> str:
    """Which accelerator will be used, as a human-readable string."""
    try:
        import torch
    except ImportError:
        return "unavailable"
    if torch.backends.mps.is_available():
        return "mps (Metal)"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve(model: str) -> str:
    """The model that will actually run: Depth Anything needs ``transformers``."""
    if model == "DepthAnythingV2_small" and \
            importlib.util.find_spec("transformers") is None:
        return "MiDaS_small"
    return model


def load(model: str = "DepthAnythingV2_small"):
    """Load and cache the depth model.  First call may download weights."""
    global _MODEL, _TRANSFORM, _DEVICE
    model = resolve(model)
    if _MODEL is not None and _MODEL[0] == model:
        return _MODEL[1], _TRANSFORM, _DEVICE

    import contextlib
    import io
    import torch
    if model == "DepthAnythingV2_small":
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        with warnings.catch_warnings(), \
             contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            processor = AutoImageProcessor.from_pretrained(_DA2)
            net = AutoModelForDepthEstimation.from_pretrained(_DA2)
        _DEVICE = _pick_device()
        net.to(_DEVICE).eval()
        _MODEL = (model, net)
        _TRANSFORM = processor
        return net, processor, _DEVICE

    # torch.hub narrates every cache hit on both stdout and stderr; the
    # caller is a web app and a CLI, neither of which wants that mixed into
    # its own output.
    with warnings.catch_warnings(), \
         contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        net = torch.hub.load("intel-isl/MiDaS", model, trust_repo=True)
        tfs = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = tfs.small_transform if model == "MiDaS_small" else tfs.dpt_transform

    _DEVICE = _pick_device()
    net.to(_DEVICE).eval()
    _MODEL = (model, net)
    _TRANSFORM = transform
    return net, transform, _DEVICE


def estimate(img_rgb: np.ndarray, size: tuple[int, int] | None = None,
             model: str = "DepthAnythingV2_small") -> np.ndarray:
    """Relative depth for an RGB image, 0..1 with 1 nearest.

    ``size`` is (width, height) to resample to — normally the working
    resolution, since the orientation field is built at that scale.
    """
    import torch

    model = resolve(model)
    net, transform, device = load(model)
    with torch.no_grad():
        if model == "DepthAnythingV2_small":
            batch = transform(images=img_rgb, return_tensors="pt")
            pred = net(pixel_values=batch["pixel_values"].to(device)).predicted_depth
        else:
            pred = net(transform(img_rgb).to(device))
        target = (img_rgb.shape[0], img_rgb.shape[1]) if size is None \
            else (size[1], size[0])
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=target, mode="bicubic",
            align_corners=False).squeeze()
    depth = pred.detach().to("cpu").numpy().astype(np.float32)

    lo, hi = float(np.percentile(depth, 1)), float(np.percentile(depth, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(depth)
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0)


def form_orientation(depth: np.ndarray, sigma: float = 6.0):
    """Orientation field for hatching that follows the surface.

    Hatching reads as form when it runs the way the surface *bends*: an
    engraver rings a cylinder around its circumference, not along its axis.
    That direction is the principal curvature of the depth surface, and for a
    height field the curvature is the Hessian, so the eigenvector with the
    larger absolute eigenvalue is the line to follow.

    Returns ``(cos 2t, sin 2t, strength)`` — double-angle form, matching the
    tone-derived field so the two can be blended.
    """
    z = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), max(0.8, sigma))
    zxx = cv2.Sobel(z, cv2.CV_32F, 2, 0, ksize=5)
    zyy = cv2.Sobel(z, cv2.CV_32F, 0, 2, ksize=5)
    zxy = cv2.Sobel(z, cv2.CV_32F, 1, 1, ksize=5)

    diff = zxx - zyy
    disc = np.sqrt(diff * diff + 4.0 * zxy * zxy) + 1e-12
    mean = 0.5 * (zxx + zyy)
    lam1 = mean + 0.5 * disc          # larger signed eigenvalue
    lam2 = mean - 0.5 * disc

    # Double angle of the eigenvector belonging to lam1.
    c2 = diff / disc
    s2 = 2.0 * zxy / disc
    # Where the *other* principal direction bends harder, take that instead;
    # a quarter turn is a sign flip in double-angle space.
    swap = np.abs(lam2) > np.abs(lam1)
    c2 = np.where(swap, -c2, c2)
    s2 = np.where(swap, -s2, s2)

    # Confidence: how far from umbilic this point is.  On a sphere both
    # curvatures match and the direction is arbitrary, so it must not be
    # allowed to outvote its neighbours when the field is smoothed.
    strength = np.abs(lam1 - lam2)
    ref = float(np.percentile(strength, 98))
    strength = np.clip(strength / ref, 0.0, 1.0) if ref > 1e-12 \
        else np.zeros_like(strength)
    return c2.astype(np.float32), s2.astype(np.float32), strength.astype(np.float32)
