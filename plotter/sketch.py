"""Optimisation-based sketching (CLIPasso-style, own differentiable renderer).

Vinker et al.'s CLIPasso (2022) fits a handful of Bezier strokes directly
against a CLIP loss by backpropagating through a differentiable vector
rasteriser, rather than matching tone pixel-by-pixel the way every other
engine in this package does. The result optimises for *recognisability*, not
coverage, which is a fundamentally different way to draw and can produce a
far sparser, more legible sketch than any tone-matching method.

This reimplements the idea rather than depending on the reference
``diffvg`` rasteriser: diffvg is not on PyPI and builds a native CUDA/CMake
extension from source, which is fragile with no CUDA present and nothing to
gain on a machine that already runs CLIP itself on Metal. The renderer here
is plain PyTorch - a soft distance-to-curve raster - so it differentiates
through ordinary autograd and needs nothing beyond ``torch``.

Everything is optional: importing this module without torch installed
raises only when ``run()`` is actually called, so the rest of the package is
unaffected.
"""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass

import numpy as np


def available() -> bool:
    return (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("open_clip") is not None)


def device_name() -> str:
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
    """CPU, deliberately, on Apple Silicon.

    Measured: backpropagating through this ViT on MPS costs ~2s/iteration
    against ~40ms/iteration on CPU - a ~47x slowdown, not a rounding error.
    Forward-only inference (what the depth estimator uses) is genuinely fast
    on MPS; it is specifically the *backward* pass through this attention
    architecture that Apple's Metal backend handles far worse than plain CPU
    does today. Iterative gradient-based optimisation lives or dies on the
    backward pass, so CPU is the faster device for this technique even
    though MPS would be the right choice for most other torch workloads.
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_CLIP = None   # (model, device) cache


def _load_clip():
    global _CLIP
    if _CLIP is not None:
        return _CLIP
    import contextlib
    import io
    import open_clip

    with warnings.catch_warnings(), \
         contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai")
    device = _pick_device()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _CLIP = (model, device)
    return _CLIP


# CLIP's own published normalisation (this is what the model was trained on;
# using anything else quietly degrades every feature it produces).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class SketchResult:
    strokes: "object"          # torch.Tensor (N, 4, 2), normalised 0-1
    width: float                # normalised stroke half-width used
    canvas: int                 # render resolution the strokes were fit at
    loss_curve: list


def render_soft(strokes, width, canvas: int, sharpness: float = 260.0,
                samples: int = 20):
    """Differentiable raster: 1.0 = paper, 0.0 = full ink.

    Distance from every pixel to a stroke's sampled points, softened by a
    sigmoid, rather than an analytic point-to-segment distance - it needs no
    per-stroke branching, so the whole batch renders as ordinary tensor ops
    and autograd differentiates it for free.
    """
    import torch

    device = strokes.device
    n = strokes.shape[0]
    t = torch.linspace(0.0, 1.0, samples, device=device).view(1, samples, 1)
    p0, p1, p2, p3 = (strokes[:, i:i + 1, :] for i in range(4))
    mt = 1.0 - t
    curve = ((mt ** 3) * p0 + 3 * (mt ** 2) * t * p1
             + 3 * mt * (t ** 2) * p2 + (t ** 3) * p3)
    curve = curve * canvas                        # (N, samples, 2), pixel units

    ys, xs = torch.meshgrid(
        torch.arange(canvas, device=device, dtype=torch.float32),
        torch.arange(canvas, device=device, dtype=torch.float32), indexing="ij")
    grid_pts = torch.stack([xs, ys], dim=-1).reshape(-1, 2)       # (H*W, 2)

    ink = torch.ones(canvas * canvas, device=device)
    # Distance-scales bound the chunk so a large stroke count does not blow
    # up memory in one shot.
    chunk = max(1, 4_000_000 // max(1, canvas * canvas * samples))
    for i in range(0, n, chunk):
        seg = curve[i:i + chunk].reshape(-1, 2)                   # (c*samples, 2)
        c = curve[i:i + chunk].shape[0]
        # A hand-rolled distance rather than torch.cdist: cdist's backward is
        # (a-b)/distance, which is 0/0 the moment two points land exactly on
        # top of each other - and as strokes converge during optimisation,
        # a control point landing exactly on a sampled grid coordinate is not
        # a corner case, it is the expected end state.  That single NaN
        # gradient then poisons every stroke in the same backward pass, which
        # is exactly the "loss holds steady, then every path vanishes at
        # once" failure this was producing.  A small epsilon inside the sqrt
        # keeps the gradient finite everywhere without changing the distance
        # anywhere that matters.
        diff = grid_pts.unsqueeze(1) - seg.unsqueeze(0)            # (H*W, c*samples, 2)
        d = torch.sqrt(diff.pow(2).sum(-1) + 1e-8)
        d = d.reshape(canvas * canvas, c, samples).amin(dim=2)    # (H*W, c)
        alpha = torch.sigmoid((width * canvas - d) * (sharpness / canvas))
        keep = (1.0 - alpha).clamp(min=1e-6)
        ink = ink * keep.prod(dim=1)
    return ink.reshape(canvas, canvas)


def _seed_strokes(edge_mask: np.ndarray, n_strokes: int, seed: int = 0):
    """Initial control points sampled along real edges.

    Random placement works eventually but wastes most of the optimisation
    budget moving strokes onto the subject; starting near actual structure
    (we already compute this for the outline engine) gets there in far fewer
    iterations.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(edge_mask)
    h, w = edge_mask.shape
    strokes = np.zeros((n_strokes, 4, 2), dtype=np.float32)
    for i in range(n_strokes):
        if len(ys) > 0:
            j = rng.integers(0, len(ys))
            cx, cy = xs[j] / w, ys[j] / h
        else:
            cx, cy = rng.random(), rng.random()
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(0.02, 0.06)
        dx, dy = np.cos(angle) * length, np.sin(angle) * length
        jitter = lambda: rng.uniform(-0.01, 0.01)
        strokes[i, 0] = [cx - dx + jitter(), cy - dy + jitter()]
        strokes[i, 1] = [cx - dx / 3 + jitter(), cy - dy / 3 + jitter()]
        strokes[i, 2] = [cx + dx / 3 + jitter(), cy + dy / 3 + jitter()]
        strokes[i, 3] = [cx + dx + jitter(), cy + dy + jitter()]
    return np.clip(strokes, 0.0, 1.0)


def run(image_rgb: np.ndarray, edge_mask: np.ndarray, n_strokes: int = 64,
        iters: int = 400, canvas: int = 224, width: float = 0.006,
        lr: float = 0.01, length_penalty: float = 0.02,
        progress=None) -> SketchResult:
    """Optimise ``n_strokes`` Bezier curves to match ``image_rgb`` under CLIP.

    Two losses drive the fit: a semantic term (final CLIP embedding, the
    same signal CLIP's zero-shot classification uses - "is this the same
    thing") and a geometric term over an intermediate transformer block
    (roughly "is it in the same place"), because the semantic term alone is
    satisfied by anything recognisable regardless of layout.
    """
    import torch
    import torch.nn.functional as F

    import cv2

    model, device = _load_clip()

    # Resized on the CPU with cv2 rather than torch's own interpolate: MPS's
    # adaptive-pool backend (what mode="area" lowers to) refuses input sizes
    # that are not an exact multiple of the output size, which an arbitrary
    # photograph essentially never is.
    resized = cv2.resize(image_rgb, (canvas, canvas), interpolation=cv2.INTER_AREA)
    target = torch.from_numpy(
        resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    ).unsqueeze(0).to(device)

    mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=device).view(1, 3, 1, 1)

    def encode(img_01, want_mid=False):
        x = (img_01 - mean) / std
        feats = {}
        if want_mid:
            h = model.visual.transformer.resblocks[3].register_forward_hook(
                lambda m, i, o: feats.__setitem__("mid", o))
            out = model.encode_image(x)
            h.remove()
            return out, feats["mid"]
        return model.encode_image(x), None

    with torch.no_grad():
        target_feat, target_mid = encode(target, want_mid=True)

    strokes = torch.tensor(_seed_strokes(edge_mask, n_strokes),
                           device=device, requires_grad=True)
    opt = torch.optim.Adam([strokes], lr=lr)

    loss_curve = []
    for i in range(iters):
        opt.zero_grad()
        ink = render_soft(strokes.clamp(0.0, 1.0), width, canvas)
        sketch = ink.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

        pred_feat, pred_mid = encode(sketch, want_mid=True)
        semantic = 1.0 - F.cosine_similarity(pred_feat, target_feat).mean()
        geometric = F.mse_loss(pred_mid, target_mid)
        length = (strokes[:, 1:] - strokes[:, :-1]).pow(2).sum(-1).sqrt().sum()

        loss = semantic + 0.5 * geometric + length_penalty * length / n_strokes
        loss.backward()

        # Differentiable stroke optimisation is known for occasional gradient
        # spikes - two control points crossing, a stroke collapsing to a
        # point - and Adam's moment estimates carry one bad step forward
        # into every later one, which is what "loss holds steady, then every
        # stroke goes to NaN at once" was: one exploding gradient poisoning
        # the shared optimizer state that every stroke is updated from.
        # Clipping bounds it; the NaN guard catches whatever a norm clip
        # cannot (a gradient that is already inf/NaN has no finite norm to
        # clip to) by treating that step as a no-op instead of a crash.
        if not torch.isfinite(strokes.grad).all():
            strokes.grad = torch.zeros_like(strokes.grad)
        torch.nn.utils.clip_grad_norm_([strokes], max_norm=1.0)
        opt.step()

        with torch.no_grad():
            strokes.clamp_(0.0, 1.0)
        loss_curve.append(float(loss.detach()))
        if progress and (i % max(1, iters // 20) == 0 or i == iters - 1):
            progress(i, iters, loss_curve[-1])

    return SketchResult(strokes=strokes.detach(), width=width, canvas=canvas,
                        loss_curve=loss_curve)


def strokes_to_polylines(result: SketchResult, out_w: int, out_h: int,
                         samples: int = 40) -> list[np.ndarray]:
    """Sampled Bezier curves in pixel coordinates, ready for the mm pipeline."""
    strokes = result.strokes.cpu().numpy()
    t = np.linspace(0.0, 1.0, samples)[:, None]
    mt = 1.0 - t
    paths = []
    for p0, p1, p2, p3 in strokes:
        curve = ((mt ** 3) * p0 + 3 * (mt ** 2) * t * p1
                 + 3 * mt * (t ** 2) * p2 + (t ** 3) * p3)
        curve = curve * np.array([out_w, out_h], dtype=np.float32)
        paths.append(curve.astype(np.float32))
    return paths
