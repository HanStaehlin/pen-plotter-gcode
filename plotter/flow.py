"""Edge tangent flow: which way should a stroke run at every pixel?

Strokes that follow the image's own structure read as drawing; strokes that
ignore it read as a screen door laid over a photo.  We recover the local
orientation from the smoothed structure tensor, whose minor eigenvector runs
*along* edges rather than across them.
"""

from __future__ import annotations

import cv2
import numpy as np


class FlowField:
    """Orientation field stored in double-angle form.

    Orientation is a line, not a vector: theta and theta+pi mean the same
    thing.  Averaging or interpolating raw angles across that wrap produces
    garbage, so everything is carried as ``(cos 2t, sin 2t)``, where the
    ambiguity disappears and linear interpolation is well defined.
    """

    __slots__ = ("c2", "s2", "coherence", "h", "w")

    def __init__(self, c2: np.ndarray, s2: np.ndarray, coherence: np.ndarray):
        self.c2 = np.ascontiguousarray(c2, dtype=np.float32)
        self.s2 = np.ascontiguousarray(s2, dtype=np.float32)
        self.coherence = np.ascontiguousarray(coherence, dtype=np.float32)
        self.h, self.w = self.c2.shape

    def tangent(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bilinearly sample the unit tangent at float pixel coordinates."""
        c = _bilinear(self.c2, x, y)
        s = _bilinear(self.s2, x, y)
        theta = 0.5 * np.arctan2(s, c)
        return np.cos(theta), np.sin(theta)

    def tangent_grid(self) -> tuple[np.ndarray, np.ndarray]:
        theta = 0.5 * np.arctan2(self.s2, self.c2)
        return np.cos(theta), np.sin(theta)


def compute(value: np.ndarray,
            sigma_grad: float = 1.2,
            sigma_tensor: float = 4.0,
            coherence_mix: float = 1.0,
            base_angle_deg: float = 45.0,
            coherence_gamma: float = 1.0) -> FlowField:
    """Build the tangent field for a 0-1 lightness image.

    ``coherence_mix`` fades between pure image structure (1.0) and a plain
    ruled field at ``base_angle_deg`` (0.0).  Flat regions - sky, studio
    backdrops - carry no orientation at all, so there we fall back to a
    coarse-scale reading of the image and finally to the base angle, rather
    than letting strokes chase sensor noise.
    """
    v = cv2.GaussianBlur(value.astype(np.float32), (0, 0), sigma_grad)
    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)

    fine = _tensor_orientation(gx, gy, sigma_tensor)
    coarse = _tensor_orientation(gx, gy, sigma_tensor * 4.0)

    c2_f, s2_f, coh_f = fine
    c2_c, s2_c, coh_c = coarse

    # Sharpen the coherence response.  Smooth shading still yields a weak but
    # non-zero orientation, and following it is what turns every highlight
    # into concentric rings; real texture - feathers, fur, grain - answers far
    # more strongly.  Raising the power suppresses the former and keeps the
    # latter, which is what lets one setting serve a portrait and a bird.
    if coherence_gamma != 1.0:
        g = max(0.1, float(coherence_gamma))
        coh_f = np.power(coh_f, g)
        coh_c = np.power(coh_c, g)

    # Where fine structure is weak, defer to the coarse field; where that is
    # weak too, defer to the base angle.  Blending happens in double-angle
    # space so the result is a proper orientation average.
    c2 = coh_f * c2_f + (1.0 - coh_f) * (coh_c * c2_c)
    s2 = coh_f * s2_f + (1.0 - coh_f) * (coh_c * s2_c)

    strength = np.sqrt(c2 * c2 + s2 * s2)
    base = np.radians(base_angle_deg)
    bc, bs = np.cos(2 * base), np.sin(2 * base)

    mix = float(np.clip(coherence_mix, 0.0, 1.0))
    fill = np.clip(1.0 - strength, 0.0, 1.0) + (1.0 - mix)
    c2 = mix * c2 + fill * bc
    s2 = mix * s2 + fill * bs

    norm = np.maximum(np.hypot(c2, s2), 1e-8)
    coherence = np.clip(np.maximum(coh_f, coh_c), 0.0, 1.0)
    return FlowField(c2 / norm, s2 / norm, coherence)


def blend_form(field: FlowField, form, weight: float,
               smooth: float = 6.0) -> FlowField:
    """Fold a depth-derived form orientation into a tone-derived field.

    Blending happens in double-angle space, weighted by how far the surface
    is from umbilic there: on a sphere both principal curvatures match and
    the direction is arbitrary, so those points must not outvote their
    neighbours.  The result is smoothed for the same reason - a per-pixel
    curvature direction is far too twitchy to hatch along.
    """
    if form is None or weight <= 0:
        return field
    fc2, fs2, strength = form
    # Depth is smooth by nature, so letting it win everywhere erases real
    # texture: on a photograph of a bird it flattens every feather, which is
    # the one thing making the subject read.  Give it the ground the tone
    # field cannot hold - a cheek, a shoulder, the side of a sphere, where
    # there is no structure to follow and strokes would otherwise just rule
    # straight across the form.
    w = (np.clip(float(weight), 0.0, 1.0) * strength
         * (1.0 - np.clip(field.coherence, 0.0, 1.0)))
    c2 = (1.0 - w) * field.c2 + w * fc2
    s2 = (1.0 - w) * field.s2 + w * fs2
    if smooth > 0:
        c2 = cv2.GaussianBlur(c2, (0, 0), smooth)
        s2 = cv2.GaussianBlur(s2, (0, 0), smooth)
    norm = np.maximum(np.hypot(c2, s2), 1e-8)
    return FlowField(c2 / norm, s2 / norm, field.coherence)


def _tensor_orientation(gx: np.ndarray, gy: np.ndarray, sigma: float):
    """Smoothed structure tensor -> (cos2t, sin2t, coherence) of the tangent."""
    E = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    F = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    G = cv2.GaussianBlur(gy * gy, (0, 0), sigma)

    diff = E - G
    disc = np.sqrt(diff * diff + 4.0 * F * F) + 1e-12
    l1 = 0.5 * (E + G + disc)          # across the edge
    l2 = 0.5 * (E + G - disc)          # along the edge
    coherence = np.clip((l1 - l2) / (l1 + l2 + 1e-8), 0.0, 1.0)

    # 2*theta of the major (gradient) eigenvector; the tangent is +90 deg,
    # which in double-angle space is simply a negation.
    c2_major = diff / disc
    s2_major = 2.0 * F / disc
    return -c2_major, -s2_major, coherence


def _bilinear(src: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = src.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = x - x0
    fy = y - y0
    return (src[y0, x0] * (1 - fx) * (1 - fy) +
            src[y0, x1] * fx * (1 - fy) +
            src[y1, x0] * (1 - fx) * fy +
            src[y1, x1] * fx * fy)
