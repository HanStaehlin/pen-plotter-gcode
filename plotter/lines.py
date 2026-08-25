"""Coherent line extraction (flow-based Difference of Gaussians).

Canny answers "is there a step here" at one fixed scale and in isolation, so
on a photograph it returns a confetti of fragments and traces both sides of
every dark line.  Kang, Lee and Chui's method asks the same question but only
ever looks *across* the local structure, then smooths the answer *along* it.
The across-pass finds the edge; the along-pass is what turns a field of noisy
responses into connected line work a pen can draw in one stroke.
"""

from __future__ import annotations

import cv2
import numpy as np

from .flow import FlowField


def _kernel(sigma: float) -> tuple[int, np.ndarray]:
    radius = max(1, int(np.ceil(2.0 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-t * t / (2.0 * sigma * sigma))
    return radius, (g / g.sum()).astype(np.float32)


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(img, x, y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def coherent_lines(value: np.ndarray, field: FlowField,
                   sigma_c: float = 1.0, ratio: float = 1.6,
                   rho: float = 0.997, sigma_m: float = 3.0,
                   tau: float = 0.5, sharpness: float = 90.0,
                   iterations: int = 2) -> np.ndarray:
    """Boolean mask of line pixels.

    ``sharpness`` scales the soft threshold.  The DoG response on 0-1 values
    is a few hundredths, so without it ``tanh`` never leaves its linear
    region and nothing is ever marked as ink.
    """
    h, w = value.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    tx, ty = field.tangent_grid()
    nx, ny = -ty, tx                       # across the structure

    rad_n, g_c = _kernel(sigma_c)
    rad_s, g_s = _kernel(sigma_c * ratio)
    # Pad the narrow kernel so both are sampled over the same offsets.
    if rad_n < rad_s:
        pad = rad_s - rad_n
        g_c = np.pad(g_c, pad)
        rad_n = rad_s
    offsets = np.arange(-rad_n, rad_n + 1, dtype=np.float32)
    rad_m, g_m = _kernel(sigma_m)

    img = value.astype(np.float32).copy()
    for _ in range(max(1, iterations)):
        acc_c = np.zeros_like(img)
        acc_s = np.zeros_like(img)
        for k, t in enumerate(offsets):
            s = _sample(img, xs + nx * t, ys + ny * t)
            acc_c += g_c[k] * s
            acc_s += g_s[k] * s
        dog = acc_c - rho * acc_s

        # Walk the streamline in both directions, accumulating the response.
        acc = g_m[rad_m] * dog
        wsum = float(g_m[rad_m])
        for sign in (1.0, -1.0):
            px, py = xs.copy(), ys.copy()
            dx, dy = tx * sign, ty * sign
            for step in range(1, rad_m + 1):
                px = px + dx
                py = py + dy
                acc += g_m[rad_m + step] * _sample(dog, px, py)
                wsum += float(g_m[rad_m + step])
                ux = _sample(tx, px, py)
                uy = _sample(ty, px, py)
                # Orientation has no sign; keep walking the same way round.
                flip = (ux * dx + uy * dy) < 0
                ux = np.where(flip, -ux, ux)
                uy = np.where(flip, -uy, uy)
                norm = np.hypot(ux, uy) + 1e-8
                dx, dy = ux / norm, uy / norm
        edge = acc / wsum

        mask = (edge < 0) & (1.0 + np.tanh(sharpness * edge) < tau)
        img = np.where(mask, 0.0, 1.0).astype(np.float32)
        img = cv2.GaussianBlur(img, (0, 0), 0.6)

    return img < 0.5
