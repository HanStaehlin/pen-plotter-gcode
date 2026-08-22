"""Photograph -> ink-density target.

The job of this module is to squeeze the dynamic range of a colour photograph
into the narrow band a pen plotter can actually reproduce, without throwing
away shadow or highlight detail.

Output is a *coverage map*: for every pixel, the fraction of paper area that
should end up covered in ink.  Coverage - not "grey level" - is the right
currency, because it is what the drawing engines can physically control by
spacing their strokes.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import ToneConfig, PenConfig


# ── colour -> perceptual lightness ───────────────────────────────────────

def to_lightness(img_rgb: np.ndarray, chroma_weight: float = 0.18) -> np.ndarray:
    """Collapse colour to a 0-1 lightness field, keeping colours separable.

    Plain luma maps a saturated red and a mid grey to nearly the same value,
    so anything relying on colour contrast dissolves into flat tone.  We use
    CIE L* as the base (perceptually uniform, unlike a gamma-ignorant luma)
    and subtract a fraction of chroma, which pushes vivid colours darker than
    equally-light neutrals so they stay distinguishable in ink.
    """
    if img_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0
    if chroma_weight <= 0:
        return np.clip(L, 0.0, 1.0)

    a = (lab[..., 1] - 128.0) / 127.0
    b = (lab[..., 2] - 128.0) / 127.0
    chroma = np.sqrt(a * a + b * b)          # 0 .. ~1.4 for real images
    return np.clip(L - chroma_weight * chroma, 0.0, 1.0)


# ── tonal conditioning ───────────────────────────────────────────────────

def local_contrast(value: np.ndarray, clip: float, grid: int) -> np.ndarray:
    """CLAHE.  This is what rescues detail buried in deep shadow / blown sky."""
    if clip <= 0:
        return value
    u8 = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    grid = max(2, int(grid))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(grid, grid))
    return clahe.apply(u8).astype(np.float32) / 255.0


def enhance_detail(value: np.ndarray, amount: float) -> np.ndarray:
    """Structure-preserving detail boost.

    A plain unsharp mask haloes every strong edge, and haloes turn into
    visible stray strokes on paper.  Splitting on a bilateral filter keeps the
    boost in the texture band where it actually buys drawable detail.
    """
    if amount <= 0:
        return value
    base = cv2.bilateralFilter(value.astype(np.float32), 9, 0.12, 9)
    detail = value - base
    return np.clip(base + detail * (1.0 + 3.0 * amount), 0.0, 1.0)


def normalise(value: np.ndarray, black_pct: float, white_pct: float) -> np.ndarray:
    """Percentile auto-levels: use the whole ink range the pen has to offer."""
    lo, hi = np.percentile(value, [black_pct, white_pct])
    if hi - lo < 1e-4:
        return value
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0)


# ── main entry point ─────────────────────────────────────────────────────

def prepare(img: np.ndarray, cfg: ToneConfig, pen: PenConfig
            ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(value, coverage)`` maps at the configured resolution.

    ``value``    - 0..1 lightness, 1 = paper white.  Used for structure analysis.
    ``coverage`` - 0..pen.max_coverage, the ink area fraction each pixel wants.
    """
    img = _resize_long_edge(img, cfg.resolution)

    if cfg.denoise > 0:
        img = cv2.bilateralFilter(img, 7, 25 * cfg.denoise, 7)

    value = to_lightness(img, cfg.chroma_weight)

    if cfg.flip_x:
        value = np.flipud(value)
    if cfg.flip_y:
        value = np.fliplr(value)

    value = local_contrast(value, cfg.clahe_clip, cfg.clahe_grid)
    value = enhance_detail(value, cfg.detail)
    value = normalise(value, cfg.black_point, cfg.white_point)

    if cfg.exposure:
        value = np.clip(value + cfg.exposure, 0.0, 1.0)
    if cfg.gamma != 1.0:
        value = np.power(value, max(0.05, cfg.gamma))
    if cfg.invert:
        value = 1.0 - value

    coverage = to_coverage(value, pen, cfg.ink_response)
    return value.astype(np.float32), coverage


def to_coverage(value: np.ndarray, pen: PenConfig,
                response: float = 0.25) -> np.ndarray:
    """Map lightness to an ink-area target.

    Two defensible mappings exist and they disagree sharply.

    *Photometric*: reflectance falls linearly with ink coverage, so matching
    the photograph's actual luminance needs ``coverage = 1 - Y``.  That is
    correct, and it is also brutal - a mid grey demands about 80% coverage,
    which on paper is a near-solid block and on the machine is an hour of
    drawing.

    *Perceptual*: spend the pen's coverage range evenly across L*.  Since L*
    is already perceptually uniform, equal steps of image tone become equal
    steps of apparent tone, which is what "capture the dynamic range" is
    really asking for - distinguishable tones, not literal luminance.

    ``response`` fades between the two, defaulting near the perceptual end
    with enough photometric weight to keep shadows from going flat.
    """
    v = np.clip(value, 0.0, 1.0)
    perceptual = 1.0 - v
    photometric = 1.0 - _lstar_to_linear(v)
    r = float(np.clip(response, 0.0, 1.0))
    cov = (1.0 - r) * perceptual + r * photometric

    cov *= pen.max_coverage
    # min_coverage is a cutoff, not a floor: tone too faint to be worth a
    # stroke becomes paper white rather than a lonely line every 20 mm.
    cov[cov < pen.min_coverage] = 0.0
    return np.clip(cov, 0.0, pen.max_coverage).astype(np.float32)


def _lstar_to_linear(L: np.ndarray) -> np.ndarray:
    """CIE L* (0-1) -> linear luminance Y (0-1)."""
    Ls = L * 100.0
    fy = (Ls + 16.0) / 116.0
    y = np.where(Ls > 8.0, fy ** 3, Ls / 903.3)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _resize_long_edge(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge == target:
        return img
    scale = target / float(long_edge)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, int(round(w * scale))),
                            max(1, int(round(h * scale)))), interpolation=interp)


def load_image(path: str) -> np.ndarray:
    """Read any image file as RGB uint8, flattening transparency onto white."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return _to_rgb(img)


def decode_image(buf: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode uploaded image")
    return _to_rgb(img)


def cutout(img_bytes: bytes) -> np.ndarray:
    """Segment the subject out and drop the background to paper white.

    Needs the optional ``rembg`` package; the model downloads on first use.
    """
    from rembg import remove
    return _to_rgb(cv2.imdecode(
        np.frombuffer(remove(img_bytes), dtype=np.uint8), cv2.IMREAD_UNCHANGED))


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        rgb = img[..., :3].astype(np.float32)
        alpha = (img[..., 3:4].astype(np.float32)) / 255.0
        flat = rgb * alpha + 255.0 * (1.0 - alpha)   # composite over white
        img = flat.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
