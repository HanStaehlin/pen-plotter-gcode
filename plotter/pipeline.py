"""Image -> G-code, end to end."""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field, replace

import numpy as np

from . import engines, gcode, paths as P, tone, transform
from .config import Config
from .geometry import Canvas


@dataclass
class Result:
    paths: list                          # polylines in machine millimetres
    canvas: Canvas
    config: Config
    value: np.ndarray                    # 0-1 lightness used for structure
    coverage: np.ndarray                 # ink target that was aimed at
    stats: dict = dc_field(default_factory=dict)

    def output_paths(self):
        """Paths as the machine will receive them, turned if the bed needs it.

        The rotation lives here rather than in the drawing stage so the
        preview keeps showing the picture the way it was composed; only what
        goes down the wire is turned.
        """
        page = self.config.page
        mode = (page.rotate_output or "auto").lower()
        if mode == "off":
            return self.paths, ""
        if mode == "auto":
            if page.height_mm <= page.width_mm:
                return self.paths, ""
            clockwise = False
        else:
            clockwise = mode == "cw"
        turned, w, h = transform.rotate_paths(
            self.paths, page.width_mm, page.height_mm, clockwise)
        note = (f"rotated 90 deg {'clockwise' if clockwise else 'anticlockwise'} "
                f"for a landscape bed; page {w:.0f} x {h:.0f} mm — "
                f"turn the paper the same way")
        return turned, note

    def gcode(self, comment: str = "") -> str:
        paths, note = self.output_paths()
        if note:
            comment = (comment + "\n" + note) if comment else note
        return gcode.write(paths, self.config, comment)

    def preview(self, dpi: int = 160, show_travel: bool = False) -> np.ndarray:
        return gcode.render_preview(self.paths, self.config, dpi=dpi,
                                    show_travel=show_travel)


def convert(img_rgb: np.ndarray, cfg: Config, progress=None) -> Result:
    """Run the whole conversion.  ``progress`` is an optional ``fn(str)``."""
    def note(msg):
        if progress:
            progress(msg)

    t_start = time.perf_counter()

    note("preparing tone")
    # The canvas is built first so tone can be told the working scale: how
    # thin "thin detail" is only means anything in millimetres on paper.
    resolution = _resolution_for_pen(img_rgb, cfg)
    w_px, h_px = tone.output_size(img_rgb, resolution)
    canvas = Canvas(w_px, h_px, cfg.page)
    tone_cfg = cfg.tone
    if resolution != cfg.tone.resolution:
        tone_cfg = replace(cfg.tone, resolution=resolution)
    value, coverage, features = tone.prepare(img_rgb, tone_cfg, cfg.pen,
                                             canvas.px_per_mm)
    if cfg.render.ink_scale != 1.0:
        coverage = np.clip(coverage * cfg.render.ink_scale,
                           0.0, cfg.pen.max_coverage)
        coverage[coverage < cfg.pen.min_coverage] = 0.0

    note(f"drawing ({cfg.render.engine})")
    engine = engines.get(cfg.render.engine)
    produced = engine(value, coverage, canvas, cfg.pen, cfg.render,
                      features=features)
    px_paths, ink = produced if isinstance(produced, tuple) else (produced, None)

    if cfg.render.outline and cfg.render.engine not in ("outline", "lineart"):
        note("tracing outlines")
        px_paths = list(px_paths) + list(engines.outline.generate(
            value, coverage, canvas, cfg.pen, cfg.render,
            strength=cfg.render.outline_strength,
            features=features, feature_min=cfg.render.outline_feature_min))

    mm_paths = canvas.paths_to_mm(px_paths)
    raw_count = len(mm_paths)
    raw_travel = P.travel_length(mm_paths)   # unordered, unwelded

    note("welding strokes")
    mm_paths = P.merge(mm_paths, cfg.paths.merge_tolerance_mm)
    mm_paths = P.simplify_all(mm_paths, cfg.paths.simplify_mm)
    mm_paths = P.drop_short(mm_paths, cfg.paths.min_path_mm)

    if cfg.paths.sort:
        note("ordering travel")
        mm_paths = P.sort_paths(mm_paths, refine=cfg.paths.refine_order)

    if cfg.paths.bridge_max_mm > 0:
        note("bridging gaps")
        mm_paths = P.bridge(mm_paths, cfg.paths.bridge_max_mm,
                            _ink_sampler(coverage, canvas),
                            cfg.paths.bridge_min_ink)
        mm_paths = P.simplify_all(mm_paths, cfg.paths.simplify_mm)

    note("estimating")
    stats = gcode.estimate_time(mm_paths, cfg)
    stats.update({
        "strokes": len(mm_paths),
        "strokes_before_merge": raw_count,
        "travel_unoptimised_m": raw_travel / 1000.0,
        "points": int(sum(len(p) for p in mm_paths)),
        "image_mm": f"{canvas.width_mm:.0f} × {canvas.height_mm:.0f} mm",
        "tone_error": float(ink.coverage_error()) if ink is not None else None,
        "elapsed_s": time.perf_counter() - t_start,
        "resolution": resolution,
        "pen_px": cfg.pen.width_mm * canvas.px_per_mm,
    })

    rotated = (cfg.page.rotate_output or "auto").lower() != "off" and (
        cfg.page.height_mm > cfg.page.width_mm
        or (cfg.page.rotate_output or "").lower() in ("cw", "ccw"))
    stats["rotated_for_bed"] = bool(rotated)

    return Result(paths=mm_paths, canvas=canvas, config=cfg,
                  value=value, coverage=coverage, stats=stats)


#: A pen narrower than one working pixel cannot be placed accurately, and the
#: ink accumulator ends up finer than the image it is judging.  Measured at
#: 1.16 px the output is indistinguishable from 1.68 px, so this only rescues
#: genuinely broken cases and leaves a deliberately coarse preview alone.
MIN_PEN_PX = 1.0
MAX_AUTO_RESOLUTION = 2600


def _resolution_for_pen(img_rgb, cfg: Config) -> int:
    """Raise the working resolution when the pen would land sub-pixel.

    Everything downstream is measured in pen widths, so a fine nib on a
    coarse working image leaves the ink accumulator finer than the picture it
    is judging: the tone stays right but the detail it is placing has nothing
    behind it.  Rather than make this the user's problem, the resolution
    follows the pen.
    """
    w_px, h_px = tone.output_size(img_rgb, cfg.tone.resolution)
    pen_px = cfg.pen.width_mm * Canvas(w_px, h_px, cfg.page).px_per_mm
    if pen_px >= MIN_PEN_PX:
        return cfg.tone.resolution
    scaled = int(round(cfg.tone.resolution * MIN_PEN_PX / max(pen_px, 1e-6)))
    return min(MAX_AUTO_RESOLUTION, max(cfg.tone.resolution, scaled))


def _ink_sampler(coverage: np.ndarray, canvas: Canvas):
    """Sample the ink target at machine-millimetre points, for bridging."""
    h, w = coverage.shape

    def sample(pts_mm: np.ndarray) -> np.ndarray:
        px = canvas.to_px(np.atleast_2d(pts_mm))
        ix = np.clip(px[:, 0].astype(np.int32), 0, w - 1)
        iy = np.clip(px[:, 1].astype(np.int32), 0, h - 1)
        return coverage[iy, ix]

    return sample
