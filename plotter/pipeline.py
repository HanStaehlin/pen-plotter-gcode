"""Image -> G-code, end to end."""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import engines, gcode, paths as P, tone
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

    def gcode(self, comment: str = "") -> str:
        return gcode.write(self.paths, self.config, comment)

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
    value, coverage = tone.prepare(img_rgb, cfg.tone, cfg.pen)
    if cfg.render.ink_scale != 1.0:
        coverage = np.clip(coverage * cfg.render.ink_scale,
                           0.0, cfg.pen.max_coverage)
        coverage[coverage < cfg.pen.min_coverage] = 0.0

    canvas = Canvas(value.shape[1], value.shape[0], cfg.page)

    note(f"drawing ({cfg.render.engine})")
    engine = engines.get(cfg.render.engine)
    produced = engine(value, coverage, canvas, cfg.pen, cfg.render)
    px_paths, ink = produced if isinstance(produced, tuple) else (produced, None)

    if cfg.render.outline and cfg.render.engine not in ("outline", "lineart"):
        note("tracing outlines")
        px_paths = list(px_paths) + list(engines.outline.generate(
            value, coverage, canvas, cfg.pen, cfg.render,
            strength=cfg.render.outline_strength))

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
    })

    return Result(paths=mm_paths, canvas=canvas, config=cfg,
                  value=value, coverage=coverage, stats=stats)


def _ink_sampler(coverage: np.ndarray, canvas: Canvas):
    """Sample the ink target at machine-millimetre points, for bridging."""
    h, w = coverage.shape

    def sample(pts_mm: np.ndarray) -> np.ndarray:
        px = canvas.to_px(np.atleast_2d(pts_mm))
        ix = np.clip(px[:, 0].astype(np.int32), 0, w - 1)
        iy = np.clip(px[:, 1].astype(np.int32), 0, h - 1)
        return coverage[iy, ix]

    return sample
