"""Image -> G-code, end to end."""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field, replace

import cv2
import numpy as np

from . import budget
from . import depth as depth_mod
from . import quality
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
    depth: np.ndarray | None = None      # 0-1 relative depth, 1 = nearest

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
    # Tone flips its own copy; every layer that reads the photo directly
    # (depth, regions, learned lines) must see the same orientation or it
    # lands mirrored against the shading.
    view_rgb = img_rgb
    if cfg.tone.flip_x:
        view_rgb = np.ascontiguousarray(np.flipud(view_rgb))
    if cfg.tone.flip_y:
        view_rgb = np.ascontiguousarray(np.fliplr(view_rgb))
    value, coverage, features = tone.prepare(img_rgb, tone_cfg, cfg.pen,
                                             canvas.px_per_mm)
    if cfg.render.ink_scale != 1.0:
        coverage = np.clip(coverage * cfg.render.ink_scale,
                           0.0, cfg.pen.max_coverage)
        coverage[coverage < cfg.pen.min_coverage] = 0.0

    form = None
    depth_map = None
    if cfg.render.form_weight > 0:
        if not depth_mod.available():
            note("depth model unavailable (torch not installed) — skipping form")
        else:
            note(f"estimating depth on {depth_mod.device_name()}")
            depth_map = depth_mod.estimate(view_rgb, size=(w_px, h_px),
                                           model=cfg.render.depth_model)
            form = depth_mod.form_orientation(depth_map,
                                              sigma=max(1.0, cfg.render.form_smooth))

    focus = float(np.clip(cfg.render.subject_focus, 0.0, 1.0))
    if cfg.render.engine == "hilbert":
        # The curve's coarsest cell has a minimum ink it cannot go below, so
        # a backdrop faded only part of the way lands right on the blank
        # cutoff and comes out as a maze full of holes.  Fading it the
        # whole way leaves a clean vignette instead.
        focus = min(1.0, 1.25 * focus)
    if focus > 0 and cfg.render.engine != "illustration":
        # The same fade illustration applies to its shading, for every
        # other engine: depth says what is subject and what is backdrop,
        # and the backdrop gets lighter tone instead of full-density fill.
        if depth_map is None and depth_mod.available():
            depth_map = depth_mod.estimate(view_rgb, size=(w_px, h_px),
                                           model=cfg.render.depth_model)
        if depth_map is not None:
            coverage = coverage * _subject_weight(depth_map, focus)

    overlay, overlay_scores = [], []
    if cfg.render.outline and cfg.render.engine not in ("outline", "lineart", "illustration"):
        note("tracing outlines")
        overlay, strengths = engines.outline.generate(
            value, coverage, canvas, cfg.pen, cfg.render,
            strength=cfg.render.outline_strength,
            features=features, feature_min=cfg.render.outline_feature_min,
            form=form, image_rgb=view_rgb, with_strength=True)
        overlay_scores = [budget.line_score(st, P.path_length(p) / canvas.px_per_mm)
                          for p, st in zip(overlay, strengths)]

    engine = engines.get(cfg.render.engine)

    def draw(ink: float = 1.0):
        """Run the engine at ``ink`` x the tone target; (mm paths, scores, ranked)."""
        target = coverage
        if ink != 1.0:
            target = np.clip(coverage * ink, 0.0, cfg.pen.max_coverage)
            target[target < cfg.pen.min_coverage] = 0.0
        produced = engine(value, target, canvas, cfg.pen, cfg.render,
                          features=features, form=form, image_rgb=view_rgb)
        if not isinstance(produced, tuple):
            produced = (produced, None)
        px_paths = list(produced[0])
        # Engines that rank their own strokes return the scores as well;
        # only those are pruned, since for the rest a low score just means
        # "light tone", which is still wanted.
        ranked = len(produced) > 2
        scores = (list(produced[2]) if ranked
                  else budget.default_scores(px_paths, target))
        px_paths += overlay
        scores += overlay_scores
        scores = [sc for sc, p in zip(scores, px_paths) if len(p) >= 2]
        return canvas.paths_to_mm(px_paths), scores, ranked

    def finish(ps):
        return _optimise(ps, cfg, note, _drawn_sampler(ps, canvas, cfg))

    note(f"drawing ({cfg.render.engine})")
    mm_paths, scores, ranked = draw()
    raw_count = len(mm_paths)
    raw_travel = P.travel_length(mm_paths)   # unordered, unwelded
    if ranked or cfg.paths.time_limit_min <= 0:
        mm_paths, cut = budget.select(
            mm_paths, scores, cfg, finish=finish,
            minutes=cfg.paths.time_limit_min,
            prune_below=cfg.paths.prune_below if ranked else 0.0, note=note)
    else:
        # An engine that doesn't rank its strokes draws the picture as a
        # few long ones - a Hilbert curve, a serpentine - and dropping any
        # of them cuts a hole in it.  Lighten the whole picture instead.
        fitted, cut = budget.lighten(lambda ink: draw(ink)[0], mm_paths, cfg,
                                     finish, cfg.paths.time_limit_min, note)
        if fitted is None:
            # Even the outline overlay alone runs over: draw no tone and
            # keep the strongest outline strokes that fit.
            bare, bare_scores, _ = draw(0.0)
            fitted, cut = budget.select(bare, bare_scores, cfg, finish=finish,
                                        minutes=cfg.paths.time_limit_min,
                                        note=note)
        mm_paths = fitted

    note("estimating")
    stats = gcode.estimate_time(mm_paths, cfg)
    look = quality.assess(mm_paths, canvas, coverage, cfg.pen.width_mm)
    stats.update({
        "strokes": len(mm_paths),
        "strokes_before_merge": raw_count,
        "travel_unoptimised_m": raw_travel / 1000.0,
        "points": int(sum(len(p) for p in mm_paths)),
        "image_mm": f"{canvas.width_mm:.0f} × {canvas.height_mm:.0f} mm",
        # Scored on the finished paths, after pruning and the budget, at
        # real pen width and viewing distance (quality.py).
        "likeness": look["likeness"],
        "tone_error": look["tone_error"],
        "elapsed_s": time.perf_counter() - t_start,
        "resolution": resolution,
        "pen_px": cfg.pen.width_mm * canvas.px_per_mm,
        "used_depth": depth_map is not None,
        "pruned": cut["pruned"],
        "budget_cut": cut["budget_cut"],
    })

    rotated = (cfg.page.rotate_output or "auto").lower() != "off" and (
        cfg.page.height_mm > cfg.page.width_mm
        or (cfg.page.rotate_output or "").lower() in ("cw", "ccw"))
    stats["rotated_for_bed"] = bool(rotated)

    return Result(paths=mm_paths, canvas=canvas, config=cfg,
                  value=value, coverage=coverage, stats=stats, depth=depth_map)


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


def _optimise(mm_paths, cfg: Config, note, ink_sampler=None):
    """Weld, simplify, order and bridge strokes - shared by every source."""
    note("welding strokes")
    mm_paths = P.merge(mm_paths, cfg.paths.merge_tolerance_mm)
    mm_paths = P.simplify_all(mm_paths, cfg.paths.simplify_mm)
    mm_paths = P.drop_short(mm_paths, cfg.paths.min_path_mm)
    if cfg.paths.sort:
        note("ordering travel")
        mm_paths = P.sort_paths(mm_paths, refine=cfg.paths.refine_order)
    if cfg.paths.bridge_max_mm > 0 and ink_sampler is not None:
        note("bridging gaps")
        mm_paths = P.bridge(mm_paths, cfg.paths.bridge_max_mm, ink_sampler,
                            cfg.paths.bridge_min_ink)
        mm_paths = P.simplify_all(mm_paths, cfg.paths.simplify_mm)
    return mm_paths


def draw_procedural(kind: str, seed: int | None = None, cfg: Config | None = None,
                    progress=None, name: str | None = None) -> Result:
    """Generate a drawing from scratch (no photo) and lay it out on the page.

    The generator works in its own millimetre space; the drawing is scaled
    to fill the printable area like a photo would be, then goes through the
    same stroke optimisation and G-code path as everything else.  Bridging
    is skipped: there is no tone map to say where a pen-down crossing would
    be hidden.
    """
    def note(msg):
        if progress:
            progress(msg)

    t_start = time.perf_counter()
    from .procedural import generate
    cfg = cfg or Config()
    note(f"drawing a {kind}")
    raw, shown = generate(kind, name=name, seed=seed)
    pts = np.vstack(raw)
    lo = pts.min(axis=0)
    size = np.maximum(pts.max(axis=0) - lo, 1e-6)
    # Canvas sizes are whole pixels; working in tenths of a millimetre keeps
    # that rounding far below the pen width.
    k = 10.0
    canvas = Canvas(int(np.ceil(size[0] * k)), int(np.ceil(size[1] * k)), cfg.page)
    mm_paths = canvas.paths_to_mm([(p - lo) * k for p in raw])
    raw_count = len(mm_paths)
    raw_travel = P.travel_length(mm_paths)
    mm_paths = _optimise(mm_paths, cfg, note)

    stats = gcode.estimate_time(mm_paths, cfg)
    rotated = (cfg.page.rotate_output or "auto").lower() != "off" and (
        cfg.page.height_mm > cfg.page.width_mm
        or (cfg.page.rotate_output or "").lower() in ("cw", "ccw"))
    stats.update({
        "strokes": len(mm_paths),
        "strokes_before_merge": raw_count,
        "travel_unoptimised_m": raw_travel / 1000.0,
        "points": int(sum(len(p) for p in mm_paths)),
        "image_mm": f"{canvas.width_mm:.0f} × {canvas.height_mm:.0f} mm",
        "likeness": None,
        "tone_error": None,
        "elapsed_s": time.perf_counter() - t_start,
        "resolution": 0,
        "pen_px": 0.0,
        "used_depth": False,
        "rotated_for_bed": bool(rotated),
        "seed": seed,
        "name": shown,
    })
    blank = np.ones((2, 2), np.float32)
    return Result(paths=mm_paths, canvas=canvas, config=cfg,
                  value=blank, coverage=1.0 - blank, stats=stats)


def _subject_weight(depth_map: np.ndarray, focus: float) -> np.ndarray:
    """1 on the subject, falling to 1 - focus on the backdrop, from depth."""
    near = np.clip((depth_map - 0.25) / 0.45, 0.0, 1.0)
    near = near * near * (3.0 - 2.0 * near)            # smoothstep
    return ((1.0 - focus) + focus * near).astype(np.float32)


def _drawn_sampler(mm_paths, canvas: Canvas, cfg: Config):
    """How dark the drawing itself is at machine-millimetre points, for bridging.

    A pen-down bridge is only hidden where the strokes around it are dense.
    Asking the photo's tone target instead is wrong whenever an engine draws
    less than the target - illustration hatches only the shadows - and there
    every bridge over a mid-tone lands on bare paper as a visible zigzag.
    """
    if cfg.paths.bridge_max_mm <= 0 or not mm_paths:
        return None
    drawn, px_per_mm = quality.render(mm_paths, canvas, cfg.pen.width_mm)
    drawn = cv2.resize(quality.viewed(drawn, px_per_mm, BRIDGE_SIGMA_MM),
                       (canvas.w_px, canvas.h_px), interpolation=cv2.INTER_AREA)
    return _ink_sampler(drawn, canvas)


#: Neighbourhood a bridge is judged over: about two hatch spacings in shadow.
BRIDGE_SIGMA_MM = 0.6


def draw_text(text: str, cfg: Config | None = None, style: str = "calligraphy",
              align: str = "left", size_mm: float | None = None, progress=None) -> Result:
    """Set text on the page in single-stroke lettering (see lettering.py).

    Unlike a photo or a frog, text is laid out at its real size on the
    printable area rather than scaled to fill it.
    """
    from . import lettering
    t_start = time.perf_counter()
    cfg = cfg or Config()
    page = cfg.page
    strokes, body_mm = lettering.compose(text, style, page.draw_w, page.draw_h,
                                         align=align, size_mm=size_mm,
                                         pen_mm=cfg.pen.width_mm)
    if not strokes:
        raise ValueError("nothing to write")
    k = 10.0                                    # canvas pixels per mm
    canvas = Canvas(int(round(page.draw_w * k)), int(round(page.draw_h * k)), page)
    mm_paths = canvas.paths_to_mm([s * k for s in strokes])
    raw_count = len(mm_paths)
    raw_travel = P.travel_length(mm_paths)
    note = progress or (lambda m: None)
    mm_paths = _optimise(mm_paths, cfg, note)
    stats = gcode.estimate_time(mm_paths, cfg)
    rotated = (cfg.page.rotate_output or "auto").lower() != "off" and (
        cfg.page.height_mm > cfg.page.width_mm
        or (cfg.page.rotate_output or "").lower() in ("cw", "ccw"))
    stats.update({
        "strokes": len(mm_paths),
        "strokes_before_merge": raw_count,
        "travel_unoptimised_m": raw_travel / 1000.0,
        "points": int(sum(len(p) for p in mm_paths)),
        "image_mm": f"{canvas.width_mm:.0f} × {canvas.height_mm:.0f} mm",
        "likeness": None,
        "tone_error": None,
        "elapsed_s": time.perf_counter() - t_start,
        "resolution": 0,
        "pen_px": 0.0,
        "used_depth": False,
        "rotated_for_bed": bool(rotated),
        "body_mm": body_mm,
    })
    blank = np.ones((2, 2), np.float32)
    return Result(paths=mm_paths, canvas=canvas, config=cfg,
                  value=blank, coverage=1.0 - blank, stats=stats)


def _ink_sampler(coverage: np.ndarray, canvas: Canvas):
    """Sample a coverage map at machine-millimetre points, for bridging."""
    h, w = coverage.shape

    def sample(pts_mm: np.ndarray) -> np.ndarray:
        px = canvas.to_px(np.atleast_2d(pts_mm))
        ix = np.clip(px[:, 0].astype(np.int32), 0, w - 1)
        iy = np.clip(px[:, 1].astype(np.int32), 0, h - 1)
        return coverage[iy, ix]

    return sample
