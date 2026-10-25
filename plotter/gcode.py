"""G-code emission, plot-time estimation and preview rendering."""

from __future__ import annotations

import math
import unicodedata

import cv2
import numpy as np

from .config import Config


def write(paths, cfg: Config, header_comment: str = "") -> str:
    """Emit GRBL-flavoured G-code for the AX4 pen plotter."""
    m = cfg.machine
    prec = m.precision
    out: list[str] = []

    if header_comment:
        for line in header_comment.strip().splitlines():
            # GRBL ends a comment at the first ")", so a bracket inside one
            # - "Bettina (frog)" - would leave the rest as a bad command.
            text = line.strip().replace("(", "[").replace(")", "]")
            # Plain ASCII: older senders and firmware choke on anything else.
            text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
            out.append(f"({text})")

    out.append("G21")                      # millimetres
    out.append("G90")                      # absolute
    out.append(m.pen_up_cmd)
    out.append(f"G4 P{m.pen_up_dwell:g}")

    for p in paths:
        if len(p) < 2:
            continue
        x0, y0 = p[0]
        out.append(f"G0 X{x0:.{prec}f} Y{y0:.{prec}f} F{m.feed_travel:g}")
        out.append(m.pen_down_cmd)
        out.append(f"G4 P{m.pen_down_dwell:g}")
        # First cut carries the feed rate; the rest inherit it, which keeps
        # the file roughly a third smaller on stroke-heavy images.
        x1, y1 = p[1]
        out.append(f"G1 X{x1:.{prec}f} Y{y1:.{prec}f} F{m.feed_draw:g}")
        for x, y in p[2:]:
            out.append(f"G1 X{x:.{prec}f} Y{y:.{prec}f}")
        out.append(m.pen_up_cmd)
        out.append(f"G4 P{m.pen_up_dwell:g}")

    out.append(f"G0 X0 Y0 F{m.feed_travel:g}")
    out.append("")
    return "\n".join(out)


def estimate_time(paths, cfg: Config) -> dict:
    """Plot-time estimate using the same look-ahead the firmware uses.

    Feed rate alone is useless here: a machine accelerating at 400 mm/s^2
    never reaches 3000 mm/min over a 2 mm hatch stroke.  Assuming a dead stop
    at every vertex is just as wrong in the other direction, and it is wrong
    precisely where it matters - it would price a Hilbert curve, which is
    nothing but corners, at triple its real cost and make the continuous
    engines look far worse than they are.

    So this plans the moves the way GRBL does: each corner gets a junction
    speed from how sharply it turns, a forward and backward pass make the
    ramps physically reachable, and every move is then timed on its own
    trapezoid.
    """
    m = cfg.machine
    v_draw = m.feed_draw / 60.0            # mm/s
    v_travel = m.feed_travel / 60.0
    accel = max(1.0, m.accel_mm_s2 * m.accel_efficiency)

    draw_mm = 0.0
    draw_s = 0.0
    travel_mm = 0.0
    travel_s = 0.0
    pos = np.zeros(2)

    for p in paths:
        if len(p) < 2:
            continue
        d = float(np.hypot(*(p[0] - pos)))
        travel_mm += d
        travel_s += _move_time(d, v_travel, accel)
        length, seconds = _path_time(np.asarray(p, dtype=np.float64),
                                     v_draw, accel, m.junction_deviation)
        draw_mm += length
        draw_s += seconds
        pos = p[-1]

    d = float(np.hypot(*pos))
    travel_mm += d
    travel_s += _move_time(d, v_travel, accel)

    lifts = sum(1 for p in paths if len(p) >= 2)
    dwell_s = lifts * (m.pen_up_dwell + m.pen_down_dwell + m.lift_overhead_s)
    total = draw_s + travel_s + dwell_s

    return {
        "total_s": total,
        "draw_s": draw_s,
        "travel_s": travel_s,
        "dwell_s": dwell_s,
        "pen_lifts": lifts,
        "draw_dist_m": draw_mm / 1000.0,
        "travel_dist_m": travel_mm / 1000.0,
        "label": format_duration(total),
        "breakdown": (f"draw {format_duration(draw_s)} \u00b7 "
                      f"travel {format_duration(travel_s)} \u00b7 "
                      f"{lifts} pen lifts"),
    }


def _path_time(p: np.ndarray, v_max: float, accel: float,
               deviation: float) -> tuple[float, float]:
    """Length and duration of one pen-down polyline under look-ahead."""
    delta = np.diff(p, axis=0)
    length = np.hypot(delta[:, 0], delta[:, 1])
    keep = length > 1e-9
    if not np.any(keep):
        return 0.0, 0.0
    delta = delta[keep]
    length = length[keep]
    unit = delta / length[:, None]
    n = len(length)

    # Junction speed at each interior vertex, from GRBL's junction-deviation
    # model: the corner is treated as an arc the path may deviate into, and
    # the speed that fits that arc is what the machine can hold through it.
    entry = np.zeros(n + 1)
    entry[1:n] = v_max
    if n > 1:
        cos_theta = -np.einsum("ij,ij->i", unit[:-1], unit[1:])
        cos_theta = np.clip(cos_theta, -0.999999, 0.999999)
        sin_half = np.sqrt(0.5 * (1.0 - cos_theta))
        with np.errstate(divide="ignore", invalid="ignore"):
            v_j = np.sqrt(accel * deviation * sin_half / (1.0 - sin_half))
        entry[1:n] = np.minimum(v_max, np.nan_to_num(v_j, nan=0.0,
                                                     posinf=v_max))

    # Ramps have to be physically reachable from both directions.
    reach = np.sqrt(2.0 * accel * length)
    for i in range(n):
        entry[i + 1] = min(entry[i + 1], math.hypot(entry[i], reach[i]))
    for i in range(n - 1, -1, -1):
        entry[i] = min(entry[i], math.hypot(entry[i + 1], reach[i]))

    e = entry[:-1]
    x = entry[1:]
    peak = np.minimum(v_max, np.sqrt(np.maximum(
        0.0, (2.0 * accel * length + e * e + x * x) * 0.5)))
    d_acc = (peak * peak - e * e) / (2.0 * accel)
    d_dec = (peak * peak - x * x) / (2.0 * accel)
    cruise = np.maximum(0.0, length - d_acc - d_dec)
    seconds = ((peak - e) + (peak - x)) / accel + cruise / np.maximum(peak, 1e-9)

    return float(length.sum()), float(seconds.sum())


def _move_time(dist: float, v_max: float, accel: float) -> float:
    """Trapezoidal (or triangular, for short moves) profile from rest to rest."""
    if dist <= 1e-9:
        return 0.0
    d_accel = v_max * v_max / accel          # ramp up + ramp down distance
    if dist < d_accel:
        return 2.0 * math.sqrt(dist / accel)
    return v_max / accel + dist / v_max


def format_duration(s: float) -> str:
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{int(s // 60)}m {int(s % 60):02d}s"
    return f"{int(s // 3600)}h {int((s % 3600) // 60):02d}m"


# ── preview ──────────────────────────────────────────────────────────────

def render_preview(paths, cfg: Config, dpi: int = 160,
                   show_travel: bool = False, supersample: int = 2) -> np.ndarray:
    """Render the plot as the pen will actually lay it down, on a page."""
    page = cfg.page
    ss = max(1, int(supersample))
    scale = dpi / 25.4 * ss
    w = max(2, int(page.width_mm * scale))
    h = max(2, int(page.height_mm * scale))
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    cv2.rectangle(canvas,
                  (int(page.margin_mm * scale), int(page.margin_mm * scale)),
                  (int((page.width_mm - page.margin_mm) * scale),
                   int((page.height_mm - page.margin_mm) * scale)),
                  (222, 222, 228), max(1, ss))

    thickness = max(1, int(round(cfg.pen.width_mm * scale)))

    if show_travel:
        pos = np.zeros(2)
        for p in paths:
            a = _to_canvas(pos[None, :], page, scale)[0]
            b = _to_canvas(p[:1], page, scale)[0]
            cv2.line(canvas, tuple(a), tuple(b), (176, 176, 176),
                     max(1, ss), cv2.LINE_AA)
            pos = p[-1]

    polys = [np.round(_to_canvas_f(p, page, scale) * 16).astype(np.int32)
             for p in paths if len(p) >= 2]
    if polys:
        cv2.polylines(canvas, polys, False, (26, 26, 30), thickness,
                      cv2.LINE_AA, shift=4)

    if ss > 1:
        canvas = cv2.resize(canvas, (w // ss, h // ss), interpolation=cv2.INTER_AREA)
    return canvas


def _to_canvas_f(pts_mm, page, scale) -> np.ndarray:
    pts = np.asarray(pts_mm, dtype=np.float64)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] * scale
    out[:, 1] = (-pts[:, 1] if page.negative_y else pts[:, 1]) * scale
    return out


def _to_canvas(pts_mm, page, scale) -> np.ndarray:
    return np.round(_to_canvas_f(pts_mm, page, scale)).astype(np.int32)


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                           if image.ndim == 3 else image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
