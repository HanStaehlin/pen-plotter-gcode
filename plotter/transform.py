"""Rotating finished work onto the machine's envelope.

A plotter whose bed is wider than it is tall can only draw landscape, so a
portrait composition has to be turned 90 degrees before it is sent and the
paper turned to match.

Note this is a rotation, not a transpose.  Swapping X and Y reflects the
drawing about the diagonal, which comes out mirrored - readable enough on an
abstract pattern that the mistake survives to the paper, and obvious the
moment there is any lettering or a face in the picture.  A rotation swaps the
axes *and* flips the sign of one of them.
"""

from __future__ import annotations

import re

import numpy as np

#: Machine coordinates put Y negative-downward, so page space (y down,
#: positive) is reached by negating Y.  Rotations are defined there and
#: converted back, which keeps the sign reasoning in one place.


def rotate_paths(paths, page_w: float, page_h: float, clockwise: bool = False):
    """Rotate mm paths 90 degrees; returns (paths, new_w, new_h).

    The page rectangle rotates with the content, so a drawing centred on a
    210x297 sheet stays centred on the resulting 297x210 one.
    """
    out = []
    for p in paths:
        p = np.asarray(p, dtype=np.float64)
        u = p[:, 0]
        v = -p[:, 1]                      # page space, y down
        if clockwise:
            u2, v2 = page_h - v, u
        else:
            u2, v2 = v, page_w - u
        out.append(np.stack([u2, -v2], axis=1))
    return out, page_h, page_w


def fit_paths(paths, page_w: float, page_h: float, margin: float = 0.0):
    """Centre paths on a page of the given size, in machine coordinates."""
    if not paths:
        return paths
    allpts = np.vstack([np.asarray(p, dtype=np.float64) for p in paths])
    x0, x1 = allpts[:, 0].min(), allpts[:, 0].max()
    v = -allpts[:, 1]
    v0, v1 = v.min(), v.max()
    dx = (page_w - (x1 - x0)) / 2.0 - x0
    dv = (page_h - (v1 - v0)) / 2.0 - v0
    return [np.stack([np.asarray(p)[:, 0] + dx,
                      np.asarray(p)[:, 1] - dv], axis=1) for p in paths]


def bounds(paths):
    """(min_x, max_x, min_y, max_y) over every point."""
    allpts = np.vstack([np.asarray(p, dtype=np.float64) for p in paths])
    return (float(allpts[:, 0].min()), float(allpts[:, 0].max()),
            float(allpts[:, 1].min()), float(allpts[:, 1].max()))


# ── rotating an existing G-code file ─────────────────────────────────────

_WORD = re.compile(r"([A-Za-z])\s*(-?\d*\.?\d+)")
_ARC = re.compile(r"\bG0?[23]\b")


def rotate_gcode(text: str, page_w: float, page_h: float,
                 clockwise: bool = False, precision: int = 3,
                 recentre: bool = True) -> tuple[str, dict]:
    """Rotate every move in a G-code program 90 degrees.

    Coordinates are modal: a line may set X alone and inherit Y from the move
    before it.  After a rotation both coordinates change, so the position has
    to be tracked and *both* written out on every move - rewriting only the
    words that happen to appear would silently drop half the geometry.
    """
    lines = text.splitlines()
    pos = [0.0, 0.0]
    pts: list[tuple[int, float, float]] = []      # line index, x, y
    arcs = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("(") or stripped.startswith(";"):
            continue
        if _ARC.search(stripped):
            arcs += 1
        words = dict(_WORD.findall(stripped))
        if "G" not in words and "X" not in words and "Y" not in words:
            continue
        moved = False
        if "X" in words:
            pos[0] = float(words["X"])
            moved = True
        if "Y" in words:
            pos[1] = float(words["Y"])
            moved = True
        if moved:
            pts.append((i, pos[0], pos[1]))

    if not pts:
        return text, {"moves": 0, "arcs": arcs}

    xy = np.array([[p[1], p[2]] for p in pts], dtype=np.float64)
    u, v = xy[:, 0], -xy[:, 1]
    if clockwise:
        u2, v2 = page_h - v, u
    else:
        u2, v2 = v, page_w - u

    new_w, new_h = page_h, page_w
    if recentre:
        u2 = u2 + (new_w - (u2.max() - u2.min())) / 2.0 - u2.min()
        v2 = v2 + (new_h - (v2.max() - v2.min())) / 2.0 - v2.min()

    out = list(lines)
    for (idx, _, _), nx, nv in zip(pts, u2, v2):
        ny = -nv
        line = out[idx]
        # Replace whichever of X/Y is present, then append the one that is not.
        has_x = re.search(r"[Xx]\s*-?\d*\.?\d+", line)
        has_y = re.search(r"[Yy]\s*-?\d*\.?\d+", line)
        if has_x:
            line = re.sub(r"[Xx]\s*-?\d*\.?\d+", f"X{nx:.{precision}f}", line, count=1)
        if has_y:
            line = re.sub(r"[Yy]\s*-?\d*\.?\d+", f"Y{ny:.{precision}f}", line, count=1)
        if not has_x:
            line = re.sub(r"([Yy]\s*-?\d*\.?\d+)", f"X{nx:.{precision}f} \\1",
                          line, count=1)
        elif not has_y:
            line = re.sub(r"([Xx]\s*-?\d*\.?\d+)", f"\\1 Y{ny:.{precision}f}",
                          line, count=1)
        out[idx] = line

    info = {
        "moves": len(pts),
        "arcs": arcs,
        "src_w": float(u.max() - u.min()),
        "src_h": float(v.max() - v.min()),
        "out_w": float(u2.max() - u2.min()),
        "out_h": float(v2.max() - v2.min()),
        "page": (new_w, new_h),
        "x_range": (float(u2.min()), float(u2.max())),
        "y_range": (float(-v2.max()), float(-v2.min())),
    }
    return "\n".join(out) + "\n", info
