"""Lettering: set text on the page in single-stroke fonts, optionally as if
written with a broad-edged nib.

A plotter cannot fill an outline font; it needs fonts whose glyphs are the
centrelines of the strokes.  The Hershey fonts are exactly that - drawn for
vector displays and plotters in the 1960s - and come in roman, italic,
script and blackletter faces.

Calligraphy gets its thick and thin from a flat nib held at a fixed angle:
a stroke across the nib is as wide as the nib, a stroke along it is a
hairline.  A round plotter pen has no edge, but the same contrast appears
when every stroke is drawn several times side by side, shifted along the
nib's edge.  The passes are joined into one zigzag, so a stroke still costs
a single pen lift and its ends come out cut at the nib angle, as a real
nib's do.

Text format: a line starting with "# " is a heading, leading spaces indent
a line (an em is two spaces), and a blank line leaves a gap.
"""

from __future__ import annotations

import unicodedata

import numpy as np

#: style -> (heading face, body face, nib width as a fraction of body size)
STYLES = {
    "calligraphy": ("gothiceng", "scripts", 0.075),
    "book": ("timesrb", "timesi", 0.0),
    "handwriting": ("cursive", "cursive", 0.0),
    "technical": ("futuram", "futural", 0.0),
}
NIB_ANGLE_DEG = 40.0
HEADING_SCALE = 1.7
LINE_GAP = 1.75            # baseline to baseline, in body sizes
MAX_BODY_MM = 9.0          # short texts stop growing here
MIN_BODY_MM = 3.0

# The fonts are ASCII only, so accented letters are a base letter plus one of
# these marks: small strokes in units of the letter size, placed relative to
# the top centre of the letter's ink.
_MARKS = {
    "̈": [[(-0.13, -0.10), (-0.13, -0.07)], [(0.13, -0.10), (0.13, -0.07)]],   # diaeresis
    "́": [[(-0.06, -0.06), (0.10, -0.20)]],                                      # acute
    "̀": [[(0.06, -0.06), (-0.10, -0.20)]],                                      # grave
    "̂": [[(-0.13, -0.06), (0.0, -0.18), (0.13, -0.06)]],                        # circumflex
    "̃": [[(-0.15, -0.10), (-0.05, -0.16), (0.05, -0.10), (0.15, -0.16)]],      # tilde
    "̊": [[(0.0, -0.08), (0.06, -0.12), (0.0, -0.18), (-0.06, -0.12), (0.0, -0.08)]],  # ring
}
_REPLACE = {"ß": "ss", "‘": "'", "’": "'", "“": '"', "”": '"',
            "„": '"', "–": "-", "—": "-", "…": "...", " ": " "}

_FONTS: dict = {}


def _font(face):
    if face not in _FONTS:
        from HersheyFonts import HersheyFonts
        f = HersheyFonts()
        f.load_default_font(face)
        _FONTS[face] = f
    return _FONTS[face]


class _Face:
    """One font at one size: glyph strokes in mm, y down, baseline at 0."""

    def __init__(self, face: str, size_mm: float):
        f = _font(face)
        opts = f.render_options
        self.f = f
        self.base = opts["base_line"]
        self.k = size_mm / (opts["bottom_line"] - opts["cap_line"])
        self.size = size_mm

    def glyph(self, ch):
        glyphs = list(self.f.glyphs_for_text(ch))
        return glyphs[0] if glyphs else None

    def word(self, text: str, x0: float = 0.0):
        """Strokes for a run of text starting at x0; returns (strokes, end x)."""
        out, x = [], x0
        for ch in text:
            base, marks = _decompose(ch)
            for b in base:
                g = self.glyph(b)
                if g is None:
                    continue
                strokes = [np.array([((px - g.left_offset) * self.k + x,
                                      (py - self.base) * self.k) for px, py in s], float)
                           for s in g.strokes if len(s) > 1]
                out += strokes
                if marks and strokes:
                    out += self._marks(marks, strokes)
                x += g.char_width * self.k
        return out, x

    def _marks(self, marks, strokes):
        pts = np.vstack(strokes)
        top = pts[:, 1].min()
        near_top = pts[pts[:, 1] < top + 0.25 * self.size]
        cx = near_top[:, 0].mean()
        out = []
        for m in marks:
            for s in _MARKS.get(m, []):
                out.append(np.array([(cx + u * self.size, top + v * self.size) for u, v in s], float))
        return out

    def width(self, text: str) -> float:
        return self.word(text)[1]


def _decompose(ch):
    """Base letters and combining marks for one character (ä -> a + ¨)."""
    if ch in _REPLACE:
        return _REPLACE[ch], []
    parts = unicodedata.normalize("NFD", ch)
    base = "".join(c for c in parts if not unicodedata.combining(c))
    marks = [c for c in parts if unicodedata.combining(c)]
    return base, marks


def _parse(text: str):
    """[(kind, indent_ems, text)], kind in heading / body / gap."""
    out = []
    for raw in text.replace("\t", "    ").splitlines():
        if not raw.strip():
            out.append(("gap", 0, ""))
        elif raw.lstrip().startswith("# "):
            out.append(("heading", 0, raw.lstrip()[2:].strip()))
        else:
            out.append(("body", (len(raw) - len(raw.lstrip(" "))) / 2.0, raw.strip()))
    while out and out[-1][0] == "gap":
        out.pop()
    return out


def _wrap(face: _Face, text: str, width: float, indent: float):
    """Greedy word wrap; continuation lines get one extra em of indent."""
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w) if cur else w
        room = width - indent - (face.size if lines else 0.0)
        if cur and face.width(trial) > room:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    return [(line, indent + (face.size if i else 0.0)) for i, line in enumerate(lines)]


def _layout(items, style, body_mm, width, wrap=True):
    """Lines as [(face, text, indent mm, baseline y)] and the block's height."""
    head_face, body_face, _ = STYLES[style]
    body = _Face(body_face, body_mm)
    head = _Face(head_face, body_mm * HEADING_SCALE)
    rows, y, first = [], 0.0, True
    for kind, indent, text in items:
        if kind == "gap":
            y += 0.8 * LINE_GAP * body_mm
            continue
        face = head if kind == "heading" else body
        lines = (_wrap(face, text, width, indent * body_mm) if wrap
                 else [(text, indent * body_mm)])
        for line, ind in lines:
            y += (face.size if first else LINE_GAP * face.size)
            first = False
            rows.append((face, line, ind, y))
        if kind == "heading":
            y += 0.6 * body_mm
    return rows, y + 0.45 * body_mm          # room for descenders


def fit_size(text: str, style: str, width: float, height: float) -> tuple[float, bool]:
    """Largest body size (mm) at which the text fits the box, and whether to wrap.

    Line breaks in a poem carry meaning, so the text first shrinks to keep
    every line whole; only when even the smallest size is too wide do lines
    wrap.
    """
    items = _parse(text)

    def fits(size, wrap):
        rows, h = _layout(items, style, size, width, wrap)
        return h <= height and all(face.width(line) + ind <= width + 1e-6
                                   for face, line, ind, _ in rows)

    for wrap in (False, True):
        if not fits(MIN_BODY_MM, wrap):
            continue
        lo, hi = MIN_BODY_MM, MAX_BODY_MM
        if fits(hi, wrap):
            return hi, wrap
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if fits(mid, wrap) else (lo, mid)
        return lo, wrap
    return MIN_BODY_MM, True


def nib(strokes, width_mm: float, pen_mm: float, angle_deg: float = NIB_ANGLE_DEG):
    """Each stroke as a zigzag of side-by-side passes along the nib's edge."""
    if width_mm <= pen_mm * 0.5:
        return strokes
    a = np.radians(angle_deg)
    edge = np.array([np.cos(a), -np.sin(a)])          # y down: rises to the right
    passes = int(np.ceil(width_mm / (0.75 * pen_mm))) + 1
    out = []
    for s in strokes:
        parts = []
        for i, k in enumerate(np.linspace(-0.5, 0.5, passes)):
            p = s + edge * k * width_mm
            parts.append(p if i % 2 == 0 else p[::-1])
        out.append(np.vstack(parts))
    return out


def compose(text: str, style: str = "calligraphy", width: float = 277.0, height: float = 190.0,
            align: str = "left", size_mm: float | None = None, pen_mm: float = 0.4):
    """Set ``text`` in a width x height box (mm, y down); returns (polylines, body size)."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {', '.join(STYLES)}")
    items = _parse(text)
    if not items:
        return [], 0.0
    if size_mm:
        body_mm, wrap = size_mm, True
    else:
        body_mm, wrap = fit_size(text, style, width, height)
    rows, block_h = _layout(items, style, body_mm, width, wrap)
    nib_frac = STYLES[style][2]
    widths = [face.width(line) + ind for face, line, ind, _ in rows]
    block_w = max(widths)
    left = (width - block_w) / 2.0                    # the block sits centred on the page
    top = max(0.0, (height - block_h) / 2.0)
    out = []
    for (face, line, ind, y), w in zip(rows, widths):
        if align == "center":
            x0 = (width - (w - ind)) / 2.0
        else:
            x0 = left + ind
        strokes, _ = face.word(line, x0)
        strokes = [s + [0.0, top + y] for s in strokes]
        if nib_frac > 0 and face.size == body_mm:     # headings carry their own weight
            strokes = nib(strokes, nib_frac * body_mm, pen_mm)
        out += strokes
    return out, body_mm
