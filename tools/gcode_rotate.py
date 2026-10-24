#!/usr/bin/env python3
"""Rotate a finished G-code file 90 degrees onto a landscape page.

    python3 tools/gcode_rotate.py drawing.gcode
    python3 tools/gcode_rotate.py drawing.gcode -o turned.gcode --cw

For a plotter whose bed only goes landscape, a portrait plot has to be turned
before sending; turn the paper the same way and the drawing comes out upright.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root

from plotter.config import PAPER_SIZES
from plotter.transform import rotate_gcode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="G-code file to rotate")
    p.add_argument("-o", "--output", help="default: <input>_landscape.gcode")
    p.add_argument("--paper", default="a4", choices=sorted(PAPER_SIZES),
                   help="target paper; the plot is centred on its landscape form")
    p.add_argument("--page", metavar="WxH",
                   help="explicit source page in mm, e.g. 210x297")
    p.add_argument("--cw", action="store_true",
                   help="turn clockwise instead of anticlockwise")
    p.add_argument("--no-recentre", action="store_true",
                   help="keep the rotated position instead of centring it")
    args = p.parse_args(argv)

    src = Path(args.input)
    if not src.exists():
        print(f"error: no such file: {src}", file=sys.stderr)
        return 1

    if args.page:
        try:
            w, h = (float(v) for v in args.page.lower().split("x"))
        except ValueError:
            print(f"error: --page wants WxH in mm, got {args.page!r}", file=sys.stderr)
            return 1
    else:
        long_side, short_side = PAPER_SIZES[args.paper]
        w, h = short_side, long_side          # portrait source

    text, info = rotate_gcode(src.read_text(), w, h, clockwise=args.cw,
                              recentre=not args.no_recentre)
    if info["moves"] == 0:
        print("error: no coordinated moves found — is this G-code?", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else src.with_name(src.stem + "_landscape.gcode")
    out.write_text(text)

    page_w, page_h = info["page"]
    print(f"  moves      {info['moves']:,}")
    print(f"  was        {info['src_w']:.1f} × {info['src_h']:.1f} mm  (portrait)")
    print(f"  now        {info['out_w']:.1f} × {info['out_h']:.1f} mm  "
          f"on a {page_w:.0f} × {page_h:.0f} mm page")
    print(f"  X          {info['x_range'][0]:.1f} .. {info['x_range'][1]:.1f} mm")
    print(f"  Y          {info['y_range'][0]:.1f} .. {info['y_range'][1]:.1f} mm")
    if info["arcs"]:
        print(f"  WARNING    {info['arcs']} arc moves (G2/G3) found; their I/J "
              f"offsets are NOT rotated. Check the result.", file=sys.stderr)
    if info["out_w"] > page_w + 0.01 or info["out_h"] > page_h + 0.01:
        print("  WARNING    the drawing is larger than the target page.",
              file=sys.stderr)
    print(f"  wrote      {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
