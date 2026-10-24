#!/usr/bin/env python3
"""Run engines over the photo test set and write comparison sheets.

    python3 tools/evaluate.py                          # default engines
    python3 tools/evaluate.py -e flowlines illustration -o ~/Downloads/eval

One sheet per photo (source + each engine's preview, labelled with plot time
and pen lifts) and a summary table, so a change is judged on every kind of
picture at once rather than on whichever one it was tuned against.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root

import cv2
import numpy as np

import plotter


def label(img, text, sub=""):
    bar = np.full((44, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_DUPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit(img, h):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--engines", nargs="+",
                    default=["flowlines", "adaptive", "illustration"])
    ap.add_argument("-p", "--photos", default="samples/photos/*.jpg")
    ap.add_argument("-o", "--out", default="eval_out")
    ap.add_argument("--height", type=int, default=520)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for path in sorted(glob.glob(args.photos)):
        name = os.path.splitext(os.path.basename(path))[0]
        img = plotter.load_image(path)
        tiles = [label(fit(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), args.height), name, "source")]
        for eng in args.engines:
            cfg = plotter.Config(); cfg.render.engine = eng
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                r = plotter.convert(img, cfg)
            cpu = time.perf_counter() - t0
            s = r.stats
            prev = r.preview(dpi=120)
            m = int(cfg.page.margin_mm / 25.4 * 120)
            prev = cv2.cvtColor(prev[m:-m, m:-m], cv2.COLOR_RGB2BGR)
            tiles.append(label(fit(prev, args.height), eng,
                               f"{s['label']}  likeness {s['likeness']:.2f}  {s['pen_lifts']} lifts"))
            rows.append((name, eng, s["label"], s["likeness"], s["tone_error"],
                         s["pen_lifts"], s["draw_dist_m"], cpu))
            print(f"{name:20s} {eng:13s} {s['label']:>9s} likeness {s['likeness']:.3f} "
                  f"{s['pen_lifts']:6d} lifts {s['draw_dist_m']:6.1f} m  {cpu:5.1f}s", flush=True)
        gap = np.full((args.height + 44, 8, 3), 235, np.uint8)
        sheet = tiles[0]
        for t in tiles[1:]:
            sheet = np.hstack([sheet, gap, t])
        cv2.imwrite(os.path.join(args.out, f"{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("| photo | engine | plot time | likeness | tone error | lifts | drawn m | compute s |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]} "
                    f"| {r[6]:.1f} | {r[7]:.1f} |\n")
    print(f"\nwrote {args.out}/")


if __name__ == "__main__":
    main()
