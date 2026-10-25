#!/usr/bin/env python3
"""Convert photographs to pen-plotter G-code.

    python3 image_to_gcode.py photo.jpg -o photo.gcode
    python3 image_to_gcode.py photo.jpg --engine waves --pen 0.3
    python3 image_to_gcode.py photo.jpg --engine hilbert --preview

Run ``--list-engines`` to see what each engine does.  The algorithms live in
the ``plotter`` package; this file is the command line around them.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2

from plotter import Config, ENGINES, PAPER_SIZES, convert, grbl, load_image
from plotter.gcode import format_duration


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert an image to pen-plotter G-code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", nargs="?", help="input image (PNG, JPEG, TIFF, ...)")
    p.add_argument("-o", "--output", help="output .gcode (default: <input>.gcode)")
    p.add_argument("--draw", metavar="SUBJECT",
                   choices=["frog", "truefrog", "toad", "treefrog", "dartfrog", "hornedfrog", "clawedfrog"],
                   help="generate a drawing from scratch instead of converting an image")
    p.add_argument("--name", help='with --draw: the animal\'s name, e.g. "Rana viridis"; '
                   "the same name always gives the same drawing (default: invent one)")
    p.add_argument("--seed", type=int, help="with --draw: pick an invented name by number")
    p.add_argument("--write", metavar="TEXT_FILE",
                   help='set the text in this file as lettering ("# " starts a heading, '
                        'leading spaces indent; "-" reads stdin)')
    p.add_argument("--style", default="calligraphy",
                   choices=["calligraphy", "book", "handwriting", "technical"],
                   help="with --write: lettering style")
    p.add_argument("--align", default="left", choices=["left", "center"],
                   help="with --write: line alignment")
    p.add_argument("--size", type=float, metavar="MM",
                   help="with --write: body letter size (default: fit the page)")
    p.add_argument("--list-engines", action="store_true",
                   help="describe the available engines and exit")

    g = p.add_argument_group("drawing")
    g.add_argument("-e", "--engine", default="illustration", choices=sorted(ENGINES),
                   help="how the image is turned into strokes")
    g.add_argument("--ink", type=float, default=1.0, metavar="SCALE",
                   help="overall ink multiplier; below 1 draws lighter")
    g.add_argument("--angle", type=float, default=45.0,
                   help="stroke angle in flat areas, degrees")
    g.add_argument("--spacing", type=float, default=0.0, metavar="MM",
                   help="engine line spacing; 0 derives it from the pen width")
    g.add_argument("--outline", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="trace landmark lines on top of the shading")
    g.add_argument("--outline-strength", type=float, default=0.6, metavar="F")
    g.add_argument("--hatch-layers", type=int, default=3,
                   help="crosshatch: how many directions may stack")
    g.add_argument("--region-threshold", type=float, default=0.25, metavar="F",
                   help="adaptive: texture energy needed to route a region "
                        "to flow-traced strokes instead of cross-hatch")
    g.add_argument("--region-segments", type=int, default=400,
                   help="adaptive: approximate number of SLIC superpixels")
    g.add_argument("--hatch-coherence", type=float, default=0.45, metavar="F",
                   help="crosshatch: how far hatching follows image texture; "
                        "raise for feathers and fur, lower for portraits")
    g.add_argument("--form-weight", type=float, default=0.8, metavar="F",
                   help="how strongly depth-estimated form steers hatching; "
                        "0 disables depth estimation entirely")
    g.add_argument("--depth-model", default="DepthAnythingV2_small",
                   choices=["DepthAnythingV2_small", "MiDaS_small", "DPT_Hybrid", "DPT_Large"],
                   help="Depth Anything V2 needs transformers; falls back to MiDaS_small")
    g.add_argument("--lines", default="learned", choices=["learned", "coherent", "canny"],
                   help="line source: learned (Informative Drawings), flow-guided DoG, or Canny")
    g.add_argument("--line-style", default="sketch", choices=["sketch", "contour"])
    g.add_argument("--shading", default="flowlines",
                   help="illustration: tone engine under the lines, or 'none'")
    g.add_argument("--shadow", type=float, default=0.40, metavar="F",
                   help="illustration: hatch only tones darker than this")
    g.add_argument("--focus", type=float, default=0.85, metavar="F",
                   help="illustration: fade shading on the background using depth")
    g.add_argument("--hatch-share", type=float, default=0.30, metavar="F",
                   help="crosshatch: coverage one direction may supply "
                        "before the next is laid across it")
    g.add_argument("--subject-focus", type=float, default=0.8, metavar="F",
                   help="every engine but illustration: lighten the background "
                        "using depth (0 = whole frame alike)")
    g.add_argument("--hilbert-smooth", type=float, default=0.5, metavar="F",
                   help="hilbert: round the right angles; plots faster")
    g.add_argument("--hilbert-detail", type=float, default=1.2, metavar="F",
                   help="hilbert: extra subdivision where fine detail lives")
    g.add_argument("--flow-coherence", type=float, default=1.0, metavar="F",
                   help="flowlines: 0 ignores image structure, 1 follows it")
    g.add_argument("--levels", type=int, default=12,
                   help="contours: number of tonal levels")
    g.add_argument("--points", type=int, default=12000,
                   help="tsp: number of stipple points")
    g.add_argument("--depth", type=int, default=9,
                   help="hilbert: maximum subdivision depth (darkest fill)")
    g.add_argument("--coarsest", type=float, default=4.0, metavar="MM",
                   help="hilbert: largest cell the curve may collapse to; "
                        "raise it for lighter, sparser fill, lower it if "
                        "pale areas come out looking bare")

    g = p.add_argument_group("pen and page")
    g.add_argument("--pen", type=float, default=0.4, metavar="MM",
                   help="ink trace width; this is what calibrates tone")
    g.add_argument("--paper", default="a4", choices=sorted(PAPER_SIZES),
                   help="paper size")
    g.add_argument("--portrait", action="store_true", help="portrait orientation")
    g.add_argument("--margin", type=float, default=10.0, metavar="MM")
    g.add_argument("--rotate-output", default="auto",
                   choices=["auto", "off", "ccw", "cw"],
                   help="turn the G-code for a landscape-only bed; auto turns "
                        "portrait pages only. The preview stays as composed")
    g.add_argument("--max-coverage", type=float, default=0.92, metavar="F",
                   help="darkest ink coverage the pen can usefully reach")

    g = p.add_argument_group("tone")
    g.add_argument("-r", "--resolution", type=int, default=1100, metavar="PX",
                   help="internal working resolution, long edge")
    g.add_argument("--contrast", type=float, default=2.0, metavar="F",
                   help="local contrast (CLAHE clip limit); 0 disables")
    g.add_argument("--detail", type=float, default=0.45,
                   help="structure-preserving detail boost, 0-1")
    g.add_argument("--gamma", type=float, default=1.0,
                   help="below 1 lightens midtones, above 1 darkens")
    g.add_argument("--exposure", type=float, default=0.0, help="-1 to 1")
    g.add_argument("--feature-gain", type=float, default=0.55, metavar="F",
                   help="push thin dark detail (lash lines, lip seams) "
                        "toward black so tone does not swallow it")
    g.add_argument("--chroma", type=float, default=0.18, metavar="F",
                   help="how much saturated colour darkens against neutrals")
    g.add_argument("--ink-response", type=float, default=0.25, metavar="F",
                   help="0 spreads tone perceptually, 1 matches luminance")
    g.add_argument("--denoise", type=float, default=0.0)
    g.add_argument("--invert", action="store_true")
    g.add_argument("--flip-x", action="store_true", help="mirror top to bottom")
    g.add_argument("--flip-y", action="store_true", help="mirror left to right")

    g = p.add_argument_group("paths")
    g.add_argument("--minutes", type=float, default=0.0, metavar="MIN",
                   help="plot-time limit: drop the strokes that add least "
                        "until the estimate fits (0 = no limit)")
    g.add_argument("--prune", type=float, default=0.2, metavar="F",
                   help="drop strokes the engine scores below this "
                        "(faint fragments); 0 keeps everything")
    g.add_argument("--merge", type=float, default=1.0, metavar="MM",
                   help="weld stroke ends closer than this into one stroke")
    g.add_argument("--bridge", type=float, default=3.0, metavar="MM",
                   help="cross gaps up to this with the pen down, over ink")
    g.add_argument("--bridge-min-ink", type=float, default=0.30, metavar="F",
                   help="minimum darkness under a bridge for it to be drawn")
    g.add_argument("--simplify", type=float, default=0.06, metavar="MM")
    g.add_argument("--no-sort", action="store_true",
                   help="skip travel-order optimisation")

    g = p.add_argument_group("machine")
    g.add_argument("--feed", type=float, default=3000.0, help="drawing feed, mm/min")
    g.add_argument("--travel-feed", type=float, default=6000.0, help="mm/min")
    g.add_argument("--accel", type=float, default=400.0, help="mm/s^2, estimate only")

    g = p.add_argument_group("output")
    g.add_argument("--preview", nargs="?", const="auto", metavar="FILE",
                   help="also write a preview PNG")
    g.add_argument("--show-travel", action="store_true",
                   help="draw pen-up moves on the preview")
    g.add_argument("--dpi", type=int, default=160)
    g.add_argument("-q", "--quiet", action="store_true")
    return p


def config_from_args(args) -> Config:
    cfg = Config()
    cfg.page.paper = args.paper
    cfg.page.portrait = args.portrait
    cfg.page.margin_mm = args.margin
    cfg.page.rotate_output = args.rotate_output
    cfg.page.resolve()

    cfg.pen.width_mm = args.pen
    cfg.pen.max_coverage = args.max_coverage

    cfg.machine.feed_draw = args.feed
    cfg.machine.feed_travel = args.travel_feed
    cfg.machine.accel_mm_s2 = args.accel

    cfg.tone.resolution = args.resolution
    cfg.tone.clahe_clip = args.contrast
    cfg.tone.detail = args.detail
    cfg.tone.gamma = args.gamma
    cfg.tone.exposure = args.exposure
    cfg.tone.chroma_weight = args.chroma
    cfg.tone.feature_gain = args.feature_gain
    cfg.tone.ink_response = args.ink_response
    cfg.tone.denoise = args.denoise
    cfg.tone.invert = args.invert
    cfg.tone.flip_x = args.flip_x
    cfg.tone.flip_y = args.flip_y

    cfg.render.engine = args.engine
    cfg.render.ink_scale = args.ink
    cfg.render.angle_deg = args.angle
    cfg.render.spacing_mm = args.spacing
    cfg.render.outline = args.outline
    cfg.render.outline_strength = args.outline_strength
    cfg.render.hatch_layers = args.hatch_layers
    cfg.render.hatch_share = args.hatch_share
    cfg.render.hatch_coherence = args.hatch_coherence
    cfg.render.region_threshold = args.region_threshold
    cfg.render.region_segments = args.region_segments
    cfg.render.form_weight = args.form_weight
    cfg.render.depth_model = args.depth_model
    cfg.render.line_method = args.lines
    cfg.render.line_style = args.line_style
    cfg.render.illustration_shading = args.shading
    cfg.render.illustration_shadow = args.shadow
    cfg.render.illustration_focus = args.focus
    cfg.render.hilbert_smooth = args.hilbert_smooth
    cfg.render.subject_focus = args.subject_focus
    cfg.render.hilbert_detail = args.hilbert_detail
    cfg.render.flow_coherence = args.flow_coherence
    cfg.render.contour_levels = args.levels
    cfg.render.tsp_points = args.points
    cfg.render.hilbert_max_depth = args.depth
    cfg.render.hilbert_coarsest_mm = args.coarsest

    cfg.paths.merge_tolerance_mm = args.merge
    cfg.paths.bridge_max_mm = args.bridge
    cfg.paths.bridge_min_ink = args.bridge_min_ink
    cfg.paths.time_limit_min = args.minutes
    cfg.paths.prune_below = args.prune
    cfg.paths.simplify_mm = args.simplify
    cfg.paths.sort = not args.no_sort
    # The plotter's own limits, if the app has read them over USB.
    return grbl.apply_profile(cfg)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_engines:
        width = max(len(k) for k in ENGINES)
        for name, (_, description, continuity) in ENGINES.items():
            print(f"  {name:<{width}}  {description}")
            print(f"  {'':<{width}}  pen lifts: {continuity}")
        return 0

    if args.draw:
        from plotter import draw_procedural
        cfg = config_from_args(args)
        result = draw_procedural(args.draw, args.seed, cfg, name=args.name)
        shown = result.stats["name"]
        slug = re.sub(r"[^a-z0-9]+", "-", shown.lower()).strip("-")
        out = Path(args.output) if args.output else Path(f"{slug}.gcode")
        out.write_text(result.gcode(f"{shown} ({args.draw})"))
        s = result.stats
        if not args.quiet:
            print(f"  {shown}: {s['image_mm']}, {s['label']}, "
                  f"{s['pen_lifts']} pen lifts -> {out}")
            print(f'  redraw with:  --draw {args.draw} --name "{shown}"')
        if args.preview is not None:
            path = str(out.with_suffix("")) + "_preview.png" if args.preview == "auto" else args.preview
            cv2.imwrite(path, cv2.cvtColor(result.preview(dpi=args.dpi), cv2.COLOR_RGB2BGR))
        return 0

    if args.write:
        from plotter import draw_text
        text = sys.stdin.read() if args.write == "-" else Path(args.write).read_text()
        cfg = config_from_args(args)
        result = draw_text(text, cfg, style=args.style, align=args.align, size_mm=args.size)
        stem = "lettering" if args.write == "-" else Path(args.write).stem
        out = Path(args.output) if args.output else Path(f"{stem}.gcode")
        out.write_text(result.gcode(f"{stem} ({args.style} lettering)"))
        s = result.stats
        if not args.quiet:
            print(f"  {args.style}, letters {s['body_mm']:.1f} mm: {s['label']}, "
                  f"{s['pen_lifts']} pen lifts -> {out}")
        if args.preview is not None:
            path = str(out.with_suffix("")) + "_preview.png" if args.preview == "auto" else args.preview
            cv2.imwrite(path, cv2.cvtColor(result.preview(dpi=args.dpi), cv2.COLOR_RGB2BGR))
        return 0

    if not args.input:
        parser.error("an input image is required (or use --draw frog / --write text.txt)")

    src = Path(args.input)
    if not src.exists():
        print(f"error: no such file: {src}", file=sys.stderr)
        return 1

    cfg = config_from_args(args)
    note = (lambda msg: None) if args.quiet else (lambda msg: print(f"  {msg} ..."))

    if not args.quiet:
        print(f"Converting {src.name} with the {args.engine} engine")
    result = convert(load_image(str(src)), cfg, progress=note)

    out = Path(args.output) if args.output else src.with_suffix(".gcode")
    header = (f"generated by image_to_gcode from {src.name}\n"
              f"engine {args.engine}, pen {args.pen} mm, "
              f"{cfg.page.paper} {'portrait' if args.portrait else 'landscape'}")
    out.write_text(result.gcode(header))

    s = result.stats
    if not args.quiet:
        print(f"\n  image      {s['image_mm']}")
        print(f"  strokes    {s['strokes']} "
              f"(welded down from {s['strokes_before_merge']})")
        print(f"  draw       {s['draw_dist_m']:.1f} m")
        print(f"  travel     {s['travel_dist_m']:.1f} m "
              f"(from {s['travel_unoptimised_m']:.1f} m unordered)")
        print(f"  plot time  {s['label']}  ({s['breakdown']})")
        if s.get("pruned") or s.get("budget_cut"):
            print(f"  left out   {s['pruned']} faint strokes"
                  + (f", {s['budget_cut']} more to fit {args.minutes:g} min"
                     if s["budget_cut"] else ""))
        if s.get("likeness") is not None:
            print(f"  likeness   {s['likeness']:.3f} SSIM at viewing distance "
                  f"(tone off by {s['tone_error']:.3f})")
        print(f"  wrote      {out}  ({out.stat().st_size / 1024:.0f} kB, "
              f"{s['points']} points)")
        print(f"  took       {format_duration(s['elapsed_s'])}")

    if args.preview is not None:
        path = (str(src.with_name(src.stem + "_preview.png"))
                if args.preview == "auto" else args.preview)
        img = result.preview(dpi=args.dpi, show_travel=args.show_travel)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not args.quiet:
            print(f"  preview    {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
