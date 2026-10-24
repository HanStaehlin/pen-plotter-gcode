#!/usr/bin/env python3
"""Local web UI for the image-to-G-code converter.

    python3 app.py      then open http://localhost:8080
"""

from __future__ import annotations

import base64
import io
import re
import traceback

from flask import Flask, jsonify, render_template_string, request, send_file

import plotter
from plotter import Config, ENGINES, PAPER_SIZES
from plotter.gcode import encode_png, format_duration

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024

_last: dict = {}


def _f(name, default):
    try:
        return float(request.form.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _i(name, default):
    try:
        return int(float(request.form.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _b(name):
    return request.form.get(name, "0") == "1"


def config_from_form() -> Config:
    cfg = Config()
    cfg.page.paper = request.form.get("paper", "a4")
    cfg.page.portrait = _b("portrait")
    cfg.page.margin_mm = _f("margin", 10)
    cfg.page.rotate_output = "auto" if _b("rotate_out") else "off"
    cfg.page.resolve()

    cfg.pen.width_mm = _f("pen", 0.4)
    cfg.pen.max_coverage = _f("max_coverage", 0.92)

    cfg.machine.feed_draw = _f("feed", 3000)
    cfg.machine.feed_travel = _f("travel_feed", 6000)

    cfg.tone.resolution = _i("resolution", 1100)
    cfg.tone.clahe_clip = _f("contrast", 2.0)
    cfg.tone.detail = _f("detail", 0.45)
    cfg.tone.gamma = _f("gamma", 1.0)
    cfg.tone.exposure = _f("exposure", 0.0)
    cfg.tone.chroma_weight = _f("chroma", 0.18)
    cfg.tone.ink_response = _f("ink_response", 0.25)
    cfg.tone.feature_gain = _f("feature_gain", 0.55)
    cfg.tone.denoise = _f("denoise", 0.0)
    cfg.tone.invert = _b("invert")
    cfg.tone.flip_x = _b("flip_x")
    cfg.tone.flip_y = _b("flip_y")

    cfg.render.engine = request.form.get("engine", "flowlines")
    cfg.render.ink_scale = _f("ink", 1.0)
    cfg.render.angle_deg = _f("angle", 45)
    cfg.render.spacing_mm = _f("spacing", 0.0)
    cfg.render.flow_coherence = _f("coherence", 1.0)
    cfg.render.outline = _b("outline")
    cfg.render.outline_strength = _f("outline_strength", 0.6)
    cfg.render.hatch_layers = _i("hatch_layers", 3)
    cfg.render.hatch_share = _f("hatch_share", 0.30)
    cfg.render.hilbert_smooth = _f("hilbert_smooth", 0.35)
    cfg.render.hilbert_detail = _f("hilbert_detail", 1.2)
    cfg.render.hatch_coherence = _f("hatch_coherence", 0.45)
    cfg.render.region_segments = _i("region_segments", 400)
    cfg.render.region_threshold = _f("region_threshold", 0.25)
    cfg.render.coherence_gamma = _f("coherence_gamma", 2.0)
    cfg.render.form_weight = _f("form_weight", 0.8) if _b("use_depth") else 0.0
    cfg.render.line_method = request.form.get("line_method", "learned")
    cfg.render.line_style = request.form.get("line_style", "sketch")
    cfg.render.illustration_shading = request.form.get("illustration_shading", "crosshatch")
    cfg.render.illustration_shadow = _f("illustration_shadow", 0.40)
    cfg.render.illustration_focus = _f("illustration_focus", 0.85)
    cfg.render.illustration_tone = _f("illustration_tone", 0.75)
    cfg.render.contour_levels = _i("levels", 12)
    cfg.render.tsp_points = _i("points", 12000)
    cfg.render.hilbert_max_depth = _i("depth", 9)
    cfg.render.hilbert_coarsest_mm = _f("coarsest", 4.0)

    cfg.paths.merge_tolerance_mm = _f("merge", 1.0)
    cfg.paths.bridge_max_mm = _f("bridge", 3.0)
    cfg.paths.bridge_min_ink = _f("bridge_min_ink", 0.30)
    cfg.paths.time_limit_min = _f("time_limit", 0.0)
    return cfg


@app.route("/")
def index():
    engines = []
    for name, meta in ENGINES.items():
        # Descriptions read "Name - blurb"; the card wants those apart.
        label, _, blurb = meta[1].partition("\u2014")
        engines.append({"id": name, "label": label.strip(),
                        "blurb": blurb.strip(), "lifts": meta[2]})
    return render_template_string(HTML, engines=engines,
                                  papers=sorted(PAPER_SIZES))


@app.route("/generate", methods=["POST"])
def generate():
    upload = request.files.get("image")
    if not upload:
        return jsonify(error="No image uploaded")

    raw = upload.read()
    try:
        if _b("remove_bg"):
            img = plotter.cutout(raw)
        else:
            img = plotter.decode_image(raw)
    except Exception as exc:
        return jsonify(error=f"Could not read the image: {exc}")

    cfg = config_from_form()
    try:
        result = plotter.convert(img, cfg)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(error=f"{type(exc).__name__}: {exc}")

    if not result.paths:
        return jsonify(error="Nothing to draw — try raising Ink or Contrast.")
    return _respond(result, (upload.filename.rsplit(".", 1)[0] or "plot") + ".gcode",
                    f"engine {cfg.render.engine}, pen {cfg.pen.width_mm} mm")


@app.route("/procedural", methods=["POST"])
def procedural():
    kind = request.form.get("subject", "frog")
    typed = (request.form.get("name") or "").strip()
    cfg = config_from_form()
    try:
        result = plotter.draw_procedural(kind, None, cfg, name=typed or None)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(error=f"{type(exc).__name__}: {exc}")
    shown = result.stats["name"]
    slug = re.sub(r"[^a-z0-9]+", "-", shown.lower()).strip("-")
    return _respond(result, f"{slug}.gcode", f"{shown} ({kind})", name=shown)


def _respond(result, filename, header, **extra):
    global _last
    preview = result.preview(dpi=150, show_travel=_b("show_travel"))
    _last = {"gcode": result.gcode(header), "name": filename}
    s = result.stats
    return jsonify(
        preview=base64.b64encode(encode_png(preview)).decode(),
        stats={
            "time": s["label"],
            "draw_time": format_duration(s["draw_s"]),
            "travel_time": format_duration(s["travel_s"]),
            "lifts": s["pen_lifts"],
            "lifts_before": s["strokes_before_merge"],
            "draw_m": round(s["draw_dist_m"], 1),
            "travel_m": round(s["travel_dist_m"], 1),
            "travel_raw_m": round(s["travel_unoptimised_m"], 1),
            "size": s["image_mm"],
            "points": s["points"],
            "kb": round(len(_last["gcode"]) / 1024),
            "tone_error": (None if s["tone_error"] is None
                           else round(s["tone_error"], 3)),
            "likeness": (None if s.get("likeness") is None
                         else round(s["likeness"], 3)),
            "rotated": s["rotated_for_bed"],
            "resolution": s["resolution"],
            "asked_resolution": _i("resolution", 1100),
            "pen_px": round(s["pen_px"], 2),
            "elapsed": f"{s['elapsed_s']:.1f}s",
            "budget_cut": s.get("budget_cut", 0),
        },
        **extra,
    )


@app.route("/download")
def download():
    if not _last:
        return "Nothing generated yet", 404
    return send_file(io.BytesIO(_last["gcode"].encode()), mimetype="text/plain",
                     as_attachment=True, download_name=_last["name"])


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Plotter — image to G-code</title>
<style>
  /* Monochrome by design: the output is one ink on one sheet of paper, and
     the interface says the same thing. Every value below is a neutral. */
  :root{
    --surface:#ececea;      /* the desk the paper sits on */
    --panel:#ffffff;
    --field:#f5f5f3;
    --ink:#101010;
    --ink-2:#5e5e5c;
    --ink-3:#93938f;
    --rule:#e3e3df;
    --rule-2:#c9c9c4;
    --sheet-shadow:0 1px 2px rgba(0,0,0,.07),0 12px 32px rgba(0,0,0,.13);
    --r:7px;
    --sans:ui-sans-serif,-apple-system,"SF Pro Text","Segoe UI",Inter,
           "Helvetica Neue",Arial,sans-serif;
    --mono:ui-monospace,"SF Mono",SFMono-Regular,"JetBrains Mono",Menlo,monospace;
  }
  @media (prefers-color-scheme:dark){
    :root{
      --surface:#0c0c0c; --panel:#161616; --field:#1e1e1e;
      --ink:#f2f2f0; --ink-2:#a3a39f; --ink-3:#6e6e6a;
      --rule:#282828; --rule-2:#3a3a3a;
      --sheet-shadow:0 2px 4px rgba(0,0,0,.5),0 18px 44px rgba(0,0,0,.6);
    }
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{
    font-family:var(--sans);font-size:13px;line-height:1.5;
    background:var(--surface);color:var(--ink);
    -webkit-font-smoothing:antialiased;
    font-variant-numeric:tabular-nums;
  }

  /* ── masthead ──────────────────────────────────────────────── */
  .top{
    display:flex;align-items:center;justify-content:space-between;
    gap:20px;padding:0 22px;height:56px;
    background:var(--panel);border-bottom:1px solid var(--rule);
    position:sticky;top:0;z-index:20;
  }
  .brand{display:flex;align-items:center;gap:11px}
  .brand svg{display:block}
  .brand h1{font-size:14px;font-weight:600;letter-spacing:-.01em}
  .brand .tag{
    font-size:10px;letter-spacing:.14em;text-transform:uppercase;
    color:var(--ink-3);padding-left:11px;margin-left:1px;
    border-left:1px solid var(--rule);
  }
  .top .spec{font-family:var(--mono);font-size:11px;color:var(--ink-3)}

  /* ── shell ─────────────────────────────────────────────────── */
  .shell{
    display:grid;grid-template-columns:326px minmax(0,1fr);
    gap:18px;padding:18px;align-items:start;
    max-width:1680px;margin:0 auto;
  }

  /* ── control column ────────────────────────────────────────── */
  .panel{
    background:var(--panel);border:1px solid var(--rule);border-radius:var(--r);
    position:sticky;top:74px;display:flex;flex-direction:column;
    max-height:calc(100vh - 92px);overflow:hidden;
  }
  .scroll{overflow-y:auto;padding:4px 0 2px;flex:1;scrollbar-width:thin}
  .scroll::-webkit-scrollbar{width:9px}
  .scroll::-webkit-scrollbar-thumb{
    background:var(--rule-2);border-radius:9px;
    border:3px solid var(--panel);
  }

  section{padding:15px 17px;border-bottom:1px solid var(--rule)}
  section:last-child{border-bottom:none}
  .sec-head{
    display:flex;align-items:baseline;justify-content:space-between;
    gap:8px;margin-bottom:12px;
  }
  .sec-head h2{
    font-size:10px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;
    color:var(--ink-3);
  }
  .sec-head .aside{font-family:var(--mono);font-size:10px;color:var(--ink-3)}

  /* ── file drop ─────────────────────────────────────────────── */
  .drop{
    border:1px dashed var(--rule-2);border-radius:var(--r);
    padding:20px 14px;text-align:center;cursor:pointer;
    transition:border-color .15s,background .15s;background:var(--field);
  }
  .drop:hover,.drop.over{border-color:var(--ink);background:var(--panel)}
  .drop input{display:none}
  .drop .glyph{color:var(--ink-3);margin-bottom:7px}
  .drop .lead{font-size:12.5px}
  .drop .lead b{font-weight:600;border-bottom:1px solid var(--rule-2)}
  .drop .hint{font-size:10.5px;color:var(--ink-3);margin-top:3px;
              letter-spacing:.05em;text-transform:uppercase}
  .drop.loaded{padding:11px;border-style:solid;text-align:left;
               display:flex;gap:11px;align-items:center}
  .drop.loaded .glyph,.drop.loaded .lead,.drop.loaded .hint{display:none}
  .drop img{display:none;width:54px;height:54px;object-fit:cover;
            border-radius:4px;border:1px solid var(--rule);flex:none}
  .drop.loaded img{display:block}
  .or{font-size:10.5px;color:var(--ink-3);text-align:center;margin:12px 0 8px;
       letter-spacing:.06em;text-transform:uppercase}
  .fname{display:none;font-size:12px;font-weight:500;overflow:hidden;
         text-overflow:ellipsis;white-space:nowrap}
  .fname small{display:block;font-family:var(--mono);font-size:10px;
               color:var(--ink-3);font-weight:400;letter-spacing:.04em;
               text-transform:uppercase;margin-top:2px}
  .drop.loaded .fname{display:block}

  /* ── engine cards ──────────────────────────────────────────── */
  .engines{display:flex;flex-direction:column;gap:5px}
  .eng{display:block;cursor:pointer;position:relative}
  .eng input{position:absolute;opacity:0;pointer-events:none}
  .eng .box{
    display:block;                 /* a span would collapse the card inline */
    border:1px solid var(--rule);border-radius:6px;padding:9px 11px;
    transition:border-color .12s,background .12s;background:var(--panel);
  }
  .eng:hover .box{border-color:var(--rule-2)}
  .eng input:checked + .box{border-color:var(--ink);background:var(--field)}
  .eng input:focus-visible + .box{outline:2px solid var(--ink);outline-offset:2px}
  .eng .line1{display:flex;align-items:baseline;justify-content:space-between;gap:8px}
  .eng .nm{font-size:12.5px;font-weight:600;letter-spacing:-.005em;min-width:0}
  .eng .lift{
    font-family:var(--mono);font-size:9.5px;color:var(--ink-3);
    letter-spacing:.02em;white-space:nowrap;
  }
  .eng input:checked + .box .lift{color:var(--ink-2)}
  .eng .bl{display:block;font-size:11px;color:var(--ink-2);margin-top:2px;
           line-height:1.4}

  /* ── controls ──────────────────────────────────────────────── */
  .ctl{margin-top:12px}
  .ctl:first-child{margin-top:0}
  .ctl > label{
    display:flex;align-items:baseline;justify-content:space-between;gap:8px;
    font-size:11.5px;color:var(--ink-2);margin-bottom:6px;
  }
  .ctl > label .v{font-family:var(--mono);font-size:11px;color:var(--ink)}
  .tip{
    display:inline-flex;align-items:center;justify-content:center;
    width:13px;height:13px;border-radius:50%;border:1px solid var(--rule-2);
    font-size:9px;color:var(--ink-3);cursor:help;flex:none;
    font-family:var(--sans);line-height:1;
  }
  .tip:hover{border-color:var(--ink);color:var(--ink)}

  input[type=range]{
    -webkit-appearance:none;appearance:none;width:100%;height:16px;
    background:transparent;display:block;cursor:pointer;
  }
  input[type=range]::-webkit-slider-runnable-track{
    height:2px;border-radius:2px;
    background:linear-gradient(to right,var(--ink) 0 var(--pct,50%),
               var(--rule-2) var(--pct,50%) 100%);
  }
  input[type=range]::-webkit-slider-thumb{
    -webkit-appearance:none;width:13px;height:13px;border-radius:50%;
    background:var(--panel);border:1.5px solid var(--ink);margin-top:-5.5px;
    transition:transform .1s;
  }
  input[type=range]:active::-webkit-slider-thumb{transform:scale(1.18)}
  input[type=range]::-moz-range-track{height:2px;background:var(--rule-2)}
  input[type=range]::-moz-range-progress{height:2px;background:var(--ink)}
  input[type=range]::-moz-range-thumb{
    width:11px;height:11px;border-radius:50%;
    background:var(--panel);border:1.5px solid var(--ink);
  }

  select,input[type=number]{
    width:100%;font-family:var(--sans);font-size:12px;color:var(--ink);
    background:var(--field);border:1px solid var(--rule);border-radius:5px;
    padding:7px 9px;
  }
  input[type=number]{font-family:var(--mono);font-size:11.5px}
  select:focus,input[type=number]:focus{outline:none;border-color:var(--ink)}
  select{
    -webkit-appearance:none;appearance:none;padding-right:26px;cursor:pointer;
    background-image:url("data:image/svg+xml;charset=utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='9' height='6'%3E%3Cpath d='M1 1l3.5 3.5L8 1' stroke='%2393938f' stroke-width='1.4' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
    background-repeat:no-repeat;background-position:right 9px center;
  }
  .pair{display:grid;grid-template-columns:1fr 1fr;gap:9px}

  .check{
    display:flex;align-items:center;gap:9px;font-size:12px;
    cursor:pointer;padding:4px 0;color:var(--ink-2);
  }
  .check:hover{color:var(--ink)}
  .check input{
    -webkit-appearance:none;appearance:none;width:14px;height:14px;flex:none;
    border:1px solid var(--rule-2);border-radius:3px;background:var(--field);
    position:relative;cursor:pointer;
  }
  .check input:checked{background:var(--ink);border-color:var(--ink)}
  .check input:checked::after{
    content:"";position:absolute;left:4px;top:1px;width:4px;height:8px;
    border:solid var(--panel);border-width:0 1.6px 1.6px 0;
    transform:rotate(42deg);
  }
  .check input:focus-visible{outline:2px solid var(--ink);outline-offset:2px}

  details > summary{
    list-style:none;cursor:pointer;display:flex;align-items:center;gap:7px;
    font-size:10px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;
    color:var(--ink-3);
  }
  details > summary::-webkit-details-marker{display:none}
  details > summary::before{
    content:"";width:5px;height:5px;flex:none;
    border-right:1.4px solid currentColor;border-bottom:1.4px solid currentColor;
    transform:rotate(-45deg);transition:transform .15s;margin-left:1px;
  }
  details[open] > summary::before{transform:rotate(45deg)}
  details > summary:hover{color:var(--ink)}
  details .body{padding-top:13px}

  /* ── actions ───────────────────────────────────────────────── */
  .foot{
    padding:13px 17px;border-top:1px solid var(--rule);background:var(--panel);
    display:flex;flex-direction:column;gap:8px;flex:none;
  }
  .btn{
    width:100%;font-family:var(--sans);font-size:13px;font-weight:600;
    padding:11px;border-radius:5px;cursor:pointer;border:1px solid var(--ink);
    background:var(--ink);color:var(--panel);transition:opacity .12s;
  }
  .btn:hover:not(:disabled){opacity:.84}
  .btn:disabled{opacity:.28;cursor:not-allowed}
  .btn.ghost{background:transparent;color:var(--ink);border-color:var(--rule-2)}
  .btn.ghost:hover{border-color:var(--ink);opacity:1}
  .note{font-size:10.5px;color:var(--ink-3);text-align:center;min-height:1.3em;
        line-height:1.45}
  .note.bad{color:var(--ink)}
  .note.bad::before{content:"⚠ ";}

  /* ── stage ─────────────────────────────────────────────────── */
  .stage{
    background:var(--panel);border:1px solid var(--rule);border-radius:var(--r);
    min-height:calc(100vh - 92px);display:flex;flex-direction:column;
    overflow:hidden;position:relative;
  }
  .bar{height:1px;background:var(--rule);overflow:hidden;flex:none}
  .bar::after{
    content:"";display:none;height:100%;width:34%;background:var(--ink);
    animation:slide 1.15s ease-in-out infinite;
  }
  .stage.busy .bar::after{display:block}
  @keyframes slide{
    0%{transform:translateX(-105%)}
    100%{transform:translateX(400%)}
  }

  .sheetwrap{
    flex:1;display:flex;flex-direction:column;align-items:center;
    justify-content:center;gap:13px;padding:32px 28px;
    background:var(--surface);min-height:340px;
  }
  .stage.busy .sheetwrap{opacity:.42;transition:opacity .2s}
  .sheet{
    background:#fff;box-shadow:var(--sheet-shadow);border-radius:2px;
    max-width:100%;line-height:0;
  }
  .sheet img{display:block;max-width:100%;max-height:calc(100vh - 330px);
             border-radius:2px}
  .caption{
    font-family:var(--mono);font-size:10px;letter-spacing:.08em;
    text-transform:uppercase;color:var(--ink-3);
  }
  .blank{text-align:center;color:var(--ink-3);max-width:340px}
  .blank svg{margin:0 auto 16px;display:block;color:var(--rule-2)}
  .blank h3{font-size:14px;font-weight:600;color:var(--ink-2);margin-bottom:5px}
  .blank p{font-size:12px;line-height:1.6}

  /* ── readout ───────────────────────────────────────────────── */
  .readout{display:none;border-top:1px solid var(--rule);background:var(--panel);
           flex:none}
  .readout.on{display:block}
  .headline{
    display:flex;align-items:baseline;gap:11px;flex-wrap:wrap;
    padding:15px 20px 12px;
  }
  .headline .big{font-size:27px;font-weight:600;letter-spacing:-.02em;
                 font-family:var(--mono)}
  .headline .cap{font-size:11px;color:var(--ink-3);letter-spacing:.04em;
                 text-transform:uppercase}
  .headline .rule{flex:1;height:1px;background:var(--rule);min-width:20px}
  .metrics{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));
    border-top:1px solid var(--rule);
  }
  .metrics > div{
    padding:11px 20px;border-right:1px solid var(--rule);
  }
  .metrics > div:last-child{border-right:none}
  .metrics dt{
    font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;
    color:var(--ink-3);margin-bottom:3px;
  }
  .metrics dd{font-family:var(--mono);font-size:14px;font-weight:500}
  .metrics dd .unit{font-size:10px;color:var(--ink-3);margin-left:2px}
  .delta{font-size:10px;color:var(--ink-3);font-family:var(--mono);margin-top:2px}

  @media (max-width:940px){
    .shell{grid-template-columns:1fr}
    .panel{position:static;max-height:none}
    .stage{min-height:0}
  }
</style>
</head>
<body>

<div class="top">
  <div class="brand">
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <rect x="1.5" y="1.5" width="19" height="19" rx="2.5"
            stroke="currentColor" stroke-width="1.2" opacity=".28"/>
      <path d="M5 15.5c2.2-6.4 4.1-8.6 5.6-6.7 1.5 1.9-1.9 5.4-.4 6.5 1.4 1.1 4-1.6 6.3-8"
            stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
    </svg>
    <h1>Plotter</h1>
    <span class="tag">image → g-code</span>
  </div>
  <div class="spec" id="spec">A4 landscape · 0.40 mm pen</div>
</div>

<div class="shell">
  <aside class="panel">
    <div class="scroll">

      <section>
        <div class="sec-head"><h2>Source</h2></div>
        <div class="drop" id="drop">
          <div class="glyph">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="4" width="18" height="16" rx="2"
                    stroke="currentColor" stroke-width="1.3"/>
              <circle cx="8.5" cy="9.5" r="1.6" fill="currentColor"/>
              <path d="M3.5 17l4.8-4.8 3.6 3.6 3-3 5.6 5.6"
                    stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
            </svg>
          </div>
          <div class="lead">Drop an image or <b>browse</b></div>
          <div class="hint">png · jpeg · tiff · bmp</div>
          <img id="thumb" alt="">
          <div class="fname" id="fname"></div>
          <input type="file" id="file" accept="image/*">
        </div>
        <div class="or">or draw one from scratch</div>
        <div class="pair">
          <select id="subject"><option value="frog" selected>Frog (any family)</option><option value="truefrog">True frog</option><option value="toad">Toad</option><option value="treefrog">Tree frog</option><option value="dartfrog">Poison dart frog</option><option value="hornedfrog">Horned frog</option><option value="clawedfrog">Clawed frog</option></select>
          <input type="text" id="frogname" placeholder="name (blank = invent one)">
        </div>
        <button class="btn ghost" id="drawSubject" style="margin-top:8px">Draw a new one</button>
      </section>

      <section>
        <div class="sec-head"><h2>Engine</h2><span class="aside">pen lifts</span></div>
        <div class="engines">
          {% for e in engines %}
          <label class="eng">
            <input type="radio" name="engine" value="{{ e.id }}"
                   {% if loop.first %}checked{% endif %}>
            <span class="box">
              <span class="line1">
                <span class="nm">{{ e.label }}</span>
                <span class="lift">{{ e.lifts }}</span>
              </span>
              <span class="bl">{{ e.blurb }}</span>
            </span>
          </label>
          {% endfor %}
        </div>

        <div class="ctl">
          <label><span>Time limit
            <span class="tip" title="Longest the plot may take. The strokes that add least to the picture are dropped first - short faint lines and hatching in the lightest shadows - until the estimate fits. Off draws everything.">i</span></span>
            <span class="v" id="time_limitV">off</span></label>
          <input type="range" id="time_limit" min="0" max="60" step="1" value="0">
        </div>
        <div class="ctl" data-for="flowlines">
          <label>Follow image structure <span class="v" id="coherenceV">100%</span></label>
          <input type="range" id="coherence" min="0" max="1" step="0.05" value="1">
        </div>
        <div class="ctl" data-for="crosshatch">
          <label>Hatch directions <span class="v" id="hatch_layersV">3</span></label>
          <input type="range" id="hatch_layers" min="1" max="5" step="1" value="3">
        </div>
        <div class="ctl" data-for="illustration">
          <label>Shading
            <span class="v"></span></label>
          <select id="illustration_shading">
            <option value="crosshatch" selected>Cross-hatch</option>
            <option value="flowlines">Flow lines</option>
            <option value="adaptive">Adaptive</option>
            <option value="none">None (lines only)</option>
          </select>
        </div>
        <div class="ctl" data-for="illustration">
          <label><span>Shade only below
            <span class="tip" title="Tones lighter than this get no hatching at all, as in pen-and-ink illustration: the paper carries the light and mid tones. Raise for a sparser drawing, lower for more tone.">i</span></span>
            <span class="v" id="illustration_shadowV">0.40</span></label>
          <input type="range" id="illustration_shadow" min="0" max="0.8" step="0.05" value="0.40">
        </div>
        <div class="ctl" data-for="illustration">
          <label><span>Subject focus
            <span class="tip" title="Uses depth to fade shading on the background so detail goes to the subject. Turn down for landscapes, where everything is far.">i</span></span>
            <span class="v" id="illustration_focusV">0.85</span></label>
          <input type="range" id="illustration_focus" min="0" max="1" step="0.05" value="0.85">
        </div>
        <div class="ctl" data-for="illustration">
          <label>Shading strength <span class="v" id="illustration_toneV">0.75</span></label>
          <input type="range" id="illustration_tone" min="0.2" max="1" step="0.05" value="0.75">
        </div>
        <div class="ctl" data-for="adaptive">
          <label>
            <span>Texture sensitivity
              <span class="tip" title="How much local tone variation it takes for a region to be routed to flow-traced strokes instead of cross-hatch. Lower catches more texture (feathers, fur) but risks routing smooth gradients there too.">i</span>
            </span>
            <span class="v" id="region_thresholdV">0.25</span>
          </label>
          <input type="range" id="region_threshold" min="0.10" max="0.55" step="0.02" value="0.25">
        </div>
        <div class="ctl" data-for="adaptive">
          <label>Region count <span class="v" id="region_segmentsV">400</span></label>
          <input type="range" id="region_segments" min="100" max="1000" step="50" value="400">
        </div>
        <div class="ctl" data-for="crosshatch">
          <label>
            <span>Follow texture
              <span class="tip" title="How far hatching bends to follow the image's own structure. High keeps feathers, fur and hair; too high and smooth skin turns into concentric rings around every highlight. No single value suits both — raise it for texture, lower it for portraits.">i</span>
            </span>
            <span class="v" id="hatch_coherenceV">0.45</span>
          </label>
          <input type="range" id="hatch_coherence" min="0" max="1" step="0.05" value="0.45">
        </div>
        <div class="ctl" data-for="crosshatch">
          <label>
            <span>Tone per direction
              <span class="tip" title="How much darkness one angle may supply before the next is laid across it. Lower it and cross-hatching starts earlier, so more of the picture shows two or three directions.">i</span>
            </span>
            <span class="v" id="hatch_shareV">0.30</span>
          </label>
          <input type="range" id="hatch_share" min="0.1" max="0.7" step="0.02" value="0.30">
        </div>
        <div class="ctl" data-for="hilbert">
          <label>
            <span>Corner rounding
              <span class="tip" title="A plotter nearly stops at every square corner, and a Hilbert curve is nothing but corners. Rounding them plots faster and softens the maze into something closer to drawn line.">i</span>
            </span>
            <span class="v" id="hilbert_smoothV">0.35</span>
          </label>
          <input type="range" id="hilbert_smooth" min="0" max="0.5" step="0.05" value="0.35">
        </div>
        <div class="ctl" data-for="hilbert">
          <label>Detail depth <span class="v" id="hilbert_detailV">1.2</span></label>
          <input type="range" id="hilbert_detail" min="0" max="2.5" step="0.1" value="1.2">
        </div>
        <div class="ctl" data-for="contours">
          <label>Tonal levels <span class="v" id="levelsV">12</span></label>
          <input type="range" id="levels" min="3" max="40" step="1" value="12">
        </div>
        <div class="ctl" data-for="tsp">
          <label>Stipple points <span class="v" id="pointsV">12 000</span></label>
          <input type="range" id="points" min="1000" max="40000" step="500" value="12000">
        </div>
        <div class="ctl" data-for="hilbert">
          <label>
            <span>Coarsest fill
              <span class="tip" title="The largest cell the curve may collapse to. A big cell carries a single squiggle, so past a few millimetres pale areas stop reading as light tone and start reading as bare patches. Lower this if the subject looks unpainted.">i</span>
            </span>
            <span class="v" id="coarsestV">4.0 mm</span>
          </label>
          <input type="range" id="coarsest" min="1.5" max="16" step="0.5" value="4">
        </div>
        <div class="ctl" data-for="hilbert">
          <label>Finest detail <span class="v" id="depthV">depth 9</span></label>
          <input type="range" id="depth" min="5" max="10" step="1" value="9">
        </div>
        <div class="ctl" data-for="waves,contours">
          <label>Line spacing <span class="v" id="spacingV">auto</span></label>
          <input type="range" id="spacing" min="0" max="6" step="0.1" value="0">
        </div>
      </section>

      <section>
        <div class="sec-head"><h2>Form &amp; line</h2></div>
        <div class="ctl pair">
          <div>
            <label style="display:block;font-size:11.5px;color:var(--ink-2);margin-bottom:6px">Line source
              <span class="tip" title="Learned: a network trained on artist drawings (Informative Drawings) picks the lines an illustrator would keep. Coherent: flow-guided edge detection. Canny: the old detector.">i</span></label>
            <select id="line_method">
              <option value="learned" selected>Learned</option>
              <option value="coherent">Coherent</option>
              <option value="canny">Canny</option>
            </select>
          </div>
          <div>
            <label style="display:block;font-size:11.5px;color:var(--ink-2);margin-bottom:6px">Line style</label>
            <select id="line_style">
              <option value="sketch" selected>Sketch</option>
              <option value="contour">Contour</option>
            </select>
          </div>
        </div>
        <label class="check">
          <input type="checkbox" id="use_depth" checked> Depth-driven form
          <span class="tip" title="Estimates depth (Depth Anything V2, or MiDaS without transformers) on the GPU and hatches along the surface's principal curvature, so strokes wrap a form instead of ruling straight across it. Only applied where the image itself has no structure to follow, so texture is left alone. Needs PyTorch; skipped if absent.">i</span>
        </label>
        <div class="ctl" data-when="use_depth">
          <label>Form strength <span class="v" id="form_weightV">0.80</span></label>
          <input type="range" id="form_weight" min="0.1" max="1" step="0.05" value="0.8">
        </div>
      </section>

      <section>
        <div class="sec-head"><h2>Ink</h2></div>
        <div class="ctl">
          <label>Amount <span class="v" id="inkV">1.00×</span></label>
          <input type="range" id="ink" min="0.3" max="2" step="0.05" value="1">
        </div>
        <div class="ctl">
          <label>Angle in flat areas <span class="v" id="angleV">45°</span></label>
          <input type="range" id="angle" min="0" max="180" step="5" value="45">
        </div>
        <label class="check">
          <input type="checkbox" id="outline" checked> Draw landmark lines
          <span class="tip" title="Traces the thin dark lines a person would draw — a lid line, a lip seam, a nostril — on top of the shading. On a portrait this is what makes an eye read as an eye rather than a dark patch.">i</span>
        </label>
        <div class="ctl" data-when="outline">
          <label>Landmark strength <span class="v" id="outline_strengthV">0.60</span></label>
          <input type="range" id="outline_strength" min="0.1" max="1" step="0.05" value="0.6">
        </div>
      </section>

      <section>
        <div class="sec-head"><h2>Pen &amp; page</h2></div>
        <div class="ctl">
          <label>
            <span>Pen width
              <span class="tip" title="The real ink trace width, not what the barrel claims. Tone is computed from this, so it is what makes greys come out grey.">i</span>
            </span>
            <span class="v" id="penV">0.40 mm</span>
          </label>
          <input type="range" id="pen" min="0.1" max="1.5" step="0.05" value="0.4">
        </div>
        <div class="ctl pair">
          <div>
            <label style="display:block;font-size:11.5px;color:var(--ink-2);margin-bottom:6px">Paper</label>
            <select id="paper">
              {% for p in papers %}
              <option value="{{ p }}"{% if p == 'a4' %} selected{% endif %}>{{ p|upper }}</option>
              {% endfor %}
            </select>
          </div>
          <div>
            <label style="display:block;font-size:11.5px;color:var(--ink-2);margin-bottom:6px">Margin, mm</label>
            <input type="number" id="margin" value="10" min="0" max="60" step="1">
          </div>
        </div>
        <label class="check" style="margin-top:9px"><input type="checkbox" id="portrait"> Portrait</label>
        <label class="check" data-when="portrait">
          <input type="checkbox" id="rotate_out" checked>
          Turn G-code to landscape
          <span class="tip" title="Plotter beds are wider than they are tall, so a portrait page cannot be drawn as composed. The G-code is turned 90 degrees anticlockwise; turn the paper the same way and the drawing comes out upright. The preview stays as composed.">i</span>
        </label>
      </section>

      <section>
        <div class="sec-head"><h2>Tone</h2></div>
        <div class="ctl">
          <label>Local contrast <span class="v" id="contrastV">2.0</span></label>
          <input type="range" id="contrast" min="0" max="6" step="0.1" value="2">
        </div>
        <div class="ctl">
          <label>Detail <span class="v" id="detailV">0.45</span></label>
          <input type="range" id="detail" min="0" max="1" step="0.05" value="0.45">
        </div>
        <div class="ctl">
          <label>
            <span>Fine detail
              <span class="tip" title="Pushes thin dark structure toward black. Without it a lip seam carries so little area that the surrounding skin tone swallows its ink budget and it disappears.">i</span>
            </span>
            <span class="v" id="feature_gainV">0.55</span>
          </label>
          <input type="range" id="feature_gain" min="0" max="1" step="0.05" value="0.55">
        </div>
        <div class="ctl">
          <label>Exposure <span class="v" id="exposureV">0.00</span></label>
          <input type="range" id="exposure" min="-0.4" max="0.4" step="0.02" value="0">
        </div>
        <div class="ctl">
          <label>Gamma <span class="v" id="gammaV">1.00</span></label>
          <input type="range" id="gamma" min="0.4" max="2.2" step="0.05" value="1">
        </div>
        <div style="display:flex;flex-direction:column;gap:1px;margin-top:11px">
          <label class="check"><input type="checkbox" id="invert"> Invert</label>
          <label class="check"><input type="checkbox" id="flip_x"> Mirror top / bottom</label>
          <label class="check"><input type="checkbox" id="flip_y"> Mirror left / right</label>
          <label class="check"><input type="checkbox" id="remove_bg"> Remove background</label>
        </div>
      </section>

      <section>
        <details>
          <summary>Colour &amp; resolution</summary>
          <div class="body">
            <div class="ctl">
              <label>
                <span>Colour separation
                  <span class="tip" title="How much saturated colour darkens against a neutral of the same lightness. At zero, a red and a grey that are equally light become the same tone.">i</span>
                </span>
                <span class="v" id="chromaV">0.18</span>
              </label>
              <input type="range" id="chroma" min="0" max="0.6" step="0.02" value="0.18">
            </div>
            <div class="ctl">
              <label>
                <span>Tone response
                  <span class="tip" title="0 spreads the pen's range evenly across perceived lightness. 1 matches the photograph's real luminance — accurate, much darker, much slower to plot.">i</span>
                </span>
                <span class="v" id="ink_responseV">0.25</span>
              </label>
              <input type="range" id="ink_response" min="0" max="1" step="0.05" value="0.25">
            </div>
            <div class="ctl">
              <label>Working resolution <span class="v" id="resolutionV">1100 px</span></label>
              <input type="range" id="resolution" min="400" max="2200" step="50" value="1100">
            </div>
          </div>
        </details>
      </section>

      <section>
        <details>
          <summary>Pen lifts &amp; machine</summary>
          <div class="body">
            <div class="ctl">
              <label>Weld ends within <span class="v" id="mergeV">1.0 mm</span></label>
              <input type="range" id="merge" min="0" max="4" step="0.1" value="1">
            </div>
            <div class="ctl">
              <label>
                <span>Bridge gaps up to
                  <span class="tip" title="Crosses short gaps with the pen down instead of lifting — but only where the image is dark enough to hide the extra line.">i</span>
                </span>
                <span class="v" id="bridgeV">3.0 mm</span>
              </label>
              <input type="range" id="bridge" min="0" max="12" step="0.5" value="3">
            </div>
            <div class="ctl">
              <label>Bridge only over ink above <span class="v" id="bridge_min_inkV">0.30</span></label>
              <input type="range" id="bridge_min_ink" min="0.05" max="0.9" step="0.05" value="0.3">
            </div>
            <div class="ctl">
              <label>Maximum ink coverage <span class="v" id="max_coverageV">0.92</span></label>
              <input type="range" id="max_coverage" min="0.4" max="1" step="0.02" value="0.92">
            </div>
            <div class="ctl pair">
              <div>
                <label style="display:block;font-size:11.5px;color:var(--ink-2);margin-bottom:6px">Draw feed</label>
                <input type="number" id="feed" value="3000" step="100">
              </div>
              <div>
                <label style="display:block;font-size:11.5px;color:var(--ink-2);margin-bottom:6px">Travel feed</label>
                <input type="number" id="travel_feed" value="6000" step="100">
              </div>
            </div>
            <label class="check" style="margin-top:9px">
              <input type="checkbox" id="show_travel"> Show pen-up moves in preview
            </label>
          </div>
        </details>
      </section>

    </div>

    <div class="foot">
      <button class="btn" id="go" disabled>Generate</button>
      <button class="btn ghost" id="dl" style="display:none">Download G-code</button>
      <p class="note" id="msg"></p>
    </div>
  </aside>

  <main class="stage" id="stage">
    <div class="bar"></div>
    <div class="sheetwrap" id="sheetwrap">
      <div class="blank" id="blank">
        <svg width="46" height="46" viewBox="0 0 24 24" fill="none">
          <rect x="3.5" y="2.5" width="17" height="19" rx="1.5"
                stroke="currentColor" stroke-width="1.1"/>
          <path d="M7 16.5c1.9-5.6 3.5-7.5 4.8-5.9 1.3 1.7-1.6 4.8-.3 5.7 1.2 1 3.4-1.4 5.4-7"
                stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
        </svg>
        <h3>No plot yet</h3>
        <p>Drop in a photograph, pick an engine and hit Generate.
           The preview shows exactly what the pen lays down.</p>
      </div>
    </div>

    <div class="readout" id="readout">
      <div class="headline">
        <span class="big" id="bigTime">—</span>
        <span class="cap">estimated plot time</span>
        <span class="rule"></span>
        <span class="cap" id="engCap"></span>
      </div>
      <dl class="metrics">
        <div><dt>Pen lifts</dt><dd id="mLifts">—</dd><div class="delta" id="dLifts"></div></div>
        <div><dt>Drawn</dt><dd id="mDraw">—</dd><div class="delta" id="dDraw"></div></div>
        <div><dt>Travel</dt><dd id="mTravel">—</dd><div class="delta" id="dTravel"></div></div>
        <div><dt title="How closely the finished plot matches the photo's tone when seen from arm's length (SSIM, 1 = identical). Compare it between settings of one engine: illustration leaves light tones to the paper on purpose, so it scores lower than the dense engines even where it looks better.">Likeness</dt><dd id="mTone">—</dd><div class="delta" id="dTone">viewed at arm's length</div></div>
        <div><dt>G-code</dt><dd id="mFile">—</dd><div class="delta" id="dPoints"></div></div>
      </dl>
    </div>
  </main>
</div>

<script>
const $ = id => document.getElementById(id);

/* Slider readouts. Each also paints its own filled track, which is the only
   way to get a progress fill in WebKit without an extra element. */
const FMT = {
  ink:v=>(+v).toFixed(2)+'×', angle:v=>v+'°',
  coherence:v=>Math.round(v*100)+'%', levels:v=>v,
  points:v=>(+v).toLocaleString('en-US').replace(/,/g,' '),
  depth:v=>'depth '+v, coarsest:v=>(+v).toFixed(1)+' mm',
  hatch_layers:v=>v, hatch_share:v=>(+v).toFixed(2),
  hilbert_smooth:v=>(+v).toFixed(2), hilbert_detail:v=>(+v).toFixed(1),
  feature_gain:v=>(+v).toFixed(2), outline_strength:v=>(+v).toFixed(2),
  hatch_coherence:v=>(+v).toFixed(2), form_weight:v=>(+v).toFixed(2),
  region_threshold:v=>(+v).toFixed(2), region_segments:v=>v,
  illustration_shadow:v=>(+v).toFixed(2), illustration_focus:v=>(+v).toFixed(2),
  illustration_tone:v=>(+v).toFixed(2),
  time_limit:v=>(+v===0?'off':v+' min'),
  spacing:v=>(+v===0?'auto':(+v).toFixed(1)+' mm'),
  pen:v=>(+v).toFixed(2)+' mm', contrast:v=>(+v).toFixed(1),
  detail:v=>(+v).toFixed(2), exposure:v=>(+v).toFixed(2), gamma:v=>(+v).toFixed(2),
  chroma:v=>(+v).toFixed(2), ink_response:v=>(+v).toFixed(2),
  resolution:v=>v+' px', merge:v=>(+v).toFixed(1)+' mm',
  bridge:v=>(+v===0?'off':(+v).toFixed(1)+' mm'),
  bridge_min_ink:v=>(+v).toFixed(2), max_coverage:v=>(+v).toFixed(2),
};
for (const [id, fmt] of Object.entries(FMT)) {
  const el = $(id), out = $(id + 'V');
  if (!el) continue;
  const sync = () => {
    if (out) out.textContent = fmt(el.value);
    const pct = (el.value - el.min) / (el.max - el.min) * 100;
    el.style.setProperty('--pct', pct + '%');
  };
  el.addEventListener('input', sync);
  sync();
}

const engineInputs = [...document.querySelectorAll('input[name=engine]')];
const engineValue = () => engineInputs.find(r => r.checked).value;

/* Only show the knobs the chosen engine actually reads. */
function syncEngine() {
  const name = engineValue();
  document.querySelectorAll('[data-for]').forEach(el => {
    el.style.display = el.dataset.for.split(',').includes(name) ? '' : 'none';
  });
}
engineInputs.forEach(r => r.addEventListener('change', syncEngine));
syncEngine();

function syncSpec() {
  const paper = $('paper').value.toUpperCase();
  const orient = $('portrait').checked ? 'portrait' : 'landscape';
  $('spec').textContent = `${paper} ${orient} · ${(+$('pen').value).toFixed(2)} mm pen`;
}
function syncWhen() {
  document.querySelectorAll('[data-when]').forEach(el => {
    const src = document.getElementById(el.dataset.when);
    el.style.display = src && src.checked ? '' : 'none';
  });
}
['portrait', 'outline', 'use_depth'].forEach(id =>
  $(id).addEventListener('change', syncWhen));
syncWhen();

['paper','portrait','pen'].forEach(id => {
  $(id).addEventListener('input', syncSpec);
  $(id).addEventListener('change', syncSpec);
});
syncSpec();

/* ── file handling ── */
let file = null;
const drop = $('drop');
drop.addEventListener('click', () => $('file').click());
$('file').addEventListener('change', e => e.target.files[0] && setFile(e.target.files[0]));
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
function setFile(f) {
  file = f;
  $('go').disabled = false;
  $('dl').style.display = 'none';
  const kb = f.size > 1048576
    ? (f.size / 1048576).toFixed(1) + ' MB' : Math.round(f.size / 1024) + ' kB';
  $('fname').innerHTML = f.name.replace(/[<>&]/g, '') + '<small>' + kb + '</small>';
  const r = new FileReader();
  r.onload = e => { $('thumb').src = e.target.result; drop.classList.add('loaded'); };
  r.readAsDataURL(f);
}

/* ── generate ── */
const FIELDS = ['ink','angle','coherence','levels','points','depth','coarsest',
  'spacing','hatch_layers','hatch_share','hilbert_smooth','hilbert_detail',
  'feature_gain','outline_strength','hatch_coherence','form_weight',
  'region_threshold','region_segments','line_method','line_style',
  'illustration_shading','illustration_shadow','illustration_focus','illustration_tone',
  'pen','paper','margin','contrast','detail','exposure','gamma','chroma',
  'ink_response','resolution','merge','bridge','bridge_min_ink','max_coverage',
  'feed','travel_feed','time_limit'];
const CHECKS = ['outline','portrait','rotate_out','invert','flip_x','flip_y',
  'remove_bg','show_travel','use_depth'];

async function submit(url, form, label) {
  $('go').disabled = true; $('drawSubject').disabled = true;
  $('dl').style.display = 'none';
  $('stage').classList.add('busy');
  $('msg').textContent = 'Working…';
  $('msg').className = 'note';
  FIELDS.forEach(k => form.append(k, $(k).value));
  CHECKS.forEach(k => form.append(k, $(k).checked ? '1' : '0'));
  try {
    const res = await fetch(url, { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) {
      $('msg').textContent = data.error;
      $('msg').className = 'note bad';
    } else {
      show(data, typeof label === 'function' ? label(data) : label);
      const st = data.stats;
      $('msg').textContent = st.resolution > st.asked_resolution
        ? `rendered in ${st.elapsed} · working image raised to ` +
          `${st.resolution} px to keep the ${$('pen').value} mm pen above a pixel`
        : `rendered in ${st.elapsed}`;
      $('msg').className = 'note';
      $('dl').style.display = 'block';
    }
  } catch (err) {
    $('msg').textContent = 'Request failed: ' + err.message;
    $('msg').className = 'note bad';
  }
  $('stage').classList.remove('busy');
  $('go').disabled = !file; $('drawSubject').disabled = false;
}

$('go').addEventListener('click', () => {
  if (!file) return;
  const form = new FormData();
  form.append('image', file);
  form.append('engine', engineValue());
  submit('/generate', form, engineValue());
});

$('drawSubject').addEventListener('click', () => {
  const form = new FormData();
  form.append('subject', $('subject').value);
  form.append('name', $('frogname').value);
  // Show the name it got, so the same frog can be drawn again by typing it.
  submit('/procedural', form, d => { $('frogname').placeholder = d.name; return d.name; });
});

function show(data, label) {
  const s = data.stats;
  const wrap = $('sheetwrap');
  wrap.innerHTML =
    '<div class="sheet"><img alt="plot preview" src="data:image/png;base64,' +
    data.preview + '"></div><div class="caption">' + s.size + '</div>';

  $('bigTime').textContent = s.time;
  $('engCap').textContent = (label || engineValue()) +
    (s.rotated ? ' · turned to landscape' : '');
  $('mLifts').textContent = s.lifts.toLocaleString('en-US').replace(/,/g, ' ');
  $('dLifts').textContent = 'from ' +
    s.lifts_before.toLocaleString('en-US').replace(/,/g, ' ') + ' strokes' +
    (s.budget_cut ? ' · ' + s.budget_cut + ' left out to fit' : '');
  $('mDraw').innerHTML = s.draw_m + '<span class="unit">m</span>';
  $('dDraw').textContent = 'pen down ' + s.draw_time;
  $('mTravel').innerHTML = s.travel_m + '<span class="unit">m</span>';
  $('dTravel').textContent = 'from ' + s.travel_raw_m + ' m unordered';
  $('mTone').textContent = s.likeness === null ? '—' : s.likeness.toFixed(2);
  $('dTone').textContent = s.tone_error === null ? "viewed at arm's length"
    : 'tone off by ' + s.tone_error.toFixed(2);
  $('mFile').innerHTML = s.kb + '<span class="unit">kB</span>';
  $('dPoints').textContent =
    s.points.toLocaleString('en-US').replace(/,/g, ' ') + ' points';
  $('readout').classList.add('on');
}

$('dl').addEventListener('click', () => { window.location.href = '/download'; });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("\n  Open http://localhost:8080 in your browser\n")
    app.run(host="127.0.0.1", port=8080, debug=False)
