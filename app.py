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

import numpy as np

import plotter
from plotter import Config, ENGINES, PAPER_SIZES, grbl, speedtest
from plotter.gcode import encode_png, format_duration

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024

_last: dict = {}
_machine = grbl.Grbl()


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
    cfg.render.hilbert_smooth = _f("hilbert_smooth", 0.5)
    cfg.render.subject_focus = _f("subject_focus", 0.8)
    cfg.render.hilbert_detail = _f("hilbert_detail", 1.2)
    cfg.render.hatch_coherence = _f("hatch_coherence", 0.45)
    cfg.render.region_segments = _i("region_segments", 400)
    cfg.render.region_threshold = _f("region_threshold", 0.25)
    cfg.render.coherence_gamma = _f("coherence_gamma", 2.0)
    cfg.render.form_weight = _f("form_weight", 0.8) if _b("use_depth") else 0.0
    cfg.render.line_method = request.form.get("line_method", "learned")
    cfg.render.line_style = request.form.get("line_style", "sketch")
    cfg.render.illustration_shading = request.form.get("illustration_shading", "flowlines")
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
    # Estimates use the plotter's own acceleration and corner tolerance once
    # it has been connected and read (machine.json).
    return grbl.apply_profile(cfg)


#: What each style is for and how long it tends to plot, in the words the
#: style picker shows.  The first three are the ones worth trying first.
ENGINE_UI = {
    "illustration": ("fast", "Pen-and-ink: outlines with shadow hatching. Light and airy."),
    "waves": ("medium", "One wavy line sweeps the page. Strong tone, graphic look."),
    "hilbert": ("medium", "One maze-like curve. Most faithful tone, subject on white."),
    "crosshatch": ("slow", "Layered engraving-style hatching."),
    "flowlines": ("slow", "Strokes follow fur, feathers and hair."),
    "adaptive": ("slow", "Cross-hatch on smooth areas, flow lines on texture."),
    "tsp": ("slow", "Stipple dots joined into one continuous line."),
    "contours": ("slow", "Tonal contour lines, like a map."),
    "outline": ("fastest", "Edges only, no shading."),
    "lineart": ("fastest", "Traces an existing drawing or scan."),
}
FEATURED = ("illustration", "waves", "hilbert")


@app.route("/")
def index():
    engines = []
    for name, meta in ENGINES.items():
        label = meta[1].partition("\u2014")[0].strip()
        speed, best = ENGINE_UI.get(name, ("", meta[1]))
        engines.append({"id": name, "label": label, "speed": speed, "best": best,
                        "featured": name in FEATURED})
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
    _last = {"gcode": result.gcode(header), "name": filename,
             "estimate_s": result.stats["total_s"], **_placement(result)}
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


def _placement(result) -> dict:
    """How machine coordinates map back onto the preview, and the plot's extent.

    The preview shows the page as composed; a portrait page goes down the
    wire turned 90 degrees.  The live pen marker needs the inverse of that.
    """
    page = result.config.page
    mode = (page.rotate_output or "auto").lower()
    if mode == "off" or (mode == "auto" and page.height_mm <= page.width_mm):
        turn = None
    else:
        turn = "cw" if mode == "cw" else "ccw"
    out, _ = result.output_paths()
    pts = np.vstack(out) if out else np.zeros((1, 2))
    return {"page_w": page.width_mm, "page_h": page.height_mm, "turn": turn,
            "negative_y": page.negative_y,
            "bbox": [float(pts[:, 0].min()), float(pts[:, 0].max()),
                     float(pts[:, 1].min()), float(pts[:, 1].max())]}


def _on_preview(x: float, y: float) -> tuple[float, float] | None:
    """Machine X/Y -> fraction across and down the preview image."""
    if not _last:
        return None
    w, h = _last["page_w"], _last["page_h"]
    if _last["turn"] == "ccw":
        u, v = w + y, x
    elif _last["turn"] == "cw":
        u, v = -y, h - x
    else:
        u, v = x, (-y if _last["negative_y"] else y)
    return u / w, v / h


# ── plotter over USB ─────────────────────────────────────────────────────

def _machine_call(fn):
    try:
        out = fn()
        return jsonify(ok=True, **(out or {}))
    except Exception as exc:
        return jsonify(ok=False, error=f"{exc}")


@app.route("/machine/ports")
def machine_ports():
    return jsonify(ports=grbl.ports(), connected=_machine.port if _machine.connected else None)


@app.route("/machine/connect", methods=["POST"])
def machine_connect():
    def go():
        info = _machine.connect(request.form["port"], int(request.form.get("baud", 115200)))
        # Keep what the speed test measured; the firmware cannot report it.
        grbl.save_profile({**(grbl.load_profile() or {}), **info})
        _machine.command(Config().machine.pen_up_cmd)
        return {"info": info}
    return _machine_call(go)


@app.route("/machine/disconnect", methods=["POST"])
def machine_disconnect():
    return _machine_call(_machine.disconnect)


@app.route("/machine/status")
def machine_status():
    st = _machine.progress()
    st["info"] = _machine.info() if _machine.connected else None
    st["marker"] = _on_preview(st["x"], st["y"]) if _machine.connected else None
    st["origin_corner"] = _origin_corner()
    job = st.get("job")
    if job and job["name"] == "speed test" and job.get("results"):
        st["speedtest"] = speedtest.summarise(job["results"])
    return jsonify(st)


@app.route("/machine/speedtest", methods=["POST"])
def machine_speedtest():
    """Draw the speed test sheet: five blocks, each faster than the last."""
    return _machine_call(lambda: _machine.run_test(speedtest.build(Config())))


@app.route("/machine/apply_speed", methods=["POST"])
def machine_apply_speed():
    """Keep the chosen speed in the firmware, and the measured lift cost."""
    def go():
        _machine.apply_speed(_f("accel", 600), _f("junction_deviation", 0.01))
        profile = {**(grbl.load_profile() or {}), **_machine.info()}
        job = _machine.job
        if job and job.get("results"):
            overhead = speedtest.summarise(job["results"])["lift_overhead_s"]
            if overhead is not None:
                profile["lift_overhead_s"] = round(max(0.0, overhead), 3)
        grbl.save_profile(profile)
        return {"info": _machine.info()}
    return _machine_call(go)


def _origin_corner() -> str:
    """Which corner of the drawing X0 Y0 is, as the paper lies on the bed.

    Unturned, it is the top-left.  A portrait page goes to the landscape bed
    turned a quarter, and the bed's origin corner is then a different corner
    of the drawing - set the origin at the top-left and the plot runs off
    the paper.
    """
    turn = _last.get("turn") if _last else None
    return {"ccw": "top-right", "cw": "bottom-left"}.get(turn, "top-left")


@app.route("/machine/jog", methods=["POST"])
def machine_jog():
    """Move the pen in the drawing's own directions (u right, v down).

    A portrait page goes down the wire turned 90 degrees, so "up" on the
    preview is a different machine axis; the arrows follow the preview.
    """
    du, dv = _f("du", 0), _f("dv", 0)
    turn = _last.get("turn") if _last else None
    neg = _last.get("negative_y", True) if _last else Config().page.negative_y
    if turn == "ccw":
        dx, dy = dv, du
    elif turn == "cw":
        dx, dy = -dv, -du
    else:
        dx, dy = du, (-dv if neg else dv)
    return _machine_call(lambda: _machine.jog(dx, dy))


@app.route("/machine/pen", methods=["POST"])
def machine_pen():
    m = Config().machine
    cmd = m.pen_down_cmd if request.form.get("down") == "1" else m.pen_up_cmd
    return _machine_call(lambda: _machine.command(cmd) and None)


@app.route("/machine/origin", methods=["POST"])
def machine_origin():
    return _machine_call(_machine.set_origin)


@app.route("/machine/home", methods=["POST"])
def machine_home():
    return _machine_call(lambda: _machine.go_origin(Config().machine.pen_up_cmd))


@app.route("/machine/unlock", methods=["POST"])
def machine_unlock():
    return _machine_call(_machine.unlock)


@app.route("/machine/frame", methods=["POST"])
def machine_frame():
    """Trace the drawing's outline with the pen up, to check it lands on the paper."""
    if not _last:
        return jsonify(ok=False, error="Generate a drawing first")
    x0, x1, y0, y1 = _last["bbox"]
    m = Config().machine
    prog = "\n".join([m.pen_up_cmd, "G90",
                      f"G0 X{x0:.2f} Y{y1:.2f}", f"G0 X{x1:.2f} Y{y1:.2f}",
                      f"G0 X{x1:.2f} Y{y0:.2f}", f"G0 X{x0:.2f} Y{y0:.2f}",
                      f"G0 X{x0:.2f} Y{y1:.2f}", "G0 X0 Y0"])
    return _machine_call(lambda: _machine.start(prog, "outline"))


@app.route("/machine/start", methods=["POST"])
def machine_start():
    if not _last:
        return jsonify(ok=False, error="Generate a drawing first")
    problem = _off_bed(_last["gcode"])
    if problem and request.form.get("force") != "1":
        return jsonify(ok=False, confirm=problem)
    return _machine_call(lambda: _machine.start(_last["gcode"], _last["name"],
                                                _last["estimate_s"]))


def _off_bed(program: str) -> str | None:
    """Would this program drive the pen past the bed, measured from the origin?

    The AX4 has no limit switches and no soft limits: nothing stops it from
    running into the frame, so the check has to happen here.
    """
    ext = grbl.extents(program.splitlines())
    travel = _machine.info()["travel"] if _machine.connected else [None, None]
    if ext is None or not all(travel):
        return None
    x0, x1, y0, y1 = ext
    tx, ty = travel
    ylo, yhi = (-ty, 0.0) if Config().page.negative_y else (0.0, ty)
    if x0 < -0.5 or x1 > tx + 0.5 or y0 < ylo - 0.5 or y1 > yhi + 0.5:
        return (f"The drawing spans X {x0:.0f}..{x1:.0f}, Y {y0:.0f}..{y1:.0f} mm "
                f"from the origin, but the bed is {tx:.0f} x {ty:.0f} mm. "
                f"Is the origin set at the paper's {_origin_corner()} corner, "
                "as the drawing appears in the preview?")
    return None


@app.route("/machine/pause", methods=["POST"])
def machine_pause():
    return _machine_call(_machine.pause)


@app.route("/machine/resume", methods=["POST"])
def machine_resume():
    return _machine_call(_machine.resume)


@app.route("/machine/stop", methods=["POST"])
def machine_stop():
    return _machine_call(lambda: _machine.stop(Config().machine.pen_up_cmd))


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
    height:calc(100vh - 92px);min-height:560px;display:flex;flex-direction:column;
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
    justify-content:center;gap:10px;padding:20px 24px;
    background:var(--surface);min-height:0;overflow:hidden;
  }
  .stage.busy .sheetwrap{opacity:.42;transition:opacity .2s}
  .sheet{
    background:#fff;box-shadow:var(--sheet-shadow);border-radius:2px;
    max-width:100%;line-height:0;
  }
  .sheet img{display:block;max-width:100%;border-radius:2px}
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
    display:flex;align-items:center;gap:11px;flex-wrap:wrap;
    padding:9px 20px;
  }
  .headline .big{font-size:21px;font-weight:600;letter-spacing:-.02em;
                 font-family:var(--mono)}
  .headline .cap{font-size:11px;color:var(--ink-3);letter-spacing:.04em;
                 text-transform:uppercase}
  .headline .rule{flex:1;height:1px;background:var(--rule);min-width:20px}
  .metrics{display:flex;gap:20px;flex-wrap:wrap}
  .metrics > div{cursor:default}
  .metrics dt{
    font-size:9px;letter-spacing:.1em;text-transform:uppercase;
    color:var(--ink-3);
  }
  .metrics dd{font-family:var(--mono);font-size:13px;font-weight:500}
  .metrics .delta{display:none}
  .metrics dd .unit{font-size:10px;color:var(--ink-3);margin-left:2px}
  .delta{font-size:10px;color:var(--ink-3);font-family:var(--mono);margin-top:2px}

  /* ── camera ──────────────────────────────────────────────── */
  .cam{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.55);
       display:flex;align-items:center;justify-content:center;padding:24px}
  .cam[hidden]{display:none}
  .cam-box{background:var(--panel);border-radius:var(--r);padding:14px;
           display:flex;flex-direction:column;gap:12px;max-width:min(980px,100%);
           box-shadow:var(--sheet-shadow)}
  .cam-view{position:relative;line-height:0;background:#000;border-radius:4px;overflow:hidden}
  .cam video{max-width:100%;max-height:70vh;transform:scaleX(-1)}
  .cam-count{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
             font:600 96px var(--mono);color:#fff;text-shadow:0 2px 14px rgba(0,0,0,.6);
             pointer-events:none}
  .cam-bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .cam-bar select{width:auto;max-width:280px}

  /* ── sidebar guidance ──────────────────────────────────────── */
  .subhead{font-size:10px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;
           color:var(--ink-3);margin:16px 0 10px;padding-top:13px;border-top:1px dashed var(--rule)}
  .subhead span{color:var(--ink)}
  details.more{margin-top:9px}
  details.more > summary{font-size:10.5px;letter-spacing:.06em}
  .finehead{padding:15px 17px 4px}
  .finehead h2{font-size:10px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;
               color:var(--ink-3)}
  .finehead p{font-size:10.5px;color:var(--ink-3);margin-top:4px;line-height:1.45}

  /* ── plotter ───────────────────────────────────────────────── */
  .machine{border-top:1px solid var(--rule);background:var(--panel);flex:none;
           padding:12px 20px 14px;display:flex;flex-direction:column;gap:10px}
  .mrow{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .mlabel{font-size:10px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;
          color:var(--ink-3);margin-right:4px}
  .machine select{width:auto;min-width:190px}
  .mbtn{font-family:var(--sans);font-size:12px;font-weight:600;padding:7px 11px;
        border-radius:5px;cursor:pointer;border:1px solid var(--rule-2);
        background:transparent;color:var(--ink);white-space:nowrap}
  .mbtn:hover:not(:disabled){border-color:var(--ink)}
  .mbtn:disabled{opacity:.3;cursor:not-allowed}
  .mbtn.solid{background:var(--ink);color:var(--panel);border-color:var(--ink)}
  .mbtn.solid:hover:not(:disabled){opacity:.84}
  .mstate{font-family:var(--mono);font-size:11px;color:var(--ink-2);
          display:flex;align-items:center;gap:6px}
  .mstate .dot{width:7px;height:7px;border-radius:50%;border:1.5px solid var(--ink-3)}
  .mstate.on .dot{background:var(--ink);border-color:var(--ink)}
  .mstate.busy .dot{animation:blink 1s steps(2) infinite}
  @keyframes blink{50%{opacity:.2}}
  .minfo{font-family:var(--mono);font-size:10px;color:var(--ink-3)}
  .jog{display:grid;grid-template-columns:repeat(3,30px);grid-template-rows:repeat(3,30px);gap:3px}
  .jog .mbtn{padding:0;font-size:13px}
  .seg{display:inline-flex;border:1px solid var(--rule-2);border-radius:5px;overflow:hidden}
  .seg button{flex:1;font-family:var(--mono);font-size:11px;padding:6px 9px;border:none;
              background:transparent;color:var(--ink-2);cursor:pointer}
  .seg button.on{background:var(--ink);color:var(--panel)}
  .mgroup{display:flex;flex-direction:column;gap:6px}
  .spacer{flex:1}
  .mprog{height:3px;background:var(--rule);border-radius:2px;overflow:hidden}
  .mprog > i{display:block;height:100%;width:0;background:var(--ink);transition:width .3s}
  .mnote{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);min-height:1.3em}
  .mnote.bad{color:var(--ink)}
  .mnote.bad::before{content:"⚠ "}
  .sheet{position:relative}
  .pen-marker{position:absolute;width:13px;height:13px;margin:-6.5px 0 0 -6.5px;
              border:2px solid #d0021b;border-radius:50%;pointer-events:none;display:none;
              box-shadow:0 0 0 2px rgba(255,255,255,.8)}
  .pen-marker::after{content:"";position:absolute;left:50%;top:50%;width:3px;height:3px;
              margin:-1.5px 0 0 -1.5px;background:#d0021b;border-radius:50%}

  @media (max-width:940px){
    .shell{grid-template-columns:1fr}
    .panel{position:static;max-height:none}
    .stage{min-height:0;height:auto}
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
        <button class="btn ghost" id="camBtn" style="margin-top:8px">Take a photo</button>
        <div class="or">or draw one from scratch</div>
        <div class="pair">
          <select id="subject"><option value="frog" selected>Frog (any family)</option><option value="truefrog">True frog</option><option value="toad">Toad</option><option value="treefrog">Tree frog</option><option value="dartfrog">Poison dart frog</option><option value="hornedfrog">Horned frog</option><option value="clawedfrog">Clawed frog</option></select>
          <input type="text" id="frogname" placeholder="name (blank = invent one)">
        </div>
        <button class="btn ghost" id="drawSubject" style="margin-top:8px">Draw a new one</button>
      </section>

      <section>
        <div class="sec-head"><h2>Style</h2><span class="aside">plot time</span></div>
        <div class="engines">
          {% for e in engines if e.featured %}
          <label class="eng">
            <input type="radio" name="engine" value="{{ e.id }}"
                   {% if e.id == 'illustration' %}checked{% endif %}>
            <span class="box">
              <span class="line1">
                <span class="nm">{{ e.label }}</span>
                <span class="lift">{{ e.speed }}</span>
              </span>
              <span class="bl">{{ e.best }}</span>
            </span>
          </label>
          {% endfor %}
        </div>
        <details class="more" id="moreEngines">
          <summary>More styles</summary>
          <div class="engines" style="margin-top:9px">
          {% for e in engines if not e.featured %}
          <label class="eng">
            <input type="radio" name="engine" value="{{ e.id }}"
                   {% if e.id == 'illustration' %}checked{% endif %}>
            <span class="box">
              <span class="line1">
                <span class="nm">{{ e.label }}</span>
                <span class="lift">{{ e.speed }}</span>
              </span>
              <span class="bl">{{ e.best }}</span>
            </span>
          </label>
          {% endfor %}
          </div>
        </details>

        <div class="subhead">Settings for <span id="engName">Illustration</span></div>
        <div class="ctl" data-for="crosshatch,flowlines,adaptive,waves,hilbert,tsp,contours">
          <label><span>Subject focus
            <span class="tip" title="Uses depth to lighten the background so the drawing - and the plot time - goes to the subject. Hilbert fades the background out completely. Turn down for landscapes or to keep a full-frame texture.">i</span></span>
            <span class="v" id="subject_focusV">0.80</span></label>
          <input type="range" id="subject_focus" min="0" max="1" step="0.05" value="0.8">
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
            <option value="crosshatch">Cross-hatch</option>
            <option value="flowlines" selected>Flow lines</option>
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
            <span class="v" id="hilbert_smoothV">0.50</span>
          </label>
          <input type="range" id="hilbert_smooth" min="0" max="0.5" step="0.05" value="0.5">
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
        <div class="sec-head"><h2>Paper, pen &amp; time</h2></div>
        <div class="ctl">
          <label><span>Time limit
            <span class="tip" title="Longest the plot may take. The strokes that add least to the picture are dropped first - short faint lines and hatching in the lightest shadows - until the estimate fits. Off draws everything.">i</span></span>
            <span class="v" id="time_limitV">off</span></label>
          <input type="range" id="time_limit" min="0" max="60" step="1" value="0">
        </div>
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

      <div class="finehead">
        <h2>Fine-tuning</h2>
        <p>The defaults are tuned on test photos. Open these only to change a specific look.</p>
      </div>

      <section>
        <details>
          <summary>Form &amp; line</summary>
          <div class="body">
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
          </div>
        </details>
      </section>

      <section>
        <details>
          <summary>Ink</summary>
          <div class="body">
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
          </div>
        </details>
      </section>

      <section>
        <details>
          <summary>Tone</summary>
          <div class="body">
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
          </div>
        </details>
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
        <span class="cap">plot time</span>
        <span class="cap" id="engCap"></span>
        <span class="rule"></span>
      <dl class="metrics">
        <div><dt>Pen lifts</dt><dd id="mLifts">—</dd><div class="delta" id="dLifts"></div></div>
        <div><dt>Drawn</dt><dd id="mDraw">—</dd><div class="delta" id="dDraw"></div></div>
        <div><dt>Travel</dt><dd id="mTravel">—</dd><div class="delta" id="dTravel"></div></div>
        <div><dt title="How closely the finished plot matches the photo's tone when seen from arm's length (SSIM, 1 = identical). Compare it between settings of one engine: illustration leaves light tones to the paper on purpose, so it scores lower than the dense engines even where it looks better.">Likeness</dt><dd id="mTone">—</dd><div class="delta" id="dTone">viewed at arm's length</div></div>
        <div><dt>G-code</dt><dd id="mFile">—</dd><div class="delta" id="dPoints"></div></div>
      </dl>
      </div>
    </div>

    <div class="machine" id="machine">
      <div class="mrow">
        <span class="mlabel">Plotter</span>
        <select id="mport" aria-label="Serial port"><option value="">no port found</option></select>
        <button class="mbtn" id="mrefresh" title="Look for ports again">↻</button>
        <button class="mbtn solid" id="mconnect">Connect</button>
        <span class="mstate" id="mstate"><span class="dot"></span><span id="mstateText">not connected</span></span>
        <span class="spacer"></span>
        <span class="minfo" id="minfo"></span>
      </div>
      <div class="mrow" id="mcontrols" style="display:none;align-items:flex-start;gap:18px">
        <div class="jog" aria-label="Move the pen">
          <span></span><button class="mbtn" data-jog="0,-1" title="Toward the top of the drawing">↑</button><span></span>
          <button class="mbtn" data-jog="-1,0" title="Left">←</button>
          <button class="mbtn" id="mhome" title="Go to origin (pen up)">⌂</button>
          <button class="mbtn" data-jog="1,0" title="Right">→</button>
          <span></span><button class="mbtn" data-jog="0,1" title="Toward the bottom of the drawing">↓</button><span></span>
        </div>
        <div class="mgroup">
          <div class="mrow" style="gap:6px"><div class="seg" id="mstep" title="How far each arrow press moves the pen"><button data-step="1">1</button><button data-step="10" class="on">10</button><button data-step="50">50</button></div><span class="minfo">mm per step</span></div>
          <div class="mrow" style="gap:6px">
            <button class="mbtn" id="mpenDown">Pen down</button>
            <button class="mbtn" id="mpenUp">Pen up</button>
          </div>
          <button class="mbtn" id="morigin" title="Put the pen on the paper's top-left corner first">Set origin here</button>
        </div>
        <div class="mgroup">
          <button class="mbtn" id="mframe" title="Moves around the drawing's outline with the pen up, so you can check it lands on the paper">Trace outline</button>
          <button class="mbtn" id="mspeed" title="Draws five test blocks along the top of a landscape sheet, each faster than the last, to find how fast this machine can safely go">Speed test</button>
          <button class="mbtn" id="munlock" style="display:none">Unlock alarm</button>
        </div>
        <span class="spacer"></span>
        <div class="mgroup" style="min-width:230px">
          <div class="mrow" style="gap:6px;justify-content:flex-end">
            <button class="mbtn solid" id="mstart">Plot</button>
            <button class="mbtn" id="mpause" disabled>Pause</button>
            <button class="mbtn" id="mstop" disabled>Stop</button>
          </div>
          <div class="mprog"><i id="mbar"></i></div>
          <div class="mnote" id="mnote">Jog the pen to the paper's top-left corner, then Set origin.</div>
        </div>
      </div>
      <div class="mrow" id="mtest" style="display:none;gap:6px">
        <span class="minfo" style="flex-basis:100%;margin-bottom:2px">Speed test done. In each block the second square should land exactly on the first; where it doesn't, the motors skipped. Keep the fastest clean one:</span>
        <span id="mtestLevels" class="mrow" style="gap:6px"></span>
      </div>
    </div>
  </main>
</div>

<div class="cam" id="cam" hidden>
  <div class="cam-box" role="dialog" aria-label="Take a photo">
    <div class="cam-view">
      <video id="camVideo" autoplay playsinline muted></video>
      <div class="cam-count" id="camCount"></div>
    </div>
    <div class="cam-bar">
      <select id="camSelect" aria-label="Camera"></select>
      <span class="spacer"></span>
      <button class="mbtn" id="camCancel">Cancel</button>
      <button class="mbtn" id="camTimer">3 s timer</button>
      <button class="mbtn solid" id="camShoot">Take photo</button>
    </div>
  </div>
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
  time_limit:v=>(+v===0?'off':v+' min'), subject_focus:v=>(+v).toFixed(2),
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
  const card = document.querySelector(`input[name=engine][value="${name}"]`);
  $('engName').textContent = card.closest('.eng').querySelector('.nm').textContent;
  if (card.closest('#moreEngines')) $('moreEngines').open = true;
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
  'feed','travel_feed','time_limit','subject_focus'];
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
    data.preview + '"><div class="pen-marker" id="penMarker"></div></div>' +
    '<div class="caption">' + s.size + '</div>';

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
  // The detail under each figure lives in its tooltip, to keep the line slim.
  document.querySelectorAll('.metrics > div').forEach(d => {
    const delta = d.querySelector('.delta');
    if (delta) d.title = delta.textContent;
  });
  $('readout').classList.add('on');
  const img = wrap.querySelector('.sheet img');
  img.addEventListener('load', fitPreview);
  fitPreview();
}

/* The preview takes whatever height the readout and plotter panel leave, so
   the drawing and the controls are on screen together without scrolling. */
function fitPreview() {
  const wrap = $('sheetwrap'), img = wrap.querySelector('.sheet img');
  if (!img) return;
  if (window.innerWidth <= 940) { img.style.maxHeight = ''; return; }
  const cap = wrap.querySelector('.caption');
  const room = wrap.clientHeight - 40 - (cap ? cap.offsetHeight + 10 : 0);
  img.style.maxHeight = Math.max(160, room) + 'px';
}
window.addEventListener('resize', fitPreview);

$('dl').addEventListener('click', () => { window.location.href = '/download'; });

/* ── camera: a photo taken here is used exactly like an uploaded one ── */
let camStream = null, camTimer = null;
function camStop() {
  clearInterval(camTimer); camTimer = null; $('camCount').textContent = '';
  if (camStream) camStream.getTracks().forEach(t => t.stop());
  camStream = null;
}
async function camStart(deviceId) {
  camStop();
  const size = {width: {ideal: 1920}, height: {ideal: 1440}};
  camStream = await navigator.mediaDevices.getUserMedia({
    audio: false, video: deviceId ? {deviceId: {exact: deviceId}, ...size} : size});
  $('camVideo').srcObject = camStream;
  // Labels are only readable once permission has been given.
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const current = camStream.getVideoTracks()[0].getSettings().deviceId;
  $('camSelect').innerHTML = '';
  cams.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = d.deviceId; o.textContent = d.label || `Camera ${i + 1}`;
    o.selected = d.deviceId === current;
    $('camSelect').appendChild(o);
  });
  $('camSelect').style.display = cams.length > 1 ? '' : 'none';
}
function camClose() { camStop(); $('cam').hidden = true; }
function camShoot() {
  const v = $('camVideo');
  if (!v.videoWidth) return;
  const c = document.createElement('canvas');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);     // the preview is mirrored, the photo is not
  c.toBlob(b => {
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    setFile(new File([b], `photo-${stamp}.jpg`, {type: 'image/jpeg'}));
    camClose();
  }, 'image/jpeg', 0.92);
}
$('camBtn').addEventListener('click', async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $('msg').textContent = 'This browser cannot use the camera here.';
    $('msg').className = 'note bad'; return;
  }
  $('cam').hidden = false;
  try { await camStart(); }
  catch (err) {
    camClose();
    $('msg').textContent = 'Camera not available: ' + (err.name === 'NotAllowedError'
      ? 'allow camera access for this page in the browser' : err.message);
    $('msg').className = 'note bad';
  }
});
$('camSelect').addEventListener('change', e => camStart(e.target.value).catch(() => {}));
$('camCancel').addEventListener('click', camClose);
$('camShoot').addEventListener('click', camShoot);
$('camTimer').addEventListener('click', () => {
  if (camTimer) return;
  let n = 3;
  $('camCount').textContent = n;
  camTimer = setInterval(() => {
    n -= 1;
    if (n > 0) { $('camCount').textContent = n; return; }
    clearInterval(camTimer); camTimer = null; $('camCount').textContent = '';
    camShoot();
  }, 1000);
});
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('cam').hidden) camClose(); });

/* ── plotter over USB ── */
const M = {connected:false, step:10, timer:null, running:false};
async function mpost(url, data) {
  const form = new FormData();
  Object.entries(data || {}).forEach(([k, v]) => form.append(k, v));
  const res = await fetch(url, {method:'POST', body:form});
  return res.json();
}
function mnote(text, bad) {
  $('mnote').textContent = text || '';
  $('mnote').className = 'mnote' + (bad ? ' bad' : '');
}
async function mports() {
  const d = await (await fetch('/machine/ports')).json();
  const sel = $('mport');
  sel.innerHTML = d.ports.length ? '' : '<option value="">no port found</option>';
  d.ports.forEach(p => {
    const o = document.createElement('option');
    o.value = p; o.textContent = p.replace('/dev/', '');
    if (/usbserial|usbmodem|wchusb|SLAB/.test(p)) o.selected = true;
    sel.appendChild(o);
  });
  if (d.connected) { sel.value = d.connected; setConnected(true); }
}
function setConnected(on) {
  M.connected = on;
  $('mcontrols').style.display = on ? '' : 'none';
  $('mconnect').textContent = on ? 'Disconnect' : 'Connect';
  $('mconnect').classList.toggle('solid', !on);
  $('mport').disabled = on;
  clearInterval(M.timer);
  if (on) { M.timer = setInterval(mpoll, 400); mpoll(); }
  else {
    $('mstateText').textContent = 'not connected';
    $('mstate').className = 'mstate';
    $('minfo').textContent = '';
    const mk = $('penMarker'); if (mk) mk.style.display = 'none';
  }
}
const fmtS = s => s == null ? '—' : (s >= 3600 ? Math.floor(s/3600) + 'h ' : '') +
  Math.floor(s % 3600 / 60) + 'm ' + String(Math.floor(s % 60)).padStart(2, '0') + 's';
async function mpoll() {
  let st;
  try { st = await (await fetch('/machine/status')).json(); } catch (e) { return; }
  if (!st.connected) { setConnected(false); mnote(st.error || 'Connection lost', true); return; }
  $('mstateText').textContent = `${st.state} · X ${st.x.toFixed(1)}  Y ${st.y.toFixed(1)}`;
  const job = st.job, running = job && (job.state === 'running' || job.state === 'paused');
  M.running = running;
  M.corner = st.origin_corner;
  $('morigin').title = `Put the pen on the paper's ${st.origin_corner} corner (as the drawing shows in the preview) first`;
  $('mstate').className = 'mstate on' + (st.state === 'Run' ? ' busy' : '');
  if (st.info) {
    const i = st.info;
    $('minfo').textContent = `${i.version} · bed ${i.travel[0]}×${i.travel[1]} mm · ` +
      `${i.accel[0]} mm/s² · max ${i.max_rate[0]} mm/min`;
  }
  $('munlock').style.display = st.state === 'Alarm' ? '' : 'none';
  ['mstart','morigin','mframe','mspeed','mpenDown','mpenUp','mhome'].forEach(id => $(id).disabled = running);
  document.querySelectorAll('[data-jog]').forEach(b => b.disabled = running);
  $('mpause').disabled = !running;
  $('mstop').disabled = !running;
  $('mpause').textContent = job && job.state === 'paused' ? 'Resume' : 'Pause';
  if (job) {
    $('mbar').style.width = (job.fraction * 100).toFixed(1) + '%';
    const what = job.name === 'outline' ? 'Outline'
      : job.name === 'speed test' ? 'Speed test' + (job.block ? ` (${job.block})` : '') : 'Plot';
    if (job.name === 'speed test' && job.state === 'done' && st.speedtest) showSpeedResults(st.speedtest);
    if (job.name === 'speed test' && job.state === 'running') $('mtestLevels').dataset.shown = '';
    if (job.errors.length) mnote(`${what} held: ${job.errors[job.errors.length-1]}. Resume or Stop.`, true);
    else if (job.state === 'running' || job.state === 'paused')
      mnote(`${what} ${job.state} · ${job.done}/${job.total} lines · ` +
            `${fmtS(job.elapsed_s)} elapsed` + (job.remaining_s != null ? ` · about ${fmtS(job.remaining_s)} left` : ''));
    else if (job.state === 'done') mnote(`${what} finished in ${fmtS(job.elapsed_s)}.`);
    else if (job.state === 'stopped') mnote(`${what} stopped at line ${job.done} of ${job.total}. Pen lifted.`);
  }
  const mk = $('penMarker');
  if (mk && st.marker) {
    const [fx, fy] = st.marker;
    const inside = fx > -0.05 && fx < 1.05 && fy > -0.05 && fy < 1.05;
    mk.style.display = inside ? 'block' : 'none';
    mk.style.left = (fx * 100) + '%'; mk.style.top = (fy * 100) + '%';
  }
}
async function mdo(url, data, busyText) {
  if (busyText) mnote(busyText);
  const d = await mpost(url, data);
  if (!d.ok && d.error) mnote(d.error, true);
  mpoll();
  return d;
}
$('mrefresh').addEventListener('click', mports);
$('mconnect').addEventListener('click', async () => {
  if (M.connected) { await mpost('/machine/disconnect'); setConnected(false); mnote(''); return; }
  const port = $('mport').value;
  if (!port) { mnote('Plug in the plotter, quit UGS, then ↻', true); return; }
  $('mconnect').disabled = true;
  $('mstateText').textContent = 'connecting — the board restarts…';
  const d = await mpost('/machine/connect', {port});
  $('mconnect').disabled = false;
  if (!d.ok) { $('mstateText').textContent = 'not connected';
               mnote(d.error + ' — is UGS still open?', true); return; }
  setConnected(true);
  const st = await (await fetch('/machine/status')).json();
  mnote('Connected. Plot-time estimates now use this machine\'s settings. ' +
        `Jog the pen to the paper's ${st.origin_corner} corner (as the drawing shows in the preview), then Set origin.`);
});
document.querySelectorAll('#mstep button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#mstep button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); M.step = +b.dataset.step;
}));
document.querySelectorAll('[data-jog]').forEach(b => b.addEventListener('click', () => {
  const [du, dv] = b.dataset.jog.split(',').map(Number);
  mdo('/machine/jog', {du: du * M.step, dv: dv * M.step});
}));
$('mpenDown').addEventListener('click', () => mdo('/machine/pen', {down: 1}));
$('mpenUp').addEventListener('click', () => mdo('/machine/pen', {down: 0}));
$('morigin').addEventListener('click', async () => {
  const d = await mdo('/machine/origin');
  if (d.ok) mnote('Origin set. Trace outline to check the drawing lands on the paper.');
});
$('mhome').addEventListener('click', () => mdo('/machine/home'));
$('munlock').addEventListener('click', () => mdo('/machine/unlock'));
$('mframe').addEventListener('click', () => mdo('/machine/frame', {}, 'Tracing the outline, pen up…'));
$('mspeed').addEventListener('click', async () => {
  if (!confirm('Speed test: draws five 42 mm blocks along the top of the sheet, each faster than ' +
               'the last (about 8 minutes).\n\nUse a landscape sheet with the origin set at its ' +
               'top-left corner. Your speed settings are restored afterwards.')) return;
  $('mtest').style.display = 'none';
  mdo('/machine/speedtest', {}, 'Starting the speed test…');
});
function showSpeedResults(sum) {
  const box = $('mtestLevels');
  if (box.dataset.shown === '1') return;
  box.innerHTML = '';
  sum.levels.forEach(lv => {
    const b = document.createElement('button');
    b.className = 'mbtn';
    b.textContent = `${lv.accel} mm/s² · ${lv.junction_deviation} mm — ${fmtS(lv.actual_s)}`;
    b.title = `estimated ${fmtS(lv.estimate_s)} without the per-lift cost`;
    b.addEventListener('click', async () => {
      if (!confirm(`Keep ${lv.accel} mm/s² and ${lv.junction_deviation} mm in the plotter?`)) return;
      const d = await mdo('/machine/apply_speed', {accel: lv.accel, junction_deviation: lv.junction_deviation});
      if (d.ok) mnote(`Saved: ${lv.accel} mm/s², ${lv.junction_deviation} mm. Generate again to see the new plot times.`);
    });
    box.appendChild(b);
  });
  box.dataset.shown = '1';
  $('mtest').style.display = '';
}
$('mpause').addEventListener('click', () =>
  mdo($('mpause').textContent === 'Resume' ? '/machine/resume' : '/machine/pause'));
$('mstop').addEventListener('click', () => mdo('/machine/stop', {}, 'Stopping…'));
$('mstart').addEventListener('click', async () => {
  const t = $('bigTime').textContent;
  if (!confirm(`Plot the current drawing (about ${t})?\n\nThe origin should be set on the paper's ${M.corner || 'top-left'} corner, as the drawing shows in the preview.`)) return;
  let d = await mpost('/machine/start', {});
  if (!d.ok && d.confirm) {
    if (!confirm(d.confirm + '\n\nPlot anyway?')) return;
    d = await mpost('/machine/start', {force: 1});
  }
  if (!d.ok && d.error) mnote(d.error, true);
  mpoll();
});
mports();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("\n  Open http://localhost:8080 in your browser\n")
    app.run(host="127.0.0.1", port=8080, debug=False)
