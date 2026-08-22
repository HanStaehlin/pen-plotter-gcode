#!/usr/bin/env python3
"""Local web UI for the image-to-G-code converter.

    python3 app.py      then open http://localhost:8080
"""

from __future__ import annotations

import base64
import io
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
    cfg.render.contour_levels = _i("levels", 12)
    cfg.render.tsp_points = _i("points", 12000)
    cfg.render.hilbert_max_depth = _i("depth", 9)

    cfg.paths.merge_tolerance_mm = _f("merge", 1.0)
    cfg.paths.bridge_max_mm = _f("bridge", 3.0)
    cfg.paths.bridge_min_ink = _f("bridge_min_ink", 0.30)
    return cfg


@app.route("/")
def index():
    engines = [(name, meta[1], meta[2]) for name, meta in ENGINES.items()]
    return render_template_string(HTML, engines=engines,
                                  papers=sorted(PAPER_SIZES))


@app.route("/generate", methods=["POST"])
def generate():
    global _last
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

    preview = result.preview(dpi=150, show_travel=_b("show_travel"))
    _last = {
        "gcode": result.gcode(f"engine {cfg.render.engine}, "
                              f"pen {cfg.pen.width_mm} mm"),
        "name": (upload.filename.rsplit(".", 1)[0] or "plot") + ".gcode",
    }

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
            "elapsed": f"{s['elapsed_s']:.1f}s",
        },
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
<title>Image → G-code</title>
<style>
  :root {
    --bg:#14161f; --card:#1b1f2e; --sunk:#131622; --line:#2b3145;
    --text:#e8eaf2; --muted:#8b93ab; --accent:#f2545b; --ok:#4ec9a5;
    --radius:10px;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:var(--bg);color:var(--text);min-height:100vh;font-size:14px}
  header{padding:16px 28px;border-bottom:1px solid var(--line);
         display:flex;align-items:baseline;gap:14px}
  header h1{font-size:1.15rem;font-weight:600}
  header span{color:var(--muted);font-size:.85rem}
  .wrap{display:grid;grid-template-columns:360px 1fr;gap:20px;
        padding:20px 28px;max-width:1600px;margin:0 auto}
  .panel{background:var(--card);border-radius:var(--radius);padding:18px;
         display:flex;flex-direction:column;gap:14px;height:fit-content;
         position:sticky;top:20px;max-height:calc(100vh - 40px);overflow-y:auto}
  fieldset{border:none;display:flex;flex-direction:column;gap:11px}
  legend{font-size:.7rem;font-weight:700;color:var(--muted);
         text-transform:uppercase;letter-spacing:.09em;padding-bottom:8px}
  .drop{border:2px dashed var(--line);border-radius:var(--radius);
        padding:22px 14px;text-align:center;cursor:pointer;transition:.15s}
  .drop:hover,.drop.over{border-color:var(--accent);background:rgba(242,84,91,.06)}
  .drop input{display:none}
  .drop .hint{color:var(--muted);font-size:.8rem;margin-top:4px}
  .drop img{max-width:100%;max-height:110px;border-radius:6px;margin-top:10px;
            object-fit:contain}
  .drop .fname{margin-top:6px;font-size:.78rem;color:var(--accent);
               word-break:break-all}
  .ctl{display:flex;flex-direction:column;gap:5px}
  .ctl label{font-size:.78rem;color:var(--muted);display:flex;
             justify-content:space-between;gap:8px}
  .ctl label b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums}
  select,input[type=number]{background:var(--sunk);color:var(--text);
    border:1px solid var(--line);border-radius:6px;padding:7px 9px;
    font-size:.85rem;width:100%;font-family:inherit}
  input[type=range]{-webkit-appearance:none;appearance:none;height:5px;
    border-radius:3px;background:var(--sunk);outline:none;width:100%}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:15px;
    height:15px;border-radius:50%;background:var(--accent);cursor:pointer}
  .engine-note{font-size:.75rem;color:var(--muted);line-height:1.45;
    background:var(--sunk);border-radius:6px;padding:8px 10px}
  .engine-note b{color:var(--ok)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .check{display:flex;align-items:center;gap:8px;font-size:.82rem;cursor:pointer}
  .check input{accent-color:var(--accent);width:15px;height:15px}
  details>summary{cursor:pointer;font-size:.7rem;font-weight:700;
    color:var(--muted);text-transform:uppercase;letter-spacing:.09em;
    list-style:none;padding:6px 0}
  details>summary::-webkit-details-marker{display:none}
  details>summary::before{content:"▸ ";color:var(--accent)}
  details[open]>summary::before{content:"▾ "}
  details>div{display:flex;flex-direction:column;gap:11px;padding-top:6px}
  .btn{background:var(--accent);color:#fff;border:none;border-radius:8px;
       padding:12px;font-size:.95rem;font-weight:600;cursor:pointer;width:100%}
  .btn:disabled{opacity:.4;cursor:not-allowed}
  .btn.alt{background:var(--sunk);border:1px solid var(--line)}
  .stage{background:var(--card);border-radius:var(--radius);padding:20px;
         display:flex;flex-direction:column;align-items:center;gap:16px;
         min-height:520px;justify-content:center}
  .stage img{max-width:100%;max-height:calc(100vh - 260px);border-radius:6px;
             box-shadow:0 6px 30px rgba(0,0,0,.45);background:#fff}
  .empty{color:var(--muted);text-align:center;line-height:1.7}
  .stats{display:none;width:100%;background:var(--sunk);border-radius:8px;
         padding:14px 16px}
  .stats .big{font-size:1.7rem;font-weight:700;color:var(--accent);
              display:flex;align-items:baseline;gap:9px}
  .stats .big small{font-size:.78rem;color:var(--muted);font-weight:400}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));
        gap:8px;margin-top:12px}
  .stat{background:rgba(255,255,255,.035);border-radius:6px;padding:8px 10px;
        font-size:.7rem;color:var(--muted)}
  .stat b{display:block;color:var(--text);font-size:.92rem;margin-bottom:2px;
          font-variant-numeric:tabular-nums}
  .msg{font-size:.8rem;color:var(--muted);min-height:1.2em;text-align:center}
  .msg.err{color:var(--accent)}
  .spin{display:none;width:30px;height:30px;border:3px solid var(--line);
        border-top-color:var(--accent);border-radius:50%;
        animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  @media (max-width:900px){.wrap{grid-template-columns:1fr}.panel{position:static;max-height:none}}
</style>
</head>
<body>
<header>
  <h1>Image → G-code</h1>
  <span>pen plotter converter</span>
</header>

<div class="wrap">
  <div class="panel">
    <div class="drop" id="drop">
      <div style="font-size:1.6rem">🖼️</div>
      <div>Drop an image or <b>click to browse</b></div>
      <div class="hint">PNG · JPEG · TIFF · BMP</div>
      <div class="fname" id="fname"></div>
      <img id="thumb" style="display:none">
      <input type="file" id="file" accept="image/*">
    </div>

    <fieldset>
      <legend>Drawing</legend>
      <div class="ctl">
        <label>Engine</label>
        <select id="engine">
          {% for name, desc, cont in engines %}
          <option value="{{name}}"{% if name=='flowlines' %} selected{% endif %}>{{desc}}</option>
          {% endfor %}
        </select>
      </div>
      <div class="engine-note" id="engineNote"></div>
      <div class="ctl">
        <label>Ink <b id="inkV">1.00×</b></label>
        <input type="range" id="ink" min="0.3" max="2" step="0.05" value="1">
      </div>
      <div class="ctl">
        <label>Stroke angle in flat areas <b id="angleV">45°</b></label>
        <input type="range" id="angle" min="0" max="180" step="5" value="45">
      </div>
      <div class="ctl" data-for="flowlines">
        <label>Follow image structure <b id="coherenceV">100%</b></label>
        <input type="range" id="coherence" min="0" max="1" step="0.05" value="1">
      </div>
      <div class="ctl" data-for="contours">
        <label>Tonal levels <b id="levelsV">12</b></label>
        <input type="range" id="levels" min="3" max="40" step="1" value="12">
      </div>
      <div class="ctl" data-for="tsp">
        <label>Stipple points <b id="pointsV">12000</b></label>
        <input type="range" id="points" min="1000" max="40000" step="500" value="12000">
      </div>
      <div class="ctl" data-for="hilbert">
        <label>Max depth <b id="depthV">9</b></label>
        <input type="range" id="depth" min="5" max="10" step="1" value="9">
      </div>
      <div class="ctl" data-for="waves,contours">
        <label>Line spacing <b id="spacingV">auto</b></label>
        <input type="range" id="spacing" min="0" max="6" step="0.1" value="0">
      </div>
      <label class="check"><input type="checkbox" id="outline"> Add traced outlines</label>
    </fieldset>

    <fieldset>
      <legend>Pen &amp; page</legend>
      <div class="ctl">
        <label>Pen width <b id="penV">0.40 mm</b>
          <span title="The real ink trace width. Tone is computed from this, so getting it right is what makes greys come out grey.">ⓘ</span>
        </label>
        <input type="range" id="pen" min="0.1" max="1.5" step="0.05" value="0.4">
      </div>
      <div class="row">
        <div class="ctl">
          <label>Paper</label>
          <select id="paper">
            {% for p in papers %}
            <option value="{{p}}"{% if p=='a4' %} selected{% endif %}>{{p|upper}}</option>
            {% endfor %}
          </select>
        </div>
        <div class="ctl">
          <label>Margin (mm)</label>
          <input type="number" id="margin" value="10" min="0" max="60" step="1">
        </div>
      </div>
      <label class="check"><input type="checkbox" id="portrait"> Portrait</label>
    </fieldset>

    <fieldset>
      <legend>Tone</legend>
      <div class="ctl">
        <label>Local contrast <b id="contrastV">2.0</b></label>
        <input type="range" id="contrast" min="0" max="6" step="0.1" value="2">
      </div>
      <div class="ctl">
        <label>Detail <b id="detailV">0.45</b></label>
        <input type="range" id="detail" min="0" max="1" step="0.05" value="0.45">
      </div>
      <div class="ctl">
        <label>Exposure <b id="exposureV">0.00</b></label>
        <input type="range" id="exposure" min="-0.4" max="0.4" step="0.02" value="0">
      </div>
      <div class="ctl">
        <label>Gamma <b id="gammaV">1.00</b></label>
        <input type="range" id="gamma" min="0.4" max="2.2" step="0.05" value="1">
      </div>
      <div class="ctl">
        <label>Colour separation <b id="chromaV">0.18</b>
          <span title="How much saturated colour darkens relative to a neutral of the same lightness. Zero collapses a red and a grey of equal lightness onto the same tone.">ⓘ</span>
        </label>
        <input type="range" id="chroma" min="0" max="0.6" step="0.02" value="0.18">
      </div>
      <div class="ctl">
        <label>Tone response <b id="ink_responseV">0.25</b>
          <span title="0 spreads the pen's range evenly across perceived lightness. 1 matches the photograph's real luminance — accurate, much darker, much slower.">ⓘ</span>
        </label>
        <input type="range" id="ink_response" min="0" max="1" step="0.05" value="0.25">
      </div>
      <div class="ctl">
        <label>Resolution <b id="resolutionV">1100 px</b></label>
        <input type="range" id="resolution" min="400" max="2200" step="50" value="1100">
      </div>
      <label class="check"><input type="checkbox" id="invert"> Invert</label>
      <label class="check"><input type="checkbox" id="flip_x"> Mirror top/bottom</label>
      <label class="check"><input type="checkbox" id="flip_y"> Mirror left/right</label>
      <label class="check"><input type="checkbox" id="remove_bg"> Remove background (AI)</label>
    </fieldset>

    <details>
      <summary>Pen lifts &amp; machine</summary>
      <div>
        <div class="ctl">
          <label>Weld ends within <b id="mergeV">1.0 mm</b></label>
          <input type="range" id="merge" min="0" max="4" step="0.1" value="1">
        </div>
        <div class="ctl">
          <label>Bridge gaps up to <b id="bridgeV">3.0 mm</b>
            <span title="Crosses short gaps with the pen down instead of lifting, but only where the image is dark enough to hide the extra line.">ⓘ</span>
          </label>
          <input type="range" id="bridge" min="0" max="12" step="0.5" value="3">
        </div>
        <div class="ctl">
          <label>Bridge only over ink darker than <b id="bridge_min_inkV">0.30</b></label>
          <input type="range" id="bridge_min_ink" min="0.05" max="0.9" step="0.05" value="0.3">
        </div>
        <div class="ctl">
          <label>Max ink coverage <b id="max_coverageV">0.92</b></label>
          <input type="range" id="max_coverage" min="0.4" max="1" step="0.02" value="0.92">
        </div>
        <div class="row">
          <div class="ctl">
            <label>Draw feed</label>
            <input type="number" id="feed" value="3000" step="100">
          </div>
          <div class="ctl">
            <label>Travel feed</label>
            <input type="number" id="travel_feed" value="6000" step="100">
          </div>
        </div>
        <label class="check"><input type="checkbox" id="show_travel"> Show pen-up moves in preview</label>
      </div>
    </details>

    <button class="btn" id="go" disabled>Generate G-code</button>
    <button class="btn alt" id="dl" style="display:none">⬇ Download G-code</button>
    <div class="msg" id="msg"></div>
  </div>

  <div class="stage" id="stage">
    <div class="empty" id="empty">Drop in a photo and hit <b>Generate</b>.<br>
      The preview shows exactly what the pen will lay down.</div>
    <div class="spin" id="spin"></div>
    <div class="stats" id="stats">
      <div class="big" id="bigTime">—</div>
      <div class="grid">
        <div class="stat"><b id="sLifts">—</b>pen lifts</div>
        <div class="stat"><b id="sDraw">—</b>drawing</div>
        <div class="stat"><b id="sTravel">—</b>travel</div>
        <div class="stat"><b id="sTone">—</b>tone error</div>
        <div class="stat"><b id="sSize">—</b>on paper</div>
        <div class="stat"><b id="sFile">—</b>G-code</div>
      </div>
    </div>
  </div>
</div>

<script>
const ENGINES = {{ engines|tojson }};
const $ = id => document.getElementById(id);
const SLIDERS = {
  ink:v=>(+v).toFixed(2)+'×', angle:v=>v+'°', coherence:v=>Math.round(v*100)+'%',
  levels:v=>v, points:v=>(+v).toLocaleString(), depth:v=>v,
  spacing:v=>(+v===0?'auto':(+v).toFixed(1)+' mm'),
  pen:v=>(+v).toFixed(2)+' mm', contrast:v=>(+v).toFixed(1),
  detail:v=>(+v).toFixed(2), exposure:v=>(+v).toFixed(2), gamma:v=>(+v).toFixed(2),
  chroma:v=>(+v).toFixed(2), ink_response:v=>(+v).toFixed(2),
  resolution:v=>v+' px', merge:v=>(+v).toFixed(1)+' mm',
  bridge:v=>(+v===0?'off':(+v).toFixed(1)+' mm'),
  bridge_min_ink:v=>(+v).toFixed(2), max_coverage:v=>(+v).toFixed(2),
};
for (const [id, fmt] of Object.entries(SLIDERS)) {
  const el = $(id), out = $(id + 'V');
  if (!el || !out) continue;
  const sync = () => out.textContent = fmt(el.value);
  el.addEventListener('input', sync); sync();
}

// Only show the knobs the chosen engine actually reads.
function syncEngine() {
  const name = $('engine').value;
  const meta = ENGINES.find(e => e[0] === name);
  $('engineNote').innerHTML = meta[1] + '<br>Pen lifts: <b>' + meta[2] + '</b>';
  document.querySelectorAll('[data-for]').forEach(el => {
    el.style.display = el.dataset.for.split(',').includes(name) ? '' : 'none';
  });
}
$('engine').addEventListener('change', syncEngine); syncEngine();

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
  file = f; $('fname').textContent = f.name; $('go').disabled = false;
  $('dl').style.display = 'none';
  const r = new FileReader();
  r.onload = e => { $('thumb').src = e.target.result; $('thumb').style.display = 'block'; };
  r.readAsDataURL(f);
}

const FIELDS = ['engine','ink','angle','coherence','levels','points','depth','spacing',
  'pen','paper','margin','contrast','detail','exposure','gamma','chroma',
  'ink_response','resolution','merge','bridge','bridge_min_ink','max_coverage',
  'feed','travel_feed'];
const CHECKS = ['outline','portrait','invert','flip_x','flip_y','remove_bg','show_travel'];

$('go').addEventListener('click', async () => {
  if (!file) return;
  $('go').disabled = true; $('dl').style.display = 'none';
  $('msg').textContent = 'Working…'; $('msg').className = 'msg';
  $('spin').style.display = 'block'; $('stats').style.display = 'none';
  $('empty').style.display = 'none';
  const old = $('stage').querySelector('img.out'); if (old) old.remove();

  const form = new FormData();
  form.append('image', file);
  FIELDS.forEach(k => form.append(k, $(k).value));
  CHECKS.forEach(k => form.append(k, $(k).checked ? '1' : '0'));

  try {
    const res = await fetch('/generate', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) {
      $('msg').textContent = data.error; $('msg').className = 'msg err';
      $('empty').style.display = 'block';
    } else {
      const img = document.createElement('img');
      img.className = 'out';
      img.src = 'data:image/png;base64,' + data.preview;
      $('stage').insertBefore(img, $('stats'));
      const s = data.stats;
      $('bigTime').innerHTML = s.time + ' <small>estimated plot time</small>';
      $('sLifts').textContent = s.lifts.toLocaleString();
      $('sDraw').textContent = s.draw_m + ' m';
      $('sTravel').textContent = s.travel_m + ' m';
      $('sTone').textContent = s.tone_error === null ? '—' : s.tone_error;
      $('sSize').textContent = s.size;
      $('sFile').textContent = s.kb + ' kB';
      $('stats').style.display = 'block';
      $('msg').textContent =
        `welded ${s.lifts_before.toLocaleString()} strokes down to ` +
        `${s.lifts.toLocaleString()} · travel ${s.travel_raw_m} m → ${s.travel_m} m · ` +
        `${s.points.toLocaleString()} points · ${s.elapsed}`;
      $('dl').style.display = 'block';
    }
  } catch (err) {
    $('msg').textContent = 'Request failed: ' + err.message;
    $('msg').className = 'msg err';
  }
  $('spin').style.display = 'none';
  $('go').disabled = false;
});

$('dl').addEventListener('click', () => { window.location.href = '/download'; });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("\n  Open http://localhost:8080 in your browser\n")
    app.run(host="127.0.0.1", port=8080, debug=False)
