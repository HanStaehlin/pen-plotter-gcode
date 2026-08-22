# Image → G-code

Turns photographs into continuous-line drawings for a pen plotter (built around
an AX4 / GRBL machine, but the G-code is plain and portable).

Two design goals drove the rewrite of the drawing algorithms:

1. **Keep the pen down.** A pen lift costs two dwells and a rapid move whether
   the gap is 40 mm or 0.05 mm. Several engines here draw the entire image as
   one unbroken stroke.
2. **Keep the dynamic range.** A colour photograph has to survive being
   flattened to bare paper and one ink. Tone is reproduced continuously rather
   than posterised into a handful of hatch layers.

```
python3 app.py                      # web UI on http://localhost:8080
python3 image_to_gcode.py photo.jpg -o photo.gcode
python3 image_to_gcode.py --list-engines
```

## Engines

| Engine      | Look                                              | Pen lifts        |
|-------------|---------------------------------------------------|------------------|
| `flowlines` | Strokes follow the image's own structure          | many long strokes|
| `waves`     | One serpentine sweep; tone from how far it detours| near-continuous  |
| `hilbert`   | A single space-filling curve, adaptive depth      | one stroke       |
| `tsp`       | Stipple density joined into one continuous tour   | one stroke       |
| `contours`  | Tonal isolines, long closed loops                 | few long loops   |
| `outline`   | Traced edges only, no shading                     | few strokes      |
| `lineart`   | Centreline trace of an existing drawing or scan   | few strokes      |

`flowlines` is the default and the best general answer for photographs.
`hilbert` and `tsp` are the ones to reach for when pen lifts really matter.

## How it works

### Tone: coverage, not grey level

The pipeline never asks "how grey is this pixel". It asks **what fraction of
the paper here should end up covered in ink** — because coverage is the thing
a plotter can actually control, by spacing its strokes.

- Colour collapses to CIE **L\***, not luma, and a fraction of chroma is
  subtracted so a saturated red and a neutral of the same lightness do not land
  on the same tone. (`--chroma`)
- **CLAHE** recovers detail buried in deep shadow or blown highlight, and a
  bilateral-split detail boost keeps texture without the haloes an unsharp mask
  would leave — haloes become visible stray strokes on paper.
- Lightness maps to coverage through a curve that fades between *perceptual*
  (spend the pen's range evenly across L\*) and *photometric* (match the
  photograph's real luminance). Photometric is correct and brutal: a mid grey
  genuinely demands ~80% coverage. `--ink-response` picks the blend.

### Density: an ink budget, not a threshold

The old converter thresholded darkness into four bands and drew a hatch layer
per band — five tones, total. Instead, engines here keep an **ink accumulator**
and lay strokes wherever the deposited ink still falls short of the target.
Line spacing is *discovered* rather than dialled in, and tone comes out
continuous.

Two details make it work, and both are easy to get wrong:

- The accumulator credits a stroke over a band **two line-spacings wide**. A
  triangular kernel of half-width *s* sampled every *s* sums to a constant —
  the partition of unity behind linear interpolation. Halve that support and
  the midpoint between two strokes reads as under-inked, pulls in a third
  stroke, and the whole image comes out 2–3× too dark.
- A stroke deposits nearly a full target's worth of coverage where it lands, so
  it is only worth drawing if **at least half of one is missing** — the same
  round-to-nearest rule error diffusion uses. Seeding on any leftover deficit
  over-inks by ~50%.

### Direction: the image's own structure

`flowlines` traces strokes through an orientation field recovered from the
smoothed **structure tensor**, whose minor eigenvector runs *along* edges rather
than across them. Orientation is carried in double-angle form throughout, since
θ and θ+π mean the same thing and averaging raw angles across that wrap gives
garbage. Flat regions fall back to a coarse-scale reading and finally to a fixed
angle, so strokes never chase sensor noise.

### Pen lifts

After the strokes exist, three passes attack the lift count:

- **Weld** — endpoints closer than `--merge` are chained into one polyline.
- **Bridge** — gaps up to `--bridge` are crossed with the pen *down*, but only
  where the image is dark enough to hide the extra line.
- **Order** — nearest-neighbour travel ordering, then windowed 2-opt; reversing
  a run flips the strokes inside it too.

On the test image that takes `flowlines` from 2 352 strokes to 565, and travel
from 205 m unordered down to 3.6 m.

### Plot-time estimates

Estimates plan the moves the way GRBL does — junction speeds from corner angle,
forward and backward passes to make the ramps reachable, then a trapezoid per
move. Feed rate alone is useless (a machine at 400 mm/s² never reaches
3000 mm/min over a 2 mm stroke), and assuming a dead stop at every vertex is
just as wrong — it would price the Hilbert curve, which is nothing but corners,
at triple its real cost.

## Notes

- **Pen width is the important setting.** It is what calibrates tone. Measure
  the real ink trace, not what the barrel claims.
- The `plotter/` package is importable on its own:

  ```python
  import plotter
  cfg = plotter.Config()
  cfg.render.engine = "hilbert"
  result = plotter.convert(plotter.load_image("photo.jpg"), cfg)
  open("photo.gcode", "w").write(result.gcode())
  ```

- Install with `pip install -r requirements.txt`. `rembg` is optional and only
  powers the "Remove background" checkbox.
