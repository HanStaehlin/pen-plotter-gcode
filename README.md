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
| `adaptive`  | Cross-hatch where smooth, flow lines where textured| many, mixed style|
| `crosshatch`| Layered engraving; directions stack as tone darkens| many medium      |
| `waves`     | One serpentine sweep; tone from how far it detours| near-continuous  |
| `hilbert`   | A single space-filling curve, adaptive depth      | one curve        |
| `tsp`       | Stipple density joined into one continuous tour   | one tour         |
| `contours`  | Tonal isolines, long closed loops                 | few long loops   |
| `outline`   | Traced edges only, no shading                     | few strokes      |
| `lineart`   | Centreline trace of an existing drawing or scan   | few strokes      |

`flowlines` is the default and the best general answer for photographs. `adaptive` is worth trying whenever a photograph mixes real texture with smooth shading in the same frame - a bird, an animal, a person against a plain background.
`hilbert` and `tsp` are the ones to reach for when pen lifts really matter;
both stay unbroken except where the paper is genuinely blank.

On `hilbert`, **Coarsest fill** (`--coarsest`) is the setting that matters.
Depth controls cell size and ink density together, so a shallow cell is not
just pale, it is *large* — at 24 mm it carries one squiggle and reads as a
bare patch rather than as light tone. Lower it if pale areas look unpainted;
raise it for a sparser, faster plot.

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

### Faces

Tone alone does not draw a face. A lip seam is 0.8 mm tall and carries almost
no area, so an engine chasing coverage averages it into the surrounding skin
and it disappears — the picture comes out as a topographic map of a face
rather than a drawing of one. Three things fix that, and all are on by
default:

- **Fine detail** (`--feature-gain`) finds thin *dark* structure and pushes it
  toward black. Being thin and dark is not enough on its own — fabric weave
  and foliage are too — so the response is measured against its own local
  average, which leaves isolated structure standing and cancels anything that
  repeats.
- The ink accumulator **narrows its measuring band** where that structure
  lives, so a feature is judged on its own tone instead of the skin around it.
- **Landmark lines** (`--outline`) trace the lines a person would draw — lid
  line, lip seam, nostril — on top of the shading.

For portraits `crosshatch` beats `flowlines`. Flow lines follow the image's
own contours, which on a face means concentric rings around every tonal blob;
hatching lays parallel strokes that describe form without ringing features.

### Line, and form

A drawing is really three separate problems, and until now this project only
solved one of them well:

| | what it is | status |
|---|---|---|
| tone | where ink goes | solved — coverage model, ~0.07 error |
| **line** | marks describing edges and detail | now: flow-guided DoG |
| **form** | hatching *direction* encoding surface shape | now: depth-estimated |

**Coherent line tracing** (`--lines coherent`, on by default) replaces plain
Canny for the `outline` engine and the landmark overlay. Canny finds edges at
one fixed scale and in isolation, so on a photograph it traces both sides of
every dark line and returns a confetti of fragments. Kang, Lee & Chui's
method (2007) asks the same question — "is there a step here" — but only ever
looks *across* the local structure, then smooths the answer *along* it. The
across-pass finds the edge; the along-pass is what turns a field of noisy
responses into connected line work a pen can draw in one stroke.

**Depth-driven form** (`--form-weight`, default 0.8) estimates monocular
depth with MiDaS and hatches along the surface's principal curvature — the
direction an engraver would ring a cylinder, not the direction along its
axis. On Apple Silicon this runs on the Metal (MPS) backend and costs about
20 ms after the model is warm. It needs PyTorch (`pip install torch`); the
app runs fine without it and the control is simply skipped.

Depth is smooth by nature, and applying it everywhere erases real texture —
tried naively, it flattened every feather on a photograph of a bird into
ruled diagonal, because feathers are texture MiDaS smooths away, not
geometry it can see. Depth is therefore blended in only where the image's own
structure is weak (`1 - coherence`), so it fills the ground shading can't
hold — a cheek, a shoulder, the side of a sphere — and leaves real texture to
speak for itself.

That tension is also why **Follow texture** (`hatch_coherence`, default
0.45) has no value that suits everything: high keeps feathers and fur
readable; too high and smooth skin turns into concentric rings around every
highlight, because the tone field is chasing faint shading gradients that
carry no real texture. `coherence_gamma` (default 2.0) sharpens the "is this
real structure" test to make one setting serve more cases, but for very
different subjects — a face versus a bird — it is worth adjusting by hand.

### Region-adaptive routing

`hatch_coherence` has no value that suits both a feathered subject and a
smooth face (see above) - that tension turned out to be a property of the
*image*, not a tuning miss: some regions are real texture (feathers, fur,
weave) and want to follow it closely, others are shading (skin, sky) and
ring into concentric contours if they do. The `adaptive` engine sidesteps
picking one global value by classifying regions and routing each to the
engine suited to it: cross-hatch for smooth shading, flow-traced strokes
(full structure-following) for texture.

Classification is local variance of lightness, aggregated over SLIC
superpixels rather than per pixel - a per-pixel threshold on noisy variance
would fragment one feather into hundreds of slivers, each triggering an
engine switch. SLIC needs real colour to find those superpixels: handed a
single grayscale channel it has no signal to cluster on, spatial compactness
wins outright, and the result is a perfect grid that ignores the image
entirely - worth knowing if you reach for `skimage.segmentation.slic`
elsewhere, since the failure is silent rather than an error.

Tested against the same photograph that broke Hilbert: routing rescued the
eye and chest barring that pure `crosshatch` alone lost, while still giving
the smooth wing membrane a clean engraved look. The honest cost is that it
draws more line than either pure engine alone (owl: 35.5 m vs 28.4 m for
crosshatch, 30.4 m for flowlines) - confining each style to a smaller,
irregularly-shaped region means shorter individual strokes than letting one
engine range over the full canvas, and that overhead does not go away by
cleaning up the mask (tried; made no measurable difference). Worth it when a
subject genuinely mixes texture and smooth shading; pure `flowlines` remains
the better choice when everything in frame is already texture.

### Hilbert variants

Corner rounding (`--hilbert-smooth`) is worth having on. A plotter nearly
stops at a square corner and a Hilbert curve is nothing but corners, so
rounding them plots about 19% faster, reproduces tone slightly *better*, and
softens the hard maze into something closer to drawn line.

`--hilbert-detail` adds subdivision levels where fine structure lives, so an
eye is not drawn at the same pitch as the flat cheek beside it.

### Portrait plots

A plotter bed that is wider than it is tall cannot draw a portrait page, so
portrait output is turned 90 degrees anticlockwise before it is written; turn
the paper the same way and the drawing comes out upright. The preview keeps
showing the picture as composed — only what goes down the wire is turned.
`--rotate-output off` disables it.

To turn a G-code file that already exists:

```
python3 gcode_rotate.py drawing.gcode          # -> drawing_landscape.gcode
```

This is a rotation, not a transpose. Swapping X and Y reflects the drawing
about the diagonal and comes out mirrored — readable enough on an abstract
pattern that the mistake reaches the paper, and obvious the moment there is
lettering or a face in the picture. A rotation swaps the axes *and* flips the
sign of one of them.

## Notes

- **Pen width is the important setting.** It is what calibrates tone. Measure
  the real ink trace, not what the barrel claims.
- **Finer hatching means a finer pen**, and the app needs telling. Tone is
  reproduced by spacing strokes `pen / coverage` apart, so halving the nib
  roughly doubles the line drawn: on one portrait, 0.40 mm gave 24 m and 18
  minutes, 0.25 mm gave 38 m and 25 minutes, 0.15 mm gave 58 m and 40 minutes.
  Tone accuracy holds throughout. A finer pen also affords more hatch
  directions, since each contributes less coverage. The working resolution
  follows the pen automatically when it would otherwise land sub-pixel.
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
