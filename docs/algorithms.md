# How it works

## Tone: coverage, not grey level

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

## Density: an ink budget, not a threshold

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

## Direction: the image's own structure

`flowlines` traces strokes through an orientation field recovered from the
smoothed **structure tensor**, whose minor eigenvector runs *along* edges rather
than across them. Orientation is carried in double-angle form throughout, since
θ and θ+π mean the same thing and averaging raw angles across that wrap gives
garbage. Flat regions fall back to a coarse-scale reading and finally to a fixed
angle, so strokes never chase sensor noise.

## Pen lifts

After the strokes exist, three passes attack the lift count:

- **Weld** — endpoints closer than `--merge` are chained into one polyline.
- **Bridge** — gaps up to `--bridge` are crossed with the pen *down*, but only
  where the image is dark enough to hide the extra line.
- **Order** — nearest-neighbour travel ordering, then windowed 2-opt; reversing
  a run flips the strokes inside it too.

On the test image that takes `flowlines` from 2 352 strokes to 565, and travel
from 205 m unordered down to 3.6 m.

## Plot-time estimates

Estimates plan the moves the way GRBL does — junction speeds from corner angle,
forward and backward passes to make the ramps reachable, then a trapezoid per
move. Feed rate alone is useless (a machine at 400 mm/s² never reaches
3000 mm/min over a 2 mm stroke), and assuming a dead stop at every vertex is
just as wrong — it would price the Hilbert curve, which is nothing but corners,
at triple its real cost.

## Hilbert and Waves: why they are slow, and what helps

Both styles earn tone by wiggling, and a wiggle is a tight curve. A plotter
can only take a curve of radius r at about √(acceleration × r): at the
default 400 mm/s² the waves' peaks (r ≈ 0.1 mm) cap the pen near 7 mm/s
against a 50 mm/s feed, and the Hilbert maze turns every 0.6 mm. A tone needs
a fixed length of line per area, so changing wave spacing or period barely
moves the time (32–34 min on the beard portrait across the settings tried).

What does help:

- **Draw less.** `--subject-focus` (default 0.8, every engine but
  illustration, which has its own) lightens the backdrop using depth. Hilbert
  also leaves paper wherever the tone wants less than half of what its
  coarsest cell lays down; before, a pale wall became a full-page maze darker
  than the photo. Hilbert fades the backdrop fully, because a partial fade
  lands on that cutoff and comes out as a maze full of holes. Beard portrait:
  Hilbert 44 → 34 min, Waves 37 → 32 min; dog: Hilbert 1h 17m → 33 min.
- **Round the corners.** `--hilbert-smooth` now defaults to 0.5.
- **The machine.** These two gain the most from higher GRBL acceleration:
  at 1600 mm/s² and 0.1 mm junction deviation Hilbert drops from 35 to 13 min
  and Waves from 32 to 15 (illustration: 13 → 7.5). See `docs/hardware.md`.
- **The time limit** lightens the whole picture for these engines (it
  re-runs the engine on less ink until the plot fits) instead of dropping
  strokes, which for a Hilbert curve or a serpentine cuts holes.

## Crossing layers

Strokes of one direction sit side by side, so their coverage adds. A second
direction lands as often on ink as on paper: two 50% layers cover 75%, not
100%. Cross-hatch therefore starts each new direction as a fresh layer that
combines with what is already down as 1 − (1 − b)(1 − l). Counting it
linearly left cross-hatched darks about 0.04 too light; the fix costs about
6% more plot time on `crosshatch` and nothing on `illustration`, which
rarely stacks layers.

## Judging the result

`plotter/quality.py` scores the finished G-code rather than the engine's own
bookkeeping: every path drawn at the pen's real width, drawing and tone
target both blurred by 1 mm (arm's length), then compared by SSIM
("likeness") and mean coverage difference ("tone error"). OpenCV only draws
odd line widths plus an antialiasing fringe, so the rendering resolution is
chosen to make one of those widths exactly the pen; asking for "4 px" would
overstate ink by a third. This runs after pruning and the time budget, so it
shows what they cost.

## Time budget and pruning

Plot time is treated as a budget rather than only reported. Every stroke gets
a score for how much it adds to the picture (`plotter/budget.py`):

- a line scores its drawn darkness (from the line network) times a factor
  that rises with length, so a 2 mm fragment, which costs a full lift and
  reads as a speck, ranks low;
- a hatch stroke scores the darkness of the shadow it shades, weighted 1.5×
  so the deepest shadows compete with mid-strength lines.

Lines and shading share one scale because the line layer is about nine
tenths of the plot time on the test photos: "lines first, hatching after"
would leave nothing to choose between.

Strokes scoring below `--prune` (default 0.2) are always dropped — the delete
step of Turk & Banks 1996. On the six test photos this saves 4–21% with no
visible loss; mostly it removes faint hatching on backgrounds and skies.

With `--minutes N` (the app's *Time limit*) the best-scoring strokes are kept
until the estimate fits. Welding and bridging change the real cost a lot, so
the cut is checked on the finished, ordered paths: each stroke is priced on
its own, the best prefix that fits is finished and timed, and the ratio
corrects the next guess. It lands within 1% under the limit in a few passes.

## Faces

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

## Line, and form

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
depth and hatches along the surface's principal curvature — the
direction an engraver would ring a cylinder, not the direction along its
axis. On Apple Silicon this runs on the Metal (MPS) backend and costs about
0.1–0.2 s after the model is warm. It needs PyTorch (`pip install torch`); the
app runs fine without it and the control is simply skipped.

The depth model is Depth Anything V2 Small (Apache-2.0, needs `transformers`),
with MiDaS small as the fallback. MiDaS returns roughly one smooth blob per
subject; Depth Anything resolves a nose, eye sockets and a muzzle, so the
shadow band hugs the cheek and jaw instead of spreading across the forehead.
A dedicated surface-normal model was considered: DSINE's licence is
research-only, and Marigold (a diffusion model) is far slower; normals from
the sharper depth were enough to get most of the gain.

Depth is smooth by nature, and applying it everywhere erases real texture —
tried naively, it flattened every feather on a photograph of a bird into
ruled diagonal, because feathers are texture depth smooths away, not
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

## Illustration: learned lines over shadow hatching

Every hand-designed line detector here (Canny, flow-guided DoG) asks "is
there an edge"; illustrators draw only the edges that describe shape and
identity. `illustration` uses **Informative Drawings** (Chan, Durand &
Isola, CVPR 2022) - a small network trained so that a photo's depth and
meaning can be recovered from its drawing - for the line layer, then adds a
separate, lighter shading layer the way pen-and-ink work is built:

- hatching only where tone is darker than `--shadow`; the paper carries the
  light and mid tones instead of every region getting strokes;
- shading faded on the background using depth (`--focus`, default 0.85), so
  detail goes to the subject rather than to blurred grass or a night sky
  behind it. The default was 0.4 while depth came from MiDaS, whose blurry
  subject mask made a strong fade risky; with Depth Anything the subject's
  edge is sharp, and on a close-up portrait against a dark background 0.4
  still let the background hatching bury the face;
- no shading on the line pixels themselves, so layers don't double-ink;
- a shadow band on the side of the subject facing away from a virtual light
  (`illustration_rim`, `illustration_light_deg`), after LingDong's
  [fishdraw](https://github.com/LingDong-/fishdraw). fishdraw finds that band
  by subtracting a light-shifted copy of each closed shape; a photo's line
  work is never closed, so it comes from relighting the depth estimate here;
- hatch strokes start at a random point up to half their length in from the
  lit end (`illustration_fade`), so shading thins out instead of stopping
  on a hard edge - also from fishdraw.

Turning the network's drawing into strokes needs care. Its marks are soft
and thick, and thresholding leaves pinholes inside them; thinning wraps
around each pinhole, so an eyelid came back as a chain of tiny loops. Holes
under 0.5 mm² are filled first. Where marks crowd into a solid patch (a
beard, curls) the centreline of the patch is a honeycomb, so the threshold
rises with local mark density and only the strongest strands become lines.

Pen-down bridging (see *Pen lifts*) judges a gap by the ink actually drawn
around it, not by the photo's tone. This engine hatches only shadows, so
the photo said "dark" over forehead and cheek mid-tones while the paper there
was nearly blank, and every bridge showed as a zigzag across the face.

The network runs in about 0.2 s on Metal; weights (MIT) download from the
authors' Hugging Face Space on first use. The same learned lines are now the
default line source for the landmark overlay on every engine
(`--lines learned|coherent|canny`).

It is a different style rather than a replacement: across the seven test
photos it plots 2-5x faster than `flowlines` with roughly half the pen lifts,
but it deliberately gives up some tonal range. Taste decides; `--shadow` and
`--focus` are the two controls that set how sparse it gets.

## Region-adaptive routing

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

## Lettering

A plotter needs fonts whose glyphs are stroke centrelines, not outlines to
fill; the Hershey fonts are exactly that, from the 1960s, in roman, italic,
script and blackletter faces (`plotter/lettering.py`).

Calligraphy's thick and thin come from a flat nib held at a fixed angle: a
stroke across the nib is as wide as the nib, a stroke along it a hairline.
A round pen has no edge, but the same contrast appears when each stroke is
drawn several times side by side, shifted along the nib's edge (40°, about
7.5% of the letter size wide). The passes are joined into one zigzag, so a
stroke still costs one pen lift and its ends come out cut at the nib angle,
as a real nib's do. The blackletter headings are not nibbed: their glyphs are
already drawn with doubled strokes, and nibbing them closes every counter.

Layout fits the largest body size, up to 9 mm, at which every line fits the
width and the block fits the height, by bisection. Line breaks in a poem
carry meaning, so text shrinks to keep lines whole and wraps, with a hanging
indent, only when even 3 mm is too wide. The block is centred on the page with
lines left-aligned inside it, or each line centred.

## Hilbert variants

On `hilbert`, **Coarsest fill** (`--coarsest`) is the setting that matters.
Depth controls cell size and ink density together, so a shallow cell is not
just pale, it is *large* — at 24 mm it carries one squiggle and reads as a
bare patch rather than as light tone. Lower it if pale areas look unpainted;
raise it for a sparser, faster plot.


Corner rounding (`--hilbert-smooth`) is worth having on. A plotter nearly
stops at a square corner and a Hilbert curve is nothing but corners, so
rounding them plots about 19% faster, reproduces tone slightly *better*, and
softens the hard maze into something closer to drawn line.

`--hilbert-detail` adds subdivision levels where fine structure lives, so an
eye is not drawn at the same pitch as the flat cheek beside it.
