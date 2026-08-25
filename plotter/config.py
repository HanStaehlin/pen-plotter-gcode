"""Geometry, pen and pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


# ── Paper presets (width_mm, height_mm) in landscape orientation ──────────
PAPER_SIZES = {
    "a3": (420.0, 297.0),
    "a4": (297.0, 210.0),
    "a5": (210.0, 148.0),
    "letter": (279.4, 215.9),
    "square": (200.0, 200.0),
}


@dataclass
class PageConfig:
    """Physical page and machine geometry."""

    paper: str = "a4"
    width_mm: float = 297.0
    height_mm: float = 210.0
    margin_mm: float = 10.0
    portrait: bool = False
    #: Machine Y axis grows downward as negative values (GRBL pen-plotter
    #: convention used by the AX4 firmware). Set False for positive-Y machines.
    negative_y: bool = True
    #: Beds that are wider than they are tall cannot draw a portrait page, so
    #: the emitted G-code is turned 90 degrees and the paper turned to match.
    #: "auto" rotates only portrait pages. The preview stays as composed.
    rotate_output: str = "auto"      # auto | off | ccw | cw

    def resolve(self) -> "PageConfig":
        if self.paper in PAPER_SIZES:
            w, h = PAPER_SIZES[self.paper]
            self.width_mm, self.height_mm = (h, w) if self.portrait else (w, h)
        elif self.portrait and self.width_mm > self.height_mm:
            self.width_mm, self.height_mm = self.height_mm, self.width_mm
        return self

    @property
    def draw_w(self) -> float:
        return max(1.0, self.width_mm - 2 * self.margin_mm)

    @property
    def draw_h(self) -> float:
        return max(1.0, self.height_mm - 2 * self.margin_mm)


@dataclass
class PenConfig:
    """Pen physics. ``width_mm`` is what makes tone reproduction calibrated."""

    width_mm: float = 0.4          # actual ink trace width of the pen
    #: Highest ink coverage the pen/paper combination can reach before the
    #: result just looks like a black blob. 1.0 == solid black.
    max_coverage: float = 0.92
    #: Coverage below this is treated as paper white (pen stays up).
    min_coverage: float = 0.02


@dataclass
class MachineConfig:
    """G-code dialect and motion parameters."""

    feed_draw: float = 3000.0      # mm/min, pen down
    feed_travel: float = 6000.0    # mm/min, pen up
    accel_mm_s2: float = 400.0     # used by the time estimator only
    #: GRBL $11. How far the tool may cut a corner; sets corner speed.
    junction_deviation: float = 0.02
    pen_up_cmd: str = "M5"
    pen_down_cmd: str = "M3S1000"
    pen_up_dwell: float = 0.15     # seconds
    pen_down_dwell: float = 0.15
    precision: int = 3             # decimals in coordinates


@dataclass
class ToneConfig:
    """Photograph -> ink-density mapping."""

    resolution: int = 1100         # internal long-edge resolution in px
    #: How much saturated colour darkens relative to its plain lightness.
    #: 0 = pure perceptual lightness, higher = colours separate more.
    chroma_weight: float = 0.18
    denoise: float = 0.0           # bilateral strength, 0 = off
    clahe_clip: float = 2.0        # local contrast, 0 = off
    clahe_grid: int = 8
    detail: float = 0.45           # structure-preserving detail boost, 0 = off
    black_point: float = 0.5       # percentile clipped to solid black
    white_point: float = 99.5      # percentile clipped to paper white
    gamma: float = 1.0             # <1 lightens midtones, >1 darkens
    #: 0 = spend the pen's range evenly across perceptual lightness,
    #: 1 = match the photograph's actual luminance (much darker, much slower).
    ink_response: float = 0.25
    #: Thin dark structures - an eyelash line, the seam between lips, a
    #: nostril - carry almost no area, so a tone-driven engine averages them
    #: into the skin around them and they vanish.  This pushes them toward
    #: black and tells the engines to measure tone locally where they are.
    feature_gain: float = 0.55
    feature_mm: float = 0.7        # how thin "thin" is, on paper
    exposure: float = 0.0          # -1..1 additive after normalisation
    invert: bool = False
    flip_x: bool = False
    flip_y: bool = False


@dataclass
class PathConfig:
    """Pen-lift minimisation and output cleanup."""

    #: Endpoints closer than this are welded into one continuous stroke.
    merge_tolerance_mm: float = 1.0
    #: Gaps up to this are crossed with the pen DOWN when the crossing runs
    #: over ink-bearing image area. This is the aggressive lift killer.
    bridge_max_mm: float = 3.0
    #: Minimum darkness (0-1) along a bridge for it to be drawn pen-down.
    bridge_min_ink: float = 0.30
    simplify_mm: float = 0.06      # Douglas-Peucker tolerance
    min_path_mm: float = 0.4       # drop strokes shorter than this
    sort: bool = True              # nearest-neighbour travel ordering
    refine_order: bool = True      # 2-opt pass over the travel ordering


@dataclass
class RenderConfig:
    """Engine selection and per-engine knobs."""

    engine: str = "flowlines"
    spacing_mm: float = 0.0        # 0 = derive from pen width
    ink_scale: float = 1.0         # global ink multiplier (0.3 - 2.0)
    angle_deg: float = 45.0        # base direction for flat regions / hatching
    # flowlines
    flow_coherence: float = 1.0    # 0 = ignore image structure, 1 = full
    flow_smooth: float = 4.0       # structure tensor smoothing sigma (px)
    #: Sharpness of the "is there really structure here" test. Above 1 it
    #: takes strong texture seriously and ignores faint shading gradients.
    coherence_gamma: float = 2.0
    stroke_min_mm: float = 1.5
    stroke_max_mm: float = 60.0
    # waves
    wave_period_mm: float = 2.4
    # hilbert
    #: Coarsest cell the curve may collapse to, in mm.  This is the real
    #: control on light areas: a cell that big carries one squiggle, so past
    #: a few millimetres it stops reading as pale tone and starts reading as
    #: a hole in the drawing.
    hilbert_coarsest_mm: float = 4.0
    hilbert_max_depth: int = 9
    #: Round the curve's right angles. 0 keeps the hard maze look; ~0.35
    #: turns every corner into a short arc, which plots markedly faster
    #: because a plotter has to slow almost to a stop for a square corner.
    hilbert_smooth: float = 0.35
    #: Extra subdivision levels where fine detail lives, so an eye is not
    #: drawn at the same pitch as the flat cheek beside it.
    hilbert_detail: float = 1.2
    # tsp
    tsp_points: int = 12000
    tsp_relax: int = 30
    # crosshatch
    hatch_layers: int = 3          # how many directions may stack
    hatch_spread_deg: float = 58.0  # angle between successive directions
    #: Coverage one direction may contribute before the next takes over.
    hatch_share: float = 0.30
    #: How much hatching follows the image's own structure.  Full coherence
    #: turns every tonal blob into concentric rings; too little and real
    #: texture - feathers, fur, hair - is flattened into ruled diagonal.
    #: 0.55 was chosen on photographs rather than on synthetic tests.
    hatch_coherence: float = 0.45
    # contours
    contour_levels: int = 12
    #: How strongly a depth-derived form field steers hatching. Shading says
    #: where the light is; depth says where the surface is, so this is what
    #: makes strokes wrap a form instead of ringing a highlight.
    form_weight: float = 0.8
    form_smooth: float = 8.0       # px; curvature direction is twitchy
    depth_model: str = "MiDaS_small"
    #: "coherent" = flow-based DoG (connected line work), "canny" = the old
    #: fixed-scale detector.
    line_method: str = "coherent"
    # overlay
    #: Trace landmark lines on top of the shading.  On a portrait this is
    #: what makes an eye read as an eye rather than as a dark patch.
    outline: bool = True
    outline_strength: float = 0.6
    #: Overlay strokes must sit on thin dark structure this strong to be
    #: kept, which draws the landmarks without chasing every texture edge.
    outline_feature_min: float = 0.35


@dataclass
class Config:
    page: PageConfig = field(default_factory=PageConfig)
    pen: PenConfig = field(default_factory=PenConfig)
    machine: MachineConfig = field(default_factory=MachineConfig)
    tone: ToneConfig = field(default_factory=ToneConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def __post_init__(self):
        self.page.resolve()

    def to_dict(self) -> dict:
        return asdict(self)
