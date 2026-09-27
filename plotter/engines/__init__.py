"""Drawing engines.

Each engine turns a coverage target into polylines in *pixel* coordinates and
returns the ink accumulator it filled, so the pipeline can report how closely
the drawing actually reproduces the requested tone.
"""

from __future__ import annotations

from . import (adaptive, contours, crosshatch, flowlines, hilbert,
               illustration, outline, tsp, waves)

#: name -> (callable, human description, roughly how continuous it is)
ENGINES = {
    "illustration": (illustration.generate,
                     "Illustration — learned line drawing over lighter shading",
                     "fewest lifts, sparse style"),
    "crosshatch": (crosshatch.generate,
                   "Cross-hatch — layered engraving that follows the form",
                   "many medium strokes"),
    "flowlines": (flowlines.generate,
                  "Flow lines — strokes follow the image's own structure",
                  "many long strokes"),
    "adaptive": (adaptive.generate,
                "Adaptive — cross-hatch where smooth, flow lines where textured",
                "many, style varies by region"),
    "waves": (waves.generate,
              "Waves — one serpentine sweep, tone from how far it detours",
              "near-continuous"),
    "hilbert": (hilbert.generate,
                "Hilbert — a single space-filling curve, adaptive depth",
                "one curve, breaks at white"),
    "tsp": (tsp.generate,
            "TSP — stipple density joined into one continuous tour",
            "one tour, few breaks"),
    "contours": (contours.generate,
                 "Contours — tonal isolines, long closed loops",
                 "few long loops"),
    "outline": (outline.generate,
                "Outline — traced edges only, no shading",
                "few strokes"),
    "lineart": (outline.generate_lineart,
                "Line art — centreline trace of an existing drawing or scan",
                "few strokes"),
}


def get(name: str):
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; "
                         f"expected one of {', '.join(sorted(ENGINES))}")
    return ENGINES[name][0]


__all__ = ["ENGINES", "get", "adaptive", "contours", "illustration", "crosshatch",
           "flowlines", "hilbert", "outline", "tsp", "waves"]
