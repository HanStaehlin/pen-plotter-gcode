"""Pseudo-Latin names, and the name <-> drawing link, after fishdraw.

In fishdraw the name *is* the seed: a typed name is hashed to seed the
generator, and when no name is given a random binomial is invented first
and the drawing is seeded from that.  Every drawing therefore carries a
name that reproduces it exactly.

Frogs add one wrinkle: with "any family" the family must be recoverable
from the name alone, or typing a name back could produce a different
animal.  Generated genera end in a family's root, as real ones often do
(Rana, Bufo, Hyla, Dendrobates, Ceratophrys, Pipa), and those endings are
read back to pick the family.  A name without a known ending gets a family
from its hash.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

#: genus endings that identify each family
ROOTS = {
    "truefrog": ("rana", "ranus"),
    "toad": ("bufo", "phryne"),
    "treefrog": ("hyla", "hylax"),
    "dartfrog": ("bates", "batis"),
    "hornedfrog": ("phrys", "ceras"),
    "clawedfrog": ("pipa", "opus"),
}

_PREFIX = ["lepto", "pachy", "micro", "megalo", "litho", "chloro", "melano", "rhino",
           "platy", "brachy", "dendro", "hydro", "xeno", "oxy", "trachy", "calo",
           "cyano", "phyllo", "eury", "steno", "pseudo", "ptero", "aglyo", "ophi",
           "scapho", "nycto", "glypho", "odonto", "lopho", "thelo"]
_ONSET = ["b", "c", "d", "f", "g", "l", "m", "n", "p", "r", "s", "t", "v", "ph", "th",
          "ch", "tr", "cr", "pl", "gl", "st", "sc", "x", "z"]
_VOWEL = ["a", "e", "i", "o", "u", "ae", "ia", "io", "y"]
_EPITHET_END = ["us", "a", "um", "is", "ensis", "atus", "ata", "icus", "ica", "ii",
                "oides", "ops", "ura", "ulus", "ula", "inus", "ina", "ans", "ax"]


def _word(rng, syllables):
    return "".join(rng.choice(_ONSET) + rng.choice(_VOWEL) for _ in range(syllables))


def binomen(rng, family):
    """A plausible-sounding genus + species for ``family``."""
    root = rng.choice(ROOTS[family])
    genus = rng.choice(_PREFIX)
    if genus[-1] in "aeiouy" and root[0] in "aeiouy":
        genus = genus[:-1]
    genus = (genus + root).capitalize()
    epithet = _word(rng, int(rng.integers(1, 3))) + rng.choice(_EPITHET_END)
    epithet = re.sub(r"([aeiouy])\1+", r"\1", epithet)
    return f"{genus} {epithet}"


def normalise(name: str) -> str:
    return " ".join(name.replace(".", " ").split()).lower()


def seed_from_name(name: str) -> int:
    """Stable 32-bit seed from a name; spacing, case and dots don't matter."""
    digest = hashlib.sha256(normalise(name).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def family_of(name: str, families, weights) -> str:
    """Family implied by the genus ending, else one picked by the name's hash."""
    genus = normalise(name).split(" ")[0] if normalise(name) else ""
    for fam, roots in ROOTS.items():
        if fam in families and any(genus.endswith(r) for r in roots):
            return fam
    rng = np.random.default_rng(seed_from_name(name) ^ 0x5EED)
    return families[int(rng.choice(len(families), p=weights))]


def resolve(kind: str, families, weights, name: str | None = None,
            seed: int | None = None) -> tuple[str, int, str]:
    """(display name, drawing seed, family) for a request.

    ``name`` wins; otherwise ``seed`` (or a random one) picks a generated
    name, and the drawing is seeded from that name so the two stay linked.
    """
    if name and name.strip():
        fam = kind if kind in families else family_of(name, families, weights)
        # binomial style - Genus capitalised, epithet lower case - so any
        # spelling of the same name gives the same caption as well as the
        # same animal
        words = normalise(name).split(" ")
        shown = " ".join([words[0].capitalize()] + words[1:])
        return shown, seed_from_name(shown), fam
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))
    rng = np.random.default_rng(seed)
    fam = kind if kind in families else families[int(rng.choice(len(families), p=weights))]
    # the name stream is keyed by family too, or seed 1 would be "xafioax"
    # in every family
    shown = binomen(np.random.default_rng([seed, families.index(fam)]), fam)
    return shown, seed_from_name(shown), fam


# ── single-stroke lettering ──────────────────────────────────────────────

_FONT = None


def lettering(text: str, height: float, centre_x: float, top_y: float,
              face: str = "timesi") -> list[np.ndarray]:
    """``text`` as single-stroke polylines (Hershey font), centred, y down.

    A plotter cannot fill an outline font cleanly; Hershey glyphs are
    centrelines, so each letter is drawn once with the pen.  Returns no
    strokes if the optional Hershey-Fonts package is missing.
    """
    global _FONT
    try:
        from HersheyFonts import HersheyFonts
    except ImportError:
        return []
    if _FONT is None:
        _FONT = HersheyFonts()
        _FONT.load_default_font(face)
        _FONT.normalize_rendering(1.0)
    strokes = [np.asarray(s, float) for s in _FONT.strokes_for_text(text)]
    strokes = [s for s in strokes if len(s) >= 2]
    if not strokes:
        return []
    pts = np.vstack(strokes)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    k = height / max(hi[1] - lo[1], 1e-9)
    out = []
    for s in strokes:
        x = (s[:, 0] - (lo[0] + hi[0]) / 2) * k + centre_x
        y = (hi[1] - s[:, 1]) * k + top_y            # font is y-up
        out.append(np.stack([x, y], axis=1))
    return out
