"""Stroke post-processing: weld, order and thin the polylines.

Everything here operates on lists of ``(N, 2)`` float arrays in millimetres.
This stage is where pen lifts actually get eliminated - the drawing engines
produce good strokes, but they produce *many* of them, and a lift costs two
dwells plus a rapid move whether the gap is 40 mm or 0.05 mm.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:                                   # pragma: no cover
    cKDTree = None


Path = np.ndarray          # (N, 2) float32/float64
Paths = list


# ── length helpers ───────────────────────────────────────────────────────

def path_length(p: Path) -> float:
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def total_length(paths: Paths) -> float:
    return float(sum(path_length(p) for p in paths))


def travel_length(paths: Paths, home=(0.0, 0.0)) -> float:
    """Pen-up distance for the given ordering, including the return home."""
    if not paths:
        return 0.0
    pos = np.asarray(home, dtype=np.float64)
    total = 0.0
    for p in paths:
        total += float(np.hypot(*(p[0] - pos)))
        pos = p[-1]
    return total + float(np.hypot(*(np.asarray(home) - pos)))


# ── simplification ───────────────────────────────────────────────────────

def simplify(p: Path, tol: float) -> Path:
    """Iterative Douglas-Peucker.

    Iterative rather than recursive because hatch strokes routinely run to
    thousands of points and Python's recursion limit is not negotiable.
    """
    n = len(p)
    if n < 3 or tol <= 0:
        return p
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        seg = b - a
        seg_len2 = float(seg @ seg)
        rel = p[i + 1:j] - a
        if seg_len2 < 1e-18:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            t = np.clip((rel @ seg) / seg_len2, 0.0, 1.0)
            proj = t[:, None] * seg
            dist = np.hypot(*(rel - proj).T)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return p[keep]


def simplify_all(paths: Paths, tol: float) -> Paths:
    if tol <= 0:
        return paths
    return [simplify(p, tol) for p in paths]


def drop_short(paths: Paths, min_len: float) -> Paths:
    if min_len <= 0:
        return paths
    return [p for p in paths if len(p) >= 2 and path_length(p) >= min_len]


# ── welding: the pen-lift killer ─────────────────────────────────────────

def merge(paths: Paths, tolerance: float) -> Paths:
    """Chain strokes whose endpoints nearly touch into single polylines.

    Greedy: take a stroke, then keep hunting for the nearest free endpoint
    within ``tolerance`` of the growing chain's tail and splice it on.  With a
    tolerance around the pen width the joins are invisible on paper but each
    one removes a lift, a rapid and two dwells.
    """
    paths = [np.asarray(p, dtype=np.float64) for p in paths if len(p) >= 2]
    if len(paths) < 2 or tolerance <= 0:
        return paths

    n = len(paths)
    # Endpoint 2i is the start of path i, endpoint 2i+1 is its end.
    ends = np.empty((2 * n, 2), dtype=np.float64)
    for i, p in enumerate(paths):
        ends[2 * i] = p[0]
        ends[2 * i + 1] = p[-1]

    tree = _tree(ends)
    used = np.zeros(n, dtype=bool)
    out: Paths = []

    for seed in range(n):
        if used[seed]:
            continue
        used[seed] = True
        chain = [paths[seed]]
        tail = paths[seed][-1]

        # Grow forwards, then flip the accumulated chain and grow again so
        # both ends of the seed stroke get a chance to pick up neighbours.
        for _ in range(2):
            while True:
                nxt = _nearest_free(tree, ends, tail, tolerance, used)
                if nxt is None:
                    break
                idx, at_end = nxt
                used[idx] = True
                piece = paths[idx][::-1] if at_end else paths[idx]
                chain.append(piece)
                tail = piece[-1]
            chain = [c[::-1] for c in reversed(chain)]
            tail = chain[-1][-1]

        joined = _concat(chain)
        out.append(joined)

    return out


def _nearest_free(tree, ends, point, tolerance, used):
    """Closest unused path endpoint within ``tolerance`` of ``point``."""
    if tree is None:
        d = np.hypot(ends[:, 0] - point[0], ends[:, 1] - point[1])
        order = np.argsort(d)
        cand = order[d[order] <= tolerance]
    else:
        cand = tree.query_ball_point(point, tolerance)
        if not cand:
            return None
        cand = np.asarray(cand)
        d = np.hypot(ends[cand, 0] - point[0], ends[cand, 1] - point[1])
        cand = cand[np.argsort(d)]

    for e in cand:
        idx = int(e) // 2
        if not used[idx]:
            return idx, bool(int(e) % 2)
    return None


def _concat(chain: Paths) -> Path:
    """Join pieces, dropping duplicated points at the seams."""
    parts = [chain[0]]
    for piece in chain[1:]:
        if np.hypot(*(piece[0] - parts[-1][-1])) < 1e-9:
            piece = piece[1:]
        if len(piece):
            parts.append(piece)
    return np.vstack(parts)


def bridge(paths: Paths, max_gap: float, ink_at, min_ink: float) -> Paths:
    """Cross short gaps with the pen DOWN when the crossing lands on ink.

    A bridge trades a tiny amount of stray ink for a whole pen lift.  It is
    only worth taking where the image is dark enough that the extra line
    disappears into surrounding tone, which ``ink_at`` decides by sampling the
    coverage target along the proposed join.
    """
    if max_gap <= 0 or len(paths) < 2:
        return paths

    out: Paths = []
    current = paths[0]
    for nxt in paths[1:]:
        gap = float(np.hypot(*(nxt[0] - current[-1])))
        if gap <= max_gap and _bridge_is_dark(current[-1], nxt[0], ink_at, min_ink):
            current = np.vstack([current, nxt])
        else:
            out.append(current)
            current = nxt
    out.append(current)
    return out


def _bridge_is_dark(a, b, ink_at, min_ink: float, samples: int = 6) -> bool:
    t = np.linspace(0.0, 1.0, samples)[:, None]
    pts = a + (b - a) * t
    return float(ink_at(pts).min()) >= min_ink


# ── travel ordering ──────────────────────────────────────────────────────

def sort_paths(paths: Paths, home=(0.0, 0.0), refine: bool = True) -> Paths:
    """Nearest-neighbour ordering with free reversal, then a 2-opt cleanup."""
    if len(paths) < 2:
        return paths

    n = len(paths)
    ends = np.empty((2 * n, 2), dtype=np.float64)
    for i, p in enumerate(paths):
        ends[2 * i] = p[0]
        ends[2 * i + 1] = p[-1]

    tree = _tree(ends)
    used = np.zeros(n, dtype=bool)
    order: list[tuple[int, bool]] = []
    pos = np.asarray(home, dtype=np.float64)

    for _ in range(n):
        pick = _nearest_any(tree, ends, pos, used)
        if pick is None:
            break
        idx, reverse = pick
        used[idx] = True
        order.append((idx, reverse))
        pos = paths[idx][0] if reverse else paths[idx][-1]

    ordered = [paths[i][::-1] if rev else paths[i] for i, rev in order]
    if refine and len(ordered) > 3:
        ordered = _two_opt(ordered, home)
    return ordered


def _nearest_any(tree, ends, point, used):
    """Nearest endpoint of any unused path, expanding the search as needed."""
    n = len(used)
    if tree is None:
        d = np.hypot(ends[:, 0] - point[0], ends[:, 1] - point[1])
        d[np.repeat(used, 2)] = np.inf
        e = int(np.argmin(d))
        if not np.isfinite(d[e]):
            return None
        return e // 2, bool(e % 2)

    k = 8
    while k <= 2 * n:
        k = min(k, 2 * n)
        _, idx = tree.query(point, k=k)
        for e in np.atleast_1d(idx):
            e = int(e)
            if e < 2 * n and not used[e // 2]:
                return e // 2, bool(e % 2)
        if k == 2 * n:
            break
        k *= 4

    remaining = np.where(~used)[0]
    if remaining.size == 0:
        return None
    d = np.hypot(ends[2 * remaining, 0] - point[0], ends[2 * remaining, 1] - point[1])
    return int(remaining[int(np.argmin(d))]), False


def _two_opt(paths: Paths, home, window: int = 40, max_rounds: int = 3) -> Paths:
    """Windowed 2-opt over the *travel* edges.

    Nearest-neighbour ordering leaves a handful of long "go back for the bit I
    missed" hops.  Reversing a run of strokes fixes those, and because a
    reversed run also draws each stroke backwards, the flip flags travel with
    it.  The search window keeps this O(n*w): a full pass would be O(n^2) and
    these lists routinely hold thousands of strokes.
    """
    n = len(paths)
    if n < 4:
        return paths

    starts = np.array([p[0] for p in paths], dtype=np.float64)
    endpts = np.array([p[-1] for p in paths], dtype=np.float64)
    order = np.arange(n)
    flip = np.zeros(n, dtype=bool)
    home = np.asarray(home, dtype=np.float64)

    for _ in range(max_rounds):
        improved = False
        # heads[i]/tails[i] = entry/exit point of the stroke at position i.
        heads = np.where(flip[:, None], endpts[order], starts[order])
        tails = np.where(flip[:, None], starts[order], endpts[order])
        for i in range(-1, n - 2):
            a_end = home if i < 0 else tails[i]
            js = np.arange(i + 2, min(n, i + 2 + window))
            if js.size == 0:
                continue
            # Successor of j; for the final stroke the successor is home.
            b_next = np.where((js >= n - 1)[:, None],
                              home, heads[np.minimum(js + 1, n - 1)])
            cur_head = heads[i + 1]
            before = (np.hypot(*(cur_head - a_end)) +
                      np.hypot(*(b_next - tails[js]).T))
            after = (np.hypot(*(tails[js] - a_end).T) +
                     np.hypot(*(b_next - cur_head).T))
            gain = before - after
            k = int(np.argmax(gain))
            if gain[k] > 1e-9:
                j = int(js[k])
                sl = slice(i + 1, j + 1)
                order[sl] = order[sl][::-1]
                flip[sl] = ~flip[sl][::-1]
                heads[sl], tails[sl] = tails[sl][::-1].copy(), heads[sl][::-1].copy()
                improved = True
        if not improved:
            break

    return [paths[order[i]][::-1] if flip[i] else paths[order[i]] for i in range(n)]


def _tree(points: np.ndarray):
    return cKDTree(points) if cKDTree is not None else None
