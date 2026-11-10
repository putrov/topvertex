"""Lattice polygons and their unimodular triangulations.

Conventions follow Konishi, math/0504188, Sec. 4.2 (see CONVENTIONS.md): a lattice point
``w = (x, y)`` stands for the ray generator ``(x, y, 1)`` of the fan of a toric Calabi-Yau
threefold; a unimodular triangulation of the polygon is the section of a smooth fan at height 1.

Everything here is exact integer combinatorics, except the linear program in :func:`heights`:
with scipy its floating point solution is rounded to integers and then verified exactly; without
scipy it is solved exactly over Q by a small rational simplex (:func:`lp_backend`).
"""
from __future__ import annotations

import math
import os
import re
from collections import deque
from fractions import Fraction
from itertools import combinations
from urllib.parse import unquote

Point = tuple[int, int]

__all__ = [
    "parse_polygon", "format_polygon", "orient", "convex_hull", "is_convex", "area2", "lattice_points",
    "boundary_points", "interior_points", "relation_coefficients", "relation_vector",
    "Triangulation", "initial_triangulation", "all_unimodular_triangulations",
    "heights", "lp_backend", "is_regular", "all_ones_weights_consistent", "plot",
]


# ----------------------------------------------------------------------------- corner strings

def parse_polygon(s: str) -> list[Point]:
    """Parse polygon corners from a compact corner string.

    The corners (x, y) are listed in order, coordinates separated by ``,`` and corners by ``_``:
    ``p=0,0_4,0_0,2`` or the bare list ``0,0_4,0_0,2``.  A URL carrying a ``p=`` parameter in its
    query or fragment (e.g. the drawing page's ``http://127.0.0.1:8765/#p=0,0_4,0_0,2``) is
    accepted too.  Rule: take the value of the ``p=`` parameter if present, reject decimals, then
    collect all integers with the regex ``(?<!\\d)-?\\d+`` and pair them up as (x, y).
    """
    m = re.search(r"[?&#]p=([^&\s]+)", s)
    if m is None:
        m = re.match(r"\s*p=([^&\s]+)", s)
    body = unquote(m.group(1)) if m else s
    if re.search(r"\d\.\d", body):
        raise ValueError("coordinates must be whole numbers")
    nums = [int(t) for t in re.findall(r"(?<!\d)-?\d+", body)]
    if len(nums) % 2:
        raise ValueError("odd number of coordinates")
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def format_polygon(corners) -> str:
    """Compact corner string of a polygon, inverse of :func:`parse_polygon`:
    ``[(0,0),(4,0),(0,2)] -> 'p=0,0_4,0_0,2'``."""
    return "p=" + "_".join(f"{int(x)},{int(y)}" for x, y in corners)


# ----------------------------------------------------------------------------- basic geometry

def orient(a: Point, b: Point, c: Point) -> int:
    """Twice the signed area of the triangle (a, b, c): ``(b-a) ^ (c-a)`` (> 0 iff ccw)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def convex_hull(points) -> list[Point]:
    """Corners of the convex hull, counterclockwise, starting at the lexicographically
    smallest point; collinear boundary points are dropped (Andrew's monotone chain)."""
    pts = sorted(set((int(x), int(y)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def chain(seq):
        out: list[Point] = []
        for p in seq:
            while len(out) >= 2 and orient(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = chain(pts), chain(reversed(pts))
    return lower[:-1] + upper[:-1]


def is_convex(corners) -> bool:
    """True iff the closed polygon with the given corners (in this order, either orientation)
    is convex and non-degenerate; collinear consecutive corners are allowed."""
    c = [tuple(p) for p in corners]
    n = len(c)
    if n < 3:
        return False
    signs = {(orient(c[i], c[(i + 1) % n], c[(i + 2) % n]) > 0) - (orient(c[i], c[(i + 1) % n], c[(i + 2) % n]) < 0)
             for i in range(n)}
    signs.discard(0)
    if len(signs) != 1:
        return False
    # simple (no winding twice): total turning = one revolution <=> hull has same corners
    hull = convex_hull(c)
    return set(hull) <= set(c) and abs(area2(c)) == area2(hull)


def area2(corners) -> int:
    """Twice the (absolute) area of the polygon with the given corners (shoelace formula)."""
    c = list(corners)
    n = len(c)
    return abs(sum(c[i][0] * c[(i + 1) % n][1] - c[(i + 1) % n][0] * c[i][1] for i in range(n)))


def _in_hull(p: Point, hull: list[Point]) -> int:
    """+1 strictly inside, 0 on the boundary, -1 outside the ccw convex polygon ``hull``."""
    n = len(hull)
    worst = 1
    for i in range(n):
        o = orient(hull[i], hull[(i + 1) % n], p)
        if o < 0:
            return -1
        if o == 0:
            worst = 0
    return worst


def lattice_points(corners) -> list[Point]:
    """All lattice points of the closed convex polygon conv(corners), sorted by (x, y)."""
    hull = convex_hull(corners)
    if len(hull) < 3:
        raise ValueError("degenerate polygon (fewer than 3 non-collinear corners)")
    xs, ys = [p[0] for p in hull], [p[1] for p in hull]
    return [(x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)
            if _in_hull((x, y), hull) >= 0]


def boundary_points(corners) -> list[Point]:
    """Lattice points on the boundary of conv(corners), sorted."""
    hull = convex_hull(corners)
    return [p for p in lattice_points(hull) if _in_hull(p, hull) == 0]


def interior_points(corners) -> list[Point]:
    """Lattice points in the interior of conv(corners), sorted."""
    hull = convex_hull(corners)
    return [p for p in lattice_points(hull) if _in_hull(p, hull) == 1]


def _vec3(p: Point) -> tuple[int, int, int]:
    return (p[0], p[1], 1)


def relation_coefficients(w1: Point, w2: Point, w3: Point, w3p: Point) -> tuple[int, int]:
    """Integers (a1, a2) with ``w3' = -a1 w1 - a2 w2 - w3`` for the 3-vectors (x, y, 1)
    (KP Sec. 4.2).  Requires (w1, w2, w3) to be a unimodular triangle; then a1 + a2 = -2.
    Solved exactly: (w1, w2, w3) is a Z-basis of Z^3, so w3' = c1 w1 + c2 w2 + c3 w3 with
    integer c given by Cramer's rule (det = +-1); c3 = -1 when w3' is on the other side of w1w2.
    """
    d = orient(w1, w2, w3)
    if abs(d) != 1:
        raise ValueError(f"non-unimodular: orbifold vertex not implemented (triangle {w1},{w2},{w3})")
    # barycentric-type coordinates of w3' w.r.t. w1, w2, w3 (exact since |d| = 1)
    c1 = orient(w3p, w2, w3) * d
    c2 = orient(w1, w3p, w3) * d
    c3 = orient(w1, w2, w3p) * d
    assert c1 + c2 + c3 == 1
    if c3 != -1:
        raise ValueError(f"{w3} and {w3p} are not the opposite vertices of adjacent unimodular triangles")
    return -c1, -c2


def relation_vector(n_points: int, i1: int, i2: int, i3: int, i3p: int, points) -> tuple[int, ...]:
    """Curve class of the interior segment tau = (w_i1, w_i2) (KP Sec. 4.2, CONVENTIONS.md):
    l_tau has +1 at w3 and w3', a1 at w1 and a2 at w2, so that sum_i l_i (x_i, y_i, 1) = 0."""
    a1, a2 = relation_coefficients(points[i1], points[i2], points[i3], points[i3p])
    lv = [0] * n_points
    lv[i3] += 1
    lv[i3p] += 1
    lv[i1] += a1
    lv[i2] += a2
    return tuple(lv)


# ----------------------------------------------------------------------------- triangulations

def _ccw(points, t) -> tuple[int, int, int]:
    a, b, c = t
    if orient(points[a], points[b], points[c]) < 0:
        b, c = c, b
    return (a, b, c)


def _canon_tri(t) -> tuple[int, int, int]:
    """Rotate a ccw triple so that the smallest index comes first."""
    k = t.index(min(t))
    return tuple(t[k:] + t[:k])


def _tri_overlap(P, Q) -> bool:
    """True iff two non-degenerate ccw triangles (as point triples) have overlapping interiors
    (separating axis test over the 6 edge lines, exact)."""
    for A, B in ((P, Q), (Q, P)):
        for i in range(3):
            a, b = A[i], A[(i + 1) % 3]
            if all(orient(a, b, q) <= 0 for q in B):
                return False
    return True


class Triangulation:
    """A unimodular triangulation of a lattice polygon.

    ``points``: list of lattice points (x, y).  For a convex polygon (``convex=True``, default)
    these must be ALL lattice points of the polygon (in any order; the order fixes the indices).
    ``triangles``: index triples, stored counterclockwise (input orientation is normalised).

    ``convex=False`` allows the support to be a non-convex union of triangles (e.g. the fan of
    O(1)+O(-3) -> P^1, KP Fig. 3 middle); then ``points`` are the vertices used.
    """

    def __init__(self, points, triangles, convex: bool = True):
        self.points: list[Point] = [(int(x), int(y)) for x, y in points]
        self.triangles: list[tuple[int, int, int]] = [_ccw(self.points, tuple(int(i) for i in t)) for t in triangles]
        self.convex = convex

    # -- constructors
    @classmethod
    def from_corners_and_triangles(cls, corners, triangles_as_point_tuples, points=None, convex: bool = True):
        """Build from polygon corners and triangles given by coordinates.  ``points`` (optional)
        fixes the order of the lattice points; default: :func:`lattice_points` order (for
        ``convex=False``: the sorted set of triangle vertices)."""
        tris = [tuple(tuple(p) for p in t) for t in triangles_as_point_tuples]
        if points is None:
            if convex:
                points = lattice_points(corners)
            else:
                points = sorted({p for t in tris for p in t})
        pts = [tuple(p) for p in points]
        index = {p: i for i, p in enumerate(pts)}
        try:
            idx = [tuple(index[p] for p in t) for t in tris]
        except KeyError as e:
            raise ValueError(f"triangle vertex {e.args[0]} is not a lattice point of the polygon") from None
        return cls(pts, idx, convex=convex)

    # -- canonical form
    def canonical(self) -> frozenset:
        """Coordinate-level canonical form: frozenset of sorted coordinate triples."""
        return frozenset(tuple(sorted(self.points[i] for i in t)) for t in self.triangles)

    def __eq__(self, other):
        return isinstance(other, Triangulation) and self.canonical() == other.canonical()

    def __hash__(self):
        return hash(self.canonical())

    def __repr__(self):
        tris = [tuple(self.points[i] for i in t) for t in self.triangles]
        return f"Triangulation(points={self.points}, triangles={tris})"

    def sorted(self) -> "Triangulation":
        """Same triangulation with triangles in canonical (sorted, rotated) order."""
        return Triangulation(self.points, sorted(_canon_tri(t) for t in self.triangles), self.convex)

    # -- combinatorics
    def corners(self) -> list[Point]:
        """Vertices of the convex hull of the point set, counterclockwise."""
        return convex_hull(self.points)

    def _edge_map(self) -> dict[tuple[int, int], list[int]]:
        em: dict[tuple[int, int], list[int]] = {}
        for k, t in enumerate(self.triangles):
            for i in range(3):
                a, b = t[i], t[(i + 1) % 3]
                em.setdefault((min(a, b), max(a, b)), []).append(k)
        return em

    def interior_edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Sorted list of ((i, j), (k, k')) : interior segment (i < j) and its two triangles (k < k')."""
        return sorted((e, tuple(sorted(ts))) for e, ts in self._edge_map().items() if len(ts) == 2)

    def boundary_edges(self) -> list[tuple[tuple[int, int], int]]:
        """Sorted list of ((i, j), k) : boundary segment (i < j) and its unique triangle."""
        return sorted((e, ts[0]) for e, ts in self._edge_map().items() if len(ts) == 1)

    def third_vertex(self, k: int, edge) -> int:
        """Index of the vertex of triangle k not on the segment `edge`."""
        (v,) = set(self.triangles[k]) - set(edge)
        return v

    def validate(self) -> "Triangulation":
        """Check unimodularity, covering and the edge structure; raise ValueError otherwise."""
        P = self.points
        if len(set(P)) != len(P):
            raise ValueError("repeated points")
        if not self.triangles:
            raise ValueError("no triangles")
        for t in self.triangles:
            if len(set(t)) != 3 or not all(0 <= i < len(P) for i in t):
                raise ValueError(f"bad triangle {t}")
            a = orient(P[t[0]], P[t[1]], P[t[2]])
            if a != 1:
                raise ValueError(f"non-unimodular: orbifold vertex not implemented "
                                 f"(triangle {[P[i] for i in t]} has twice-area {a})")
        if len(set(frozenset(t) for t in self.triangles)) != len(self.triangles):
            raise ValueError("repeated triangle (overlap)")
        em = self._edge_map()
        hull = convex_hull(P)
        for (i, j), ts in em.items():
            if len(ts) > 2:
                raise ValueError(f"segment {P[i]}-{P[j]} lies in {len(ts)} triangles (overlap)")
            on_bd = (_in_hull(P[i], hull) == 0 and _in_hull(P[j], hull) == 0 and
                     any(orient(hull[k], hull[(k + 1) % len(hull)], P[i]) == 0 and
                         orient(hull[k], hull[(k + 1) % len(hull)], P[j]) == 0 for k in range(len(hull))))
            if len(ts) == 2:
                if on_bd:
                    raise ValueError(f"boundary segment {P[i]}-{P[j]} lies in 2 triangles (overlap)")
                o = [orient(P[i], P[j], P[self.third_vertex(k, (i, j))]) for k in ts]
                if o[0] * o[1] >= 0:
                    raise ValueError(f"triangles on segment {P[i]}-{P[j]} lie on the same side (overlap)")
            elif self.convex and not on_bd:
                raise ValueError(f"interior segment {P[i]}-{P[j]} lies in only 1 triangle (not covering)")
        tot = len(self.triangles)
        if self.convex:
            if tot != area2(hull):
                raise ValueError(f"triangles do not cover the polygon: total twice-area {tot} "
                                 f"!= polygon twice-area {area2(hull)}")
            lp = lattice_points(hull)
            if set(lp) != set(P):
                raise ValueError("points must be exactly the lattice points of the polygon")
        else:
            tps = [tuple(P[i] for i in t) for t in self.triangles]
            for x, y in combinations(range(len(tps)), 2):
                if _tri_overlap(tps[x], tps[y]):
                    raise ValueError(f"triangles {tps[x]} and {tps[y]} overlap")
        return self

    def is_flippable(self, edge) -> bool:
        """True if `edge` is interior and the two adjacent triangles form a strictly convex
        quadrilateral, so that the diagonal can be flipped (the flipped triangles are then
        again unimodular)."""
        i, j = min(edge), max(edge)
        ts = self._edge_map().get((i, j), [])
        if len(ts) != 2:
            return False
        k, kp = ts
        a, b = self.third_vertex(k, (i, j)), self.third_vertex(kp, (i, j))
        P = self.points
        # convex quadrilateral i, a, j, b: segment a-b must cross the open segment i-j
        return orient(P[a], P[b], P[i]) * orient(P[a], P[b], P[j]) < 0

    def flip(self, edge) -> "Triangulation":
        """Replace the diagonal ``edge`` of its (strictly convex) quadrilateral by the other one."""
        i, j = min(edge), max(edge)
        if not self.is_flippable((i, j)):
            raise ValueError(f"edge {(i, j)} is not flippable")
        k, kp = self._edge_map()[(i, j)]
        a, b = self.third_vertex(k, (i, j)), self.third_vertex(kp, (i, j))
        tris = [t for n, t in enumerate(self.triangles) if n not in (k, kp)]
        tris += [(a, b, i), (a, b, j)]
        return Triangulation(self.points, tris, self.convex)

    def flippable_edges(self) -> list[tuple[int, int]]:
        """Interior segments whose diagonal can be flipped (see :meth:`is_flippable`)."""
        return [e for e, _ in self.interior_edges() if self.is_flippable(e)]

    def transformed(self, A, t=(0, 0)) -> "Triangulation":
        """Image under x -> A x + t with A in GL(2, Z) (det = -1 reverses orientation;
        triangles are re-oriented ccw)."""
        (p, q), (r, s) = A
        if abs(p * s - q * r) != 1:
            raise ValueError("A must be in GL(2,Z)")
        pts = [(p * x + q * y + t[0], r * x + s * y + t[1]) for x, y in self.points]
        return Triangulation(pts, self.triangles, self.convex)


# ----------------------------------------------------------------------------- enumeration

def initial_triangulation(points) -> list[tuple[int, int, int]]:
    """A triangulation of the point set using ALL points as vertices (lexicographic placing /
    incremental hull construction).  Its triangles contain no other points, hence are
    unimodular when ``points`` are all lattice points of a lattice polygon (Pick)."""
    P = list(points)
    order = sorted(range(len(P)), key=lambda i: P[i])
    k = 2
    while k < len(order) and orient(P[order[0]], P[order[1]], P[order[k]]) == 0:
        k += 1
    if k == len(order):
        raise ValueError("all points are collinear")
    apex = order[k]
    line = order[:k]
    tris = [_ccw(P, (line[m], line[m + 1], apex)) for m in range(k - 1)]
    # hull as ccw cycle (keeping all collinear boundary points)
    hull = line + [apex] if orient(P[line[0]], P[line[-1]], P[apex]) > 0 else [apex] + line[::-1]
    if orient(P[hull[0]], P[hull[1]], P[hull[2]]) < 0:
        hull = hull[::-1]
    for p in order[k + 1:]:
        n = len(hull)
        vis = [orient(P[hull[m]], P[hull[(m + 1) % n]], P[p]) < 0 for m in range(n)]
        # rotate so that the visible chain is contiguous and starts at index 0
        s = next(m for m in range(n) if vis[m] and not vis[m - 1])
        hull = hull[s:] + hull[:s]
        vis = vis[s:] + vis[:s]
        e = 0
        while e < n and vis[e]:
            tris.append(_ccw(P, (hull[e], hull[(e + 1) % n], p)))
            e += 1
        # hull[0] .. hull[e] is the visible chain; replace interior by p
        hull = [hull[0], p] + hull[e:]
    return tris


def all_unimodular_triangulations(points_or_corners, convex_check: bool = True) -> list[Triangulation]:
    """All unimodular triangulations of the lattice polygon conv(points_or_corners).

    Breadth-first search on the flip graph (Lawson: the flip graph of triangulations of a planar
    point set using all points is connected; for all lattice points of a lattice polygon these
    are exactly the unimodular triangulations).  Flips of interior edges whose quadrilateral is
    strictly convex.  Output: canonicalised triangulations in BFS order from the placing
    triangulation; the point order is :func:`lattice_points` order.
    """
    pts = lattice_points(points_or_corners)
    start = Triangulation(pts, initial_triangulation(pts)).sorted()
    if convex_check:
        start.validate()
    seen = {start.canonical(): start}
    out = [start]
    queue = deque([start])
    while queue:
        T = queue.popleft()
        for e in T.flippable_edges():
            U = T.flip(e).sorted()
            c = U.canonical()
            if c not in seen:
                seen[c] = U
                out.append(U)
                queue.append(U)
    return out


# ----------------------------------------------------------------------------- regularity

def _edge_relations(tri: Triangulation) -> list[tuple[int, ...]]:
    """Relation vectors l_tau of all interior edges (orientation-independent: l_tau is
    symmetric under swapping the two triangles, cf. KP a1 <-> a2 with w1 <-> w2)."""
    out = []
    n = len(tri.points)
    for (i, j), (k, kp) in tri.interior_edges():
        w3, w3p = tri.third_vertex(k, (i, j)), tri.third_vertex(kp, (i, j))
        out.append(relation_vector(n, i, j, w3, w3p, tri.points))
    return out


def _check_heights(h, rels) -> bool:
    return all(sum(a * b for a, b in zip(h, l)) > 0 for l in rels)


def lp_backend() -> str:
    """Solver used by :func:`heights`: ``"scipy"`` (HiGHS via ``scipy.optimize.linprog``) if scipy
    is importable and the environment variable ``TOPVERTEX_NO_SCIPY`` is unset, empty or ``0``;
    otherwise ``"exact"`` (a pure-Python rational simplex, :func:`_heights_exact`)."""
    if os.environ.get("TOPVERTEX_NO_SCIPY", "") not in ("", "0"):
        return "exact"
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        return "exact"
    return "scipy"


def heights(tri: Triangulation) -> list[int] | None:
    """Integer heights h (one per point) whose lower convex hull induces ``tri``, or None if
    ``tri`` is not regular.

    Condition (local convexity, sufficient on a convex support): for every interior edge,
    ``<h, l_tau> > 0`` with l_tau the relation vector (+1 at w3, w3', a1 at w1, a2 at w2).
    Derivation: the affine function f agreeing with h on w1, w2, w3 has
    f(w3') = -a1 h1 - a2 h2 - h3, and convexity across tau means h(w3') > f(w3').
    E.g. the conifold square with diagonal (1,0)-(0,1): l = +1 at (0,0),(1,1), -1 at (1,0),(0,1),
    heights 1 at (1,1), 0 elsewhere.  Solved as the LP  min sum h, <h,l> >= 1, h >= 0  by scipy
    linprog (floating point; rounded to integers and re-verified exactly) or, without scipy (see
    :func:`lp_backend`), exactly over Q by a rational simplex and scaled to integers.  The two
    solvers may return different optimal vertices, hence different (equally valid) heights.
    """
    rels = _edge_relations(tri)
    n = len(tri.points)
    if not rels:
        return [0] * n
    if lp_backend() == "exact":
        return _heights_exact(rels, n)
    return _heights_scipy(rels, n)


def _heights_exact(rels, n: int) -> list[int] | None:
    """Exact LP  min sum h, <h, l> >= 1 (l in rels), h >= 0  over Q, scaled to integers.

    Solved through its dual  max sum y, sum_i y_i l_i <= 1 (componentwise), y >= 0, whose origin
    is feasible, so a single-phase tableau simplex suffices; Bland's rule guarantees termination
    on these highly degenerate problems.  The primal optimum h is read off the reduced costs of
    the dual slack variables; the primal is infeasible (``tri`` not regular) iff the dual is
    unbounded.  (sympy.solvers.simplex was tried first but returns infeasible points on
    degenerate instances such as the SU(3) ladder, so it is not used.)"""
    rels = sorted(set(rels))
    m = len(rels)
    ncol = m + n
    # tableau rows j = 0..n-1:  sum_i rels[i][j] y_i + s_j = 1;  basis = slacks
    rows = [[Fraction(rels[i][j]) for i in range(m)] + [Fraction(int(k == j)) for k in range(n)]
            + [Fraction(1)] for j in range(n)]
    basis = [m + j for j in range(n)]
    cost = [Fraction(1)] * m + [Fraction(0)] * n  # reduced costs (maximisation)
    while True:
        enter = next((c for c in range(ncol) if cost[c] > 0), None)
        if enter is None:
            break
        best = None
        for r, row in enumerate(rows):
            a = row[enter]
            if a > 0:
                key = (row[-1] / a, basis[r])
                if best is None or key < best[0]:
                    best = (key, r)
        if best is None:
            return None  # dual unbounded <=> no heights
        r = best[1]
        piv = rows[r]
        inv = 1 / piv[enter]
        piv = rows[r] = [v * inv for v in piv]
        for k, row in enumerate(rows):
            f = row[enter]
            if k != r and f:
                rows[k] = [x - f * y for x, y in zip(row, piv)]
        f = cost[enter]
        cost = [x - f * y for x, y in zip(cost, piv)]
        basis[r] = enter
    fr = [-cost[m + j] for j in range(n)]
    L = math.lcm(*(f.denominator for f in fr))
    h = [int(f * L) for f in fr]
    if not _check_heights(h, rels):  # pragma: no cover (LP duality guarantees <h, l> >= 1)
        raise AssertionError("exact LP for the heights returned an infeasible point")
    return _normalise_heights(h)


def _heights_scipy(rels, n: int) -> list[int] | None:
    """Floating-point LP (scipy HiGHS), rounded to integers and re-verified exactly."""
    from scipy.optimize import linprog

    res = linprog(c=[1.0] * n, A_ub=[[-float(x) for x in l] for l in rels], b_ub=[-1.0] * len(rels),
                  bounds=[(0, None)] * n, method="highs")
    if res.status != 0:
        return None
    x = list(res.x)
    # 1) try small rationals
    for den in (1, 2, 3, 4, 6, 12, 60, 1000):
        fr = [Fraction(v).limit_denominator(den) for v in x]
        L = 1
        for f in fr:
            L = L * f.denominator // math.gcd(L, f.denominator)
        h = [int(f * L) for f in fr]
        if _check_heights(h, rels):
            return _normalise_heights(h)
    # 2) guaranteed: <x,l> >= 1 and |round(Kx) - Kx| <= 1/2 give <round(Kx),l> >= K - |l|_1/2
    K = max(sum(abs(c) for c in l) for l in rels) // 2 + 1
    while True:
        h = [round(K * v) for v in x]
        if _check_heights(h, rels):
            return _normalise_heights(h)
        K *= 2  # pragma: no cover (only if the LP solution violates constraints numerically)


def _normalise_heights(h):
    """Shift to min 0 (sum l_i = 0, so <h, l> is unchanged) and divide by the gcd."""
    m = min(h)
    h = [v - m for v in h]
    g = 0
    for v in h:
        g = math.gcd(g, v)
    return [v // g for v in h] if g > 1 else h


def is_regular(tri: Triangulation) -> bool:
    """True iff ``tri`` is induced by a lifting (see :func:`heights`)."""
    return heights(tri) is not None


def _solve_rational(rows, rhs):
    """Exact Gaussian elimination: one solution x of rows . x = rhs over Q, or None."""
    m = [[Fraction(v) for v in r] + [Fraction(b)] for r, b in zip(rows, rhs)]
    ncol = len(rows[0]) if rows else 0
    piv = []
    r = 0
    for c in range(ncol):
        p = next((k for k in range(r, len(m)) if m[k][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        inv = 1 / m[r][c]
        m[r] = [v * inv for v in m[r]]
        for k in range(len(m)):
            if k != r and m[k][c] != 0:
                f = m[k][c]
                m[k] = [a - f * b for a, b in zip(m[k], m[r])]
        piv.append(c)
        r += 1
    if any(all(v == 0 for v in row[:-1]) and row[-1] != 0 for row in m):
        return None
    x = [Fraction(0)] * ncol
    for k, c in enumerate(piv):
        x[c] = m[k][-1]
    return x


def all_ones_weights_consistent(tri: Triangulation) -> bool:
    """True iff there are heights h with <h, l_tau> = 1 for every interior edge, i.e. assigning
    Kahler weight 1 to every compact curve C_tau is consistent with the linear relations among
    the classes (true for local P^2, F0; false for F1 since [section] = [B] + [F])."""
    rels = _edge_relations(tri)
    if not rels:
        return True
    return _solve_rational(rels, [1] * len(rels)) is not None


# ----------------------------------------------------------------------------- plotting

def plot(tri: Triangulation, ax=None, show_web: bool = True):
    """Draw the polygon, its lattice points and the triangulation; with ``show_web`` a second
    panel shows the dual web: one node per triangle (at its centroid), internal edges between
    adjacent triangles and external legs along the outward normals of boundary sides."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axs = plt.subplots(1, 2 if show_web else 1, figsize=(10 if show_web else 5, 5))
        axs = list(axs) if show_web else [axs]
    else:
        axs = list(ax) if isinstance(ax, (list, tuple)) else [ax]
        fig = axs[0].figure
    P = tri.points
    a0 = axs[0]
    hull = convex_hull(P)
    a0.fill([p[0] for p in hull], [p[1] for p in hull], color="0.92", zorder=0)
    for t in tri.triangles:
        xs = [P[i][0] for i in t] + [P[t[0]][0]]
        ys = [P[i][1] for i in t] + [P[t[0]][1]]
        a0.plot(xs, ys, "k-", lw=1)
    a0.scatter([p[0] for p in P], [p[1] for p in P], c="k", zorder=3)
    for i, p in enumerate(P):
        a0.annotate(str(i), p, textcoords="offset points", xytext=(4, 4), fontsize=8)
    a0.set_aspect("equal")
    a0.set_title("triangulation")
    if show_web and len(axs) > 1:
        a1 = axs[1]
        cen = [(sum(P[i][0] for i in t) / 3, sum(P[i][1] for i in t) / 3) for t in tri.triangles]
        for (i, j), (k, kp) in tri.interior_edges():
            a1.plot([cen[k][0], cen[kp][0]], [cen[k][1], cen[kp][1]], "b-", lw=1.5)
            a1.annotate(f"{i}-{j}", ((cen[k][0] + cen[kp][0]) / 2, (cen[k][1] + cen[kp][1]) / 2), fontsize=7)
        for (i, j), k in tri.boundary_edges():
            t = tri.triangles[k]
            # oriented side of the ccw triangle
            a, b = (i, j) if any(t[m] == i and t[(m + 1) % 3] == j for m in range(3)) else (j, i)
            dx, dy = P[b][0] - P[a][0], P[b][1] - P[a][1]
            nx, ny = dy, -dx
            s = 0.6 / math.hypot(nx, ny)
            a1.plot([cen[k][0], cen[k][0] + s * nx], [cen[k][1], cen[k][1] + s * ny], "b-", lw=1)
        a1.scatter([c[0] for c in cen], [c[1] for c in cen], c="b", zorder=3)
        a1.set_aspect("equal")
        a1.set_title("dual web (centroid placeholder positions)")
    return fig
