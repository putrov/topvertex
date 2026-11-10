"""GT graphs (toric graphs / (p,q)-webs) of toric Calabi-Yau threefolds.

Konishi, math/0504188, Sec. 4.2 (conventions fixed in CONVENTIONS.md):

* one vertex v_sigma per triangle sigma of a unimodular triangulation;
* the three legs of a vertex are the three sides of the triangle in COUNTERCLOCKWISE order; the
  (p,q)-vector of a leg is the outward normal of the side: for the side w_a -> w_b of a ccw
  triangle, (dx, dy) = w_b - w_a  |->  (dy, -dx);
* one internal edge e_tau per interior segment tau = (w1, w2) shared by sigma (third vertex w3)
  and sigma' (third vertex w3'); boundary segments are external legs (empty partition);
* writing w3' = -a1 w1 - a2 w2 - w3 for the 3-vectors (x, y, 1) with (w1, w2, w3) ccw, the
  framing of the flag (v_sigma, e_tau) is n_f = (a1 - a2)/2 = a1 + 1, and n_{-f} = -n_f;
* each internal edge is directed; its label n_e is the framing of the flag at its TAIL;
* curve class l_tau in Z^{#points}: +1 at w3, w3', a1 at w1, a2 at w2.

Web-side formula (derived here, pinned by the round-trip test web_of <-> from_web), AKMV
hep-th/0305132 eq. (3.15): with all leg vectors taken OUTGOING from their vertex, for an edge
from vertex a (leg i) to vertex b (leg j)

    n_e = v_{k'} ^ v_k,     v_k = leg of a preceding leg i in ccw order (index i-1 mod 3),
                            v_{k'} = leg of b preceding leg j in ccw order (index j-1 mod 3),

and (x1, y1) ^ (x2, y2) = x1 y2 - y1 x2.  See :func:`web_framing` for the derivation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import polygon as pg
from .polygon import Triangulation, orient, relation_coefficients, relation_vector

Vec = tuple[int, int]

__all__ = [
    "Leg", "Edge", "GTGraph", "wedge", "web_framing", "from_triangulation", "from_web", "web_of",
    "flag_framings", "relation_lattice_basis", "class_coordinates", "kahler_weights",
]


def wedge(u: Vec, v: Vec) -> int:
    """u ^ v = u_x v_y - u_y v_x."""
    return u[0] * v[1] - u[1] * v[0]


def _rot(d: Vec) -> Vec:
    """Outward normal of a side (dx, dy) of a ccw triangle: (dy, -dx)."""
    return (d[1], -d[0])


def _unrot(v: Vec) -> Vec:
    """Inverse of :func:`_rot`: (p, q) -> (-q, p)."""
    return (-v[1], v[0])


@dataclass
class Leg:
    """One of the three legs at a vertex.  ``kind``: 'int' (internal edge, ``edge`` indexes
    GTGraph.edges) or 'ext' (``edge`` indexes GTGraph.external_legs).  ``outgoing``: the edge is
    directed away from this vertex (always True for external legs).  ``vector``: outward (p,q)
    direction of the leg at this vertex (= outward normal of the triangle side)."""
    kind: str
    edge: int
    outgoing: bool
    vector: Vec


@dataclass
class Edge:
    """Internal edge from vertex ``tail`` to vertex ``head``.  ``n``: framing label = framing of
    the flag at the TAIL (KP Sec. 4.2).  ``segment``: interior segment (i, j) (point indices, i < j)
    or None for web input.  ``curve_class``: relation vector l_tau (None if no points).
    ``tail_leg_index`` / ``head_leg_index``: position of the edge among the 3 legs at tail / head."""
    tail: int
    head: int
    n: int
    segment: tuple[int, int] | None
    name: str
    curve_class: tuple[int, ...] | None
    tail_leg_index: int
    head_leg_index: int


@dataclass
class GTGraph:
    """A GT graph (KP Def. 2.1 / toric graph Sec. 4.2).

    ``vertices[v]``: its 3 legs in COUNTERCLOCKWISE order.  ``edges``: internal edges.
    ``external_legs[k] = (vertex, vector)``.  ``points``: lattice points (None for pure web input),
    ``n_points`` = len(points).  ``triangulation``: the Triangulation (vertex v = triangle v) when
    available.  ``basis_edge_indices``: set by :func:`relation_lattice_basis` when the basis
    consists of edge curve classes.
    """
    vertices: list[list[Leg]]
    edges: list[Edge]
    external_legs: list[tuple[int, Vec]]
    points: list[tuple[int, int]] | None = None
    n_points: int | None = None
    triangulation: Triangulation | None = None
    basis_edge_indices: list[int] | None = field(default=None)

    def edge_names(self) -> list[str]:
        """Variable names Q_e of the internal edges, in edge order."""
        return [e.name for e in self.edges]

    def framings(self) -> list[int]:
        """Labels n_e (framing of the flag at the tail of each edge)."""
        return [e.n for e in self.edges]

    def curve_classes(self) -> list[tuple[int, ...] | None]:
        """Relation vectors l_e (curve classes in H_2) of the internal edges, or None for web input."""
        return [e.curve_class for e in self.edges]

    def check(self) -> "GTGraph":
        """Consistency: every internal edge occurs exactly twice (once outgoing at its tail, once
        incoming at its head, with opposite vectors), every external leg once; at each vertex the
        leg vectors sum to zero and consecutive legs (ccw) have positive wedge; the web-side
        framing formula reproduces the stored labels."""
        seen_int: dict[int, list[tuple[int, int, bool]]] = {}
        seen_ext: dict[int, int] = {}
        for v, legs in enumerate(self.vertices):
            if len(legs) != 3:
                raise ValueError(f"vertex {v} is not trivalent")
            if tuple(map(sum, zip(*[l.vector for l in legs]))) != (0, 0):
                raise ValueError(f"leg vectors at vertex {v} do not sum to zero")
            for m in range(3):
                if wedge(legs[m].vector, legs[(m + 1) % 3].vector) <= 0:
                    raise ValueError(f"legs at vertex {v} are not counterclockwise")
            for m, l in enumerate(legs):
                if l.kind == "int":
                    seen_int.setdefault(l.edge, []).append((v, m, l.outgoing))
                elif l.kind == "ext":
                    if not l.outgoing:
                        raise ValueError("external legs must be outgoing")
                    seen_ext[l.edge] = seen_ext.get(l.edge, 0) + 1
                else:
                    raise ValueError(f"bad leg kind {l.kind!r}")
        if sorted(seen_int) != list(range(len(self.edges))):
            raise ValueError("internal edges and legs do not match")
        for e, occ in seen_int.items():
            E = self.edges[e]
            if len(occ) != 2 or sorted(o for _, _, o in occ) != [False, True]:
                raise ValueError(f"edge {e} must appear once outgoing and once incoming")
            (vt, mt, _), = [x for x in occ if x[2]]
            (vh, mh, _), = [x for x in occ if not x[2]]
            if (vt, mt, vh, mh) != (E.tail, E.tail_leg_index, E.head, E.head_leg_index):
                raise ValueError(f"edge {e}: tail/head data inconsistent with legs")
            vt_vec, vh_vec = self.vertices[vt][mt].vector, self.vertices[vh][mh].vector
            if (vt_vec[0] + vh_vec[0], vt_vec[1] + vh_vec[1]) != (0, 0):
                raise ValueError(f"edge {e}: leg vectors at the two ends are not opposite")
            nw = web_framing(self.vertices[vt][(mt - 1) % 3].vector, self.vertices[vh][(mh - 1) % 3].vector)
            if nw != E.n:
                raise ValueError(f"edge {e}: label n={E.n} but web formula gives {nw}")
        if sorted(seen_ext) != list(range(len(self.external_legs))) or any(c != 1 for c in seen_ext.values()):
            raise ValueError("external legs inconsistent")
        for k, (v, vec) in enumerate(self.external_legs):
            if not any(l.kind == "ext" and l.edge == k and l.vector == tuple(vec) for l in self.vertices[v]):
                raise ValueError(f"external leg {k} inconsistent")
        return self


def web_framing(v_k: Vec, v_kp: Vec) -> int:
    """Framing label of an internal edge from the web (AKMV (3.15) in KP conventions).

    ``v_k``: the leg at the TAIL vertex immediately preceding the edge in ccw order;
    ``v_kp``: the leg at the HEAD vertex immediately preceding the edge in ccw order; both
    outgoing from their own vertex.  Returns ``n_e = v_kp ^ v_k``.

    Derivation: the map side -> outward normal, d -> (d_y, -d_x), is a rotation, so it preserves
    ^.  At the tail triangle (w1, w2, w3) (ccw, edge = side w1w2) the preceding side is w3w1, so
    v_k = rot(w1 - w3); at the head triangle (w2, w1, w3') (ccw) the preceding side is w3'w2,
    v_k' = rot(w2 - w3').  Hence n = (w2 - w3') ^ (w1 - w3).  Both sides are invariant under
    affine SL(2,Z), so take w1 = (0,0), w2 = (1,0), w3 = (0,1), w3' = (a,-1): then
    (w2 - w3') ^ (w1 - w3) = (1-a, 1) ^ (0, -1) = a - 1, while w3' = -a1 w1 - a2 w2 - w3 gives
    a2 = -a, a1 = a - 2, n_f = a1 + 1 = a - 1.  This is AKMV's n_i = |v_k' ^ v_k| with both
    v_k, v_k' outgoing, AKMV's v_i being outgoing at the tail -- with the STANDARD determinant
    orientation (x1 y2 - y1 x2).  AKMV normalise their wedge by v_2 ^ v_1 = 1 for ccw-ordered
    legs (opposite orientation) and accordingly write q^{-n kappa/2} in (3.15), while Konishi
    writes q^{+n kappa/2}; the two agree.  Flipping the sign of n is not harmless (local P^2
    would give n^0_3 = 15 instead of 27).
    """
    return wedge(v_kp, v_k)


# ----------------------------------------------------------------------------- constructors

def from_triangulation(tri: Triangulation) -> GTGraph:
    """GT graph of a unimodular triangulation (KP Sec. 4.2, CONVENTIONS.md).

    Vertex v = triangle v (stored ccw (a, b, c)); legs = sides (a,b), (b,c), (c,a) with vector
    = outward normal.  Internal edges in the order of ``tri.interior_edges()`` (sorted segments),
    directed from the lower- to the higher-index triangle; n_e = a1 + 1 computed at the tail with
    (w1, w2, w3) ccw in the tail triangle.  External legs ordered by (vertex, leg position).
    """
    tri.validate()
    P = tri.points
    T = tri.triangles
    nP = len(P)
    sides = [[(t[m], t[(m + 1) % 3]) for m in range(3)] for t in T]
    vecs = [[_rot((P[b][0] - P[a][0], P[b][1] - P[a][1])) for a, b in s] for s in sides]
    legs: list[list[Leg | None]] = [[None] * 3 for _ in T]
    edges: list[Edge] = []
    for e, ((i, j), (k, kp)) in enumerate(tri.interior_edges()):
        mt = next(m for m in range(3) if set(sides[k][m]) == {i, j})
        mh = next(m for m in range(3) if set(sides[kp][m]) == {i, j})
        w1, w2 = sides[k][mt]
        w3 = T[k][(mt + 2) % 3]
        w3p = T[kp][(mh + 2) % 3]
        a1, a2 = relation_coefficients(P[w1], P[w2], P[w3], P[w3p])
        cls = relation_vector(nP, w1, w2, w3, w3p, P)
        edges.append(Edge(tail=k, head=kp, n=a1 + 1, segment=(i, j), name=f"Q{e}", curve_class=cls,
                          tail_leg_index=mt, head_leg_index=mh))
        legs[k][mt] = Leg("int", e, True, vecs[k][mt])
        legs[kp][mh] = Leg("int", e, False, vecs[kp][mh])
    ext: list[tuple[int, Vec]] = []
    for v in range(len(T)):
        for m in range(3):
            if legs[v][m] is None:
                legs[v][m] = Leg("ext", len(ext), True, vecs[v][m])
                ext.append((v, vecs[v][m]))
    g = GTGraph(vertices=legs, edges=edges, external_legs=ext, points=list(P), n_points=nP, triangulation=tri)
    return g.check()


def web_of(graph: GTGraph) -> tuple[list[list[Vec]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    """(vertex_vectors, gluings) of a GT graph, for :func:`from_web`.  Gluing e is
    ((tail, tail_leg_index), (head, head_leg_index)) of edge e."""
    vv = [[tuple(l.vector) for l in legs] for legs in graph.vertices]
    gl = [((e.tail, e.tail_leg_index), (e.head, e.head_leg_index)) for e in graph.edges]
    return vv, gl


def _reconstruct(vertex_vectors, gluings):
    """Rebuild the dual triangles (up to translation) from the web: leg vector v of a side is
    rot(d) for the side vector d, so d = (-q, p); corners of vertex 0 start at (0,0)."""
    nv = len(vertex_vectors)
    adj: dict[int, list] = {}
    for (a, i), (b, j) in gluings:
        adj.setdefault(a, []).append((i, b, j))
        adj.setdefault(b, []).append((j, a, i))
    corners: list[list | None] = [None] * nv

    def place(v, start_idx, start_pt):
        c = [None] * 3
        c[start_idx] = start_pt
        for s in range(2):
            m = (start_idx + s) % 3
            d = _unrot(vertex_vectors[v][m])
            c[(m + 1) % 3] = (c[m][0] + d[0], c[m][1] + d[1])
        return c

    for root in range(nv):
        if corners[root] is not None:
            continue
        if root != 0:
            raise ValueError("web is disconnected; give a connected web")
        corners[root] = place(root, 0, (0, 0))
        stack = [root]
        while stack:
            a = stack.pop()
            for i, b, j in adj.get(a, []):
                # side i of a runs corner_a[i] -> corner_a[i+1]; side j of b runs the other way
                start = corners[a][(i + 1) % 3]
                cb = place(b, j, start)
                if cb[(j + 1) % 3] != corners[a][i]:
                    raise ValueError("inconsistent gluing")
                if corners[b] is None:
                    corners[b] = cb
                    stack.append(b)
                elif corners[b] != cb:
                    raise ValueError("web does not close up consistently")
    return corners


def from_web(vertex_vectors, gluings, n_points: int | None = None) -> GTGraph:
    """GT graph from a (p,q)-web.

    ``vertex_vectors[v]`` = the three OUTGOING primitive (p,q) vectors at vertex v in ccw order
    (sum zero, consecutive wedges positive; = 1 for a smooth vertex).  ``gluings[e] =
    ((a, i), (b, j))`` glues leg i of vertex a to leg j of vertex b (opposite vectors) into
    internal edge e directed from a to b.  Unglued legs become external legs, ordered by
    (vertex, leg position).

    Framing label (AKMV (3.15) with KP's conventions; see :func:`web_framing`):
    ``n_e = v_b[j-1] ^ v_a[i-1]``.

    If ``n_points`` is given, the dual triangles are reconstructed (translation fixed by putting
    a corner of vertex 0 at the origin), points are sorted, and curve classes are computed; the
    number of distinct points must equal ``n_points``.  Otherwise ``points`` and all
    ``curve_class`` are None.
    """
    vv = [[(int(p), int(q)) for p, q in vs] for vs in vertex_vectors]
    for v, vs in enumerate(vv):
        if len(vs) != 3 or tuple(map(sum, zip(*vs))) != (0, 0):
            raise ValueError(f"vertex {v}: need three vectors summing to zero")
        if any(wedge(vs[m], vs[(m + 1) % 3]) <= 0 for m in range(3)):
            raise ValueError(f"vertex {v}: vectors are not in counterclockwise order")
    legs: list[list[Leg | None]] = [[None] * 3 for _ in vv]
    edges: list[Edge] = []
    for e, ((a, i), (b, j)) in enumerate(gluings):
        if legs[a][i] is not None or legs[b][j] is not None:
            raise ValueError(f"gluing {e}: leg glued twice")
        if (vv[a][i][0] + vv[b][j][0], vv[a][i][1] + vv[b][j][1]) != (0, 0):
            raise ValueError(f"gluing {e}: vectors {vv[a][i]} and {vv[b][j]} are not opposite")
        n = web_framing(vv[a][(i - 1) % 3], vv[b][(j - 1) % 3])
        edges.append(Edge(tail=a, head=b, n=n, segment=None, name=f"Q{e}", curve_class=None,
                          tail_leg_index=i, head_leg_index=j))
        legs[a][i] = Leg("int", e, True, vv[a][i])
        legs[b][j] = Leg("int", e, False, vv[b][j])
    ext = []
    for v in range(len(vv)):
        for m in range(3):
            if legs[v][m] is None:
                legs[v][m] = Leg("ext", len(ext), True, vv[v][m])
                ext.append((v, vv[v][m]))
    g = GTGraph(vertices=legs, edges=edges, external_legs=ext)
    if n_points is not None:
        corners = _reconstruct(vv, gluings)
        pts = sorted({p for c in corners for p in c})
        if len(pts) != n_points:
            raise ValueError(f"web has {len(pts)} dual lattice points, expected {n_points}")
        idx = {p: k for k, p in enumerate(pts)}
        tris = [tuple(idx[p] for p in c) for c in corners]
        convex = (len(tris) == pg.area2(pg.convex_hull(pts)) and len(pts) == len(pg.lattice_points(pts)))
        tri = Triangulation(pts, tris, convex=convex).validate()
        if tri.triangles != tris:
            raise ValueError("reconstructed web triangles are not counterclockwise")
        for E in edges:
            c = corners[E.tail]
            w1, w2, w3 = (idx[c[(E.tail_leg_index + s) % 3]] for s in range(3))
            w3p = idx[corners[E.head][(E.head_leg_index + 2) % 3]]
            a1, _ = relation_coefficients(pts[w1], pts[w2], pts[w3], pts[w3p])
            if a1 + 1 != E.n:
                raise ValueError("web framing formula disagrees with the triangulation rule")
            E.segment = (min(w1, w2), max(w1, w2))
            E.curve_class = relation_vector(n_points, w1, w2, w3, w3p, pts)
        g.points, g.n_points, g.triangulation = pts, n_points, tri
    return g.check()


def flag_framings(graph: GTGraph) -> dict[tuple[int, int], int]:
    """{(vertex, edge): framing of the flag}: n_e at the tail, -n_e at the head (KP Sec. 4.2)."""
    out = {}
    for k, e in enumerate(graph.edges):
        out[(e.tail, k)] = e.n
        out[(e.head, k)] = -e.n
    return out


# ----------------------------------------------------------------------------- curve classes

def _triangle_basis(graph: GTGraph):
    """Reference unimodular triangle (first triangle) and, for every other point s, the integer
    relation b_s = e_s - (c_p e_p + c_q e_q + c_r e_r), w_s = c_p w_p + c_q w_q + c_r w_r.
    The b_s form a Z-basis of the relation lattice L (their s-components are the identity), and
    the coordinates of any l in L in this basis are its components at the non-reference points."""
    if graph.points is None:
        raise ValueError("curve classes need lattice points (build from a triangulation or pass n_points)")
    P = graph.points
    n = len(P)
    p, q, r = graph.triangulation.triangles[0] if graph.triangulation else _find_unimodular(P)
    d = orient(P[p], P[q], P[r])
    if abs(d) != 1:
        raise ValueError("non-unimodular triangle in _triangle_basis")
    others = [s for s in range(n) if s not in (p, q, r)]
    basis = []
    for s in others:
        w = P[s]
        cp, cq, cr = orient(w, P[q], P[r]) * d, orient(P[p], w, P[r]) * d, orient(P[p], P[q], w) * d
        b = [0] * n
        b[s] = 1
        b[p] -= cp
        b[q] -= cq
        b[r] -= cr
        basis.append(tuple(b))
    return others, basis


def _find_unimodular(P):
    from itertools import combinations
    for a, b, c in combinations(range(len(P)), 3):
        if abs(orient(P[a], P[b], P[c])) == 1:
            return a, b, c
    raise ValueError("no unimodular triangle")


def _saturated_independent(rows) -> bool:
    """Rows are linearly independent and span a saturated sublattice (all invariant factors 1)."""
    from sympy import Matrix, ZZ
    from sympy.matrices.normalforms import invariant_factors
    M = Matrix(rows)
    if M.rank() != len(rows):
        return False
    return all(abs(int(f)) == 1 for f in invariant_factors(M, domain=ZZ))


def relation_lattice_basis(graph: GTGraph) -> list[tuple[int, ...]]:
    """Z-basis of L = {l in Z^{n_points} : sum_i l_i (x_i, y_i, 1) = 0} (rank n_points - 3),
    L = H_2(X, Z) (KP Sec. 4, CONVENTIONS.md).

    Preferred: edge curve classes, chosen greedily in order of increasing Kahler weight, then
    edge index (an edge class is added iff the
    enlarged set stays linearly independent and spans a saturated sublattice; saturation tested
    by Smith invariant factors, sympy).  If this reaches rank n_points - 3, the basis is these
    classes and ``graph.basis_edge_indices`` records the edges; otherwise the basis {b_s} of
    :func:`_triangle_basis` is returned and ``graph.basis_edge_indices = None``.
    """
    others, tbasis = _triangle_basis(graph)
    rank = len(others)
    chosen: list[int] = []
    rows: list[tuple[int, ...]] = []
    # Prefer edges of smallest Kahler weight (so that e.g. for F_m the basis is (B, F) rather
    # than (B + mF, F) and the other edge classes have non-negative coordinates).
    order = list(range(len(graph.edges)))
    try:
        w = kahler_weights(graph)
        order.sort(key=lambda k: (w[k], k))
    except (ValueError, TypeError):
        pass
    for k in order:
        e = graph.edges[k]
        if len(chosen) == rank:
            break
        coords = tuple(e.curve_class[s] for s in others)
        if _saturated_independent(rows + [coords]):
            rows.append(coords)
            chosen.append(k)
    if len(chosen) == rank:
        graph.basis_edge_indices = chosen
        return [tuple(graph.edges[k].curve_class) for k in chosen]
    graph.basis_edge_indices = None
    return tbasis


def class_coordinates(graph: GTGraph, basis=None) -> dict[int, tuple[int, ...]]:
    """{edge index: integer coordinates of l_e in ``basis``} (exact solve; raises if some class
    is not an integer combination).  Default basis: :func:`relation_lattice_basis`."""
    if basis is None:
        basis = relation_lattice_basis(graph)
    if not basis:
        return {k: () for k in range(len(graph.edges))}
    n = len(basis[0])
    rows = [[b[i] for b in basis] for i in range(n)]
    out = {}
    for k, e in enumerate(graph.edges):
        x = pg._solve_rational(rows, list(e.curve_class))
        if x is None or any(v.denominator != 1 for v in x):
            raise ValueError(f"class of edge {k} is not an integer combination of the basis")
        out[k] = tuple(int(v) for v in x)
    return out


def kahler_weights(graph: GTGraph, heights=None) -> list[int]:
    """Integer Kahler weights w_e = <h, l_e> > 0 (CONVENTIONS.md), with h = polygon.heights of the
    triangulation unless given."""
    if heights is None:
        if graph.triangulation is None:
            raise ValueError("need heights or a triangulation")
        heights = pg.heights(graph.triangulation)
        if heights is None:
            raise ValueError("triangulation is not regular")
    if graph.points is not None and len(heights) != len(graph.points):
        raise ValueError(f"heights must have one entry per lattice point ({len(graph.points)}), got {len(heights)}")
    w = [sum(a * b for a, b in zip(heights, e.curve_class)) for e in graph.edges]
    if not all(x > 0 for x in w):
        raise ValueError(f"heights {heights} do not give positive weights {w}")
    return w
