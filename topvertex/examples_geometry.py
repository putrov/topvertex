"""Standard toric Calabi-Yau geometries as GT graphs (KP Sec. 4.2 conventions, see CONVENTIONS.md).

Each constructor returns a checked :class:`~topvertex.toric.GTGraph` built by
:func:`~topvertex.toric.from_triangulation`; ``graph.triangulation`` is the triangulation.
Edges are ordered by sorted interior segments (point indices), directed from the lower- to the
higher-index triangle, named Q0, Q1, ...; the point order is chosen so that the edge order is
meaningful (documented per example).  Normal bundle of C_e: O(n_e - 1) + O(-n_e - 1).
"""
from __future__ import annotations

from .polygon import Triangulation
from .toric import GTGraph, from_triangulation

__all__ = ["conifold", "local_p2", "local_f0", "hirzebruch", "local_f1", "local_f2", "op1_bundle",
           "su_n_ladder", "kp_flop_f1", "local_dp3", "star_triangulation"]


def star_triangulation(centre, rays_ccw, convex: bool = True) -> Triangulation:
    """Star triangulation: points [centre] + rays (given ccw around the centre), triangles
    (centre, ray_k, ray_{k+1}).  Edge k = segment (0, k+1)."""
    pts = [tuple(centre)] + [tuple(r) for r in rays_ccw]
    n = len(rays_ccw)
    tris = [(0, 1 + k, 1 + (k + 1) % n) for k in range(n)]
    return Triangulation(pts, tris, convex=convex)


def conifold() -> GTGraph:
    """Resolved conifold O(-1)+O(-1) -> P^1: unit square (0,0),(1,0),(0,1),(1,1) with diagonal
    (1,0)-(0,1).  One edge, n = 0, class l = (1, -1, -1, 1)."""
    return from_triangulation(Triangulation([(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 1, 2), (1, 3, 2)]))


def local_p2() -> GTGraph:
    """K_{P^2}: points w0..w3 = (0,0),(1,0),(0,1),(-1,-1) (KP Fig. 3 left), star triangulation.
    Three edges (segments w0-w1, w0-w2, w0-w3), all in the class of a line H, l = (-3,1,1,1);
    framings n = (-2, 2, 2) (normal bundle O(1)+O(-3))."""
    pts = [(0, 0), (1, 0), (0, 1), (-1, -1)]
    return from_triangulation(Triangulation(pts, [(0, 1, 2), (0, 2, 3), (0, 3, 1)]))


def local_f0() -> GTGraph:
    """K_{P^1 x P^1}: centre (0,0), rays (1,0),(0,1),(-1,0),(0,-1).  Edges Q0..Q3 = spokes to
    (1,0),(0,1),(-1,0),(0,-1); Q0, Q2 in one class, Q1, Q3 in the other; all |n| = 1."""
    return from_triangulation(star_triangulation((0, 0), [(1, 0), (0, 1), (-1, 0), (0, -1)]))


def hirzebruch(m: int) -> GTGraph:
    """K_{F_m}, 0 <= m <= 2 (fan rays (1,0),(0,1),(-1,m),(0,-1)).  Points: centre (0,0), then
    (0,1), (-1,m), (0,-1), (1,0), so the edges are
      Q0: ray (0,1)  -> base B (B^2 = -m),
      Q1: ray (-1,m) -> fiber F,
      Q2: ray (0,-1) -> section B + mF (self-intersection +m),
      Q3: ray (1,0)  -> fiber F.
    A curve with self-intersection s in the surface has normal bundle O(s)+O(-2-s): |n| = |s+1|.
    (For m = 2 the point (0,1) lies on the boundary; the polygon is a triangle.)"""
    if not 0 <= m <= 2:
        raise ValueError("only 0 <= m <= 2 give a convex lattice polygon (K_{F_m} toric CY)")
    return from_triangulation(star_triangulation((0, 0), [(0, 1), (-1, m), (0, -1), (1, 0)]))


def local_f1() -> GTGraph:
    """K_{F_1}: :func:`hirzebruch` (1).  Edge classes B, F, B+F, F; n = 0, +-1, +-2, +-1
    (normal bundles (-1,-1), (0,-2), (1,-3), (0,-2))."""
    return hirzebruch(1)


def local_f2() -> GTGraph:
    """K_{F_2}: :func:`hirzebruch` (2).  Edge classes B, F, B+2F, F; |n| = 1, 1, 3, 1."""
    return hirzebruch(2)


def op1_bundle(n: int) -> GTGraph:
    """O(n-1) + O(-n-1) -> P^1: two triangles (w1, w2, w3) = ((0,0),(1,0),(0,1)) and
    (w2, w1, w3') with w3' = (n+1, -1); the single edge has label n_e = n (tail = first triangle).
    The support is convex only for -1 <= n <= 1 (for n = 0: unit square, the conifold); otherwise
    the triangulation is flagged ``convex=False`` (e.g. n = -2 is KP Fig. 3 middle up to
    relabelling: two of the three triangles of local P^2)."""
    pts = [(0, 0), (1, 0), (0, 1), (n + 1, -1)]
    convex = -1 <= n <= 1
    return from_triangulation(Triangulation(pts, [(0, 1, 2), (1, 0, 3)], convex=convex))


def su_n_ladder(N: int, m: int) -> GTGraph:
    """Ladder geometry (SU(N) geometric engineering): points (0,j), 0 <= j <= N, then (-1,1) and
    (1,N-m-1); triangles {(0,j),(0,j+1),(-1,1)} and {(0,j),(0,j+1),(1,N-m-1)}, 0 <= j < N.
    N = 2, m = 0, 1, 2 reproduce local F0, F1, F2 (up to affine SL(2,Z) and relabelling)."""
    pts = [(0, j) for j in range(N + 1)] + [(-1, 1), (1, N - m - 1)]
    L, R = N + 1, N + 2
    tris = [(j, j + 1, L) for j in range(N)] + [(j, j + 1, R) for j in range(N)]
    return from_triangulation(Triangulation(pts, tris))


def kp_flop_f1() -> GTGraph:
    """Flop of K_{F_1} (KP Fig. 3 right): w0..w4 = (0,0),(1,0),(0,1),(-1,-1),(1,1); triangles
    (w0,w1,w2), (w0,w2,w3), (w0,w3,w1), (w1,w4,w2).  Edges: w0w1, w0w2, w0w3 (P^2 lines, n = -2,
    2, 2) and w1w2 (the flopped (-1,-1) curve, n = 0)."""
    pts = [(0, 0), (1, 0), (0, 1), (-1, -1), (1, 1)]
    return from_triangulation(Triangulation(pts, [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 4, 2)]))


def local_dp3() -> GTGraph:
    """Local dP3: hexagon with corners (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1) and centre (0,0),
    star triangulation.  Edge k = spoke to the k-th corner; each is a (-1)-curve of dP3
    (normal bundle O(-1)+O(-1)), n = 0.  Rank of H_2 = 4."""
    return from_triangulation(star_triangulation((0, 0), [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]))
