r"""5d Nekrasov partition functions of linear quivers U(N_1) x ... x U(N_r) and their comparison
with the topological vertex of ladder triangulations (Iqbal--Kashani-Poor hep-th/0306032, Sec. 5;
Chern--Simons factor: Tachikawa hep-th/0401184).

The Nekrasov formula is the 5d (sinh) instanton partition function at eps1 = -eps2 = hbar of
IKP eqs. (73)-(74), in the arm/leg form below, extended to bifundamental and fundamental
hypermultiplets.  The dictionary (columns -> gauge nodes, brane heights -> Coulomb parameters,
kappa_i, eps_i, phi_i) is read off from the (p,q)-web of the ladder and was verified numerically
(tests/test_nekrasov.py); it is not taken from the literature.

**Nekrasov factor.**  With arm a_Y(i, j) = Y_i - j - 1 and leg l_Y(i, j) = Y^t_j - i - 1 (0-based
boxes (i, j); both may be negative for a box outside Y),

    n_{mu nu}(u) = prod_{s in mu} [u + a_mu(s) + l_nu(s) + 1] prod_{s in nu} [u - a_nu(s) - l_mu(s) - 1]

(shifts c in units of hbar; :func:`arm_leg_shifts` lists the shifts c of its factors [u + c]).
``[u]`` is the 5d (sinh) version [u] = 2 sinh(beta u / 2), written multiplicatively with
e^{beta hbar} = q = x^2 and e^{-beta u} = Q^D a monomial in the Kahler parameters (see
**Classes**), so that [u + c] = Q^{-D/2} x^c - Q^{D/2} x^{-c}; for u = 0 it is the q-number
[c] = x^c - x^{-c} of CONVENTIONS.md.  The arm/leg form equals the (regularised) infinite product of IKP eq. (74),

    prod_{l,n} prod_{i,j>=1} [a_ln + hbar(mu_{l,i} - mu_{n,j} + j - i)] / [a_ln + hbar(j - i)],

up to boundary factors: truncating i, j <= L the double product equals (boundary)/(arm-leg
product) where all boundary arguments a_ln + hbar c have |c| >= L - max(rows, columns); see
:func:`ij_product_shifts` and tests/test_nekrasov.py.

**Ladder structure** (:func:`quiver_structure`).  A triangulation T is a ladder along a
primitive direction d if every triangle has a side parallel to d.  With e such that
det(e, d) = e_x d_y - e_y d_x = +1 and coordinates X(p) = det(p - p0, d) (shifted to min X = 0),
Y(p) = det(e, p - p0), column i is the set of points with X = i (sorted by Y), b_i = min Y,
N_i = #column - 1, i = 0..r+1 (r >= 1).  Columns 1..r are the gauge nodes U(N_i), columns 0 and
r+1 give N_0 and N_{r+1} fundamental flavours.  Brane (i, k), k = 1..N_i from the bottom, is the
segment between the (k-1)-th and k-th point of column i; for a gauge node it is a graph edge (the
"horizontal edge" T_{i,k}, carrying the partition Y^(i)_k), for a flavour column a boundary
segment (empty partition).  Then

    kappa_i = (b_{i+1} - 2 b_i + b_{i-1}) - N_i + (N_{i+1} + N_{i-1})/2,
    eps_i   = (-1)^{b_{i+1} + b_{i-1} + N_i + N_{i-1}}.

**Classes.**  Curve classes are relation vectors l in Z^{#points} (:attr:`Edge.curve_class`),
wt(l) = <h, l> with h the heights.  The brane height vector is y(i, k) = e_top - e_bottom (unit
vectors of the two endpoints); a^(i)_k = -y(i, k), so a^(i)_k - a^(j)_l corresponds to
D = y(j, l) - y(i, k) (e^{-beta a} -> Q^{-y}).  |D| means D if wt(D) > 0 and -D otherwise.  The
instanton counting parameter phi_i = eps_i Q^{Phi_i} with (for every k = 1..N_i; independence of
k is asserted)

    2 Phi_i = 2 T_{i,k} + 2 kappa_i (y_k - y_1) - 2 sum_{k' != k} |y_k - y_k'|
              + sum_{l in columns i-1, i+1} |y_k - y_l|.

**Nekrasov sum** (:func:`nekrasov_partition_function`).  With n_{mu nu} as above,

    Z_Nek = sum_Y prod_{i=1..r} eps_i^{|Y^(i)|} Q^{Phi_i |Y^(i)|} CS_i(Y^(i))
                 / prod_{k,l} n_{Y^(i)_k Y^(i)_l}(a^(i)_k - a^(i)_l)
            * prod_{i=0..r} prod_{k,l} n_{Y^(i)_k Y^(i+1)_l}(a^(i)_k - a^(i+1)_l),

with the Chern--Simons factor (Tachikawa hep-th/0401184) in the frame a^(i)_1 = 0, where
e^{-beta a^(i)_k} = Q^{y_1 - y_k},

    CS_i(Y) = prod_k prod_{(a,b) in Y_k} (Q^{y_1 - y_k} q^{a - b})^{kappa_i},   (a, b) = (row, column),
            = prod_k Q^{-kappa_i (y_k - y_1) |Y_k|} x^{-kappa_i kappa(Y_k)}

(sum_{(a,b) in Y} (b - a) = kappa(Y)/2).  Each factor [u + c] with D the vector of u is
expanded as P^{-1/2} x^c (1 - P q^{-c}) (P = D, wt(D) > 0) or -P^{-1/2} x^{-c} (1 - P q^c)
(P = -D, wt(D) < 0); for k = l in one node u = 0 and [c] is the q-number.  The monomial of each
term (before expanding the (1 - P q^..) factors) is asserted to be prod Q_{T_{i,k}}^{|Y^(i)_k|}.

**Vertex side.**  Z_top = :func:`topvertex.partition_function.partition_function` with w_e =
wt(l_e), regrouped by class vector; instanton numbers n_i(l) = sum_P l_P max(0, X_P - i); the
perturbative part Z_pert collects the classes with all n_i = 0.  Checked statement:
Z_top = Z_pert Z_Nek for all classes of weight <= D (:func:`compare_with_nekrasov`).

**Pure SU(N)** is the case r = 1, N_0 = N_2 = 0 of the above (no separate code).  For
:func:`topvertex.examples_geometry.su_n_ladder` (N, m) (column c_j = (0, j), L = (-1, 1),
R = (1, N - m - 1); N = 2, m = 0, 1, 2 is local F0, F1, F2) one has b = (1, 0, N - m - 1), hence

    kappa = -m,   eps = (-1)^m,   phi = (-1)^m Q_{B_bot} prod_{j=1}^{N-1} Q_{F_j}^{-(N-j)}   (U(N) frame a_1 = 0),

with F_j the class of the segment (c_j, L) (the diagonal of strip 0 between the branes j and
j + 1, F_j = y_{j+1} - y_j) and B_bot = T_{1,1} the class of the edge dual to (c_0, c_1).  In the
SU(N) frame (sum_k a_k = 0) this reads phi = (-1)^m Q_{B_bot} prod_j Q_{F_j}^{-(N-m)(N-j)/N}.
"""
from __future__ import annotations

import time
from fractions import Fraction
from math import gcd

from . import polygon as pg
from .partition_function import partition_function
from .partitions import Partition, conjugate, kappa as kappa_of, partitions_of, size
from .polygon import Triangulation
from .qring import QFrac
from .series import Series
from .toric import GTGraph, from_triangulation, kahler_weights

__all__ = ["arm_leg_shifts", "ij_product_shifts", "quiver_structures", "quiver_structure", "quiver_dictionary",
           "nekrasov_partition_function", "class_dict", "instanton_numbers", "perturbative_part", "instanton_part",
           "compare_with_nekrasov", "format_comparison"]

Vec = tuple[int, ...]
Brane = tuple[int, int]


def _add(a, b, s=1):
    return tuple(x + s * y for x, y in zip(a, b))


def _dot(h, l) -> int:
    return sum(a * b for a, b in zip(h, l))


def _as_graph(obj) -> GTGraph:
    if isinstance(obj, GTGraph):
        if obj.triangulation is None:
            raise ValueError("graph has no triangulation")
        return obj
    if isinstance(obj, Triangulation):
        return from_triangulation(obj)
    raise TypeError("expected a GTGraph or a Triangulation")


# ----------------------------------------------------------------------------- Nekrasov factor

def _arm(Y: Partition, i: int, j: int) -> int:
    """Arm length a_Y(i, j) = Y_i - j - 1 (0-based; Y_i = 0 beyond the length; may be < 0)."""
    return (Y[i] if i < len(Y) else 0) - j - 1


def _leg(Y: Partition, i: int, j: int) -> int:
    """Leg length l_Y(i, j) = Y^t_j - i - 1 (0-based; may be < 0)."""
    Yt = conjugate(Y)
    return (Yt[j] if j < len(Yt) else 0) - i - 1


def _boxes(Y: Partition):
    for i, r in enumerate(Y):
        for j in range(r):
            yield i, j


def arm_leg_shifts(mu: Partition, nu: Partition) -> list[int]:
    """Shifts c of the factors [u + hbar c] of the arm/leg Nekrasov factor n_{mu nu}(u) (module
    docstring; IKP hep-th/0306032 eq. (73)): c = a_mu(s) + l_nu(s) + 1 for s in mu and
    c = -(a_nu(s) + l_mu(s) + 1) for s in nu.  The vector multiplet weight of the pair
    (Y_l, Y_n) is prod 1/[a_ln + hbar c]."""
    out = [_arm(mu, i, j) + _leg(nu, i, j) + 1 for i, j in _boxes(mu)]
    out += [-(_arm(nu, i, j) + _leg(mu, i, j) + 1) for i, j in _boxes(nu)]
    return out


def ij_product_shifts(mu: Partition, nu: Partition, L: int, same: bool) -> tuple[list[int], list[int]]:
    """Numerator and denominator shifts of the truncated IKP hep-th/0306032 eq. (74) double product
    prod_{i,j=1}^L [x + hbar(mu_i - nu_j + j - i)] / [x + hbar(j - i)] (x = a_ln).  For
    ``same`` (l = n, x = 0) the 0/0 factors i = j are dropped.  Up to boundary factors it is
    1/prod_{c in arm_leg_shifts(mu, nu)} [x + hbar c] (module docstring)."""
    m = list(mu) + [0] * (L - len(mu))
    n = list(nu) + [0] * (L - len(nu))
    num, den = [], []
    for i in range(1, L + 1):
        for j in range(1, L + 1):
            if same and i == j:
                continue
            num.append(m[i - 1] - n[j - 1] + j - i)
            den.append(j - i)
    return num, den


# ----------------------------------------------------------------------------- lattice helpers

def _as_triangulation(obj) -> Triangulation:
    if isinstance(obj, GTGraph):
        if obj.triangulation is None:
            raise ValueError("graph has no triangulation")
        return obj.triangulation
    if isinstance(obj, Triangulation):
        return obj
    raise TypeError("expected a Triangulation or a GTGraph")


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    """(g, x, y) with a x + b y = g = gcd(a, b) >= 0."""
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y


def _primitive_dirs(points):
    """Primitive directions of all point differences, canonical sign (d > (0, 0)), sorted."""
    dirs = set()
    for a in points:
        for b in points:
            if a < b:
                dx, dy = b[0] - a[0], b[1] - a[1]
                g = gcd(dx, dy)
                dx, dy = dx // g, dy // g
                if (dx, dy) < (0, 0):
                    dx, dy = -dx, -dy
                dirs.add((dx, dy))
    return sorted(dirs)

# ----------------------------------------------------------------------------- structure

def _structure_along(T: Triangulation, d: tuple[int, int]) -> dict | None:
    """Ladder description of T along the primitive direction d (module docstring), or None."""
    if not T.convex:
        return None
    g, xx, yy = _egcd(d[0], d[1])
    if g != 1:
        raise ValueError(f"direction {d} is not primitive")
    e = (yy, -xx)  # det(e, d) = xx d_x + yy d_y = 1
    pts = T.points
    p0 = pts[0]
    X, Y = [], []
    for p in pts:
        v = (p[0] - p0[0], p[1] - p0[1])
        X.append(v[0] * d[1] - v[1] * d[0])
        Y.append(e[0] * v[1] - e[1] * v[0])
    # every triangle has a side parallel to d <=> its vertices take exactly two values of X
    # (unimodularity then forces them to differ by 1)
    if any(len({X[v] for v in t}) != 2 for t in T.triangles):
        return None
    x0 = min(X)
    X = [x - x0 for x in X]
    ncol = max(X) + 1
    if ncol < 3:
        return None
    columns = [sorted((v for v in range(len(pts)) if X[v] == i), key=lambda v: Y[v]) for i in range(ncol)]
    for c in columns:
        if not c or any(Y[c[k + 1]] - Y[c[k]] != 1 for k in range(len(c) - 1)):
            return None
    N = [len(c) - 1 for c in columns]
    r = ncol - 2
    if any(N[i] < 1 for i in range(1, r + 1)):
        return None
    b = [Y[c[0]] for c in columns]
    kappa = [Fraction(b[i + 1] - 2 * b[i] + b[i - 1]) - N[i] + Fraction(N[i + 1] + N[i - 1], 2)
             for i in range(1, r + 1)]
    eps = [(-1) ** ((b[i + 1] + b[i - 1] + N[i] + N[i - 1]) % 2) for i in range(1, r + 1)]
    return {"direction": tuple(d), "frame": e, "columns": columns, "X": X, "Y": Y, "b": b, "N": N, "r": r,
            "kappa": kappa, "eps": eps}


def quiver_structures(tri) -> list[dict]:
    """All linear-quiver (ladder) descriptions of the triangulation (or GTGraph), one per primitive
    direction d with every triangle having a side parallel to d (d in the canonical sign of
    :func:`_primitive_dirs`; -d gives the description rotated by pi).  Each is
    a dict {direction, frame e (det(e, d) = 1), columns (point indices, bottom to top), X, Y (per
    point), b, N, r, kappa (Fractions), eps}; formulas for kappa_i and eps_i in the module
    docstring (dictionary derived from the web and verified numerically).  Convex polygons only."""
    T = _as_triangulation(tri)
    out = []
    for d in _primitive_dirs(T.points):
        s = _structure_along(T, d)
        if s is not None:
            out.append(s)
    return out


def quiver_structure(tri, direction=None) -> dict | None:
    """The ladder description along ``direction`` (either sign; -d reverses the columns and
    flips kappa -> -kappa), default: the one of :func:`quiver_structures` with the fewest gauge
    nodes (ties -> first).  None if there is none."""
    T = _as_triangulation(tri)
    if direction is None:
        structs = quiver_structures(T)
        return min(structs, key=lambda s: s["r"]) if structs else None
    dx, dy = int(direction[0]), int(direction[1])
    g = gcd(dx, dy)
    if g == 0:
        raise ValueError("direction must be non-zero")
    return _structure_along(T, (dx // g, dy // g))


# ----------------------------------------------------------------------------- dictionary

def _phi_double(struct: dict, y: dict, hor_cls: dict, wt, i: int, kap: Fraction, check: bool) -> Vec:
    """2 Phi_i (integer vector) from the formula of the module docstring, evaluated for every
    k = 1..N_i; with ``check`` the k-independence is asserted.  Returns the k = 1 value (where
    the kappa term vanishes)."""
    N = struct["N"]
    vals = []
    for k in range(1, N[i] + 1):
        v = [2 * c for c in hor_cls[(i, k)]]
        dk = _add(y[(i, k)], y[(i, 1)], -1)
        v = [a + 2 * kap * c for a, c in zip(v, dk)]
        for kk in range(1, N[i] + 1):
            if kk != k:
                D = _add(y[(i, k)], y[(i, kk)], -1)
                s = 1 if wt(D) > 0 else -1
                v = [a - 2 * s * c for a, c in zip(v, D)]
        for j in (i - 1, i + 1):
            for l in range(1, N[j] + 1):
                D = _add(y[(i, k)], y[(j, l)], -1)
                w = wt(D)
                if w == 0:
                    raise AssertionError(f"brane heights ({i},{k}) and ({j},{l}) have zero weight difference")
                s = 1 if w > 0 else -1
                v = [a + s * c for a, c in zip(v, D)]
        if any(Fraction(a).denominator != 1 for a in v):
            raise AssertionError("2 Phi is not integral")
        vals.append(tuple(int(a) for a in v))
    if check and any(v != vals[0] for v in vals):
        raise AssertionError(f"Phi_{i} depends on k: {vals}")
    return vals[0]


def _strips(struct: dict, T: Triangulation, seg_edge: dict, y: dict, classes) -> list[dict]:
    """Bottom-to-top walk through the triangles of every strip s (between columns s and s+1):
    the brane order of the strip (the chamber) and the diagonal edges between consecutive
    triangles, whose class is asserted to be y(upper) - y(lower)."""
    cols, N = struct["columns"], struct["N"]
    tri_set = {frozenset(t) for t in T.triangles}
    out = []
    for s in range(struct["r"] + 1):
        L, R = cols[s], cols[s + 1]
        a = b = 0
        order: list[Brane] = []
        diags: list[int] = []
        while a < N[s] or b < N[s + 1]:
            inL = a < N[s] and frozenset((L[a], L[a + 1], R[b])) in tri_set
            inR = b < N[s + 1] and frozenset((L[a], R[b], R[b + 1])) in tri_set
            if inL == inR:
                raise AssertionError(f"strip {s} is not walked by its triangles at ({a}, {b})")
            if order:
                diags.append(seg_edge[frozenset((L[a], R[b]))])
            if inL:
                order.append((s, a + 1))
                a += 1
            else:
                order.append((s + 1, b + 1))
                b += 1
        for j, e in enumerate(diags):
            if tuple(classes[e]) != _add(y[order[j + 1]], y[order[j]], -1):
                raise AssertionError(f"diagonal edge {e} of strip {s}: class != y{order[j + 1]} - y{order[j]}")
        out.append({"branes": order, "diagonals": diags})
    return out


def quiver_dictionary(graph, struct: dict | None = None, direction=None, heights=None,
                      kappa=None) -> dict:
    """Gauge-theory dictionary of a ladder geometry (module docstring; derived from the web and
    verified numerically against the Nekrasov partition function, tests/test_nekrasov.py).

    Returns {structure, heights, horizontal: {(i,k): edge index} (gauge branes), horizontal_classes:
    {(i,k): T_{i,k}}, y: {(i,k):
    brane height vector} (all columns), strips: [{branes (bottom to top), diagonals (edge
    indices)}], kappa, Phi: [Fraction vector Phi_i], Phi2: [2 Phi_i], phi: [{sign: eps_i, edges:
    {edge index: Fraction}}]}.  The edge form of phi_i (k = 1, the bottom brane of node i;
    within a strip y(upper) - y(lower) = sum of the diagonal classes between them):

        Phi_i = T_{i,1} - sum_{k'>=2} (y_k' - y_1) [strip i-1]
                + 1/2 sum_{l in col i-1} |y_1 - y_l| [strip i-1] + 1/2 sum_{l in col i+1} |y_1 - y_l| [strip i],

    asserted to equal Phi_i.  ``kappa``: override of the CS levels (tests only); Phi_i is then
    computed from k = 1 without the k-independence check.  ``heights`` default to
    :func:`topvertex.polygon.heights`."""
    graph = _as_graph(graph)
    T = graph.triangulation
    if struct is None:
        struct = quiver_structure(T, direction)
        if struct is None:
            raise ValueError("geometry is not recognised as a linear quiver (ladder triangulation)")
    h = list(heights) if heights is not None else pg.heights(T)
    if h is None:
        raise ValueError("triangulation is not regular; give heights")
    kahler_weights(graph, h)  # raises unless all edge weights are positive

    def wt(l):
        return _dot(h, l)

    n = len(T.points)
    cols, N, r = struct["columns"], struct["N"], struct["r"]
    seg_edge = {frozenset(e.segment): k for k, e in enumerate(graph.edges)}
    classes = [tuple(e.curve_class) for e in graph.edges]
    y: dict[Brane, Vec] = {}
    hor: dict[Brane, int] = {}
    for i in range(r + 2):
        for k in range(1, N[i] + 1):
            v = [0] * n
            v[cols[i][k]] += 1
            v[cols[i][k - 1]] -= 1
            y[(i, k)] = tuple(v)
            if 1 <= i <= r:
                hor[(i, k)] = seg_edge[frozenset((cols[i][k - 1], cols[i][k]))]
    hor_cls = {br: classes[e] for br, e in hor.items()}
    strips = _strips(struct, T, seg_edge, y, classes)
    check = kappa is None
    kap = list(struct["kappa"]) if kappa is None else [Fraction(k) for k in kappa]
    if len(kap) != r:
        raise ValueError(f"need {r} CS levels")
    Phi2 = [_phi_double(struct, y, hor_cls, wt, i, kap[i - 1], check) for i in range(1, r + 1)]

    pos = [{br: j for j, br in enumerate(st["branes"])} for st in strips]

    def path(s, A, B, coeff, acc):
        # |y_A - y_B| = sum of the diagonals of strip s between A and B
        p, q = sorted((pos[s][A], pos[s][B]))
        for j in range(p, q):
            e = strips[s]["diagonals"][j]
            acc[e] = acc.get(e, Fraction(0)) + coeff

    phis = []
    for i in range(1, r + 1):
        acc: dict[int, Fraction] = {hor[(i, 1)]: Fraction(1)}
        for kk in range(2, N[i] + 1):
            path(i - 1, (i, 1), (i, kk), Fraction(-1), acc)
        for l in range(1, N[i - 1] + 1):
            path(i - 1, (i, 1), (i - 1, l), Fraction(1, 2), acc)
        for l in range(1, N[i + 1] + 1):
            path(i, (i, 1), (i + 1, l), Fraction(1, 2), acc)
        e1 = hor[(i, 1)]  # the horizontal edge first, then the diagonals by index
        acc = {e: c for e, c in sorted(acc.items(), key=lambda t: (t[0] != e1, t[0])) if c}
        vec = [Fraction(0)] * n
        for e, c in acc.items():
            vec = [a + c * b for a, b in zip(vec, classes[e])]
        if [2 * a for a in vec] != list(Phi2[i - 1]):
            raise AssertionError(f"edge form of phi_{i} does not reproduce Phi_{i}")
        phis.append({"sign": struct["eps"][i - 1], "edges": acc})
    return {"structure": struct, "heights": h, "horizontal": hor, "horizontal_classes": hor_cls, "y": y,
            "strips": strips, "kappa": kap,
            "Phi2": Phi2, "Phi": [tuple(Fraction(a, 2) for a in v) for v in Phi2], "phi": phis}


# ----------------------------------------------------------------------------- Nekrasov side

def _nek_term(dic: dict, Y: dict, kap: list, Phi2: list, wt, n: int, check: bool = True):
    """Sinh-form weight of the tuple Y (module docstring) without the eps signs, as (sign,
    monomial, xexp, qnums, num, den): weight = sign Q^monomial x^xexp / prod [qnums]
    * prod_{(P, e) in num} (1 - Q^P q^e) / prod_{(P, e) in den} (1 - Q^P q^e).  The monomial is
    asserted to be integral (with ``check``; a wrong kappa override may give Fraction entries)."""
    N, r, y = dic["structure"]["N"], dic["structure"]["r"], dic["y"]
    sign = 1
    xexp = 0
    qn: list[int] = []
    num: list[tuple[Vec, int]] = []
    den: list[tuple[Vec, int]] = []
    mono2 = [0] * n

    def factor(D, c, numerator):
        nonlocal sign, xexp
        w = wt(D)
        if w == 0:
            raise AssertionError("zero-weight Coulomb parameter difference")
        if w > 0:  # [u + c] = P^{-1/2} x^c (1 - P q^{-c}),  P = D
            P, xe, e = D, c, -c
        else:      # [u + c] = -P^{-1/2} x^{-c} (1 - P q^{c}),  P = -D
            P, xe, e = tuple(-a for a in D), -c, c
            sign = -sign
        if numerator:
            xexp += xe
            for t in range(n):
                mono2[t] -= P[t]
            num.append((P, e))
        else:
            xexp -= xe
            for t in range(n):
                mono2[t] += P[t]
            den.append((P, e))

    for i in range(1, r + 1):
        k2 = 2 * kap[i - 1]
        for k in range(1, N[i] + 1):
            Yk = Y[(i, k)]
            s = size(Yk)
            d = _add(y[(i, k)], y[(i, 1)], -1)
            for t in range(n):
                mono2[t] += Phi2[i - 1][t] * s - k2 * s * d[t]
            xe = -kap[i - 1] * kappa_of(Yk)
            if Fraction(xe).denominator != 1:
                raise AssertionError("half-integer CS power of x")
            xexp += int(xe)
            for l in range(1, N[i] + 1):
                for c in arm_leg_shifts(Yk, Y[(i, l)]):
                    if k == l:
                        if c == 0:
                            raise AssertionError("vanishing diagonal Nekrasov factor")
                        if c < 0:
                            sign = -sign
                        qn.append(abs(c))
                    else:
                        factor(_add(y[(i, l)], y[(i, k)], -1), c, False)
    for i in range(r + 1):
        for k in range(1, N[i] + 1):
            for l in range(1, N[i + 1] + 1):
                for c in arm_leg_shifts(Y[(i, k)], Y[(i + 1, l)]):
                    factor(_add(y[(i + 1, l)], y[(i, k)], -1), c, True)
    if any(Fraction(m).denominator != 1 or m % 2 for m in mono2):
        if check:
            raise AssertionError("half-integer monomial in a Nekrasov term")
        return sign, tuple(Fraction(m) / 2 for m in mono2), xexp, qn, num, den
    return sign, tuple(int(m) // 2 for m in mono2), xexp, qn, num, den


def _expand(lead: Vec, pref: QFrac, num, den, D: int, wt) -> dict[Vec, QFrac]:
    """pref Q^lead prod_num (1 - Q^P q^e) / prod_den (1 - Q^P q^e), all terms of weight <= D
    (every P has positive weight)."""
    if wt(lead) > D:
        return {}
    ser = {lead: pref}
    for P, e in num:
        new = dict(ser)
        for d, c in ser.items():
            dd = _add(d, P)
            if wt(dd) <= D:
                t = -(c * QFrac.x_pow(2 * e))
                new[dd] = new[dd] + t if dd in new else t
        ser = new
    for P, e in den:
        new = {}
        wP = wt(P)
        for d, c in ser.items():
            k = 0
            while wt(d) + k * wP <= D:
                dd = _add(d, P, k)
                t = c * QFrac.x_pow(2 * e * k)
                new[dd] = new[dd] + t if dd in new else t
                k += 1
        ser = new
    return {d: c for d, c in ser.items() if not c.is_zero()}


def _gauge_tuples(dic: dict, D: int, wt):
    """All tuples Y of partitions on the gauge branes with sum wt(T_{i,k}) |Y^(i)_k| <= D (flavour
    branes carry the empty partition)."""
    st = dic["structure"]
    N, r = st["N"], st["r"]
    branes = sorted(dic["horizontal"])
    ws = [wt(dic["horizontal_classes"][br]) for br in branes]

    def rec(j, budget):
        if j == len(branes):
            yield {}
            return
        for s in range(budget // ws[j] + 1):
            for p in partitions_of(s):
                for rest in rec(j + 1, budget - ws[j] * s):
                    out = dict(rest)
                    out[branes[j]] = p
                    yield out

    for Y in rec(0, D):
        for i in (0, r + 1):
            for k in range(1, N[i] + 1):
                Y[(i, k)] = ()
        yield Y


def _nekrasov_from_dictionary(graph: GTGraph, dic: dict, D: int, eps, check: bool) -> dict[Vec, QFrac]:
    h = dic["heights"]

    def wt(l):
        return _dot(h, l)

    st = dic["structure"]
    N, r = st["N"], st["r"]
    n = len(graph.points)
    out: dict[Vec, QFrac] = {}
    for Y in _gauge_tuples(dic, D, wt):
        sign, mono, xexp, qn, num, den = _nek_term(dic, Y, dic["kappa"], dic["Phi2"], wt, n, check)
        lead = [0] * n
        for i in range(1, r + 1):
            sz = sum(size(Y[(i, k)]) for k in range(1, N[i] + 1))
            if eps[i - 1] < 0 and sz % 2:
                sign = -sign
            for k in range(1, N[i] + 1):
                lead = [a + size(Y[(i, k)]) * c for a, c in zip(lead, dic["horizontal_classes"][(i, k)])]
        if check and tuple(lead) != mono:
            raise AssertionError(f"Nekrasov monomial {mono} != instanton class {tuple(lead)} for Y = {Y}")
        pref = QFrac.inv_qnums(qn, sign, xexp)
        for d, c in _expand(mono, pref, num, den, D, wt).items():
            out[d] = out[d] + c if d in out else c
    return {d: c.cancel() for d, c in out.items() if not c.is_zero()}


def nekrasov_partition_function(graph, max_weight: int, heights=None, struct: dict | None = None,
                                direction=None, kappa=None, eps=None) -> dict[Vec, QFrac]:
    """5d Nekrasov partition function of the linear quiver of the ladder geometry (IKP
    hep-th/0306032 Sec. 5 for the vector multiplet in the arm/leg form, with bifundamentals and
    fundamentals from the flavour columns; dictionary of :func:`quiver_dictionary`):

        Z_Nek = sum_Y prod_{i=1..r} phi_i^{|Y^(i)|} CS_i(Y^(i))
                     / prod_{k,l} n_{Y^(i)_k Y^(i)_l}(a^(i)_k - a^(i)_l)
                * prod_{i=0..r} prod_{k,l} n_{Y^(i)_k Y^(i+1)_l}(a^(i)_k - a^(i+1)_l)

    (module docstring for n_{mu nu}, CS_i and the expansion), as {class vector: QFrac}, all
    classes of weight <= max_weight.  For a pure SU(N) ladder (r = 1, no flavours) this is the
    pure SU(N) Nekrasov partition function with kappa = -m (module docstring).  ``kappa`` /
    ``eps`` override the CS levels and signs of the dictionary (only to show that wrong values
    fail: Phi_i is then recomputed from the overridden kappa at k = 1 and the monomial check is
    skipped)."""
    graph = _as_graph(graph)
    dic = quiver_dictionary(graph, struct, direction, heights, kappa=kappa)
    e = list(dic["structure"]["eps"]) if eps is None else list(eps)
    return _nekrasov_from_dictionary(graph, dic, int(max_weight), e, check=kappa is None)


# ----------------------------------------------------------------------------- vertex side

def class_dict(graph: GTGraph, Z: Series) -> dict[Vec, QFrac]:
    """Regroup a Series in the edge variables Q_e by curve class: {sum_e d_e l_e: coefficient}
    (l_e = relation vector of edge e, CONVENTIONS.md); zero sums are dropped."""
    classes = [tuple(e.curve_class) for e in graph.edges]
    n = len(classes[0]) if classes else (graph.n_points or 0)
    out: dict[Vec, QFrac] = {}
    for d, c in Z.coefficients.items():
        l = (0,) * n
        for e, m in enumerate(d):
            if m:
                l = _add(l, classes[e], m)
        out[l] = out[l] + c if l in out else c
    return {l: c for l, c in out.items() if not c.is_zero()}


def instanton_numbers(struct: dict, l) -> tuple[int, ...]:
    """n_i(l) = sum_P l_P max(0, X_P - i), i = 1..r: the instanton numbers of the class l
    (n_j(T_{i,k}) = delta_ij)."""
    X = struct["X"]
    return tuple(sum(a * max(0, X[t] - i) for t, a in enumerate(l) if a) for i in range(1, struct["r"] + 1))


def _mul(A: dict, B: dict, D: int, wt) -> dict:
    out: dict[Vec, QFrac] = {}
    for d1, c1 in A.items():
        w1 = wt(d1)
        for d2, c2 in B.items():
            if w1 + wt(d2) <= D:
                d = _add(d1, d2)
                t = c1 * c2
                out[d] = out[d] + t if d in out else t
    return {d: c for d, c in out.items() if not c.is_zero()}


def perturbative_part(graph, Z, struct: dict | None = None, direction=None) -> dict[Vec, QFrac]:
    """Z_pert: the terms of Z (a Series in edge variables or a {class: coefficient} dict) whose
    classes have all instanton numbers n_i = 0 (:func:`instanton_numbers`)."""
    graph = _as_graph(graph)
    if struct is None:
        struct = quiver_structure(graph.triangulation, direction)
        if struct is None:
            raise ValueError("geometry is not recognised as a linear quiver (ladder triangulation)")
    top = class_dict(graph, Z) if isinstance(Z, Series) else Z
    zero = (0,) * struct["r"]
    return {l: c for l, c in top.items() if instanton_numbers(struct, l) == zero}


def _edge_weights(graph: GTGraph, h) -> list[int]:
    return [_dot(h, e.curve_class) for e in graph.edges]


def instanton_part(graph, Z, heights=None, struct: dict | None = None, direction=None) -> dict[Vec, QFrac]:
    """Z_top / Z_pert as {class vector: QFrac}, all classes of weight <= D (Z_pert of
    :func:`perturbative_part`; IKP hep-th/0306032 Sec. 5).  ``Z``: the vertex
    Series (computed with the weights wt(l_e) of ``heights``; D = Z.max_weight) or the order D
    (then Z is computed).  Compare with :func:`nekrasov_partition_function`."""
    graph = _as_graph(graph)
    T = graph.triangulation
    if struct is None:
        struct = quiver_structure(T, direction)
        if struct is None:
            raise ValueError("geometry is not recognised as a linear quiver (ladder triangulation)")
    h = list(heights) if heights is not None else pg.heights(T)
    if h is None:
        raise ValueError("triangulation is not regular; give heights")
    w = kahler_weights(graph, h)
    if not isinstance(Z, Series):
        Z = partition_function(graph, int(Z), w)
    elif list(Z.weights) != w:
        raise ValueError("Z was computed with other weights")
    D = Z.max_weight

    def wt(l):
        return _dot(h, l)

    top = class_dict(graph, Z)
    pert = perturbative_part(graph, top, struct)
    zero = (0,) * len(h)
    u = {d: -c for d, c in pert.items() if d != zero}
    inv = {zero: QFrac.one()}
    term = {zero: QFrac.one()}
    while term:
        term = _mul(term, u, D, wt)
        for d, c in term.items():
            inv[d] = inv[d] + c if d in inv else c
    return {d: c.cancel() for d, c in _mul(top, inv, D, wt).items()}


# ----------------------------------------------------------------------------- comparison

def _group_string(N, r) -> str:
    return " x ".join(f"U({N[i]})" for i in range(1, r + 1))


def compare_with_nekrasov(graph_or_triangulation, max_weight: int, heights=None, direction=None,
                          Z: Series | None = None) -> dict:
    """Check Z_top = Z_pert * Z_Nek for all curve classes of Kahler weight <= max_weight (module
    docstring; IKP hep-th/0306032 Sec. 5; Z_Nek of :func:`nekrasov_partition_function`, dictionary
    of :func:`quiver_dictionary`, derived from the web and verified numerically).  Pure SU(N)
    ladders are the case r = 1 without fundamentals.  ``Z``: the vertex Series if already computed with the
    weights wt(l_e) of the same heights at order >= max_weight (it is then reused).

    Returns {structure, dictionary, direction, gauge_group, flavours (N_0, N_{r+1}), kappa, eps,
    phi, match, n_classes (classes compared with some n_i > 0), max_instanton (per node),
    mismatches (classes), other_directions, max_weight, times {vertex, nekrasov, total}}."""
    t0 = time.perf_counter()
    graph = _as_graph(graph_or_triangulation)
    T = graph.triangulation
    struct = quiver_structure(T, direction)
    if struct is None:
        raise ValueError("geometry is not recognised as a linear quiver (ladder triangulation)")
    D = int(max_weight)
    dic = quiver_dictionary(graph, struct, heights=heights)
    h = dic["heights"]

    def wt(l):
        return _dot(h, l)

    w = _edge_weights(graph, h)
    if Z is None or list(Z.weights) != w or Z.max_weight < D:
        Z = partition_function(graph, D, w)
    top = {l: c for l, c in class_dict(graph, Z).items() if wt(l) <= D}
    t1 = time.perf_counter()
    nek = _nekrasov_from_dictionary(graph, dic, D, struct["eps"], check=True)
    pert = perturbative_part(graph, top, struct)
    prod = _mul(pert, nek, D, wt)
    t2 = time.perf_counter()
    zero_c = QFrac.zero()
    r = struct["r"]
    keys = sorted(set(top) | set(prod))
    bad = [l for l in keys if not (top.get(l, zero_c) - prod.get(l, zero_c)).is_zero()]
    inst = {l: instanton_numbers(struct, l) for l in keys}
    compared = [l for l in keys if any(inst[l])]
    max_inst = tuple(max([inst[l][j] for l in top] + [0]) for j in range(r))
    d = struct["direction"]
    others = [s["direction"] for s in quiver_structures(T)
              if s["direction"] not in (d, (-d[0], -d[1]))]
    N = struct["N"]
    return {"structure": struct, "dictionary": dic, "direction": d, "gauge_group": _group_string(N, r),
            "flavours": (N[0], N[r + 1]), "kappa": list(struct["kappa"]), "eps": list(struct["eps"]),
            "phi": dic["phi"], "match": not bad, "n_classes": len(compared), "max_instanton": max_inst,
            "mismatches": bad, "other_directions": others, "max_weight": D,
            "edge_names": [e.name for e in graph.edges],
            "times": {"vertex": t1 - t0, "nekrasov": t2 - t1, "total": time.perf_counter() - t0}}


def _fmt_exp(name: str, c: Fraction) -> str:
    if c == 1:
        return name
    if c.denominator == 1:
        return f"{name}^{c.numerator}"
    return f"{name}^({c})"


def format_comparison(res: dict) -> str:
    """Human-readable summary of :func:`compare_with_nekrasov`: gauge group, fundamentals ("no
    fundamentals" if N_0 = N_{r+1} = 0), CS level(s) kappa_i, per node the branes and
    phi_i = eps_i Q^{Phi_i} in edge variables, and the result of the check."""
    st = res["structure"]
    r, N = st["r"], st["N"]
    names = res["edge_names"]
    kap = ", ".join(str(k) for k in res["kappa"])
    kap = f"levels kappa = ({kap})" if r > 1 else f"level kappa = {kap}"
    fund = f"fundamentals {N[0]} + {N[r + 1]}" if N[0] or N[r + 1] else "no fundamentals"
    lines = [f"linear quiver along {res['direction']}: {res['gauge_group']}, {fund}, CS {kap}"]
    hor = res["dictionary"]["horizontal"]
    for i in range(1, r + 1):
        ph = res["phi"][i - 1]
        mono = " ".join(_fmt_exp(names[e], c) for e, c in ph["edges"].items())
        branes = ", ".join(names[hor[(i, k)]] for k in range(1, N[i] + 1))
        lines.append(f"  node {i}: branes (bottom to top) {branes}; phi_{i} = {'+' if ph['sign'] > 0 else '-'}{mono}")
    D = res["max_weight"]
    up = ", ".join(str(m) for m in res["max_instanton"])
    up = f"({up})" if r > 1 else up
    if res["match"]:
        lines.append(f"Nekrasov comparison: MATCH for all {res['n_classes']} instanton classes of weight <= {D} "
                     f"(instanton numbers up to {up})")
    else:
        lines.append(f"Nekrasov comparison: MISMATCH in {len(res['mismatches'])} of {res['n_classes']} instanton "
                     f"classes of weight <= {D}")
    if res["other_directions"]:
        lines.append("also a linear quiver along " + ", ".join(str(d) for d in res["other_directions"]))
    lines.append(f"time: {res['times']['total']:.2f}s")
    return "\n".join(lines)
