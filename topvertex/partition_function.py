r"""Gluing: the topological vertex partition function of a GT graph, its free energy and the
Gopakumar--Vafa invariants.

Konishi math/0504188, eq. (4) / Def. 2.3 (conventions in CONVENTIONS.md):

    Z = sum_{lam} prod_e (-1)^{|lam_e|(n_e+1)} q^{n_e kappa(lam_e)/2} Q_e^{|lam_e|}  prod_v C_{lam_v}(q),

lam_e a partition on every internal edge e (external legs carry the empty partition), the triple
lam_v at a vertex read counterclockwise with lam_e for an edge pointing AWAY from v and lam_e^t
for an edge pointing TOWARD v.  The sum is truncated by sum_e w_e |lam_e| <= D with positive
Kahler weights w_e = <h, l_e> (h = heights of the regular triangulation) or all ones.

The sum is evaluated by variable elimination over the vertices (a tensor-network contraction):
vertices are processed one by one; the partial sums are indexed by the partitions on the
"frontier" edges (exactly one endpoint processed) and the degree vector of the closed edges.
Edges are closed (their partitions summed over) as soon as both endpoints are processed.  This
is mathematically identical to the brute-force sum over all multipartitions (tested), but
merges terms early.

GV invariants (physical convention, CONVENTIONS.md):

    F = log Z = sum_{beta, g, k>=1} n^g_beta (-1)^{g-1} (1/k) [k]^{2g-2} Q^{k beta}.
"""
from __future__ import annotations

import time
from fractions import Fraction
from functools import lru_cache
from math import gcd
from typing import Sequence

from . import polygon as pg
from .partitions import Partition, conjugate, multipartitions, partitions_of, size
from .polygon import Triangulation
from .qring import QFrac
from .series import Series
from .toric import GTGraph, class_coordinates, from_triangulation, kahler_weights, relation_lattice_basis
from .vertex import Vertex, default_vertex, framing_factor as _framing_factor

__all__ = [
    "partition_function", "partition_function_bruteforce", "class_partition_function", "class_series",
    "class_weights", "free_energy", "gv_invariants", "gv_table", "gv_check_integrality", "compute",
    "summary", "default_weights",
]


@lru_cache(maxsize=None)
def framing_factor(n: int, lam: Partition) -> QFrac:
    """Memoised :func:`topvertex.vertex.framing_factor`: (-1)^{|lam|(n+1)} q^{n kappa(lam)/2}."""
    return _framing_factor(n, lam)


def default_weights(graph: GTGraph, heights=None) -> list[int]:
    """Kahler weights w_e = <h, l_e> (:func:`topvertex.toric.kahler_weights`) if the graph has
    curve classes, else all ones."""
    if graph.edges and all(e.curve_class is not None for e in graph.edges) and (
            heights is not None or graph.triangulation is not None):
        return kahler_weights(graph, heights)  # raises ValueError if not regular / bad heights
    return [1] * len(graph.edges)


def _vertex_order(graph: GTGraph) -> list[int]:
    """Greedy elimination order: start at vertex 0, then repeatedly take the vertex with the most
    internal legs into the processed set (ties: fewest new edges, then lowest index)."""
    nv = len(graph.vertices)
    nbrs = []
    for legs in graph.vertices:
        s = []
        for l in legs:
            if l.kind == "int":
                E = graph.edges[l.edge]
                s.append(E.head if l.outgoing else E.tail)
        nbrs.append(s)
    done = [0] if nv else []
    seen = {0} if nv else set()
    while len(done) < nv:
        best = max((v for v in range(nv) if v not in seen),
                   key=lambda v: (sum(u in seen for u in nbrs[v]), -sum(u not in seen for u in nbrs[v]), -v))
        done.append(best)
        seen.add(best)
    return done


def _partitions_upto(budget: int, w: int, bound: int | None):
    """All partitions lam with w |lam| <= budget (and |lam| <= bound)."""
    top = budget // w
    if bound is not None:
        top = min(top, bound)
    for s in range(top + 1):
        yield from partitions_of(s)


def _contract(graph: GTGraph, weights: Sequence[int], max_weight: int, gradings: Sequence[Sequence[int]],
              vertex: Vertex, bounds: Sequence[int | None] | None = None) -> dict[tuple, QFrac]:
    """Variable-elimination evaluation of KP eq. (4).

    ``gradings[e]``: integer vector g_e (length r); a configuration contributes to the degree
    sum_e |lam_e| g_e.  ``bounds``: optional box d_i <= bounds[i] on the degree (requires all g_e
    >= 0, so that pruning partial configurations is valid).  Returns {degree: coefficient}."""
    ne = len(graph.edges)
    r = len(gradings[0]) if ne else 0
    if bounds is not None and any(c < 0 for g in gradings for c in g):
        raise ValueError("box bounds need non-negative gradings (curve classes)")
    zero = (0,) * r
    # state: (frontier partitions tuple, closed degree, closed weight) -> value
    states: dict[tuple, QFrac] = {((), zero, 0): QFrac.one()}
    frontier: list[int] = []
    for v in _vertex_order(graph):
        legs = graph.vertices[v]
        pos_in_front = {e: k for k, e in enumerate(frontier)}
        new_edges = [l.edge for l in legs if l.kind == "int" and l.edge not in pos_in_front]
        closing = [l.edge for l in legs if l.kind == "int" and l.edge in pos_in_front]
        keep = [e for e in frontier if e not in closing]
        new_frontier = keep + new_edges
        keep_pos = [pos_in_front[e] for e in keep]
        out: dict[tuple, QFrac] = {}
        for (fparts, deg, wclosed), val in states.items():
            wfront = sum(weights[e] * size(p) for e, p in zip(frontier, fparts))
            budget = max_weight - wclosed - wfront
            # degree used by the frontier (for box pruning)
            if bounds is not None:
                used = list(deg)
                for e, p in zip(frontier, fparts):
                    s = size(p)
                    for i in range(r):
                        used[i] += gradings[e][i] * s
            assign = {e: p for e, p in zip(frontier, fparts)}
            # closing contribution
            cdeg = list(deg)
            cw = wclosed
            for e in closing:
                s = size(assign[e])
                cw += weights[e] * s
                for i in range(r):
                    cdeg[i] += gradings[e][i] * s
            kept = tuple(fparts[k] for k in keep_pos)

            def rec(k: int, budget: int, used_box, chosen: list):
                if k == len(new_edges):
                    yield list(chosen)
                    return
                e = new_edges[k]
                bnd = None
                if bounds is not None:
                    bnd = min((((b - u) // gradings[e][i]) for i, (b, u) in enumerate(zip(bounds, used_box))
                               if b is not None and gradings[e][i] > 0), default=None)
                    if bnd is not None and bnd < 0:
                        return
                for p in _partitions_upto(budget, weights[e], bnd):
                    s = size(p)
                    ub = used_box if bounds is None else [u + gradings[e][i] * s for i, u in enumerate(used_box)]
                    chosen.append(p)
                    yield from rec(k + 1, budget - weights[e] * s, ub, chosen)
                    chosen.pop()

            for choice in rec(0, budget, used if bounds is not None else None, []):
                assign_new = dict(zip(new_edges, choice))
                triple = []
                for l in legs:
                    if l.kind != "int":
                        triple.append(())
                    else:
                        p = assign_new[l.edge] if l.edge in assign_new else assign[l.edge]
                        triple.append(p if l.outgoing else conjugate(p))
                c = vertex(*triple)
                if c.is_zero():
                    continue
                term = val * c
                for e, p in assign_new.items():
                    if p:
                        term = term * framing_factor(graph.edges[e].n, p)
                key = (kept + tuple(choice), tuple(cdeg), cw)
                prev = out.get(key)
                out[key] = term if prev is None else prev + term
        states = {k: x.cancel() for k, x in out.items() if not x.is_zero()}
        frontier = new_frontier
    assert not frontier
    result: dict[tuple, QFrac] = {}
    for (fparts, deg, _), val in states.items():
        result[deg] = result[deg] + val if deg in result else val
    return {d: c.cancel() for d, c in result.items() if not c.is_zero()}


def partition_function(graph: GTGraph, max_weight: int, weights: Sequence[int] | None = None,
                       vertex: Vertex = default_vertex) -> Series:
    """Topological vertex partition function Z of the GT graph (KP eq. (4), Def. 2.3):

    Z = sum_lam prod_e (-1)^{|lam_e|(n_e+1)} q^{n_e kappa(lam_e)/2} Q_e^{|lam_e|} prod_v C_{lam_v}(q),

    truncated at sum_e w_e |lam_e| <= max_weight.  Variables Q_e named ``edge.name``; ``weights``
    default to :func:`default_weights` (Kahler weights <h, l_e> if the graph has curve classes,
    else all 1).  ``vertex``: any callable (l1, l2, l3) -> QFrac (default: KP's C).
    Constant term 1."""
    if weights is None:
        weights = default_weights(graph)
    weights = [int(w) for w in weights]
    if len(weights) != len(graph.edges) or any(w <= 0 for w in weights):
        raise ValueError("need one positive weight per internal edge")
    ne = len(graph.edges)
    gradings = [tuple(1 if i == e else 0 for i in range(ne)) for e in range(ne)]
    coeffs = _contract(graph, weights, max_weight, gradings, vertex)
    return Series(graph.edge_names(), coeffs, weights, max_weight)


def partition_function_bruteforce(graph: GTGraph, max_weight: int, weights: Sequence[int] | None = None,
                                  vertex: Vertex = default_vertex) -> Series:
    """Same as :func:`partition_function`, by the literal sum over :func:`multipartitions`
    (reference implementation, used in tests)."""
    if weights is None:
        weights = default_weights(graph)
    names = graph.edge_names()
    out: dict[tuple, QFrac] = {}
    for lams in multipartitions(weights, max_weight):
        term = QFrac.one()
        for e, lam in enumerate(lams):
            if lam:
                term = term * framing_factor(graph.edges[e].n, lam)
        for legs in graph.vertices:
            triple = tuple(() if l.kind != "int" else (lams[l.edge] if l.outgoing else conjugate(lams[l.edge]))
                           for l in legs)
            term = term * vertex(*triple)
        d = tuple(size(l) for l in lams)
        out[d] = out[d] + term if d in out else term
    return Series(names, {d: c.cancel() for d, c in out.items()}, weights, max_weight)


# ----------------------------------------------------------------------------- classes

def _as_graph(obj) -> GTGraph:
    if isinstance(obj, GTGraph):
        return obj
    if isinstance(obj, Triangulation):
        return from_triangulation(obj)
    raise TypeError("expected a GTGraph or a Triangulation")


def class_weights(graph: GTGraph, edge_weights: Sequence[int], basis=None):
    """Weights w_i of the basis classes with w_e = sum_i c_{e,i} w_i for every edge (exact solve;
    raises if no integer solution).  Returns (w, basis, coords)."""
    if basis is None:
        basis = relation_lattice_basis(graph)
    coords = class_coordinates(graph, basis)
    r = len(basis)
    if r == 0:
        return [], basis, coords
    rows = [list(coords[e]) for e in range(len(graph.edges))]
    sol = pg._solve_rational(rows, list(edge_weights))
    if sol is None:
        raise ValueError("edge weights are not linear in the curve classes")
    if any(x.denominator != 1 for x in sol):
        raise ValueError(f"class weights {sol} are not integers; rescale the heights")
    w = [int(x) for x in sol]
    for e in range(len(graph.edges)):
        if sum(c * x for c, x in zip(coords[e], w)) != edge_weights[e]:
            raise ValueError("edge weights are not linear in the curve classes")
    return w, basis, coords


def _class_names(r: int) -> list[str]:
    return [f"t{i + 1}" for i in range(r)]


def class_series(graph: GTGraph, series: Series, basis=None, names: Sequence[str] | None = None):
    """Substitute Q_e -> prod_i t_i^{c_{e,i}} (c_e = :func:`class_coordinates` of l_e in ``basis``,
    default :func:`relation_lattice_basis`).  New weights w_i solve w_e = sum_i c_{e,i} w_i.
    Returns (Series in t1..tr, basis, coords)."""
    w, basis, coords = class_weights(graph, series.weights, basis)
    names = list(names) if names is not None else _class_names(len(basis))
    rows = [list(coords[e]) for e in range(len(graph.edges))]
    return series.substitute(rows, names, w), basis, coords


def class_partition_function(graph: GTGraph, max_weight: int | None = None, bounds=None, basis=None,
                             weights: Sequence[int] | None = None, names=None, coords=None,
                             vertex: Vertex = default_vertex):
    """Z directly in class variables t_i (Q_e = prod_i t_i^{c_{e,i}}), optionally truncated to the
    box d_i <= bounds[i] (needs all c_{e,i} >= 0).  ``coords``: optional explicit {edge: class
    vector} overriding :func:`class_coordinates` (to test alternative monomial substitutions);
    then ``basis`` is only used for its length/labels.  Weights: edge weights (default
    :func:`default_weights`) and class weights solving w_e = sum_i c_{e,i} w_i.  If max_weight is
    None it is set to sum_i w_i bounds[i].  Returns (Series, basis, coords)."""
    if weights is None:
        weights = default_weights(graph)
    if coords is None:
        w, basis, coords = class_weights(graph, weights, basis)
    else:
        r = len(next(iter(coords.values())))
        rows = [list(coords[e]) for e in range(len(graph.edges))]
        sol = pg._solve_rational(rows, list(weights))
        if sol is None or any(x.denominator != 1 or x <= 0 for x in sol):
            # weights not linear in the given classes: use weights 1 on classes
            sol = [1] * r
            weights = [sum(coords[e]) for e in range(len(graph.edges))]
            if any(x <= 0 for x in weights):
                raise ValueError("cannot find positive weights for the given classes")
        w = [int(x) for x in sol]
        if basis is None:
            basis = [None] * r
    r = len(w)
    if max_weight is None:
        if bounds is None or any(b is None for b in bounds):
            raise ValueError("give max_weight or finite bounds")
        max_weight = sum(a * b for a, b in zip(w, bounds))
    gradings = [tuple(coords[e]) for e in range(len(graph.edges))]
    coeffs = _contract(graph, weights, max_weight, gradings, vertex, bounds)
    names = list(names) if names is not None else _class_names(r)
    return Series(names, coeffs, w, max_weight, bounds), basis, coords


# ----------------------------------------------------------------------------- free energy / GV

def free_energy(series: Series) -> Series:
    """F = log Z (:meth:`Series.log`)."""
    return series.log()


def _divide(beta: tuple, k: int):
    if all(b % k == 0 for b in beta):
        return tuple(b // k for b in beta)
    return None


def gv_invariants(F: Series, check: bool = True) -> dict[tuple, dict[int, int]]:
    """Gopakumar--Vafa invariants n^g_beta from F = log Z (physical convention, CONVENTIONS.md):

    f_beta = F_beta - sum_{k>=2, k|beta} (1/k) sum_g n^g_{beta/k} (-1)^{g-1} [k]^{2g-2},
    P = f_beta [1]^2 = sum_g n^g_beta (-1)^{g-1} y^g,  y = [1]^2 = q - 2 + q^{-1}.

    Classes beta are processed by increasing weight.  Raises ValueError (naming beta) if P is
    not a q-symmetric Laurent polynomial in q or some n^g_beta is not an integer.  Returns
    {beta: {g: n^g_beta}} with zero invariants dropped (classes with all n = 0 omitted)."""
    gv: dict[tuple, dict[int, int]] = {}
    zero = (0,) * len(F.variables)
    one_sq = QFrac.qnum(1) ** 2
    for beta, Fb in F.terms():
        if beta == zero:
            continue
        g0 = 0
        for b in beta:
            g0 = gcd(g0, abs(b))
        f = Fb
        for k in range(2, g0 + 1):
            sub = _divide(beta, k)
            if sub is None or sub not in gv:
                continue
            for g, n in gv[sub].items():
                sign = -1 if (g - 1) % 2 else 1
                term = QFrac.qnum(k) ** (2 * g - 2) * Fraction(sign * n, k)
                f = f - term
        P = (f * one_sq).cancel()
        if P.is_zero():
            continue
        if not P.is_laurent():
            raise ValueError(f"GV extraction failed for class {beta}: f_beta [1]^2 = {P} is not a Laurent polynomial")
        if not P.is_q_symmetric():
            raise ValueError(f"GV extraction failed for class {beta}: f_beta [1]^2 is not symmetric under q -> 1/q")
        try:
            cs = P.to_y_polynomial()
        except ValueError as exc:
            raise ValueError(f"GV extraction failed for class {beta}: {exc}") from None
        ns = {}
        for g, c in enumerate(cs):
            n = c * (-1 if (g - 1) % 2 else 1)
            if n.denominator != 1:
                if check:
                    raise ValueError(f"non-integer GV invariant n^{g}_{beta} = {n}")
            if n != 0:
                ns[g] = int(n) if n.denominator == 1 else n
        if ns:
            gv[beta] = ns
    return gv


def gv_check_integrality(gv: dict) -> bool:
    """True iff every n^g_beta in ``gv`` is an integer."""
    return all(isinstance(n, int) or (isinstance(n, Fraction) and n.denominator == 1)
               for d in gv.values() for n in d.values())


def gv_table(gv: dict, genus_max: int | None = None, variables: Sequence[str] | None = None) -> str:
    """Plain-text table: one row per class beta (sorted by total degree), columns g = 0..genus_max."""
    if not gv:
        return "(no nonzero GV invariants)"
    gmax = max(g for d in gv.values() for g in d) if genus_max is None else genus_max
    head_beta = "beta" if variables is None else "(" + ",".join(variables) + ")"
    rows = [[head_beta] + [f"g={g}" for g in range(gmax + 1)]]
    for beta in sorted(gv, key=lambda b: (sum(b), b)):
        rows.append([str(beta[0]) if len(beta) == 1 else str(beta)] +
                    [str(gv[beta].get(g, 0)) for g in range(gmax + 1)])
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    lines = ["  ".join(s.rjust(wd) for s, wd in zip(r, widths)) for r in rows]
    lines.insert(1, "-" * len(lines[0]))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- convenience

def compute(graph_or_triangulation, max_weight: int, classes: bool = True, heights=None, basis=None,
            weights=None) -> dict:
    """Z (edge variables), and, with ``classes``, Z and F in class variables and the GV invariants.
    Returns dict with keys graph, weights, Z, (Z_classes, basis, coords, F, GV, time)."""
    t0 = time.perf_counter()
    g = _as_graph(graph_or_triangulation)
    if weights is None:
        weights = default_weights(g, heights)
    Z = partition_function(g, max_weight, weights)
    out = {"graph": g, "weights": list(weights), "Z": Z}
    if classes and g.edges and all(e.curve_class is not None for e in g.edges):
        Zc, basis, coords = class_series(g, Z, basis)
        F = Zc.log()
        out.update(Z_classes=Zc, basis=basis, coords=coords, F=F, GV=gv_invariants(F))
    else:
        F = Z.log()
        out.update(F=F)
    out["time"] = time.perf_counter() - t0
    return out


def summary(graph_or_triangulation, max_weight: int, **kw) -> str:
    """Human-readable summary of :func:`compute`: framings, classes, Z, GV table."""
    res = compute(graph_or_triangulation, max_weight, **kw)
    g = res["graph"]
    lines = [f"edges: {g.edge_names()}", f"framings: {g.framings()}", f"weights: {res['weights']}"]
    if "coords" in res:
        lines.append("curve classes (basis coordinates): " +
                     ", ".join(f"{g.edges[e].name}={res['coords'][e]}" for e in sorted(res["coords"])))
        lines.append(str(res["Z_classes"]))
        lines.append(gv_table(res["GV"], variables=res["Z_classes"].variables))
    else:
        lines.append(str(res["Z"]))
    lines.append(f"time: {res['time']:.2f}s")
    return "\n".join(lines)
