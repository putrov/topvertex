"""Basic computations with topvertex: partition functions and Gopakumar-Vafa invariants of
the resolved conifold, local P^2, local F0, local F1 and AKMV's Example VI (hep-th/0305132,
Sec. 9, Fig. 16).  Conventions: Konishi math/0504188 (see CONVENTIONS.md); GV numbers in the
physical convention (conifold n^0_1 = +1, local P^2 n^0_1 = 3).

Run:  .venv/bin/python examples/basic_examples.py
"""
import time

from topvertex import examples_geometry as ex
from topvertex.partition_function import (class_partition_function, class_series, gv_invariants, gv_table,
                                          partition_function)
from topvertex.polygon import Triangulation
from topvertex.toric import class_coordinates, from_triangulation


def banner(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def run(name, graph, max_weight=None, bounds=None, basis=None, names=None):
    banner(name)
    print("framings n_e:", graph.framings())
    print("edge classes:", class_coordinates(graph, basis))
    t = time.perf_counter()
    Zc, _, _ = class_partition_function(graph, max_weight=max_weight, bounds=bounds, basis=basis, names=names)
    F = Zc.log()
    gv = gv_invariants(F)
    print(f"[{time.perf_counter() - t:.2f}s]  first terms of F = log Z:")
    for d, c in F.terms()[:4]:
        print("   ", d, c)
    print(gv_table(gv, variables=Zc.variables))
    return gv


def main():
    # Resolved conifold: Z = sum_lam (-Q)^|lam| q^{-kappa/2} s_lam(q^rho)^2, F = -sum Q^k/(k [k]^2)
    g = ex.conifold()
    banner("Resolved conifold O(-1)+O(-1) -> P^1")
    Z = partition_function(g, 4)
    print(Z)
    print(Z.log())
    print(gv_table(gv_invariants(class_series(g, Z)[0].log())))

    run("Local P^2 (degree <= 6)", ex.local_p2(), max_weight=6)
    run("Local F0 = P^1 x P^1 (class box 4 x 4)", ex.local_f0(), bounds=(4, 4))
    run("Local F1 (B, F; class box 4 x 4)", ex.local_f1(), bounds=(4, 4))

    # AKMV Example VI: points (0,0),(1,1),(1,2),(2,1),(2,2),(3,3); basis (s, t1, t2) given by the
    # edges (1,1)-(2,2) [s], (1,1)-(1,2) [t1], (1,2)-(2,2) [t2].  The vertex reproduces AKMV
    # eq. (9.14) up to the sign of the l = 1 row.
    P = [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]
    tris = [[(0, 0), (2, 1), (1, 1)], [(0, 0), (1, 1), (1, 2)], [(1, 1), (2, 1), (2, 2)],
            [(1, 1), (2, 2), (1, 2)], [(3, 3), (2, 2), (2, 1)], [(3, 3), (1, 2), (2, 2)]]
    g = from_triangulation(Triangulation.from_corners_and_triangles(P, tris, points=P))
    seg = {frozenset((g.points[e.segment[0]], g.points[e.segment[1]])): e.curve_class for e in g.edges}
    basis = [seg[frozenset(((1, 1), (2, 2)))], seg[frozenset(((1, 1), (1, 2)))], seg[frozenset(((1, 2), (2, 2)))]]
    run("AKMV Example VI (s, t1, t2) in the box l <= 3, d1, d2 <= 4", g, bounds=(3, 4, 4), basis=basis,
        names=("s", "t1", "t2"))


if __name__ == "__main__":
    main()
