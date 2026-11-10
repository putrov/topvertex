"""Tests for topvertex.partition_function: gluing (KP eq. 4), free energy, GV invariants.

All literature numbers come from tests/data.py."""
from fractions import Fraction

import pytest

from tests.data import (AKMV_EXVI, LOCAL_F0_GENUS0, LOCAL_F1_GENUS0, LOCAL_F2_GENUS0, LOCAL_P2_GENUS0,
                        LOCAL_P2_GV)
from topvertex import examples_geometry as ex
from topvertex.partition_function import (class_partition_function, class_series, compute, default_weights,
                                          free_energy, gv_check_integrality, gv_invariants, gv_table,
                                          partition_function, partition_function_bruteforce, summary)
from topvertex.partitions import kappa, partitions_of
from topvertex.polygon import Triangulation
from topvertex.qring import QFrac
from topvertex.schur import schur_rho
from topvertex.toric import class_coordinates, from_triangulation, from_web, web_of


def qn(k):
    return QFrac.qnum(k)


def single_edge_Zd(n, d):
    """CONVENTIONS.md: Z_d = (-1)^{(n+1)d} sum_{lam |- d} q^{(n-1) kappa(lam)/2} s_lam(q^rho)^2."""
    tot = QFrac.zero()
    for lam in partitions_of(d):
        tot = tot + QFrac.x_pow((n - 1) * kappa(lam)) * schur_rho(lam) ** 2
    return tot * (-1) ** ((n + 1) * d)


def gv_of(graph, **kw):
    Zc, basis, coords = class_partition_function(graph, **kw)
    return gv_invariants(Zc.log())


# ------------------------------------------------------------------------- single edge

def test_conifold():
    g = ex.conifold()
    Z = partition_function(g, 4)
    assert Z.variables == ("Q0",)
    for d in range(5):
        assert Z.coeff((d,)) == (-1) ** d * sum(
            (QFrac.x_pow(-kappa(l)) * schur_rho(l) ** 2 for l in partitions_of(d)), QFrac.zero())
    F = free_energy(Z)
    for d in range(1, 5):
        assert F.coeff((d,)) == QFrac(Fraction(-1, d)) / qn(d) ** 2
    Zc, _, _ = class_series(g, Z)
    assert gv_invariants(Zc.log()) == {(1,): {0: 1}}


def test_o0_m2_curve():
    g = ex.op1_bundle(1)
    F = partition_function(g, 4).log()
    for d in range(1, 5):
        assert F.coeff((d,)) == QFrac(Fraction(1, d)) / qn(d) ** 2
    assert gv_invariants(F) == {(1,): {0: -1}}


@pytest.mark.parametrize("n", [-1, 0, 1, 2])
def test_op1_bundle_formula(n):
    g = ex.op1_bundle(n)
    assert g.framings() == [n]
    Z = partition_function(g, 3, weights=[1])
    for d in range(4):
        assert Z.coeff((d,)) == single_edge_Zd(n, d)


# ------------------------------------------------------------------------- algorithm cross-checks

@pytest.mark.parametrize("name,graph,D", [
    ("conifold", ex.conifold, 4), ("P2", ex.local_p2, 4), ("F0", ex.local_f0, 4), ("F1", ex.local_f1, 4),
    ("F2", ex.local_f2, 4), ("dP3", ex.local_dp3, 3), ("flop", ex.kp_flop_f1, 3),
    ("ladder", lambda: ex.su_n_ladder(3, 1), 3)])
def test_elimination_equals_bruteforce(name, graph, D):
    g = graph()
    assert partition_function(g, D) == partition_function_bruteforce(g, D)


def test_class_partition_function_equals_substitution():
    g = ex.local_f1()
    Z = partition_function(g, 6)
    Zc, basis, coords = class_series(g, Z)
    Zc2, _, _ = class_partition_function(g, 6)
    assert Zc == Zc2 and Zc.weights == (1, 1)
    # box truncation agrees on the box
    Zb, _, _ = class_partition_function(g, bounds=(2, 3))
    for d, c in Zb.terms():
        assert Zc.coeff(d) == c


# ------------------------------------------------------------------------- local P^2

def test_local_p2_gv_all_genera():
    g = ex.local_p2()
    assert set(class_coordinates(g).values()) == {(1,)}
    Zc, _, _ = class_series(g, partition_function(g, 6))
    assert Zc.variables == ("t1",)
    gv = gv_invariants(Zc.log())
    assert gv_check_integrality(gv)
    for d, row in LOCAL_P2_GV.items():
        ours = gv[(d,)]
        assert [ours.get(gg, 0) for gg in range(len(row))] == row, d
    # genera beyond the table: d = 4 stops at g = 3; top genus (d-1)(d-2)/2
    assert max(gv[(4,)]) == 3 and max(gv[(5,)]) == 6 and max(gv[(6,)]) == 10


def test_local_p2_genus0_degree10():
    gv = gv_of(ex.local_p2(), max_weight=10)
    assert [gv[(d,)][0] for d in range(1, 11)] == LOCAL_P2_GENUS0


def _p2_gv(T):
    g = from_triangulation(T)
    Zc, _, _ = class_series(g, partition_function(g, 5))
    return Zc, g.framings()


def test_local_p2_invariance():
    T = ex.local_p2().triangulation
    Z0, fr0 = _p2_gv(T)
    # SL(2,Z) + translation, GL(2,Z) reflections (flip all framings)
    for A, t, flip in [(((2, 1), (1, 1)), (3, -2), False), (((1, 0), (0, -1)), (0, 0), True),
                       (((0, 1), (1, 0)), (1, 1), True), (((-1, 0), (0, -1)), (0, 0), False)]:
        Z, fr = _p2_gv(T.transformed(A, t))
        assert Z == Z0
        if flip:
            assert sorted(fr) == sorted(-n for n in fr0)
    # relabel the points
    perm = [2, 0, 3, 1]
    pts = [None] * 4
    for old, new in enumerate(perm):
        pts[new] = T.points[old]
    Z, _ = _p2_gv(Triangulation(pts, [tuple(perm[i] for i in t) for t in T.triangles]))
    assert Z == Z0


def test_web_input():
    g = ex.local_p2()
    gw = from_web(*web_of(g), n_points=4)
    assert partition_function(gw, 4) == partition_function(g, 4)
    gw2 = from_web(*web_of(g))  # no classes: weights all one
    assert default_weights(gw2) == [1, 1, 1]
    assert partition_function(gw2, 4).coefficients == partition_function(g, 4).coefficients


# ------------------------------------------------------------------------- Hirzebruch surfaces

def test_local_f0_genus0_fast():
    gv = gv_of(ex.local_f0(), max_weight=6)
    for beta, n in LOCAL_F0_GENUS0.items():
        if sum(beta) <= 6:
            assert gv.get(beta, {}).get(0, 0) == n, beta
    for beta, ns in gv.items():
        assert gv[beta[::-1]] == ns  # B <-> F symmetry
        if sum(beta) <= 3:
            assert ns.get(1, 0) == 0
    assert gv_check_integrality(gv)


@pytest.mark.slow
def test_local_f0_genus0_box6():
    gv = gv_of(ex.local_f0(), bounds=(6, 6))
    for beta, n in LOCAL_F0_GENUS0.items():
        assert gv.get(beta, {}).get(0, 0) == n, beta


def test_local_f1():
    g = ex.local_f1()
    assert class_coordinates(g) == {0: (1, 0), 1: (0, 1), 2: (1, 1), 3: (0, 1)}
    gv = gv_of(g, bounds=(6, 6))
    for beta, n in LOCAL_F1_GENUS0.items():
        assert gv.get(beta, {}).get(0, 0) == n, beta
    # d(B+F) is the pull-back of d H from P^2: n^g_{(d,d)}(F1) = n^g_d(P^2)
    for d in range(1, 7):
        row = LOCAL_P2_GV[d]
        assert [gv[(d, d)].get(gg, 0) for gg in range(len(row))] == row


def test_local_f2():
    g = ex.local_f2()
    assert class_coordinates(g) == {0: (1, 0), 1: (0, 1), 2: (1, 2), 3: (0, 1)}
    gv = gv_of(g, bounds=(6, 6))
    for beta, n in LOCAL_F2_GENUS0.items():
        if beta != (1, 0):
            assert gv.get(beta, {}).get(0, 0) == n, beta
    # the (-2)-curve B (normal bundle O(0)+O(-2)) behaves like op1_bundle(1): n = -1 (CKYZ: -1/2)
    assert gv[(1, 0)] == {0: -1}


# ------------------------------------------------------------------------- AKMV Example VI

AKMV_POINTS = [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]
AKMV_TRIS = [[(0, 0), (2, 1), (1, 1)], [(0, 0), (1, 1), (1, 2)], [(1, 1), (2, 1), (2, 2)],
             [(1, 1), (2, 2), (1, 2)], [(3, 3), (2, 2), (2, 1)], [(3, 3), (1, 2), (2, 2)]]


def akmv_graph():
    T = Triangulation.from_corners_and_triangles(AKMV_POINTS, AKMV_TRIS, points=AKMV_POINTS)
    return from_triangulation(T)


def akmv_table():
    """{(g, (l, d1, d2)): n} symmetrised in d1 <-> d2."""
    out = {}
    for gg, tab in AKMV_EXVI.items():
        for (l, d1, d2), v in tab.items():
            out[(gg, (l, d1, d2))] = v
            out[(gg, (l, d2, d1))] = v
    return out


def akmv_seg_index(g):
    P = g.points
    return {frozenset((P[e.segment[0]], P[e.segment[1]])): k for k, e in enumerate(g.edges)}


def akmv_basis(g):
    idx = akmv_seg_index(g)
    s = g.edges[idx[frozenset(((1, 1), (2, 2)))]].curve_class
    t1 = g.edges[idx[frozenset(((1, 1), (1, 2)))]].curve_class
    t2 = g.edges[idx[frozenset(((1, 2), (2, 2)))]].curve_class
    return [s, t1, t2]


def test_akmv_exvi_geometry():
    g = akmv_graph()
    idx = akmv_seg_index(g)
    coords = class_coordinates(g, akmv_basis(g))
    expect = {((1, 1), (2, 2)): ((1, 0, 0), 0), ((1, 1), (1, 2)): ((0, 1, 0), 1), ((1, 1), (2, 1)): ((0, 1, 0), 1),
              ((1, 2), (2, 2)): ((0, 0, 1), 1), ((2, 1), (2, 2)): ((0, 0, 1), 1),
              ((0, 0), (1, 1)): ((1, 1, 0), 2), ((2, 2), (3, 3)): ((1, 0, 1), 2)}
    assert len(g.edges) == 7
    for seg, (cls, absn) in expect.items():
        k = idx[frozenset(seg)]
        assert coords[k] == cls and abs(g.edges[k].n) == absn


def _akmv_gv(coords=None, bounds=(3, 4, 4)):
    g = akmv_graph()
    Zc, _, _ = class_partition_function(g, bounds=bounds, basis=akmv_basis(g), coords=coords,
                                        names=("s", "t1", "t2"))
    return gv_invariants(Zc.log(), check=False)


def test_akmv_exvi():
    bounds = (3, 4, 4)
    gv = _akmv_gv(bounds=bounds)
    assert gv_check_integrality(gv)
    akmv = akmv_table()
    ours = {(gg, b): n for b, d in gv.items() for gg, n in d.items()}
    in_box = lambda b: all(x <= y for x, y in zip(b, bounds))
    # magnitudes: same support and same |n^g| everywhere in the box
    assert {k for k in akmv if in_box(k[1])} == set(ours)
    for k, v in ours.items():
        assert abs(v) == abs(akmv[k]), k
    # physical signs (CONVENTIONS.md conventions)
    assert gv[(0, 1, 0)] == {0: -2} and gv[(2, 2, 0)][0] == -6 and gv[(3, 3, 0)][0] == 27
    # observed sign relation: identical to AKMV for l = 0, 2, 3 (all genera), opposite for l = 1
    for (gg, b), v in ours.items():
        assert v == (-akmv[(gg, b)] if b[0] == 1 else akmv[(gg, b)]), (gg, b)
    # t2 -> infinity limit is local F1 (s = base, t1 = fiber): matches CKYZ, fixing the l = 1 sign
    for (l, d1), n in LOCAL_F1_GENUS0.items():
        if l <= 3 and d1 <= 4 and (l, d1) != (0, 0):
            assert gv.get((l, d1, 0), {}).get(0, 0) == n, (l, d1)


def test_akmv_exvi_alternative_reading_fails():
    """Q_{R1}, Q_{R4} -> Q_s (instead of Q_s Q_t1, Q_s Q_t2) does not reproduce AKMV."""
    g = akmv_graph()
    idx = akmv_seg_index(g)
    alt = {}
    for seg, cls in {((1, 1), (2, 2)): (1, 0, 0), ((0, 0), (1, 1)): (1, 0, 0), ((2, 2), (3, 3)): (1, 0, 0),
                     ((1, 1), (1, 2)): (0, 1, 0), ((1, 1), (2, 1)): (0, 1, 0),
                     ((1, 2), (2, 2)): (0, 0, 1), ((2, 1), (2, 2)): (0, 0, 1)}.items():
        alt[idx[frozenset(seg)]] = cls
    gv = _akmv_gv(coords=alt, bounds=(2, 2, 2))
    akmv = akmv_table()
    assert abs(gv[(1, 0, 0)][0]) != abs(akmv[(0, (1, 0, 0))])  # 3 vs 1


# ------------------------------------------------------------------------- convenience API

def test_compute_and_summary():
    res = compute(ex.local_p2(), 3)
    expected = {(d,): {g: n for g, n in enumerate(LOCAL_P2_GV[d]) if n} for d in (1, 2, 3)}
    assert res["GV"] == expected
    s = summary(ex.local_p2().triangulation, 3)
    assert "27" in s and "-10" in s and "framings" in s
    t = gv_table(res["GV"])
    assert t.splitlines()[0].split() == ["beta", "g=0", "g=1"]


def test_gv_rejects_nonintegral():
    from topvertex.series import Series
    F = Series(("Q",), {(1,): QFrac(Fraction(-1, 2)) / qn(1) ** 2}, max_weight=1)
    with pytest.raises(ValueError, match=r"\(1,\)"):
        gv_invariants(F)


@pytest.mark.slow
def test_larger_orders_integrality():
    """Deeper consistency checks: integrality (and literature values) at higher order."""
    gv = gv_of(ex.local_p2(), max_weight=12)
    assert gv_check_integrality(gv)
    assert [gv[(d,)][0] for d in range(1, 11)] == LOCAL_P2_GENUS0
    gv = gv_of(ex.local_f0(), bounds=(8, 8))
    assert gv_check_integrality(gv)
    gv = _akmv_gv(bounds=(4, 5, 5))
    assert gv_check_integrality(gv)
