"""Tests for topvertex.nekrasov: the arm/leg Nekrasov factor vs IKP hep-th/0306032 eq. (74), and the
comparison Z_top = Z_pert Z_Nekrasov of the topological vertex with the 5d Nekrasov partition function
of the linear quiver of a ladder triangulation (pure SU(N) included as one node without fundamentals)."""
import cmath
import json
import os
import random
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path

import pytest

from tests.data import QUIVER_CASES
from topvertex import polygon as pg
from topvertex.cli import geometry_report, main
from topvertex.examples_geometry import conifold, hirzebruch, local_p2, su_n_ladder
from topvertex.nekrasov import (arm_leg_shifts, class_dict, compare_with_nekrasov, format_comparison, ij_product_shifts,
                                instanton_numbers, instanton_part, nekrasov_partition_function, perturbative_part,
                                quiver_dictionary, quiver_structure, quiver_structures)
from topvertex.partition_function import partition_function
from topvertex.partitions import partitions_of
from topvertex.qring import QFrac
from topvertex.series import Series
from topvertex.toric import from_triangulation

ROOT = Path(__file__).resolve().parents[1]
SMALL = [p for d in range(5) for p in partitions_of(d)]
CASES = {name: (cols, [Fraction(k) for k in kap], eps) for name, cols, kap, eps in QUIVER_CASES}
UP = (0, 1)


def poly(cols):
    """Points (i, b_i + j), 0 <= j <= N_i, of the columns [(b_i, N_i)]."""
    return [(i, b + j) for i, (b, N) in enumerate(cols) for j in range(N + 1)]


@lru_cache(maxsize=None)
def ladders(name):
    """All unimodular triangulations of the case's polygon that are ladders along (0, 1)."""
    pts = poly(CASES[name][0])
    tris = pg.all_unimodular_triangulations(pg.convex_hull(pts))
    return tuple(T for T in tris if quiver_structure(T, UP) is not None)


def chambers(name, k=2):
    """k triangulations spread deterministically over the ladder triangulations."""
    ts = ladders(name)
    step = max(1, len(ts) // k)
    return ts[::step][:k]


def same(a: dict, b: dict) -> bool:
    z = QFrac.zero()
    return all((a.get(l, z) - b.get(l, z)).is_zero() for l in set(a) | set(b))


# ----------------------------------------------------------------------------- Nekrasov factor

def test_ij_form_equals_arm_leg_form_up_to_boundary():
    """prod_{i,j<=L} [x + mu_i - nu_j + j - i]/[x + j - i] = (boundary) / prod_AL [x + c]: as signed
    multisets of shifts, num - den + AL consists of |mu| + |nu| shifts with |c| > L - K."""
    L = 10
    for mu in SMALL:
        for nu in SMALL:
            for same_ in ([False, True] if mu == nu else [False]):
                num, den = ij_product_shifts(mu, nu, L, same_)
                B = Counter(num)
                B.subtract(Counter(den))
                B.update(Counter(arm_leg_shifts(mu, nu)))
                B = {c: n for c, n in B.items() if n}
                K = max([len(mu), len(nu)] + list(mu[:1]) + list(nu[:1]))
                assert all(n > 0 for n in B.values())
                assert sum(B.values()) == sum(mu) + sum(nu)
                assert all(abs(c) > L - K for c in B)


def _sh(u):
    return 2 * cmath.sinh(u / 2)


def _tuples(N, k):
    """All N-tuples of partitions of total size k."""
    for sizes in product(range(k + 1), repeat=N):
        if sum(sizes) == k:
            yield from product(*(partitions_of(s) for s in sizes))


def test_two_forms_numerically():
    """At random (beta, hbar, a_l): the truncated IKP eq. (74) product times the arm/leg product
    equals the boundary factors (|c| > L - K), checked numerically for U(2), U(3)."""
    rng = random.Random(1)
    L = 9
    for N in (2, 3):
        for k in (1, 2, 3):
            for Ys in _tuples(N, k):
                beta, hbar = rng.uniform(0.3, 0.9), rng.uniform(0.2, 0.7)
                a = [rng.uniform(-1, 1) for _ in range(N)]
                K = max([len(Y) for Y in Ys] + [Y[0] for Y in Ys if Y])
                ratio = 1
                for l in range(N):
                    for n in range(N):
                        x = a[l] - a[n]
                        num, den = ij_product_shifts(Ys[l], Ys[n], L, l == n)
                        for c in num:
                            ratio *= _sh(beta * (x + hbar * c))
                        for c in den:
                            ratio /= _sh(beta * (x + hbar * c))
                        for c in arm_leg_shifts(Ys[l], Ys[n]):
                            ratio *= _sh(beta * (x + hbar * c))
                        B = Counter(num)
                        B.subtract(Counter(den))
                        B.update(Counter(arm_leg_shifts(Ys[l], Ys[n])))
                        for c, mult in B.items():
                            assert mult == 0 or abs(c) > L - K
                            ratio /= _sh(beta * (x + hbar * c)) ** mult
                assert abs(ratio - 1) < 1e-9


# ----------------------------------------------------------------------------- structure

@pytest.mark.parametrize("name", sorted(CASES))
def test_structure(name):
    cols, kap, eps = CASES[name]
    ts = ladders(name)
    assert ts
    pts = poly(cols)
    for T in ts:
        s = quiver_structure(T, UP)
        assert s["direction"] == UP and s["r"] == len(cols) - 2
        assert s["N"] == [N for _, N in cols]
        assert [[T.points[v] for v in c] for c in s["columns"]] == \
            [[(i, b + j) for j in range(N + 1)] for i, (b, N) in enumerate(cols)]
        assert s["kappa"] == kap and s["eps"] == eps
        assert UP in [x["direction"] for x in quiver_structures(T)]
        # -d: columns reversed and kappa -> -kappa
        m = quiver_structure(T, (0, -1))
        assert m["kappa"] == [-k for k in reversed(kap)] and m["N"] == list(reversed(s["N"]))
    assert len(pts) == len(ts[0].points)


def _phi_doubled_all_k(dic, graph, i):
    """2 Phi_i from the defining formula, separately for every k (independent re-implementation)."""
    h, y, st = dic["heights"], dic["y"], dic["structure"]
    N, kap = st["N"], st["kappa"][i - 1]
    wt = lambda v: sum(a * b for a, b in zip(h, v))
    sub = lambda a, b: tuple(p - q for p, q in zip(a, b))
    absv = lambda v: v if wt(v) > 0 else tuple(-a for a in v)
    out = []
    for k in range(1, N[i] + 1):
        v = [Fraction(2 * c) for c in graph.edges[dic["horizontal"][(i, k)]].curve_class]
        v = [a + 2 * kap * b for a, b in zip(v, sub(y[(i, k)], y[(i, 1)]))]
        for kk in range(1, N[i] + 1):
            if kk != k:
                v = [a - 2 * b for a, b in zip(v, absv(sub(y[(i, k)], y[(i, kk)])))]
        for j in (i - 1, i + 1):
            for l in range(1, N[j] + 1):
                v = [a + b for a, b in zip(v, absv(sub(y[(i, k)], y[(j, l)])))]
        out.append(tuple(v))
    return out


@pytest.mark.parametrize("name", ["SU2 Nf=4", "SU3 Nf=2", "SU2 x SU3", "T3", "SU2 x SU2 flavours"])
def test_dictionary_assertions(name):
    """In every chamber: diagonal classes = y(upper) - y(lower) along each strip, Phi_i independent
    of k, and the edge form of phi_i reproduces Phi_i."""
    for T in ladders(name):
        g = from_triangulation(T)
        dic = quiver_dictionary(g, direction=UP)
        y = dic["y"]
        for st in dic["strips"]:
            for j, e in enumerate(st["diagonals"]):
                lo, hi = st["branes"][j], st["branes"][j + 1]
                assert g.edges[e].curve_class == tuple(a - b for a, b in zip(y[hi], y[lo]))
        for i in range(1, dic["structure"]["r"] + 1):
            vals = _phi_doubled_all_k(dic, g, i)
            assert all(v == vals[0] for v in vals) and vals[0] == dic["Phi2"][i - 1]
            ph = dic["phi"][i - 1]
            vec = [sum(c * g.edges[e].curve_class[t] for e, c in ph["edges"].items()) for t in range(len(y[(1, 1)]))]
            assert [2 * a for a in vec] == list(dic["Phi2"][i - 1])
            assert next(iter(ph["edges"])) == dic["horizontal"][(i, 1)]


def test_not_a_quiver():
    cols = [(0, 0), (0, 1), (-1, 2), (-1, 1), (-1, 0)]  # some triangle spans two strips
    tris = pg.all_unimodular_triangulations(pg.convex_hull(poly(cols)))
    assert tris and all(quiver_structure(T, UP) is None for T in tris)
    assert quiver_structures(local_p2()) == [] and quiver_structure(local_p2()) is None
    with pytest.raises(ValueError):
        compare_with_nekrasov(local_p2(), 2)


def _random_gl2z(rng):
    A = [[1, 0], [0, 1]]
    for _ in range(6):
        t = rng.randint(-2, 2)
        E = rng.choice([[[1, t], [0, 1]], [[1, 0], [t, 1]], [[0, 1], [1, 0]], [[-1, 0], [0, 1]]])
        A = [[sum(A[i][k] * E[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    return A


def test_su_n_ladder_recognised():
    """su_n_ladder(N, m) and GL(2,Z) images are one-node quivers U(N) without fundamentals; along the
    image of (0, 1), kappa = -m and eps = (-1)^m for either orientation of the transformation; the
    other triangulations of the same polygon are not ladders along (0, 1)."""
    rng = random.Random(3)
    for N in range(2, 5):
        for m in range(N + 1):
            T = su_n_ladder(N, m).triangulation
            s = quiver_structure(T)
            assert (s["direction"], s["N"], s["kappa"], s["eps"]) == ((0, 1), [0, N, 0], [-m], [(-1) ** m])
            for _ in range(3):
                A = _random_gl2z(rng)
                T2 = T.transformed(A, (rng.randint(-3, 3), rng.randint(-3, 3)))
                s2 = quiver_structure(T2, (A[0][1], A[1][1]))
                assert (s2["N"], s2["kappa"], s2["eps"]) == ([0, N, 0], [-m], [(-1) ** m])
                d = quiver_structure(T2)
                assert d["N"] == [0, N, 0] and abs(d["kappa"][0]) == m
            for e in T.flippable_edges():
                assert quiver_structure(T.flip(e), (0, 1)) is None


# ----------------------------------------------------------------------------- comparison

@pytest.mark.parametrize("name", sorted(CASES))
def test_compare(name):
    r = len(CASES[name][0]) - 2
    for T in chambers(name):
        res = compare_with_nekrasov(T, 4, direction=UP)
        assert res["match"] and res["mismatches"] == [], format_comparison(res)
        assert res["n_classes"] > 0 and len(res["max_instanton"]) == r and min(res["max_instanton"]) >= 1
        assert res["kappa"] == CASES[name][1] and res["eps"] == CASES[name][2]
        assert "MATCH for all" in format_comparison(res)
    # the default direction (fewest gauge nodes) also matches
    assert compare_with_nekrasov(chambers(name, 1)[0], 4)["match"]


@pytest.mark.parametrize("name", ["SU2 Nf=2 (1+1)", "U1 x U1", "SU2 Nf=1"])
def test_compare_weight6(name):
    for T in chambers(name, 3):
        assert compare_with_nekrasov(T, 6, direction=UP)["match"]


def test_gl2z_invariance():
    """Sheared, rotated and reflected copies are recognised along the image direction and match."""
    T = chambers("SU2 x SU2 flavours", 1)[0]
    s = quiver_structure(T, UP)
    for A, t in ((((1, 2), (0, 1)), (3, -1)), (((0, -1), (1, 0)), (0, 0)), (((2, 1), (1, 1)), (-2, 5))):
        T2 = T.transformed(A, t)
        d = (A[0][1], A[1][1])  # image of (0, 1)
        s2 = quiver_structure(T2, d)
        assert s2 is not None and (s2["kappa"], s2["eps"], s2["N"]) == (s["kappa"], s["eps"], s["N"])
        assert compare_with_nekrasov(T2, 4, direction=d)["match"]
    T3 = chambers("SU2 Nf=2 (2+0)", 1)[0].transformed(((0, 1), (1, 0)))  # reflection
    assert compare_with_nekrasov(T3, 4, direction=(1, 0))["match"]


def test_user_heights():
    """Other heights inducing the same triangulation (here 3 h + affine): weights scale by 3."""
    T = chambers("SU2 Nf=2 (1+1)", 1)[0]
    h = [3 * a + p[0] - 2 * p[1] + 1 for a, p in zip(pg.heights(T), T.points)]
    res = compare_with_nekrasov(T, 12, heights=h, direction=UP)
    assert res["match"] and res["n_classes"] == compare_with_nekrasov(T, 4, direction=UP)["n_classes"]


def test_su_n_ladder_dictionary():
    """Pure SU(N): kappa = -m, eps = (-1)^m, phi = (-1)^m Q_B prod_j Q_{F_j}^{-(N-j)} with B the bottom
    column segment and F_j the left diagonal (c_j, L) (module docstring of topvertex.nekrasov)."""
    for N in (2, 3, 4):
        for m in range(N + 1):
            g = su_n_ladder(N, m)
            s = quiver_structure(g)
            assert s["direction"] == UP and s["kappa"] == [-m] and s["eps"] == [(-1) ** m]
            dic = quiver_dictionary(g, s)
            seg = {frozenset(g.points[i] for i in e.segment): k for k, e in enumerate(g.edges)}
            want = {seg[frozenset(((0, 0), (0, 1)))]: 1}
            for j in range(1, N):
                want[seg[frozenset(((0, j), (-1, 1)))]] = -(N - j)
            assert dic["phi"][0] == {"sign": (-1) ** m, "edges": want}
    res = compare_with_nekrasov(su_n_ladder(3, 2), 4)
    assert res["match"] and res["gauge_group"] == "U(3)" and res["flavours"] == (0, 0)


# ----------------------------------------------------------------------------- pure SU(N) and U(1)

#: Kahler weight of the SU(N) comparison (instanton numbers reached: up to D with the default heights).
SU_N_WEIGHT = {2: 8, 3: 7, 4: 6}


@pytest.mark.parametrize("N,m", [(N, m) for N in (2, 3, 4) for m in range(N + 1)])
def test_pure_su_n(N, m):
    """Pure SU(N) at CS level kappa = -m: Z_top = Z_pert Z_Nek for all classes up to the weight."""
    D = SU_N_WEIGHT[N]
    res = compare_with_nekrasov(su_n_ladder(N, m), D)
    assert res["match"], format_comparison(res)
    assert res["gauge_group"] == f"U({N})" and res["flavours"] == (0, 0) and res["kappa"] == [-m]
    assert res["max_instanton"][0] >= 4 and res["n_classes"] > 0
    assert f"U({N}), no fundamentals, CS level kappa = {-m}" in format_comparison(res)


def test_su2_one_instanton():
    """SU(2): Y = ((1), ()) and ((), (1)) give 1/([1]^2 [a_12]^2) = Q_F/([1]^2 (1 - Q_F)^2), the CS
    factors 1 and Q_F^{-kappa} (frame a_1 = 0), so the instanton-number-1 part of Z_Nek is
    phi (1 + Q_F^{-kappa}) Q_F / ([1]^2 (1 - Q_F)^2) with phi = (-1)^m Q_B Q_F^{-1}, kappa = -m,
    i.e. the coefficient of B + d F is (-1)^m ((d + 1) + (d - m + 1)_+) / [1]^2 (m = 0: local F0,
    n^0_{B + dF} = -2(d + 1))."""
    D = 7
    inv = QFrac.qnum(1) ** -2
    for m in range(3):
        g = su_n_ladder(N=2, m=m)
        seg = {frozenset(g.points[i] for i in e.segment): e for e in g.edges}
        B = seg[frozenset(((0, 0), (0, 1)))].curve_class
        F = seg[frozenset(((0, 1), (-1, 1)))].curve_class
        h = pg.heights(g.triangulation)
        wt = lambda l: sum(a * b for a, b in zip(h, l))
        s = quiver_structure(g)
        nek = nekrasov_partition_function(g, D, struct=s)
        one = {l: c for l, c in nek.items() if instanton_numbers(s, l) == (1,)}
        want = {}
        for d in range(D + 1):
            l = tuple(b + d * f for b, f in zip(B, F))
            if wt(l) <= D:
                want[l] = inv * ((-1) ** m * ((d + 1) + max(0, d - m + 1)))
        assert want and set(one) == set(want)
        assert all(one[l] == want[l] for l in want), m


def test_su2_wrong_kappa_fails():
    """SU(2) at m = 1: only the dictionary's kappa = -1 and eps = -1 reproduce Z_top / Z_pert."""
    g = su_n_ladder(2, 1)
    inst = instanton_part(g, 5)
    assert same(inst, nekrasov_partition_function(g, 5))
    for k in (0, 1, 2, -2):
        assert not same(inst, nekrasov_partition_function(g, 5, kappa=[k]))
    assert not same(inst, nekrasov_partition_function(g, 5, eps=[1]))


def test_su_n_transformed_and_hirzebruch():
    """An SL(2,Z)-transformed ladder matches along the image direction with the same dictionary;
    hirzebruch(m) (local F_m, the mirror image of su_n_ladder(2, m)) matches with kappa = +m."""
    T = su_n_ladder(3, 2).triangulation.transformed(((3, -1), (-5, 2)), (1, 7))
    res = compare_with_nekrasov(T, 5, direction=(-1, 2))
    assert res["match"] and (res["kappa"], res["eps"]) == ([-2], [1])
    for m in range(3):
        res = compare_with_nekrasov(hirzebruch(m), 5)
        assert res["match"] and res["kappa"] == [m] and res["direction"] == (0, 1)
    assert compare_with_nekrasov(hirzebruch(0), 5)["other_directions"] == [(1, 0)]


def test_u1_no_flavours():
    """U(1) without fundamentals on the triangle (0,0),(2,0),(1,1) (kappa = -1, phi = -Q): Z_pert = 1,
    so the instanton part is the vertex Z itself, and it equals Z_Nek."""
    T = pg.all_unimodular_triangulations([(0, 0), (2, 0), (1, 1)])
    assert len(T) == 1
    g = from_triangulation(T[0])
    s = quiver_structure(g)
    assert (s["N"], s["kappa"], s["eps"]) == ([0, 1, 0], [-1], [-1])
    Z = partition_function(g, 6)
    zero = (0,) * len(g.points)
    assert perturbative_part(g, Z, s) == {zero: QFrac.one()}
    assert same(instanton_part(g, Z, struct=s), class_dict(g, Z))
    res = compare_with_nekrasov(g, 6, Z=Z)
    assert res["match"] and res["max_instanton"] == (6,)
    assert "U(1), no fundamentals, CS level kappa = -1" in format_comparison(res)
    assert dict(res["phi"][0]) == {"sign": -1, "edges": {0: 1}}


def test_conifold_is_u1():
    """The conifold is U(1) at kappa = 0 (along the diagonal): Z_Nek = sum_Y (-Q)^{|Y|} / prod_s [h(s)]^2
    = exp(-sum_k Q^k / (k [k]^2)) = Z_top."""
    g = conifold()
    s = quiver_structure(g)
    assert (s["N"], s["kappa"]) == ([0, 1, 0], [0])
    assert compare_with_nekrasov(g, 5)["match"]
    nek = nekrasov_partition_function(g, 5, struct=s)
    T = g.edges[0].curve_class
    coeffs = {(k,): nek[tuple(k * t for t in T)] for k in range(6)}
    assert len(nek) == 6
    F = Series(("Q",), coeffs, [1], 5, (5,)).log()
    for k in range(1, 6):
        assert F[(k,)] == -QFrac.qnum(k) ** -2 * Fraction(1, k)


def test_wrong_dictionary_fails():
    """The check is not vacuous: Z_top/Z_pert = Z_Nek only for the right kappa and eps."""
    for name, bad_eps, bad_kappa in (("SU2 Nf=2 (1+1)", [1], [0]), ("SU2 x SU2 flavours", [-1, -1], [0, Fraction(-1, 2)]),
                                     ("U1 x U1", [1, 1], [Fraction(1, 2), Fraction(-1, 2)])):
        g = from_triangulation(chambers(name, 1)[0])
        inst = instanton_part(g, 4, direction=UP)
        assert same(inst, nekrasov_partition_function(g, 4, direction=UP))
        assert not same(inst, nekrasov_partition_function(g, 4, direction=UP, eps=bad_eps))
        assert not same(inst, nekrasov_partition_function(g, 4, direction=UP, kappa=bad_kappa))


def test_vertex_side_helpers():
    from topvertex.partition_function import partition_function
    g = from_triangulation(chambers("SU2 x SU2", 1)[0])
    s = quiver_structure(g, UP)
    dic = quiver_dictionary(g, s)
    for (i, k), e in dic["horizontal"].items():
        assert instanton_numbers(s, g.edges[e].curve_class) == tuple(int(j == i) for j in (1, 2))
    for st in dic["strips"]:
        for e in st["diagonals"]:
            assert instanton_numbers(s, g.edges[e].curve_class) == (0, 0)
    Z = partition_function(g, 4)
    top = class_dict(g, Z)
    pert = perturbative_part(g, Z, s)
    assert pert and set(pert) <= set(top) and len(pert) < len(top)
    assert same(instanton_part(g, Z, struct=s), instanton_part(g, 4, struct=s))
    res = compare_with_nekrasov(g, 4, direction=UP, Z=Z)
    assert res["match"]


# ----------------------------------------------------------------------------- CLI and backends

def test_cli_quiver(capsys):
    assert main(["p=0,0_3,0_3,1_2,2_1,2_0,1", "--pick", "0", "--order", "3", "--nekrasov"]) == 0
    out = capsys.readouterr().out
    assert "linear quiver along (0, 1): U(2) x U(2), fundamentals 1 + 1, CS levels kappa = (-1/2, -1/2)" in out
    assert "Nekrasov comparison: MATCH for all" in out
    T = pg.all_unimodular_triangulations(pg.parse_polygon("p=0,0_3,0_3,1_2,2_1,2_0,1"))[0]
    assert geometry_report(T)["heights"] == pg.heights(T)


def test_cli_nekrasov_subprocess():
    """Plain --nekrasov on local F1 (pure SU(2), kappa = -1) and on a geometry that is not a quiver."""
    tris = "-1,1 0,0 0,1; -1,1 0,1 0,2; 0,0 1,0 0,1; 0,1 1,0 0,2"
    r = subprocess.run([sys.executable, "-m", "topvertex", "p=0,0_0,2_-1,1_1,0", "--triangles", tris,
                        "--order", "3", "--nekrasov"], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "linear quiver along (0, 1): U(2), no fundamentals, CS level kappa = -1" in r.stdout
    assert "phi_1 = -" in r.stdout and "Nekrasov comparison: MATCH for all" in r.stdout
    r = subprocess.run([sys.executable, "-m", "topvertex", "p=1,0_0,1_-1,-1", "--order", "1", "--nekrasov"],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "not recognised as a linear quiver" in r.stdout
    r = subprocess.run([sys.executable, "-m", "topvertex", "p=1,0_0,1_-1,-1", "--nekrasov", "2,3"],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode != 0  # --nekrasov takes no argument any more


_CHILD = r"""
import json
from topvertex import polygon as pg, qring
from topvertex.nekrasov import compare_with_nekrasov, quiver_structure
from topvertex.examples_geometry import su_n_ladder
out = {"backend": qring.BACKEND, "lp": pg.lp_backend(), "match": []}
for cols in ([(0, 1), (0, 2), (0, 1)], [(0, 0), (0, 1), (0, 1), (0, 0)], [(1, 0), (0, 2), (0, 2), (1, 0)]):
    pts = [(i, b + j) for i, (b, N) in enumerate(cols) for j in range(N + 1)]
    for T in pg.all_unimodular_triangulations(pg.convex_hull(pts))[:4]:
        if quiver_structure(T, (0, 1)):
            out["match"].append(compare_with_nekrasov(T, 4, direction=(0, 1))["match"])
for N, m in ((2, 1), (3, 2)):
    out["match"].append(compare_with_nekrasov(su_n_ladder(N, m), 5)["match"])
print(json.dumps(out))
"""


def test_pure_python_backends():
    env = dict(os.environ, TOPVERTEX_NO_FLINT="1", TOPVERTEX_NO_SCIPY="1")
    r = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, cwd=ROOT,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["backend"] == "python" and out["lp"] == "exact"
    assert out["match"] and all(out["match"])
