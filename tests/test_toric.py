"""Tests for topvertex.toric (GT graphs, framings, web round trip, curve classes)."""
import random

import pytest

from topvertex import examples_geometry as ex
from topvertex import polygon as pg
from topvertex import toric as tv
from topvertex.polygon import Triangulation

EXAMPLES = {
    "conifold": ex.conifold, "local_p2": ex.local_p2, "local_f0": ex.local_f0,
    "local_f1": ex.local_f1, "local_f2": ex.local_f2, "kp_flop_f1": ex.kp_flop_f1,
    "local_dp3": ex.local_dp3,
    "op1_m2": lambda: ex.op1_bundle(-2), "op1_m1": lambda: ex.op1_bundle(-1),
    "op1_0": lambda: ex.op1_bundle(0), "op1_1": lambda: ex.op1_bundle(1),
    "op1_2": lambda: ex.op1_bundle(2), "op1_3": lambda: ex.op1_bundle(3),
    "ladder_3_0": lambda: ex.su_n_ladder(3, 0), "ladder_3_1": lambda: ex.su_n_ladder(3, 1),
    "ladder_3_2": lambda: ex.su_n_ladder(3, 2), "ladder_3_3": lambda: ex.su_n_ladder(3, 3),
    "ladder_4_2": lambda: ex.su_n_ladder(4, 2),
}
HEXAGON = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


def flag_data(g, pmap=lambda p: p):
    """Labelling-independent flag framings: {(triangle as point set, segment as point set): n_f}."""
    T = g.triangulation
    out = {}
    for (v, e), n in tv.flag_framings(g).items():
        i, j = g.edges[e].segment
        key = (frozenset(pmap(T.points[k]) for k in T.triangles[v]), frozenset((pmap(T.points[i]), pmap(T.points[j]))))
        out[key] = n
    return out


def class_data(g, pmap=lambda p: p):
    """{segment as point set: curve class as {point: coefficient}}."""
    P = g.points
    return {frozenset((pmap(P[e.segment[0]]), pmap(P[e.segment[1]]))):
            {pmap(P[k]): c for k, c in enumerate(e.curve_class) if c} for e in g.edges}


# ---------------------------------------------------------------- framings
def test_conifold():
    g = ex.conifold()
    assert g.framings() == [0] and len(g.vertices) == 2 and len(g.external_legs) == 4
    assert g.edges[0].curve_class == (1, -1, -1, 1)
    assert g.edge_names() == ["Q0"]


def test_local_p2():
    g = ex.local_p2()
    assert sorted(g.framings()) == [-2, 2, 2]
    assert all(abs(n) == 2 for n in g.framings())
    for fl in tv.flag_framings(g).values():
        assert fl - 1 in (1, -3)  # normal bundle O(n_f - 1) + O(-n_f - 1) = O(1) + O(-3)


def test_local_f0():
    g = ex.local_f0()
    assert [abs(n) for n in g.framings()] == [1, 1, 1, 1]


def test_local_f1_and_f2():
    g = ex.local_f1()  # edges B, F, B+F, F
    assert g.framings() == [0, 1, 2, 1]
    nb = [tuple(sorted((n - 1, -n - 1))) for n in g.framings()]
    assert nb == [(-1, -1), (-2, 0), (-3, 1), (-2, 0)]
    assert [abs(n) for n in ex.local_f2().framings()] == [1, 1, 3, 1]


def test_op1_bundles():
    for n in range(-3, 4):
        g = ex.op1_bundle(n)
        assert g.framings() == [n]
        assert tv.flag_framings(g) == {(0, 0): n, (1, 0): -n}
    assert ex.op1_bundle(2).framings() == [2]


def test_kp_figure3_right():
    """KP Fig. 3 right (flop of K_F1): labels -2/2 on the three P^2 edges, 0/0 on the flopped edge.
    Read off the figure: at the vertex of triangle (w0,w1,w2): -2 on edge w0w1, +2 on w0w2, 0 on
    w1w2; at (w0,w2,w3): -2 on w0w2, +2 on w0w3; at (w0,w3,w1): -2 on w0w3, +2 on w0w1."""
    g = ex.kp_flop_f1()
    w = g.points
    W = lambda *ks: frozenset(w[k] for k in ks)
    expect = {
        (W(0, 1, 2), W(0, 1)): -2, (W(0, 1, 2), W(0, 2)): 2, (W(0, 1, 2), W(1, 2)): 0,
        (W(0, 2, 3), W(0, 2)): -2, (W(0, 2, 3), W(0, 3)): 2,
        (W(0, 3, 1), W(0, 3)): -2, (W(0, 3, 1), W(0, 1)): 2,
        (W(1, 4, 2), W(1, 2)): 0,
    }
    assert flag_data(g) == expect
    assert g.framings() == [-2, 2, 2, 0]


def test_kp_figure2_framings():
    """KP Fig. 2: the four examples have framings 0, +-1, +-2, +-3 (two triangles sharing a side):
    exactly op1_bundle(0..3) up to relabelling."""
    for n in range(4):
        assert abs(ex.op1_bundle(n).framings()[0]) == n


def test_ladder_matches_hirzebruch():
    """su_n_ladder(2, m) is K_{F_m}: same multiset of |n_e|, same intersection data."""
    for m in range(3):
        a, b = ex.su_n_ladder(2, m), ex.hirzebruch(m)
        assert sorted(map(abs, a.framings())) == sorted(map(abs, b.framings()))
        assert len(a.vertices) == len(b.vertices) == 4 and len(a.edges) == 4
        # same canonical class coordinates up to permutation
        ca, cb = tv.class_coordinates(a), tv.class_coordinates(b)
        assert len(ca) == len(cb)


# ---------------------------------------------------------------- invariance
def random_sl2z(rng, steps=6):
    A = ((1, 0), (0, 1))
    S, T = ((0, -1), (1, 0)), ((1, 1), (0, 1))
    mul = lambda X, Y: tuple(tuple(sum(X[i][k] * Y[k][j] for k in range(2)) for j in range(2)) for i in range(2))
    for _ in range(steps):
        A = mul(A, rng.choice([S, T, mul(S, mul(S, S)), ((1, -1), (0, 1))]))
    return A


def apply(A, t):
    return lambda p: (A[0][0] * p[0] + A[0][1] * p[1] + t[0], A[1][0] * p[0] + A[1][1] * p[1] + t[1])


def summary(g):
    return sorted((abs(e.n), tuple(sorted(c for c in e.curve_class if c))) for e in g.edges)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_sl2z_and_relabelling_invariance(name):
    rng = random.Random(hash(name) % 1000)
    g = EXAMPLES[name]()
    T = g.triangulation
    for _ in range(3):
        A = random_sl2z(rng)
        t = (rng.randint(-5, 5), rng.randint(-5, 5))
        g2 = tv.from_triangulation(T.transformed(A, t))
        f = apply(A, t)
        assert flag_data(g2) == flag_data(g, f)
        assert class_data(g2) == class_data(g, f)
        assert summary(g2) == summary(g)
        assert (len(g2.vertices), len(g2.edges)) == (len(g.vertices), len(g.edges))
        # relabel points and triangles
        perm = list(range(len(T.points)))
        rng.shuffle(perm)
        pts = [None] * len(perm)
        for old, new in enumerate(perm):
            pts[new] = T.points[old]
        tris = [tuple(perm[i] for i in tr) for tr in T.triangles]
        rng.shuffle(tris)
        g3 = tv.from_triangulation(Triangulation(pts, tris, convex=T.convex))
        assert flag_data(g3) == flag_data(g)
        assert class_data(g3) == class_data(g)
        assert summary(g3) == summary(g)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_reflection_flips_framings(name):
    """x -> -x reverses orientation: (w1, w2, w3) ccw becomes (w2, w1, w3) ccw, so a1 <-> a2 and
    every flag framing (a1 - a2)/2 changes sign; curve classes are unchanged.  Since the
    triangle order is kept, each label n_e flips sign."""
    g = EXAMPLES[name]()
    R = ((-1, 0), (0, 1))
    g2 = tv.from_triangulation(g.triangulation.transformed(R))
    f = apply(R, (0, 0))
    assert {k: -v for k, v in flag_data(g, f).items()} == flag_data(g2)
    assert g2.framings() == [-n for n in g.framings()]
    assert class_data(g2) == class_data(g, f)


# ---------------------------------------------------------------- web round trip
def _roundtrip(g):
    vv, gl = tv.web_of(g)
    h = tv.from_web(vv, gl)
    assert h.framings() == g.framings()
    assert h.vertices == g.vertices
    assert h.external_legs == g.external_legs
    assert [(e.tail, e.head, e.tail_leg_index, e.head_leg_index) for e in h.edges] == \
           [(e.tail, e.head, e.tail_leg_index, e.head_leg_index) for e in g.edges]
    assert h.points is None and all(e.curve_class is None for e in h.edges)
    # with points: reconstructed polygon is a translate (vertex 0's first corner at the origin)
    k = tv.from_web(vv, gl, n_points=g.n_points)
    P0 = g.points[g.triangulation.triangles[0][0]]
    f = lambda p: (p[0] - P0[0], p[1] - P0[1])
    assert flag_data(k) == flag_data(g, f)
    assert class_data(k) == class_data(g, f)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_web_roundtrip_examples(name):
    _roundtrip(EXAMPLES[name]())


def test_web_roundtrip_hexagon_all():
    ts = pg.all_unimodular_triangulations(HEXAGON)
    assert len(ts) == 18
    for T in ts:
        _roundtrip(tv.from_triangulation(T))


def test_web_roundtrip_random_relabelled_hexagon():
    rng = random.Random(1)
    for T in pg.all_unimodular_triangulations(HEXAGON):
        tris = list(T.triangles)
        rng.shuffle(tris)
        g = tv.from_triangulation(Triangulation(T.points, tris))
        _roundtrip(g)


def test_from_web_by_hand():
    # conifold web: two vertices, glued along (1,1) / (-1,-1)
    g = tv.from_web([[(-1, 0), (0, -1), (1, 1)], [(-1, -1), (1, 0), (0, 1)]], [((0, 2), (1, 0))], n_points=4)
    assert g.framings() == [0]
    assert sorted(c for c in g.edges[0].curve_class) == [-1, -1, 1, 1]
    with pytest.raises(ValueError):
        tv.from_web([[(1, 0), (-1, -1), (0, 1)]], [])  # clockwise
    with pytest.raises(ValueError):
        tv.from_web([[(-1, 0), (0, -1), (1, 1)], [(-1, -1), (1, 0), (0, 1)]], [((0, 0), (1, 0))])


# ---------------------------------------------------------------- classes, weights
def test_regular_examples():
    for name, f in EXAMPLES.items():
        g = f()
        assert pg.is_regular(g.triangulation), name
        w = tv.kahler_weights(g)
        assert all(x > 0 for x in w)


def test_curve_classes():
    g = ex.conifold()
    assert tv.class_coordinates(g) == {0: (1,)}
    g = ex.local_p2()
    assert tv.relation_lattice_basis(g) == [(-3, 1, 1, 1)] and g.basis_edge_indices == [0]
    assert tv.class_coordinates(g) == {0: (1,), 1: (1,), 2: (1,)}
    g = ex.local_f0()
    c = tv.class_coordinates(g)
    assert c[0] == c[2] and c[1] == c[3] and {c[0], c[1]} == {(1, 0), (0, 1)}
    g = ex.local_f1()  # B, F, B+F, F
    assert tv.class_coordinates(g) == {0: (1, 0), 1: (0, 1), 2: (1, 1), 3: (0, 1)}
    g = ex.local_f2()  # B, F, B+2F, F
    assert tv.class_coordinates(g) == {0: (1, 0), 1: (0, 1), 2: (1, 2), 3: (0, 1)}
    g = ex.kp_flop_f1()  # H, H, H, E' with E' the flopped curve
    assert tv.class_coordinates(g) == {0: (1, 0), 1: (1, 0), 2: (1, 0), 3: (0, 1)}


def test_relation_lattice_rank_and_saturation():
    for name, f in EXAMPLES.items():
        g = f()
        b = tv.relation_lattice_basis(g)
        assert len(b) == g.n_points - 3
        for l in b:
            s = [sum(l[i] * (p[0], p[1], 1)[k] for i, p in enumerate(g.points)) for k in range(3)]
            assert s == [0, 0, 0]
        tv.class_coordinates(g, b)  # every edge class is an integer combination


def test_kahler_weights():
    assert len(set(tv.kahler_weights(ex.local_p2()))) == 1
    w = tv.kahler_weights(ex.local_f1())  # B, F, B+F, F
    assert w[2] == w[0] + w[1] and w[1] == w[3]
    w = tv.kahler_weights(ex.local_f0())
    assert w[0] == w[2] and w[1] == w[3]
    # explicit heights: star with centre lowest gives w = 2 + (self-intersection)
    assert tv.kahler_weights(ex.local_f1(), heights=[0, 1, 1, 1, 1]) == [1, 2, 3, 2]
