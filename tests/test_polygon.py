"""Tests for topvertex.polygon (lattice polygons, unimodular triangulations, regularity)."""
import pytest

from topvertex import polygon as pg
from topvertex.polygon import Triangulation

P2 = [(-1, -1), (1, 0), (0, 1)]
F0 = [(1, 0), (0, 1), (-1, 0), (0, -1)]
F1 = [(1, 0), (0, 1), (-1, 1), (0, -1)]
DP3 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


# ---------------------------------------------------------------- corner strings
@pytest.mark.parametrize("s", [
    "http://127.0.0.1:8765/#p=0,0_4,0_0,2",
    "http://127.0.0.1:8765/?order=4&p=0,0_4,0_0,2",
    "p=0,0_4,0_0,2",
    "0,0_4,0_0,2",
])
def test_parse_polygon_forms(s):
    assert pg.parse_polygon(s) == [(0, 0), (4, 0), (0, 2)]


def test_parse_polygon_negative_and_roundtrip():
    c = [(-1, -1), (1, 0), (0, 1)]
    s = pg.format_polygon(c)
    assert s == "p=-1,-1_1,0_0,1"
    assert pg.parse_polygon(s) == c
    assert pg.parse_polygon("http://127.0.0.1:8765/?x=1&" + s + "&x=3") == c
    assert pg.parse_polygon("http://127.0.0.1:8765/#" + s + "&d=0,0_1,1") == c


def test_parse_polygon_errors():
    with pytest.raises(ValueError):
        pg.parse_polygon("p=0,0_1.5,0_0,2")
    with pytest.raises(ValueError):
        pg.parse_polygon("p=0,0_1,0_0")


# ---------------------------------------------------------------- polygons
def test_lattice_point_counts():
    assert len(pg.lattice_points(P2)) == 4
    assert pg.interior_points(P2) == [(0, 0)]
    assert len(pg.lattice_points(F0)) == 5
    # example: triangle (0,0),(4,0),(0,2).  Pick: A = 4, B = 4 + 2 + 2 = 8, I = A - B/2 + 1 = 1.
    c = pg.parse_polygon("p=0,0_4,0_0,2")
    A2, B, I = pg.area2(c), len(pg.boundary_points(c)), len(pg.interior_points(c))
    assert (A2, B, I) == (8, 8, 1)
    assert A2 == 2 * I + B - 2  # Pick
    assert len(pg.lattice_points(c)) == 9


def test_hull_and_convexity():
    assert pg.convex_hull(pg.lattice_points(DP3)) == [(-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)]
    assert pg.is_convex(SQUARE) and pg.is_convex(SQUARE[::-1])
    assert pg.is_convex([(0, 0), (1, 0), (2, 0), (0, 2)])  # collinear corner allowed
    assert not pg.is_convex([(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)])
    assert pg.area2(DP3) == 6


# ---------------------------------------------------------------- triangulations
def test_validate_rejects():
    with pytest.raises(ValueError, match="non-unimodular"):
        Triangulation([(0, 0), (1, 0), (2, 0), (0, 1)], [(0, 2, 3)]).validate()
    with pytest.raises(ValueError, match="cover"):
        Triangulation([(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 1, 2)]).validate()
    with pytest.raises(ValueError, match="overlap"):
        Triangulation([(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 1, 3), (0, 1, 2)]).validate()
    with pytest.raises(ValueError, match="non-unimodular"):  # P^2 polygon without its centre
        Triangulation([(-1, -1), (1, 0), (0, 1)], [(0, 1, 2)]).validate()


def test_from_corners_and_triangles_and_eq():
    T1 = Triangulation.from_corners_and_triangles(SQUARE, [((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))])
    T2 = Triangulation([(1, 1), (0, 1), (1, 0), (0, 0)], [(3, 1, 2), (2, 0, 1)])  # same, relabelled, cw input
    T1.validate(), T2.validate()
    assert T1 == T2 and hash(T1) == hash(T2)
    assert all(pg.orient(*(T2.points[i] for i in t)) == 1 for t in T2.triangles)
    U = T1.flip((1, 2))
    assert U != T1 and U.flip(U.interior_edges()[0][0]) == T1
    assert [e for e, _ in T1.interior_edges()] == [(1, 2)]
    assert len(T1.boundary_edges()) == 4


def test_triangulation_counts():
    """Counts (all unimodular == fine triangulations; checked by hand where small):
    * unit square: 2 (two diagonals);
    * P^2 polygon: 1 (centre is interior and must be joined to all 3 corners);
    * F0 polygon: 1 (both diagonals pass through the centre, so the centre is joined to all 4
      corners);
    * F1 polygon: 2 (star; or the diagonal (1,0)-(-1,1), which avoids the centre, with the centre
      joined to the three corners below it: the flop of the exceptional curve);
    * dP3 hexagon: 18 (star; flips of the six spokes, all (-1)-curves, ...);
    * triangle (0,0),(4,0),(0,2): 24; 3x3 grid [0,2]^2: 64 (known value).
    """
    expect = {tuple(SQUARE): 2, tuple(P2): 1, tuple(F0): 1, tuple(F1): 2, tuple(DP3): 18,
              ((0, 0), (4, 0), (0, 2)): 24, ((0, 0), (2, 0), (2, 2), (0, 2)): 64}
    for corners, n in expect.items():
        ts = pg.all_unimodular_triangulations(corners)
        assert len(ts) == n, corners
        assert len(set(ts)) == n
        for T in ts:
            T.validate()
            assert T.points == pg.lattice_points(corners)
            assert len(T.triangles) == pg.area2(corners)
            if n <= 24:
                assert pg.is_regular(T)


def test_enumeration_deterministic():
    a = pg.all_unimodular_triangulations(DP3)
    b = pg.all_unimodular_triangulations(list(reversed(DP3)))
    assert [T.triangles for T in a] == [T.triangles for T in b]


# ---------------------------------------------------------------- regularity
def _rels(T):
    return pg._edge_relations(T)


def test_heights_conifold_sign():
    """Conifold square, diagonal (1,0)-(0,1): l = +1 at (0,0),(1,1), -1 at (1,0),(0,1).
    Lifting (1,1) to height 1 makes the lower hull use exactly this diagonal, and <h,l> = 1 > 0."""
    T = Triangulation([(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 1, 2), (1, 3, 2)])
    assert _rels(T) == [(1, -1, -1, 1)]
    h = pg.heights(T)
    assert h is not None and sum(a * b for a, b in zip(h, (1, -1, -1, 1))) > 0
    assert sum(a * b for a, b in zip([0, 0, 0, 1], (1, -1, -1, 1))) > 0
    # the other diagonal needs the opposite sign
    T2 = T.flip((1, 2))
    assert sum(a * b for a, b in zip(pg.heights(T2), (1, -1, -1, 1))) < 0


def test_heights_valid_everywhere():
    for T in pg.all_unimodular_triangulations(DP3):
        h = pg.heights(T)
        assert all(isinstance(v, int) for v in h)
        assert all(sum(a * b for a, b in zip(h, l)) > 0 for l in _rels(T))


# One of the two (mirror) non-regular unimodular triangulations of the 4-fold dilated unit
# triangle 4*Delta (found by the enumerator: 7424 unimodular triangulations, 2 non-regular): the
# "twisted" configuration around the inner triangle (1,1),(2,1),(1,2).
NONREG = [((0, 0), (1, 0), (2, 1)), ((0, 0), (1, 1), (0, 1)), ((0, 0), (2, 1), (1, 1)),
          ((0, 1), (1, 1), (0, 2)), ((0, 2), (1, 1), (0, 3)), ((0, 3), (1, 1), (0, 4)),
          ((0, 4), (1, 1), (1, 2)), ((0, 4), (1, 2), (1, 3)), ((1, 0), (2, 0), (2, 1)),
          ((1, 1), (2, 1), (1, 2)), ((1, 2), (2, 1), (4, 0)), ((1, 2), (2, 2), (1, 3)),
          ((1, 2), (3, 1), (2, 2)), ((1, 2), (4, 0), (3, 1)), ((2, 0), (3, 0), (2, 1)),
          ((2, 1), (3, 0), (4, 0))]


def test_non_regular():
    T = Triangulation.from_corners_and_triangles([(0, 0), (4, 0), (0, 4)], NONREG).validate()
    assert pg.heights(T) is None and not pg.is_regular(T)
    # its mirror image under (x, y) -> (y, x) is non-regular as well
    assert not pg.is_regular(T.transformed(((0, 1), (1, 0))))


def test_all_ones_weights():
    star = lambda rays: Triangulation([(0, 0)] + rays, [(0, 1 + k, 1 + (k + 1) % len(rays)) for k in range(len(rays))])
    assert pg.all_ones_weights_consistent(star([(1, 0), (0, 1), (-1, -1)]))
    assert pg.all_ones_weights_consistent(star([(1, 0), (0, 1), (-1, 0), (0, -1)]))
    assert not pg.all_ones_weights_consistent(star([(0, 1), (-1, 1), (0, -1), (1, 0)]))  # F1
    assert pg.all_ones_weights_consistent(star(DP3))


def test_plot_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    T = pg.all_unimodular_triangulations(DP3)[3]
    fig = pg.plot(T)
    assert len(fig.axes) == 2
