"""Optional-dependency fallbacks: the pure-Python polynomial backend of topvertex.qring (used when
python-flint is missing or TOPVERTEX_NO_FLINT is set) and the exact rational LP of
topvertex.polygon.heights (used when scipy is missing or TOPVERTEX_NO_SCIPY is set)."""
import json
import os
import random
import subprocess
import sys
from fractions import Fraction

import pytest

from topvertex import _pypoly
from topvertex import polygon as pg
from topvertex.polygon import Triangulation

from tests.test_polygon import DP3, NONREG

_CHILD = r"""
import json
from fractions import Fraction
from topvertex import qring
from topvertex.qring import QFrac
from topvertex.schur import schur_rho
from topvertex.vertex import vertex as C
x = QFrac.x_pow(1)
one = QFrac.one()
out = {
    "backend": qring.BACKEND,
    "C111": C((1,), (1,), (1,)) == QFrac.from_laurent({8: 1, 6: -1, 4: 1, 2: -1, 0: 1}) / (x * (x**2 - 1)**3),
    "C11": str(C((1,), (1,), ()).cancel()),
    "s21": str(schur_rho((2, 1))),
    "div": str((QFrac.qnum(6) * QFrac.qnum(2) / QFrac.qnum(3)).cancel()),
    "frac": str(QFrac(Fraction(3, 4)) * QFrac.qnum(1) ** 2 / 6),
    "val": str(QFrac.inv_qnums([1, 2], coeff=Fraction(5, 3)).exact_value_at(Fraction(2))),
    "laurent": {str(k): str(v) for k, v in ((QFrac.qnum(2) / QFrac.qnum(1)) ** 3).as_laurent().items()},
}
print(json.dumps(out))
"""


def _run_child(no_flint: bool) -> dict:
    env = dict(os.environ)
    env.pop("TOPVERTEX_NO_FLINT", None)
    if no_flint:
        env["TOPVERTEX_NO_FLINT"] = "1"
    res = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True,
                         check=True)
    return json.loads(res.stdout)


def test_python_backend_selected_and_values():
    py = _run_child(no_flint=True)
    assert py["backend"] == "python"
    assert py["C111"]  # KP: C_{(1),(1),(1)} = (q^4-q^3+q^2-q+1)/(q^{1/2}(q-1)^3)
    assert py["C11"] == "(x^-2 - 1 + x^2) / ([1]^2)"  # (q^2-q+1)/(1-q)^2
    assert py["s21"] == "(1) / ([1]^2 [3])"
    assert py["div"] == "-x^-5 + x^-1 - x + x^5"  # [6][2]/[3] = [2] (x^3 + x^-3)
    assert py["frac"] == "1/8*x^-2 - 1/4 + 1/8*x^2"  # (3/4) [1]^2 / 6
    assert py["val"] == "8/27"  # (5/3) / ([1][2]) at x = 2: [1] = 3/2, [2] = 15/4
    assert py["laurent"] == {"-3/2": "1", "-1/2": "3", "1/2": "3", "3/2": "1"}  # ([2]/[1])^3
    try:
        import flint  # noqa: F401
    except ImportError:
        return
    fl = _run_child(no_flint=False)
    assert fl["backend"] == "flint"
    assert {k: v for k, v in fl.items() if k != "backend"} == {k: v for k, v in py.items() if k != "backend"}


# ---------------------------------------------------------------- PyPoly against a naive reference
def _naive_mul(a, b):
    r = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] += x * y
    return r


def _trim(c):
    c = list(c)
    while c and c[-1] == 0:
        c.pop()
    return c


def _rand_poly(rng, n, big=False, frac=False):
    hi = 10 ** 30 if big else 5
    c = [rng.randint(-hi, hi) for _ in range(n)]
    if frac:
        c = [Fraction(v, rng.randint(1, 12)) for v in c]
    if c and c[-1] == 0:
        c[-1] = 1
    return c


@pytest.mark.parametrize("seed", range(6))
def test_pypoly_arithmetic(seed):
    rng = random.Random(seed)
    for _ in range(40):
        la, lb = rng.choice([1, 3, 11, 12, 40]), rng.choice([1, 2, 13, 50])
        big, frac = rng.random() < 0.3, rng.random() < 0.3
        a, b = _rand_poly(rng, la, big, frac), _rand_poly(rng, lb, big, frac)
        A, B = _pypoly.from_fractions(a), _pypoly.from_fractions(b)
        assert (A * B).coeffs() == _trim(_naive_mul(a, b))
        m = max(la, lb)
        pad = lambda c: [Fraction(v) for v in c] + [Fraction(0)] * (m - len(c))
        assert (A + B).coeffs() == _trim(x + y for x, y in zip(pad(a), pad(b)))
        assert (A - B).coeffs() == _trim(x - y for x, y in zip(pad(a), pad(b)))
        assert (A * B - B * A) == _pypoly.ZERO
        # divmod: (A*B + R) / B with deg R < deg B, for integral monic-like and general divisors
        R = _pypoly.from_fractions(_rand_poly(rng, lb - 1, False, frac)) if lb > 1 else _pypoly.ZERO
        q, r = _pypoly.pdivmod(A * B + R, B)
        assert q == A and r == R
        k = rng.randint(1, 6)
        assert _pypoly.pmul_xpow_minus_one(A, k) == A * _pypoly.from_fractions([-1] + [0] * (k - 1) + [1])
        x = Fraction(rng.randint(-7, 7), rng.randint(1, 5))
        assert _pypoly.peval(A, x) == sum(Fraction(c) * x ** i for i, c in enumerate(a))
        assert (A ** 3) == A * A * A


# ---------------------------------------------------------------- heights without scipy
@pytest.mark.parametrize("solver", ["scipy", "exact"])
def test_heights_solvers(monkeypatch, solver):
    if solver == "scipy":
        pytest.importorskip("scipy")
        monkeypatch.delenv("TOPVERTEX_NO_SCIPY", raising=False)
    else:
        monkeypatch.setenv("TOPVERTEX_NO_SCIPY", "1")
    assert pg.lp_backend() == solver
    for T in pg.all_unimodular_triangulations(DP3) + pg.all_unimodular_triangulations([(0, 0), (4, 0), (0, 2)]):
        h = pg.heights(T)
        assert h is not None and all(isinstance(v, int) for v in h) and min(h) == 0
        assert all(sum(a * b for a, b in zip(h, l)) > 0 for l in pg._edge_relations(T))
    T = Triangulation.from_corners_and_triangles([(0, 0), (4, 0), (0, 4)], NONREG).validate()
    assert pg.heights(T) is None
    assert pg.heights(T.transformed(((0, 1), (1, 0)))) is None


def test_heights_exact_degenerate():
    """Duplicate, highly degenerate constraint rows (SU(3) ladder: two pairs of equal relation
    vectors) on which sympy.solvers.simplex returns an infeasible point."""
    from topvertex import examples_geometry as ex
    T = ex.su_n_ladder(3, 0).triangulation
    rels = pg._edge_relations(T)
    h = pg._heights_exact(rels, len(T.points))
    assert all(sum(a * b for a, b in zip(h, l)) >= 1 for l in rels)
