"""Tests for topvertex.series (truncated power series with QFrac coefficients)."""
from fractions import Fraction

import pytest
import sympy as sp

from topvertex.qring import QFrac
from topvertex.series import Series


def qn(k):
    return QFrac.qnum(k)


def sample_F():
    """A 2-variable series with zero constant term and q-dependent coefficients."""
    return Series(("a", "b"), {
        (1, 0): QFrac.one() / qn(1) ** 2,
        (0, 1): QFrac.q_pow(1, 3) - QFrac.x_pow(-1),
        (1, 1): QFrac.inv_qnums([2], Fraction(1, 2), xexp=3),
        (2, 0): QFrac.qnum(3) / qn(2),
        (0, 3): QFrac(Fraction(-5, 7)),
    }, weights=(1, 2), max_weight=6)


def test_log_exp_round_trip():
    F = sample_F()
    Z = F.exp()
    assert Z.constant_term() == 1
    assert Z.log() == F
    assert Z.log().exp() == Z
    # weight-2 coefficient a^2 of exp F: F_{(2,0)} + F_{(1,0)}^2 / 2
    assert Z.coeff((2, 0)) == F.coeff((2, 0)) + F.coeff((1, 0)) ** 2 * Fraction(1, 2)


def test_exp_one_variable():
    F = Series(("Q",), {(1,): QFrac.one()}, max_weight=6)
    Z = F.exp()
    from math import factorial
    for n in range(7):
        assert Z.coeff((n,)) == Fraction(1, factorial(n))
    assert Z.log() == F


def test_mul_add_truncation():
    A = Series(("Q",), {(0,): 1, (1,): 1}, max_weight=3)
    B = A * A * A * A
    assert [B.coeff((n,)) for n in range(4)] == [1, 4, 6, 4]
    assert (0,) in B.coefficients and (4,) not in B.coefficients
    assert (A + 1).coeff((0,)) == 2
    assert (A - A).coefficients == {}
    with pytest.raises(ValueError):
        B.coeff((4,))
    assert len(B.truncate(1)) == 2


def test_box_truncation():
    A = Series(("x", "y"), {(0, 0): 1, (1, 0): 1, (0, 1): 1}, max_weight=10, max_degrees=(2, 1))
    B = A * A * A
    assert all(d[0] <= 2 and d[1] <= 1 for d in B.coefficients)
    assert B.coeff((2, 1)) == 3
    assert (A.log().exp()) == A


def test_substitute():
    Z = Series(("Q0", "Q1", "Q2"), {(0, 0, 0): 1, (1, 0, 0): QFrac.x_pow(1), (0, 1, 0): 2,
                                    (0, 0, 1): QFrac.qnum(2), (1, 0, 1): 7}, weights=(1, 2, 1), max_weight=3)
    # Q0 -> t1, Q1 -> t1 t2, Q2 -> t2 ; weights t1 = t2 = 1 reproduce (1, 2, 1)
    S = Z.substitute({"Q0": {"t1": 1}, "Q1": {"t1": 1, "t2": 1}, "Q2": {"t2": 1}}, ("t1", "t2"), (1, 1))
    assert S.coeff((1, 1)) == 2 + 7
    assert S.coeff((1, 0)) == QFrac.x_pow(1) and S.coeff((0, 1)) == QFrac.qnum(2)
    assert S.max_weight == 3
    # matrix form and negative exponents
    S2 = Z.substitute([[1, 0], [1, 1], [-1, 2]], ("u", "v"), (1, 1))
    assert S2.coeff((-1, 2)) == QFrac.qnum(2) and S2.coeff((0, 2)) == 7
    with pytest.raises(ValueError):
        Z.substitute([[1, 0], [1, 1], [0, 1]], ("u", "v"), (1, 2))  # w(Q0)=1 ok, w(Q1)=3 != 2


def test_log_requires_constant_one():
    with pytest.raises(ValueError):
        Series(("Q",), {(0,): 2, (1,): 1}, max_weight=2).log()


def test_output_formats():
    Z = Series(("Q",), {(0,): 1, (1,): QFrac.one() / qn(1) ** 2}, max_weight=1)
    q, Q = sp.Symbol("q", positive=True), sp.Symbol("Q")
    expr = Z.to_sympy(q, symbols=(Q,))
    assert sp.simplify(expr - (1 + Q / (sp.sqrt(q) - 1 / sp.sqrt(q)) ** 2)) == 0
    assert sp.simplify(Z.to_sympy().subs(sp.Symbol("q", positive=True), 4).subs(sp.Symbol("Q"), 1) - (1 + Fraction(4, 9))) == 0
    m = Z.to_mathematica()
    assert "q^(1/2)" in m and "*Q" in m
    assert "[1]^{2}" in Z.latex()
    assert Z.latex(max_terms=1).endswith(r"\dots")
    assert "Q" in str(Z) and "2 terms" in str(Z)
    assert [d for d, _ in sample_F().terms()] == [(1, 0), (0, 1), (2, 0), (1, 1), (0, 3)]
