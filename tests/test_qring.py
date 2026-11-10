"""Tests for topvertex.qring (exact arithmetic in Q(q^{1/2}) with q-number denominators)."""
from fractions import Fraction

import pytest
import sympy as sp

from topvertex.qring import QFrac

Q = sp.Symbol("q", positive=True)
x = QFrac.x_pow(1)
one = QFrac.one()


def qn(k):
    return QFrac.qnum(k)


def same_sympy(f: QFrac, expr) -> bool:
    return sp.simplify(f.to_sympy(Q) - expr) == 0


def test_constructors():
    assert QFrac(3) == 3
    assert QFrac(Fraction(1, 2)) * 2 == one
    assert QFrac.zero().is_zero() and not one.is_zero()
    assert QFrac.q_pow(1) == x * x
    assert QFrac.x_pow(-3) * QFrac.x_pow(3) == one
    assert qn(1) == x - 1 / x
    assert qn(3) == QFrac.x_pow(3) - QFrac.x_pow(-3)
    assert qn(-2) == -qn(2)


def test_ring_identities():
    a = (x + 3) / qn(2)
    b = (QFrac.x_pow(-2) - Fraction(1, 3)) / (qn(1) ** 2)
    c = QFrac.q_pow(2) / qn(3)
    assert a + b == b + a
    assert a * b == b * a
    assert (a + b) * c == a * c + b * c
    assert (a * b) * c == a * (b * c)
    assert a - a == 0
    assert -(a - b) == b - a
    # exact division by elements whose numerator is a monomial times q-numbers
    u = QFrac.x_pow(-3, Fraction(2, 7)) * qn(2) * qn(5) / qn(1) ** 2
    assert (a * u) / u == a
    assert (a * b) / b == a  # general divisor numerator: exact polynomial division
    assert a ** 3 == a * a * a
    assert u ** -2 * u ** 2 == one
    assert a ** 0 == one


def test_qnumber_products():
    # [2] = [1] (x + 1/x), [2k] = [k] (x^k + x^-k), [3] = [1] (q + 1 + 1/q)
    assert qn(2) == qn(1) * (x + 1 / x)
    for k in range(1, 6):
        assert qn(2 * k) == qn(k) * (QFrac.x_pow(k) + QFrac.x_pow(-k))
    assert qn(3) == qn(1) * (QFrac.q_pow(1) + 1 + QFrac.q_pow(-1))
    # [a][b] = [a+b]... identity: [a][b] = (x^{a+b} + x^{-a-b}) - (x^{a-b} + x^{b-a})
    for a in range(1, 4):
        for b in range(1, 4):
            rhs = QFrac.x_pow(a + b) + QFrac.x_pow(-a - b) - QFrac.x_pow(a - b) - QFrac.x_pow(b - a)
            assert qn(a) * qn(b) == rhs


def test_cancel():
    f = qn(2) * qn(3) / (qn(1) * qn(2) * qn(3) * qn(3))
    g = f.cancel()
    assert g == f
    assert g.denominator == {1: 1, 3: 1}
    h = (qn(2) / (qn(1) * qn(2))).cancel()
    assert h.denominator == {1: 1} and h == 1 / qn(1)
    # [6]/([2][3]) cancels [3] (or [2]), leaving a denominator of one q-number
    r = (qn(6) / (qn(2) * qn(3))).cancel()
    assert sum(r.denominator.values()) == 1
    # nothing to cancel
    s = (x + 2) / qn(1)
    assert s.cancel().denominator == {1: 1}
    # full cancellation
    assert (qn(4) * qn(1) / (qn(1) * qn(4))).cancel().denominator == {}


def test_division():
    assert (one / qn(2)) * qn(2) == one
    assert (qn(4) / (qn(1) * qn(2))) * qn(1) * qn(2) == qn(4)
    # divisor numerator a monomial times q-numbers
    d = QFrac.x_pow(3, 5) * qn(2) * qn(2) * qn(1)
    assert (one / d) * d == one
    # divisor (x^2+1)/[1] with x^2+1 not a product of q-numbers but dividing the dividend
    assert qn(2) / (x + 1 / x) == qn(1)
    with pytest.raises(ValueError):
        one / (x + 1)
    with pytest.raises(ZeroDivisionError):
        one / QFrac.zero()


def test_hash_consistent_with_eq():
    a = qn(2) / (qn(1) * qn(2))
    b = 1 / qn(1)
    assert a == b and hash(a) == hash(b)
    assert len({a, b, a.cancel()}) == 1


def test_laurent_and_symmetry():
    assert QFrac.x_pow(-3).as_laurent() == {Fraction(-3, 2): 1}
    assert (qn(2) / qn(1)).as_laurent() == {Fraction(-1, 2): 1, Fraction(1, 2): 1}
    with pytest.raises(ValueError):
        (1 / qn(1)).as_laurent()
    assert (qn(1) ** 2).is_q_symmetric()
    assert not qn(1).is_q_symmetric()
    assert (1 / qn(1) ** 2).is_q_symmetric()
    assert (qn(1) / qn(3)).is_q_symmetric()
    assert not (x + 2).is_q_symmetric()
    assert (x + 1 / x).is_q_symmetric()


def test_to_y_polynomial():
    y = qn(1) ** 2
    assert y == QFrac.q_pow(1) - 2 + QFrac.q_pow(-1)
    P = y ** 2 - 3 * y
    assert P.to_y_polynomial() == [0, -3, 1]
    assert (QFrac(7)).to_y_polynomial() == [7]
    R = 5 - y + Fraction(1, 2) * y ** 3
    assert R.to_y_polynomial() == [5, -1, 0, Fraction(1, 2)]
    # q + 1/q = y + 2
    assert (QFrac.q_pow(1) + QFrac.q_pow(-1)).to_y_polynomial() == [2, 1]
    with pytest.raises(ValueError):
        (x + 1 / x).to_y_polynomial()  # half-integer powers
    with pytest.raises(ValueError):
        QFrac.q_pow(1).to_y_polynomial()  # not symmetric
    with pytest.raises(ValueError):
        (1 / y).to_y_polynomial()


def test_sympy_roundtrip():
    s = sp.sqrt(Q)
    cases = [
        (1 / qn(1), 1 / (s - 1 / s)),
        (QFrac.x_pow(1) / (qn(1) * qn(2)), Q ** 2 / ((Q - 1) * (Q ** 2 - 1))),
        ((qn(1) ** 2 + 1) / qn(1) ** 2, (Q ** 2 - Q + 1) / (1 - Q) ** 2),
        (QFrac.x_pow(-3, Fraction(2, 3)) - 1, sp.Rational(2, 3) * Q ** sp.Rational(-3, 2) - 1),
    ]
    for f, expr in cases:
        assert same_sympy(f, expr)
    a = (x + 3) / qn(2)
    b = (QFrac.x_pow(-2) - Fraction(1, 3)) / (qn(1) ** 2)
    u = QFrac.x_pow(1, 3) * qn(3) / qn(2)
    lhs = (a * b + a / u - b * b).to_sympy(Q)
    rhs = a.to_sympy(Q) * b.to_sympy(Q) + a.to_sympy(Q) / u.to_sympy(Q) - b.to_sympy(Q) ** 2
    assert sp.simplify(lhs - rhs) == 0


def test_output_formats():
    f = (QFrac.q_pow(2) - x) / (qn(1) ** 2 * qn(3))
    assert abs(f.evaluate(1.3) - complex(f.to_sympy(Q).subs(Q, 1.3 ** 2))) < 1e-12
    m = f.to_mathematica()
    assert "q^(1/2)" in m and "q^(3/2)" in m
    expr = sp.parse_expr(m.replace("^", "**"), local_dict={"q": Q})
    assert sp.simplify(expr - f.to_sympy(Q)) == 0
    assert "[3]" in f.latex() and "[1]^{2}" in f.latex()
    assert "[1]^2" in str(f)
    assert repr(QFrac.zero()).startswith("QFrac(0")
