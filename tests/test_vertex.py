"""Tests for topvertex.vertex: KP values, cyclic symmetry, AKMV (3.12), special cases."""
from itertools import product

import sympy as sp

from tests.data import KONISHI_VERTEX
from topvertex.partitions import (conjugate, intersection, kappa, partitions_up_to, size,
                                  subpartitions)
from topvertex.qring import QFrac
from topvertex.schur import schur_at, schur_rho, skew_schur_at
from topvertex.vertex import framing_factor, vertex

Q = sp.Symbol("q", positive=True)
PS4 = partitions_up_to(4)
TRIPLES4 = [t for t in product(PS4, repeat=3) if sum(map(size, t)) <= 4]


def assert_equal_sympy(f: QFrac, expr: str):
    """Assert that the QFrac f equals the sympy expression string (variable q)."""
    e = sp.sympify(expr, locals={"q": Q})
    assert sp.simplify(f.to_sympy(Q) - e) == 0, (str(f), expr)


def test_konishi_values():
    for triple, expr in KONISHI_VERTEX.items():
        assert_equal_sympy(vertex(*triple), expr)


def test_cyclic_symmetry():
    for a, b, c in TRIPLES4:
        v = vertex(a, b, c)
        assert v == vertex(b, c, a)
        assert v == vertex(c, a, b)


def test_one_leg():
    for lam in partitions_up_to(6):
        assert vertex(lam, (), ()) == schur_rho(lam)


def test_akmv_3_12():
    # C_{l1 l2 l3} = q^{(k1+k2+k3)/2} C_{l1^t l3^t l2^t}
    for a, b, c in TRIPLES4:
        k = kappa(a) + kappa(b) + kappa(c)
        assert vertex(a, b, c) == QFrac.x_pow(k) * vertex(conjugate(a), conjugate(c), conjugate(b))


def test_two_leg_special_cases():
    # From the definition with one empty leg:
    #   C_{l1,l2,()} = s_{l2}(q^rho) s_{l1}(q^{l2^t+rho})
    #   C_{(),l2,l3} = q^{kappa(l3)/2} s_{l2}(q^rho) s_{l3^t}(q^{l2+rho})
    # and by cyclic symmetry C_{(),mu,nu} = C_{mu,nu,()} = s_nu(q^rho) s_mu(q^{nu^t+rho}).
    ps = partitions_up_to(4)
    for mu, nu in product(ps, repeat=2):
        if size(mu) + size(nu) > 5:
            continue
        v = vertex((), mu, nu)
        assert v == QFrac.x_pow(kappa(nu)) * schur_rho(mu) * schur_at(conjugate(nu), mu)
        assert v == schur_rho(nu) * schur_at(mu, conjugate(nu))
        assert vertex(mu, nu, ()) == schur_rho(nu) * schur_at(mu, conjugate(nu))
        # C_{l1,(),l3} = q^{kappa(l3)/2} sum_eta s_{l1/eta}(q^rho) s_{l3^t/eta}(q^rho)
        s = sum((skew_schur_at(mu, eta, ()) * skew_schur_at(conjugate(nu), eta, ())
                 for eta in subpartitions(intersection(mu, conjugate(nu)))), QFrac.zero())
        assert vertex(mu, (), nu) == QFrac.x_pow(kappa(nu)) * s


def test_results_reduced_and_integral_shape():
    # numerators have integer coefficients and denominators are products of q-numbers
    for t in TRIPLES4:
        v = vertex(*t)
        assert all(c.denominator == 1 for c in v.numerator_laurent().values())


def test_framing_factor():
    assert framing_factor(0, (1,)) == -1
    assert framing_factor(1, (1,)) == 1
    assert framing_factor(1, (2,)) == QFrac.q_pow(1)
    assert framing_factor(2, (1, 1)) == QFrac.q_pow(-2)
    assert framing_factor(-1, (3,)) == QFrac.q_pow(-3)


def test_conifold_single_edge():
    # Resolved conifold (n = 0): Z_d = (-1)^d sum_{lam |- d} q^{-kappa/2} s_lam(q^rho)^2 and
    # log Z = -sum_k Q^k/(k [k]^2); check d = 1, 2 with Z_d = sum_lam C_{(),(),lam} n C_{lam^t,(),()}
    def Zd(d):
        return sum((framing_factor(0, lam) * vertex((), (), lam) * vertex(conjugate(lam), (), ())
                    for lam in partitions_up_to(d) if size(lam) == d), QFrac.zero())
    q1 = QFrac.qnum(1)
    assert Zd(1) == -1 / q1 ** 2
    # log Z at order 2: Z_2 - Z_1^2/2 = -1/(2 [2]^2)
    assert Zd(2) - Zd(1) ** 2 / 2 == -1 / (2 * QFrac.qnum(2) ** 2)
