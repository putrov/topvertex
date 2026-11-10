"""Tests for topvertex.schur: hook formula vs Jacobi-Trudi, h-form vs e-form, symmetries."""
from topvertex.partitions import conjugate, kappa, partitions_up_to, size, subpartitions
from topvertex.qring import QFrac
from topvertex.schur import (e_at, e_rho, h_at, h_rho, schur_at, schur_rho, skew_schur_at,
                             skew_schur_e_form, skew_schur_h_form)


def test_h_e_rho_small():
    assert h_rho(0) == 1 and e_rho(0) == 1 and h_rho(-1) == 0
    assert h_rho(1) == 1 / QFrac.qnum(1)
    assert h_rho(2) == QFrac.x_pow(1) / (QFrac.qnum(1) * QFrac.qnum(2))
    assert e_rho(2) == QFrac.x_pow(-1) / (QFrac.qnum(1) * QFrac.qnum(2))


def test_h_e_duality_rho():
    # sum_{k} (-1)^k e_k h_{n-k} = 0 for n >= 1, at q^rho and at q^{nu+rho}
    for nu in [(), (1,), (2, 1), (3,), (1, 1, 1)]:
        for n in range(1, 7):
            s = sum(((-1) ** k * e_at(k, nu) * h_at(n - k, nu) for k in range(n + 1)), QFrac.zero())
            assert s == 0


def test_nu_empty_reduces_to_rho():
    for i in range(7):
        assert h_at(i, ()) == h_rho(i)
        assert e_at(i, ()) == e_rho(i)


def test_h1_at_nu():
    # h_1(q^{nu+rho}) = e_1 = p_1 = h_1(q^rho) + sum_i (q^{nu_i-i+1/2} - q^{-i+1/2})
    for nu in [(1,), (2, 1), (3, 3), (4, 2, 1)]:
        delta = sum((QFrac.x_pow(2 * p - 2 * i + 1) - QFrac.x_pow(1 - 2 * i)
                     for i, p in enumerate(nu, start=1)), QFrac.zero())
        assert h_at(1, nu) == h_rho(1) + delta == e_at(1, nu)


def test_hook_formula_equals_jacobi_trudi():
    for lam in partitions_up_to(5):
        jt_h = skew_schur_h_form(lam, (), ())
        jt_e = skew_schur_e_form(lam, (), ())
        assert schur_rho(lam) == jt_h == jt_e
        assert schur_at(lam, ()) == schur_rho(lam)


def test_h_form_equals_e_form_skew():
    nus = [(), (1,), (2,), (1, 1), (2, 1), (3, 1)]
    for mu in partitions_up_to(5):
        for eta in subpartitions(mu):
            for nu in nus:
                assert skew_schur_h_form(mu, eta, nu) == skew_schur_e_form(mu, eta, nu)


def test_transpose_rho():
    for lam in partitions_up_to(6):
        assert schur_rho(conjugate(lam)) == QFrac.x_pow(-kappa(lam)) * schur_rho(lam)


def test_skew_trivial_cases():
    for mu in partitions_up_to(5):
        for nu in [(), (1,), (2, 1)]:
            assert skew_schur_at(mu, mu, nu) == 1
            assert skew_schur_at(mu, (), nu) == schur_at(mu, nu)
    assert skew_schur_at((1,), (2,), ()) == 0
    assert skew_schur_at((2, 1), (1, 1, 1), (1,)) == 0
    assert skew_schur_at((), (), (3,)) == 1


def test_symmetry_W():
    # s_mu(q^rho) s_nu(q^{mu+rho}) is symmetric in (mu, nu)  (W_{mu nu}, AKMV eq. 3.3 / KP)
    ps = partitions_up_to(3)
    for mu in ps:
        for nu in ps:
            assert schur_rho(mu) * schur_at(nu, mu) == schur_rho(nu) * schur_at(mu, nu)


def test_skew_branching():
    # s_mu(q^{nu+rho}) = sum_{eta subset mu} s_{mu/eta}(z) s_eta(w) is not available for a
    # single specialisation; instead check the one-box branching s_{mu/(1)} = sum over removable
    # boxes s_{mu - box}... at q^{rho}: s_{mu/(1)} = sum_{nu = mu minus a corner} s_nu.
    for mu in partitions_up_to(5):
        if not mu:
            continue
        corners = []
        for i in range(len(mu)):
            if i == len(mu) - 1 or mu[i] > mu[i + 1]:
                nu = list(mu)
                nu[i] -= 1
                corners.append(tuple(p for p in nu if p))
        rhs = sum((schur_rho(nu) for nu in corners), QFrac.zero())
        assert skew_schur_at(mu, (1,), ()) == rhs
        assert size(mu) >= 1
