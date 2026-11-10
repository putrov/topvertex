r"""Specialised (skew) Schur functions at q^rho and q^{nu+rho}, as exact :class:`QFrac`.

Conventions: Konishi, math/0504188, Sec. 2.3 (see CONVENTIONS.md), x = q^{1/2},
q^rho = (q^{-i+1/2})_{i>=1}, q^{nu+rho} = (q^{nu_i - i + 1/2})_{i>=1}.

* KP eq. (2):  h_i(q^rho) = q^{i(i-1)/4} / ([1]...[i]),   e_i(q^rho) = q^{-i(i-1)/4} / ([1]...[i]).
* KP eq. (3):  sum_i h_i(q^{nu+rho}) z^i = prod_{i=1}^{l(nu)} (1 - q^{-i+1/2} z)/(1 - q^{nu_i-i+1/2} z)
                                           * sum_k h_k(q^rho) z^k,
               sum_i e_i(q^{nu+rho}) z^i = prod_{i=1}^{l(nu)} (1 + q^{nu_i-i+1/2} z)/(1 + q^{-i+1/2} z)
                                           * sum_k e_k(q^rho) z^k.
* KP eq. (1), Jacobi--Trudi:  s_{mu/eta} = det(h_{mu_i - eta_j - i + j})_{l(mu) x l(mu)}
                                        = det(e_{mu^t_i - eta^t_j - i + j})_{mu_1 x mu_1}.
* Hook formula:  s_lam(q^rho) = q^{kappa(lam)/4} / prod_{boxes} [hook]   (Macdonald I.3 Ex. 2,
  rewritten with q-numbers; verified against Jacobi--Trudi in tests/test_schur.py).
"""
from __future__ import annotations

from functools import lru_cache

from .partitions import Partition, conjugate, contains, hooks, kappa
from .qring import QFrac


@lru_cache(maxsize=None)
def h_rho(i: int) -> QFrac:
    """h_i(q^rho) = q^{i(i-1)/4} / ([1][2]...[i])  (KP eq. 2); h_0 = 1, h_{<0} = 0."""
    if i < 0:
        return QFrac.zero()
    return QFrac.inv_qnums(range(1, i + 1), xexp=i * (i - 1) // 2)


@lru_cache(maxsize=None)
def e_rho(i: int) -> QFrac:
    """e_i(q^rho) = q^{-i(i-1)/4} / ([1][2]...[i])  (KP eq. 2); e_0 = 1, e_{<0} = 0."""
    if i < 0:
        return QFrac.zero()
    return QFrac.inv_qnums(range(1, i + 1), xexp=-i * (i - 1) // 2)


def _series_mul(a: list[QFrac], b: list[QFrac], n: int) -> list[QFrac]:
    """Product of two power series in z truncated at order n (inclusive)."""
    out = [QFrac.zero()] * (n + 1)
    for i, ai in enumerate(a[: n + 1]):
        if ai.is_zero():
            continue
        for j in range(min(len(b), n + 1 - i)):
            if not b[j].is_zero():
                out[i + j] = out[i + j] + ai * b[j]
    return out


@lru_cache(maxsize=None)
def _prefactor(nu: Partition, n: int, kind: str) -> tuple[QFrac, ...]:
    """Coefficients z^0..z^n of the finite products in KP eq. (3) (Laurent polynomials in x).

    kind 'h': prod_i (1 - a_i z) / (1 - b_i z);  kind 'e': prod_i (1 + b_i z) / (1 + a_i z),
    with a_i = q^{-i+1/2} = x^{1-2i}, b_i = q^{nu_i-i+1/2} = x^{2 nu_i - 2i + 1}.
    """
    series = [QFrac.one()] + [QFrac.zero()] * n
    sgn = 1 if kind == "h" else -1
    for i, part in enumerate(nu, start=1):
        a_exp, b_exp = 1 - 2 * i, 2 * part - 2 * i + 1
        if kind == "h":
            num_exp, den_exp = a_exp, b_exp      # (1 - a z) / (1 - b z)
        else:
            num_exp, den_exp = b_exp, a_exp      # (1 + b z) / (1 + a z)
        # 1/(1 - s c z) = sum_m (s c)^m z^m with s = +1 ('h') or -1 ('e')
        geo = [QFrac.x_pow(den_exp * m, sgn ** m) for m in range(n + 1)]
        lin = [QFrac.one(), QFrac.x_pow(num_exp, -sgn)]
        series = _series_mul(_series_mul(series, lin, n), geo, n)
    return tuple(series)


@lru_cache(maxsize=None)
def h_at(i: int, nu: Partition) -> QFrac:
    """h_i(q^{nu+rho}) via the generating function KP eq. (3) (h_{<0} = 0, h_0 = 1)."""
    if i < 0:
        return QFrac.zero()
    if not nu:
        return h_rho(i)
    pre = _prefactor(nu, i, "h")
    out = QFrac.zero()
    for k in range(i + 1):
        if not pre[i - k].is_zero():
            out = out + pre[i - k] * h_rho(k)
    return out


@lru_cache(maxsize=None)
def e_at(i: int, nu: Partition) -> QFrac:
    """e_i(q^{nu+rho}) via the generating function KP eq. (3) (e_{<0} = 0, e_0 = 1)."""
    if i < 0:
        return QFrac.zero()
    if not nu:
        return e_rho(i)
    pre = _prefactor(nu, i, "e")
    out = QFrac.zero()
    for k in range(i + 1):
        if not pre[i - k].is_zero():
            out = out + pre[i - k] * e_rho(k)
    return out


def _det(mat: list[list[QFrac]]) -> QFrac:
    """Determinant by Laplace expansion along rows, memoised on the set of used columns
    (O(n 2^n) QFrac operations)."""
    n = len(mat)
    if n == 0:
        return QFrac.one()
    memo: dict[tuple[int, int], QFrac] = {}

    def minor(row: int, used: int) -> QFrac:
        # determinant of rows row..n-1 with columns not in `used`
        if row == n:
            return QFrac.one()
        key = (row, used)
        if key in memo:
            return memo[key]
        acc = QFrac.zero()
        sign = 1
        for col in range(n):
            if used >> col & 1:
                continue
            entry = mat[row][col]
            if not entry.is_zero():
                term = entry * minor(row + 1, used | (1 << col))
                acc = acc + term if sign > 0 else acc - term
            sign = -sign
        memo[key] = acc
        return acc

    return minor(0, 0)


def skew_schur_h_form(mu: Partition, eta: Partition, nu: Partition) -> QFrac:
    """s_{mu/eta}(q^{nu+rho}) = det(h_{mu_i - eta_j - i + j}(q^{nu+rho}))_{1<=i,j<=l(mu)}  (KP eq. 1)."""
    if not contains(mu, eta):
        return QFrac.zero()
    n = len(mu)
    eta_ = list(eta) + [0] * (n - len(eta))
    mat = [[h_at(mu[i] - eta_[j] - i + j, nu) for j in range(n)] for i in range(n)]
    return _det(mat)


def skew_schur_e_form(mu: Partition, eta: Partition, nu: Partition) -> QFrac:
    """s_{mu/eta}(q^{nu+rho}) = det(e_{mu^t_i - eta^t_j - i + j}(q^{nu+rho}))_{1<=i,j<=mu_1}  (KP eq. 1)."""
    if not contains(mu, eta):
        return QFrac.zero()
    mt, et = conjugate(mu), conjugate(eta)
    n = len(mt)
    et_ = list(et) + [0] * (n - len(et))
    mat = [[e_at(mt[i] - et_[j] - i + j, nu) for j in range(n)] for i in range(n)]
    return _det(mat)


@lru_cache(maxsize=None)
def skew_schur_at(mu: Partition, eta: Partition, nu: Partition) -> QFrac:
    """s_{mu/eta}(q^{nu+rho}) by Jacobi--Trudi (KP eq. 1), using the h-determinant of size l(mu)
    or the e-determinant of size mu_1, whichever is smaller.  Zero unless eta is contained in mu.
    The result is passed through :meth:`QFrac.cancel`."""
    if not contains(mu, eta):
        return QFrac.zero()
    if mu == eta:
        return QFrac.one()
    if len(mu) <= (mu[0] if mu else 0):
        res = skew_schur_h_form(mu, eta, nu)
    else:
        res = skew_schur_e_form(mu, eta, nu)
    return res.cancel()


def schur_at(mu: Partition, nu: Partition) -> QFrac:
    """s_mu(q^{nu+rho}) = skew_schur_at(mu, (), nu)."""
    return skew_schur_at(mu, (), nu)


@lru_cache(maxsize=None)
def schur_rho(lam: Partition) -> QFrac:
    """s_lam(q^rho) = q^{kappa(lam)/4} / prod_{boxes x in lam} [h(x)]  (hook-content formula,
    Macdonald I.3 Ex. 2; e.g. s_(1) = 1/[1], s_(2) = q^{1/2}/([1][2]), s_(1,1) = q^{-1/2}/([1][2])).
    Equal to the Jacobi--Trudi value skew_schur_at(lam, (), ())."""
    return QFrac.inv_qnums(hooks(lam), xexp=kappa(lam) // 2)
