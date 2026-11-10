r"""Exact rational functions of x = q^{1/2} with q-number denominators.

Every element is stored as

    f = x^{offset} * P(x) / prod_k [k]^{m_k},      [k] = x^k - x^{-k} = q^{k/2} - q^{-k/2},

with ``P`` a polynomial with rational coefficients (backend-dependent, see below), ``offset`` an integer
and the denominator a finite multiset of q-numbers ``{k: m_k}`` (k >= 1).  This is exactly the
class of functions produced by the topological vertex in the Konishi (math/0504188) conventions
(math/0504188, Sec. 2.2--2.3): h_i(q^rho), e_i(q^rho) have denominators [1]...[i] and all other
ingredients are Laurent polynomials in q^{1/2}.

Normal form of the numerator: P has nonzero constant term (all powers of x are absorbed into
``offset``), and the zero element is ``(0, 0, {})``.  The denominator is NOT reduced
automatically; :meth:`QFrac.cancel` removes [k]-factors that divide the numerator.  Equality
``==`` is mathematical equality of rational functions (tested by cross-multiplication), and
``hash`` is the hash of the exact value at x = 2, so both are independent of the chosen
representation.

Polynomial backends.  All polynomial-specific code lives in the ``_p*`` helpers at the top of this
module.  Two interchangeable backends are provided, chosen once at import time and recorded in
``BACKEND``:

* ``"flint"``: ``flint.fmpq_poly`` from python-flint (fast; used whenever ``import flint``
  succeeds and the environment variable ``TOPVERTEX_NO_FLINT`` is unset, empty or ``0``);
* ``"python"``: :class:`topvertex._pypoly.PyPoly`, dense integer coefficient lists with a common
  denominator (pure Python, e.g. for Pyodide where python-flint is not available).

Both give identical results; the test suite runs with either.
"""
from __future__ import annotations

import os
from fractions import Fraction
from numbers import Rational

flint = None
if os.environ.get("TOPVERTEX_NO_FLINT", "") in ("", "0"):
    try:
        import flint
    except ImportError:  # pragma: no cover - depends on the environment
        flint = None

BACKEND = "flint" if flint is not None else "python"

# ---------------------------------------------------------------------------------------------
# Polynomial backend.  Polynomials are in the variable x = q^{1/2}.  Helper interface:
#   _pconst(c), _pfrom_coeffs(list), _pcoeffs(p) -> [Fraction], _pdeg(p) (-1 for 0), _pis_zero,
#   _pshift(p, n) = x^n p, _plow_order(p), _pstrip_low(p, n) = p / x^n, _pdivmod(a, b),
#   _peval(p, Fraction) -> Fraction, _pcoeff(p, i) -> Fraction,
#   _pmul_qnums(p, {k: m}) = p prod_k (x^{2k}-1)^m,
#   _qnum_poly(k) = x^{2k}-1; polynomial objects support unary -, +, -, *, ** (int >= 0) and ==.
# ---------------------------------------------------------------------------------------------
if flint is not None:
    _Poly = flint.fmpq_poly

    def _to_fmpq(c) -> "flint.fmpq":
        if isinstance(c, flint.fmpq):
            return c
        if isinstance(c, int):
            return flint.fmpq(c)
        c = Fraction(c)
        return flint.fmpq(c.numerator, c.denominator)

    def _pconst(c):
        return _Poly([_to_fmpq(c)])

    def _pfrom_coeffs(coeffs):
        """Polynomial sum_i coeffs[i] x^i."""
        return _Poly([_to_fmpq(c) for c in coeffs])

    def _pcoeffs(p) -> list[Fraction]:
        """Dense coefficient list [c_0, ..., c_deg] as Fractions (empty for 0)."""
        return [Fraction(int(c.p), int(c.q)) for c in p.coeffs()]

    def _pdeg(p) -> int:
        return p.degree()  # -1 for the zero polynomial

    def _pis_zero(p) -> bool:
        return p.is_zero()

    def _pshift(p, n: int):
        """Multiply by x^n, n >= 0."""
        return p.left_shift(n) if n else p

    def _plow_order(p) -> int:
        """Largest n with x^n | p (p nonzero)."""
        n = 0
        while p[n] == 0:
            n += 1
        return n

    def _pstrip_low(p, n: int):
        """Divide by x^n where x^n | p."""
        return p.right_shift(n) if n else p

    def _pdivmod(a, b):
        return divmod(a, b)

    def _peval(p, val: Fraction) -> Fraction:
        v = p(flint.fmpq(val.numerator, val.denominator))
        return Fraction(int(v.p), int(v.q))

    def _pcoeff(p, i: int) -> Fraction:
        c = p[i]
        return Fraction(int(c.p), int(c.q))

    def _qnum_poly_new(k: int):
        return _Poly([-1] + [0] * (2 * k - 1) + [1])

    def _pmul_qnums(p, den: dict[int, int]):
        """p * prod_k (x^{2k} - 1)^{den[k]}."""
        for k, m in den.items():
            for _ in range(m):
                p = p * _qnum_poly(k)
        return p

else:
    from . import _pypoly

    _Poly = _pypoly.PyPoly

    def _pconst(c):
        if isinstance(c, int):
            return _Poly([c], 1) if c else _pypoly.ZERO
        return _pypoly.from_fractions([c])

    _pfrom_coeffs = _pypoly.from_fractions

    def _pcoeffs(p) -> list[Fraction]:
        """Dense coefficient list [c_0, ..., c_deg] as Fractions (empty for 0)."""
        return p.coeffs()

    def _pdeg(p) -> int:
        return len(p.c) - 1  # -1 for the zero polynomial

    def _pis_zero(p) -> bool:
        return not p.c

    def _pshift(p, n: int):
        """Multiply by x^n, n >= 0."""
        return _Poly([0] * n + p.c, p.d) if n and p.c else p

    def _plow_order(p) -> int:
        """Largest n with x^n | p (p nonzero)."""
        c = p.c
        n = 0
        while not c[n]:
            n += 1
        return n

    def _pstrip_low(p, n: int):
        """Divide by x^n where x^n | p."""
        return _Poly(p.c[n:], p.d) if n else p

    _pdivmod = _pypoly.pdivmod
    _peval = _pypoly.peval

    def _pcoeff(p, i: int) -> Fraction:
        return p.coeff(i)

    def _qnum_poly_new(k: int):
        return _Poly([-1] + [0] * (2 * k - 1) + [1], 1)

    _QNUMS_PROD_CACHE: dict = {}

    def _pmul_qnums(p, den: dict[int, int]):
        """p * prod_k (x^{2k} - 1)^{den[k]}: one product with the cached (short) polynomial
        prod_k (x^{2k}-1)^{m_k} instead of one O(deg p) pass per factor."""
        key = tuple(sorted((k, m) for k, m in den.items() if m))
        if len(key) == 1 and key[0][1] == 1:
            return _pypoly.pmul_xpow_minus_one(p, 2 * key[0][0])
        prod = _QNUMS_PROD_CACHE.get(key)
        if prod is None:
            prod = _pypoly.ONE
            for k, m in key:
                for _ in range(m):
                    prod = _pypoly.pmul_xpow_minus_one(prod, 2 * k)
            _QNUMS_PROD_CACHE[key] = prod
        return p * prod


_QNUM_POLY_CACHE: dict = {}


def _qnum_poly(k: int):
    """x^{2k} - 1, so that [k] = x^{-k} (x^{2k} - 1)."""
    p = _QNUM_POLY_CACHE.get(k)
    if p is None:
        p = _qnum_poly_new(k)
        _QNUM_POLY_CACHE[k] = p
    return p


# ---------------------------------------------------------------------------------------------


def _den_key(den: dict[int, int]) -> tuple:
    return tuple(sorted((k, m) for k, m in den.items() if m))


def _times_qnums(off: int, p: _Poly, den: dict[int, int]) -> tuple[int, _Poly]:
    """Multiply x^off p by prod_k [k]^{den[k]} = x^{-sum k m_k} prod_k (x^{2k}-1)^{m_k}."""
    shift = sum(k * m for k, m in den.items())
    if not shift:
        return off, p
    return off - shift, _pmul_qnums(p, den)


class QFrac:
    r"""Exact element x^{offset} P(x) / prod_k [k]^{m_k} of Q(q^{1/2}), x = q^{1/2}.

    Constructors: ``QFrac(c)`` for int/Fraction c, :meth:`x_pow`, :meth:`q_pow`, :meth:`qnum`,
    :meth:`one`, :meth:`zero`, :meth:`from_laurent`.  Arithmetic: ``+ - * / **`` (integer
    exponents), unary minus.  Division is exact-only, see :meth:`__truediv__`.
    """

    __slots__ = ("_off", "_num", "_den", "_hash")

    # -- construction -----------------------------------------------------------------------
    def __init__(self, value=0):
        if isinstance(value, QFrac):
            self._off, self._num, self._den = value._off, value._num, value._den
        elif isinstance(value, (int, Rational)):
            self._off, self._num, self._den = 0, _pconst(value), ()
        else:
            raise TypeError(f"cannot build QFrac from {type(value).__name__}")
        self._hash = None

    @classmethod
    def _make(cls, off: int, num: _Poly, den) -> "QFrac":
        """Build from raw data, normalising the numerator.  ``den`` is a dict or sorted tuple."""
        self = object.__new__(cls)
        if _pis_zero(num):
            self._off, self._num, self._den = 0, num, ()
        else:
            n = _plow_order(num)
            self._off, self._num = off + n, _pstrip_low(num, n)
            self._den = _den_key(den) if isinstance(den, dict) else den
        self._hash = None
        return self

    @classmethod
    def zero(cls) -> "QFrac":
        return cls(0)

    @classmethod
    def one(cls) -> "QFrac":
        return cls(1)

    @classmethod
    def x_pow(cls, n: int, coeff=1) -> "QFrac":
        """coeff * x^n = coeff * q^{n/2} (n any integer)."""
        return cls._make(int(n), _pconst(coeff), ())

    @classmethod
    def q_pow(cls, n: int, coeff=1) -> "QFrac":
        """coeff * q^n (n integer)."""
        return cls.x_pow(2 * int(n), coeff)

    @classmethod
    def qnum(cls, k: int) -> "QFrac":
        """The q-number [k] = q^{k/2} - q^{-k/2} = x^k - x^{-k} (KP Sec. 2.2), as a numerator."""
        if k == 0:
            return cls.zero()
        if k < 0:
            return -cls.qnum(-k)
        return cls._make(-k, _qnum_poly(k), ())

    @classmethod
    def inv_qnums(cls, ks, coeff=1, xexp: int = 0) -> "QFrac":
        """coeff * x^xexp / prod_{k in ks} [k]  (ks an iterable of positive ints, repeats allowed)."""
        den: dict[int, int] = {}
        for k in ks:
            if k <= 0:
                raise ValueError("q-numbers in the denominator must have k >= 1")
            den[k] = den.get(k, 0) + 1
        return cls._make(xexp, _pconst(coeff), den)

    @classmethod
    def from_laurent(cls, coeffs: dict[int, object]) -> "QFrac":
        """Laurent polynomial sum_n coeffs[n] x^n (keys are x-exponents, i.e. 2 * q-exponent)."""
        coeffs = {int(n): c for n, c in coeffs.items() if c != 0}
        if not coeffs:
            return cls.zero()
        lo = min(coeffs)
        dense = [0] * (max(coeffs) - lo + 1)
        for n, c in coeffs.items():
            dense[n - lo] = c
        return cls._make(lo, _pfrom_coeffs(dense), ())

    # -- accessors --------------------------------------------------------------------------
    @property
    def denominator(self) -> dict[int, int]:
        """The denominator multiset {k: multiplicity} of q-numbers [k]."""
        return dict(self._den)

    def numerator_laurent(self) -> dict[int, Fraction]:
        """Numerator as {x-exponent: coefficient} (x = q^{1/2})."""
        return {self._off + i: c for i, c in enumerate(_pcoeffs(self._num)) if c != 0}

    def is_zero(self) -> bool:
        return _pis_zero(self._num)

    def is_laurent(self) -> bool:
        """True if the stored denominator is empty (call :meth:`cancel` first if needed)."""
        return not self._den

    # -- arithmetic -------------------------------------------------------------------------
    @staticmethod
    def _coerce(other):
        if isinstance(other, QFrac):
            return other
        if isinstance(other, (int, Rational)):
            return QFrac(other)
        return NotImplemented

    def __neg__(self) -> "QFrac":
        return QFrac._make(self._off, -self._num, self._den)

    def __pos__(self) -> "QFrac":
        return self

    def _add(self, other: "QFrac", sign: int) -> "QFrac":
        """a +- b over the lcm of the denominators (max multiplicities)."""
        if other.is_zero():
            return self
        if self.is_zero():
            return other if sign > 0 else -other
        da, db = dict(self._den), dict(other._den)
        if self._den == other._den:
            lcm = da
            oa, pa, ob, pb = self._off, self._num, other._off, other._num
        else:
            lcm = dict(da)
            for k, m in db.items():
                if m > lcm.get(k, 0):
                    lcm[k] = m
            oa, pa = _times_qnums(self._off, self._num,
                                  {k: m - da.get(k, 0) for k, m in lcm.items() if m > da.get(k, 0)})
            ob, pb = _times_qnums(other._off, other._num,
                                  {k: m - db.get(k, 0) for k, m in lcm.items() if m > db.get(k, 0)})
        lo = min(oa, ob)
        pa, pb = _pshift(pa, oa - lo), _pshift(pb, ob - lo)
        return QFrac._make(lo, pa + pb if sign > 0 else pa - pb, lcm)

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        return self._add(other, 1)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        return self._add(other, -1)

    def __rsub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        return other._add(self, -1)

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        if self.is_zero() or other.is_zero():
            return QFrac.zero()
        if not other._den:
            den = self._den
        elif not self._den:
            den = other._den
        else:
            d = dict(self._den)
            for k, m in other._den:
                d[k] = d.get(k, 0) + m
            den = d
        return QFrac._make(self._off + other._off, self._num * other._num, den)

    __rmul__ = __mul__

    def _peel_qnums(self) -> tuple[_Poly, dict[int, int]]:
        """Write the numerator polynomial as R(x) * prod_k (x^{2k}-1)^{m_k}, greedily from the
        largest k (this decomposition into q-numbers is unique: the largest cyclotomic index
        present can only come from [k] with that k).  Returns (R, {k: m_k})."""
        p = self._num
        peeled: dict[int, int] = {}
        k = _pdeg(p) // 2
        while k >= 1:
            qq, r = _pdivmod(p, _qnum_poly(k))
            if _pis_zero(r):
                p = qq
                peeled[k] = peeled.get(k, 0) + 1
                k = min(k, _pdeg(p) // 2)
            else:
                k -= 1
        return p, peeled

    def __truediv__(self, other):
        """Exact division a / b.

        (a) if the numerator of b is c x^m times a product of q-numbers [k] (detected by exact
            division by x^{2k}-1), the [k]'s move to the denominator (cancelling against b's own
            denominator factors where possible) -- this always succeeds;
        (b) otherwise the remaining polynomial factor R of b's numerator must divide
            num(a) * den(b) (as polynomials, via polynomial divmod); if it does not, ValueError.
        General rational-function division is deliberately not supported.
        """
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        if other.is_zero():
            raise ZeroDivisionError("QFrac division by zero")
        if self.is_zero():
            return QFrac.zero()
        if _pdeg(other._num) == 0:
            rem, peeled = other._num, {}
        else:
            rem, peeled = other._peel_qnums()
        # x^{2k} - 1 = x^k [k], hence other = x^{other_off} * rem * prod_k [k]^{m_k}
        other_off = other._off + sum(k * m for k, m in peeled.items())
        # self / other = x^{a.off - other_off} * a.num * prod_{den b}[k] / (den a * prod peeled[k] * rem)
        den = dict(self._den)
        for k, m in peeled.items():
            den[k] = den.get(k, 0) + m
        extra: dict[int, int] = {}
        for k, m in other._den:
            c = min(m, den.get(k, 0))
            if c:
                den[k] -= c
            if m - c:
                extra[k] = m - c
        off, num = _times_qnums(self._off - other_off, self._num, extra)
        if _pdeg(rem) == 0:
            num = num * _pconst(1 / _pcoeff(rem, 0))
        else:
            quo, r = _pdivmod(num, rem)
            if not _pis_zero(r):
                raise ValueError("QFrac division is not exact (divisor numerator is not a "
                                 "monomial times q-numbers and does not divide the dividend)")
            num = quo
        return QFrac._make(off, num, {k: m for k, m in den.items() if m})

    def __rtruediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        return other.__truediv__(self)

    def __pow__(self, n: int) -> "QFrac":
        if not isinstance(n, int):
            raise TypeError("QFrac exponent must be an integer")
        if n < 0:
            return QFrac.one() / (self ** (-n))
        if n == 0:
            return QFrac.one()
        if self.is_zero():
            return QFrac.zero()
        return QFrac._make(self._off * n, self._num ** n, {k: m * n for k, m in self._den})

    # -- comparison / hashing ---------------------------------------------------------------
    def __eq__(self, other) -> bool:
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        if (self._off, self._den) == (other._off, other._den) and self._num == other._num:
            return True
        return self._add(other, -1).is_zero()

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.exact_value_at(Fraction(2)))
        return self._hash

    # -- simplification ---------------------------------------------------------------------
    def cancel(self) -> "QFrac":
        """Cancel every q-number [k] of the denominator that divides the numerator (as a Laurent
        polynomial, i.e. x^{2k}-1 | P).  Larger k are tried first, which makes the result
        independent of the input representation whenever the reduced denominator is a product
        of q-numbers coprime to the numerator in the sense of cyclotomic factors."""
        if not self._den or self.is_zero():
            return self
        off, p = self._off, self._num
        den = dict(self._den)
        changed = True
        while changed:
            changed = False
            for k in sorted(den, reverse=True):
                while den[k] and _pdeg(p) >= 2 * k:
                    qq, r = _pdivmod(p, _qnum_poly(k))
                    if not _pis_zero(r):
                        break
                    p, off = qq, off + k
                    den[k] -= 1
                    changed = True
            den = {k: m for k, m in den.items() if m}
        return QFrac._make(off, p, den)

    # -- inspection -------------------------------------------------------------------------
    def as_laurent(self) -> dict[Fraction, Fraction]:
        """Return {q-exponent (Fraction, half-integers allowed): coefficient} if self is a Laurent
        polynomial in q^{1/2} (after :meth:`cancel`); raise ValueError otherwise."""
        f = self.cancel()
        if f._den:
            raise ValueError(f"not a Laurent polynomial: denominator {f._den}")
        return {Fraction(n, 2): c for n, c in f.numerator_laurent().items()}

    def is_q_symmetric(self) -> bool:
        """True iff f(x) = f(1/x), i.e. invariant under q -> 1/q.  (Each [k] is odd under x->1/x,
        so the denominator contributes the sign (-1)^{sum m_k}.)"""
        if self.is_zero():
            return True
        num = self.numerator_laurent()
        sign = -1 if sum(m for _, m in self._den) % 2 else 1
        return all(num.get(-n, 0) == sign * c for n, c in num.items())

    def to_y_polynomial(self) -> list[Fraction]:
        """For a q-symmetric Laurent polynomial P in q (integer powers of q only), return
        [c_0, c_1, ..., c_G] with P = sum_g c_g y^g, y = [1]^2 = q - 2 + q^{-1} (CONVENTIONS.md, GV
        extraction).  Obtained by repeatedly peeling off the top power of q.  Raises ValueError
        if self is not a Laurent polynomial, not symmetric, or has half-integer q-powers."""
        f = self.cancel()
        if f._den:
            raise ValueError("to_y_polynomial: not a Laurent polynomial")
        if not f.is_q_symmetric():
            raise ValueError("to_y_polynomial: not symmetric under q -> 1/q")
        coeffs = f.numerator_laurent()  # x-exponents
        if any(n % 2 for n in coeffs):
            raise ValueError("to_y_polynomial: half-integer powers of q present")
        rest = {n // 2: c for n, c in coeffs.items()}  # q-exponents
        if not rest:
            return [Fraction(0)]
        top = max(rest)
        out = [Fraction(0)] * (top + 1)
        from math import comb
        while rest:
            g = max(rest)
            if g < 0:
                raise ValueError("to_y_polynomial: not expressible")  # cannot happen if symmetric
            c = rest[g]
            out[g] = c
            # y^g = (x - 1/x)^{2g} = sum_j (-1)^j C(2g, j) q^{g - j}
            for j in range(2 * g + 1):
                e = g - j
                v = rest.get(e, 0) - c * (-1) ** j * comb(2 * g, j)
                if v:
                    rest[e] = v
                else:
                    rest.pop(e, None)
        return out

    # -- evaluation / output ----------------------------------------------------------------
    def exact_value_at(self, xval: Fraction) -> Fraction:
        """Exact value at a rational point x = xval (must not be a zero of any [k])."""
        xv = Fraction(xval)
        num = _peval(self._num, xv) * xv ** self._off
        den = Fraction(1)
        for k, m in self._den:
            den *= (xv ** k - xv ** (-k)) ** m
        return num / den

    def evaluate(self, xval) -> complex:
        """Numerical value at x = q^{1/2} = xval (complex)."""
        xv = complex(xval)
        s = sum(float(c) * xv ** n for n, c in self.numerator_laurent().items())
        d = complex(1)
        for k, m in self._den:
            d *= (xv ** k - xv ** (-k)) ** m
        return complex(s / d)

    def to_sympy(self, q=None):
        """sympy expression in the symbol q (default Symbol('q', positive=True)); odd powers of x
        become powers of sqrt(q).  Not simplified: numerator / prod (q^{k/2} - q^{-k/2})^m."""
        import sympy as sp
        if q is None:
            q = sp.Symbol("q", positive=True)
        num = sum((sp.Rational(c.numerator, c.denominator) * q ** sp.Rational(n, 2)
                   for n, c in self.numerator_laurent().items()), sp.Integer(0))
        den = sp.Integer(1)
        for k, m in self._den:
            den *= (q ** sp.Rational(k, 2) - q ** sp.Rational(-k, 2)) ** m
        return num / den

    def to_mathematica(self) -> str:
        """Mathematica input string in the variable q, e.g. (q^(1/2) - q^(-1/2))^(-1)."""
        def mono(n: int) -> str:
            return f"q^({n}/2)" if n % 2 else f"q^({n // 2})"
        terms = []
        for n, c in sorted(self.numerator_laurent().items()):
            cs = f"({c.numerator}/{c.denominator})" if c.denominator != 1 else f"({c.numerator})"
            terms.append(f"{cs}*{mono(n)}")
        num = " + ".join(terms) if terms else "0"
        if not self._den:
            return f"({num})"
        den = "*".join(f"({mono(k)} - {mono(-k)})^{m}" for k, m in self._den)
        return f"({num})/({den})"

    def latex(self) -> str:
        """LaTeX string; the denominator is written with q-numbers [k] = q^{k/2}-q^{-k/2}."""
        import sympy as sp
        q = sp.Symbol("q", positive=True)
        num = QFrac._make(self._off, self._num, ()).to_sympy(q)
        ns = sp.latex(sp.expand(num))
        if not self._den:
            return ns
        ds = "".join(f"[{k}]" + (f"^{{{m}}}" if m > 1 else "") for k, m in self._den)
        return rf"\frac{{{ns}}}{{{ds}}}"

    def __str__(self) -> str:
        def mono(n: int) -> str:
            return "" if n == 0 else ("x" if n == 1 else f"x^{n}")
        parts = []
        for n, c in sorted(self.numerator_laurent().items()):
            m = mono(n)
            if not m:
                parts.append(str(c))
            elif c == 1:
                parts.append(m)
            elif c == -1:
                parts.append("-" + m)
            else:
                parts.append(f"{c}*{m}")
        num = " + ".join(parts).replace("+ -", "- ") if parts else "0"
        if not self._den:
            return num
        den = " ".join(f"[{k}]" + (f"^{m}" if m > 1 else "") for k, m in self._den)
        return f"({num}) / ({den})"

    def __repr__(self) -> str:
        return f"QFrac({self}; x=q^(1/2))"

