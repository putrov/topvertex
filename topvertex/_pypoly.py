r"""Pure-Python dense univariate polynomials over Q: the fallback backend of :mod:`topvertex.qring`
when python-flint is unavailable (e.g. under Pyodide) or disabled by ``TOPVERTEX_NO_FLINT``.

A polynomial ``sum_i c_i x^i / d`` is stored as a list ``c`` of Python ints (no trailing zeros; the
empty list is 0) and a positive int ``d`` with ``gcd(d, c_0, c_1, ...) = 1`` (``d = 1`` for 0).
This normal form is unique, so ``==`` is a comparison of the stored data.  The numerators produced
by the topological vertex are integer polynomials (d = 1) except for ``F = log Z`` and the GV
extraction, so the common case runs on plain int arithmetic without any gcd.

Multiplication is schoolbook for short or sparse inputs and Kronecker substitution (pack the
coefficients into one big int at 2^b, multiply with CPython's Karatsuba big-int product, unpack)
for long dense ones.
Division by x^{2k}-1 and other divisors with leading coefficient +-1 is integer long division over
the sparse divisor; any other divisor falls back to long division over ``Fraction``.

Only the operations needed by the ``_p*`` helpers of :mod:`topvertex.qring` are implemented.
"""
from __future__ import annotations

import sys
from array import array as _array
from fractions import Fraction
from functools import lru_cache
from math import gcd, lcm

__all__ = ["PyPoly", "ZERO", "from_fractions", "pdivmod", "peval", "pmul_xpow_minus_one"]

# 64-bit-slot fast path of the Kronecker product (needs an 8-byte little-endian array('Q');
# true on x86, ARM and WebAssembly/Pyodide).
_H64 = 1 << 63
_ARRAY_OK = _array("Q").itemsize == 8 and sys.byteorder == "little"


@lru_cache(maxsize=4096)
def _h64rep(n: int) -> int:
    """sum_{i<n} 2^63 2^(64 i)."""
    return int.from_bytes(_H64.to_bytes(8, "little") * n, "little")


# Kronecker substitution is used only if the shorter factor has at least _KRONECKER_MIN_LEN
# coefficients of which more than _KRONECKER_MIN_TERMS are nonzero (measured crossovers).
_KRONECKER_MIN_LEN = 10
_KRONECKER_MIN_TERMS = 6


class PyPoly:
    """Polynomial ``sum_i c[i] x^i / d`` in normal form (see the module docstring); immutable."""

    __slots__ = ("c", "d")

    def __init__(self, c: list[int], d: int = 1):
        # Trusted constructor: ``c`` and ``d`` must already be in normal form.
        self.c = c
        self.d = d

    # -- arithmetic --------------------------------------------------------------------------
    def __neg__(self) -> "PyPoly":
        return PyPoly([-v for v in self.c], self.d)

    def _addsub(self, other: "PyPoly", sign: int) -> "PyPoly":
        a, b, d = self.c, other.c, self.d
        if not b:
            return self
        if not a:
            return other if sign > 0 else -other
        if d != other.d:
            d = lcm(d, other.d)
            fa, fb = d // self.d, d // other.d
            if fa != 1:
                a = [v * fa for v in a]
            if fb != 1:
                b = [v * fb for v in b]
        la, lb = len(a), len(b)
        if sign > 0:
            r = [x + y for x, y in zip(a, b)]
        else:
            r = [x - y for x, y in zip(a, b)]
        if la > lb:
            r.extend(a[lb:])
        elif lb > la:
            r.extend(b[la:] if sign > 0 else [-v for v in b[la:]])
        return _normalise(r, d)

    def __add__(self, other: "PyPoly") -> "PyPoly":
        return self._addsub(other, 1)

    def __sub__(self, other: "PyPoly") -> "PyPoly":
        return self._addsub(other, -1)

    def __mul__(self, other: "PyPoly") -> "PyPoly":
        a, b = self.c, other.c
        if not a or not b:
            return ZERO
        r = _imul(a, b)  # Z[x] is a domain: no trailing zeros appear
        d = self.d * other.d
        if d == 1:
            return PyPoly(r, 1)
        return _reduce_content(r, d)

    def __pow__(self, n: int) -> "PyPoly":
        if n < 0:
            raise ValueError("negative power of a polynomial")
        result, base = ONE, self
        while n:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def __eq__(self, other) -> bool:
        if not isinstance(other, PyPoly):
            return NotImplemented
        return self.d == other.d and self.c == other.c

    __hash__ = None  # mutable-looking container; QFrac hashes by value instead

    # -- inspection --------------------------------------------------------------------------
    def degree(self) -> int:
        return len(self.c) - 1

    def coeffs(self) -> list[Fraction]:
        d = self.d
        return [Fraction(v, d) for v in self.c]

    def coeff(self, i: int) -> Fraction:
        return Fraction(self.c[i], self.d) if 0 <= i < len(self.c) else Fraction(0)

    def __repr__(self) -> str:
        return f"PyPoly({self.c}, {self.d})"


ZERO = PyPoly([], 1)
ONE = PyPoly([1], 1)


def _reduce_content(c: list[int], d: int) -> PyPoly:
    """Divide c and d by gcd(d, c_0, c_1, ...); c nonzero without trailing zeros, d > 0."""
    g = gcd(d, *c)
    if g != 1:
        c = [v // g for v in c]
        d //= g
    return PyPoly(c, d)


def _normalise(c: list[int], d: int) -> PyPoly:
    """Normal form of c / d (c a fresh list that may be modified in place, d > 0)."""
    while c and not c[-1]:
        c.pop()
    if not c:
        return ZERO
    if d == 1:
        return PyPoly(c, 1)
    return _reduce_content(c, d)


def from_fractions(coeffs) -> PyPoly:
    """Normal form of sum_i coeffs[i] x^i (ints, Fractions or other numbers.Rational)."""
    nums, dens = [], []
    for v in coeffs:
        if isinstance(v, int):
            nums.append(v)
            dens.append(1)
        else:
            v = Fraction(v)
            nums.append(v.numerator)
            dens.append(v.denominator)
    d = lcm(*dens) if dens else 1
    if d != 1:
        nums = [n * (d // e) for n, e in zip(nums, dens)]
    return _normalise(nums, d)


# -- integer polynomial multiplication ---------------------------------------------------------
def _imul(a: list[int], b: list[int]) -> list[int]:
    """Product of two nonzero integer coefficient lists: row-wise schoolbook (one list
    comprehension per nonzero coefficient of the shorter factor) when that factor is short or
    sparse -- e.g. products of a few x^{2k}-1 -- and Kronecker substitution otherwise."""
    if len(a) < len(b):
        a, b = b, a
    lb = len(b)
    if lb == 1:
        y = b[0]
        return a[:] if y == 1 else [x * y for x in a]
    nz = [(j, y) for j, y in enumerate(b) if y]
    if lb >= _KRONECKER_MIN_LEN and len(nz) > _KRONECKER_MIN_TERMS:
        return _kronecker(a, b)
    la = len(a)
    r = [0] * (la + lb - 1)
    for j, y in nz:
        seg = r[j:j + la]
        if y == 1:
            r[j:j + la] = [u + v for u, v in zip(seg, a)]
        elif y == -1:
            r[j:j + la] = [u - v for u, v in zip(seg, a)]
        else:
            r[j:j + la] = [u + y * v for u, v in zip(seg, a)]
    return r


def _kronecker(a: list[int], b: list[int]) -> list[int]:
    """Integer polynomial product via Kronecker substitution x -> 2^(8*nb).

    Every coefficient of the product is bounded by ``bound = max|a| * max|b| * min(len)``; with
    ``H = 2^(8 nb - 1) > bound`` each digit ``c_i + H`` lies in [0, 2^(8 nb)), so the shifted
    product unpacks byte-wise without carries."""
    ma = max(max(a), -min(a))
    mb = max(max(b), -min(b))
    bound = ma * mb * min(len(a), len(b))
    n = len(a) + len(b) - 1
    if bound < _H64 and _ARRAY_OK:
        # 64-bit slots: pack / unpack in C via array('Q') (the usual case)
        rep = _h64rep
        pa = int.from_bytes(_array("Q", [v + _H64 for v in a]).tobytes(), "little") - rep(len(a))
        pb = int.from_bytes(_array("Q", [v + _H64 for v in b]).tobytes(), "little") - rep(len(b))
        out = _array("Q")
        out.frombytes((pa * pb + rep(n)).to_bytes(8 * n, "little"))
        return [v - _H64 for v in out]
    nb = (bound.bit_length() + 2 + 7) // 8
    H = 1 << (8 * nb - 1)
    unit = (H.to_bytes(nb, "little"))

    def pack(c):
        return (int.from_bytes(b"".join((v + H).to_bytes(nb, "little") for v in c), "little")
                - int.from_bytes(unit * len(c), "little"))

    prod = pack(a) * pack(b) + int.from_bytes(unit * n, "little")
    raw = prod.to_bytes(n * nb, "little")
    fb = int.from_bytes
    return [fb(raw[i:i + nb], "little") - H for i in range(0, n * nb, nb)]


def pmul_xpow_minus_one(p: PyPoly, m: int) -> PyPoly:
    """p * (x^m - 1), m >= 1 (content and denominator unchanged)."""
    c = p.c
    if not c:
        return p
    L = len(c)
    if L <= m:
        r = [-v for v in c]
        r.extend([0] * (m - L))
        r.extend(c)
    else:
        r = [-v for v in c[:m]]
        r.extend([c[i - m] - c[i] for i in range(m, L)])
        r.extend(c[L - m:])
    return PyPoly(r, p.d)


# -- division ----------------------------------------------------------------------------------
def pdivmod(a: PyPoly, b: PyPoly) -> tuple[PyPoly, PyPoly]:
    """(q, r) with a = q b + r, deg r < deg b, over Q (b nonzero)."""
    bc = b.c
    if not bc:
        raise ZeroDivisionError("polynomial division by zero")
    m = len(bc) - 1
    ac = a.c
    if len(ac) <= m:
        return ZERO, a
    lead = bc[-1]
    if b.d == 1 and lead in (1, -1):
        # integer long division; a = (A / da), b integral with unit leading coefficient
        r = ac[:]
        q = [0] * (len(ac) - m)
        nz = [(j, v) for j, v in enumerate(bc[:-1]) if v]
        for i in range(len(ac) - 1 - m, -1, -1):
            t = r[i + m]
            if t:
                if lead == -1:
                    t = -t
                q[i] = t
                for j, v in nz:
                    r[i + j] -= t * v
        del r[m:]
        return _normalise(q, a.d), _normalise(r, a.d)
    # general divisor: long division over Fraction
    r = a.coeffs()
    bf = b.coeffs()
    inv_lead = 1 / bf[-1]
    q = [Fraction(0)] * (len(r) - m)
    nz = [(j, v) for j, v in enumerate(bf[:-1]) if v]
    for i in range(len(r) - 1 - m, -1, -1):
        t = r[i + m]
        if t:
            t *= inv_lead
            q[i] = t
            for j, v in nz:
                r[i + j] -= t * v
    return from_fractions(q), from_fractions(r[:m])


def peval(p: PyPoly, x: Fraction) -> Fraction:
    """Exact value p(x) at a rational point."""
    c = p.c
    if not c:
        return Fraction(0)
    x = Fraction(x)
    u, w = x.numerator, x.denominator
    # sum_i c_i u^i w^(n-i) by Horner, n = deg p
    v, wp = 0, 1
    for ci in reversed(c):
        v = v * u + ci * wp
        wp *= w
    # after the loop wp = w^(n+1); the value is v / (w^n d)
    return Fraction(v, (wp // w) * p.d)
