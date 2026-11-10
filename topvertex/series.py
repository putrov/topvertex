r"""Truncated multivariate formal power series with exact :class:`~topvertex.qring.QFrac`
coefficients.

A :class:`Series` is

    S = sum_{d} S_d Q^d,       Q^d = prod_i Q_i^{d_i},   S_d in Q(q^{1/2}),

stored sparsely as ``{exponent vector d: S_d}``.  It is truncated by a positive grading: every
variable ``Q_i`` carries an integer weight ``w_i`` and only exponents with
``|d| = sum_i w_i d_i <= max_weight`` are kept (CONVENTIONS.md: Kahler weights, sum_e w_e |lam_e| <= D).
Optionally, a box truncation ``d_i <= max_degrees[i]`` is imposed in addition (meaningful when all
exponents are non-negative); both truncations are order ideals for non-negative exponents, so
products, ``log`` and ``exp`` computed inside the truncation are exact there.

Exponent vectors may contain negative entries after a monomial substitution (curve classes in a
basis that is not the Mori cone); the grading then still has to be positive on every exponent
that occurs with nonzero coefficient (other than 0) for ``log``/``exp`` to make sense.

``log`` uses the Euler-operator recursion (graded by the weight |d|):

    |d| F_d = |d| Z_d - sum_{d' + d'' = d, d', d'' != 0} |d'| F_{d'} Z_{d''},

and ``exp`` the inverse recursion |d| Z_d = sum_{d' + d'' = d, d' != 0} |d'| F_{d'} Z_{d''}.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

from .qring import QFrac

__all__ = ["Series"]

Exp = tuple[int, ...]


def _coerce_q(c) -> QFrac:
    return c if isinstance(c, QFrac) else QFrac(c)


class Series:
    """Truncated multivariate power series with QFrac coefficients.

    Parameters: ``variables`` (names), ``coefficients`` ({exponent tuple: QFrac or number};
    zero coefficients are dropped, out-of-truncation terms are dropped), ``weights`` (integer
    weight of each variable; default all 1), ``max_weight`` (keep |d| <= max_weight),
    ``max_degrees`` (optional per-variable upper bounds, None entries = unbounded).
    """

    __slots__ = ("variables", "coefficients", "weights", "max_weight", "max_degrees")

    def __init__(self, variables: Sequence[str], coefficients: Mapping[Exp, object] | None = None,
                 weights: Sequence[int] | None = None, max_weight: int = 0,
                 max_degrees: Sequence[int | None] | None = None):
        self.variables: tuple[str, ...] = tuple(variables)
        n = len(self.variables)
        self.weights: tuple[int, ...] = tuple(int(w) for w in weights) if weights is not None else (1,) * n
        if len(self.weights) != n:
            raise ValueError("need one weight per variable")
        self.max_weight: int = int(max_weight)
        if max_degrees is not None:
            max_degrees = tuple(None if m is None else int(m) for m in max_degrees)
            if len(max_degrees) != n:
                raise ValueError("need one max degree per variable")
            if all(m is None for m in max_degrees):
                max_degrees = None
        self.max_degrees: tuple[int | None, ...] | None = max_degrees
        self.coefficients: dict[Exp, QFrac] = {}
        for d, c in (coefficients or {}).items():
            d = tuple(int(x) for x in d)
            if len(d) != n:
                raise ValueError(f"exponent {d} has wrong length (variables {self.variables})")
            c = _coerce_q(c)
            if not c.is_zero() and self.in_range(d):
                self.coefficients[d] = c

    # ------------------------------------------------------------------ basic accessors
    @classmethod
    def one(cls, variables, weights=None, max_weight: int = 0, max_degrees=None) -> "Series":
        """The series 1."""
        n = len(tuple(variables))
        return cls(variables, {(0,) * n: QFrac.one()}, weights, max_weight, max_degrees)

    def weight(self, d: Exp) -> int:
        """|d| = sum_i w_i d_i."""
        return sum(w * x for w, x in zip(self.weights, d))

    def in_range(self, d: Exp) -> bool:
        """d lies inside the truncation (weight and box)."""
        if self.weight(d) > self.max_weight:
            return False
        if self.max_degrees is not None:
            return all(m is None or x <= m for x, m in zip(d, self.max_degrees))
        return True

    def coeff(self, exp) -> QFrac:
        """Coefficient of Q^exp (zero if absent).  Raises if exp is outside the truncation."""
        exp = tuple(int(x) for x in exp)
        if not self.in_range(exp):
            raise ValueError(f"exponent {exp} is outside the truncation")
        return self.coefficients.get(exp, QFrac.zero())

    def __getitem__(self, exp) -> QFrac:
        return self.coeff(exp)

    def terms(self) -> list[tuple[Exp, QFrac]]:
        """Nonzero terms sorted by weight, then lexicographically by exponent."""
        return [(d, self.coefficients[d]) for d in sorted(self.coefficients, key=lambda d: (self.weight(d), d))]

    def __len__(self) -> int:
        return len(self.coefficients)

    def constant_term(self) -> QFrac:
        """Coefficient of the monomial 1 (exponent vector 0)."""
        return self.coefficients.get((0,) * len(self.variables), QFrac.zero())

    def _like(self, coefficients, max_weight=None, max_degrees="same") -> "Series":
        return Series(self.variables, coefficients, self.weights,
                      self.max_weight if max_weight is None else max_weight,
                      self.max_degrees if max_degrees == "same" else max_degrees)

    def _check_compatible(self, other: "Series"):
        if self.variables != other.variables or self.weights != other.weights:
            raise ValueError("series have different variables or weights")

    def _joint_truncation(self, other: "Series"):
        mw = min(self.max_weight, other.max_weight)
        if self.max_degrees is None:
            md = other.max_degrees
        elif other.max_degrees is None:
            md = self.max_degrees
        else:
            md = tuple(b if a is None else a if b is None else min(a, b)
                       for a, b in zip(self.max_degrees, other.max_degrees))
        return mw, md

    # ------------------------------------------------------------------ arithmetic
    def _coerce(self, other) -> "Series":
        if isinstance(other, Series):
            self._check_compatible(other)
            return other
        c = _coerce_q(other)
        return self._like({(0,) * len(self.variables): c})

    def __add__(self, other) -> "Series":
        other = self._coerce(other)
        mw, md = self._joint_truncation(other)
        out = dict(self.coefficients)
        for d, c in other.coefficients.items():
            out[d] = out[d] + c if d in out else c
        return self._like(out, mw, md)

    __radd__ = __add__

    def __neg__(self) -> "Series":
        return self._like({d: -c for d, c in self.coefficients.items()})

    def __sub__(self, other) -> "Series":
        return self + (-self._coerce(other))

    def __rsub__(self, other) -> "Series":
        return self._coerce(other) - self

    def __mul__(self, other) -> "Series":
        """Product truncated at the joint truncation (min of max_weight, min of box bounds)."""
        other = self._coerce(other)
        mw, md = self._joint_truncation(other)
        res = self._like({}, mw, md)
        out: dict[Exp, QFrac] = {}
        for d1, c1 in self.coefficients.items():
            for d2, c2 in other.coefficients.items():
                d = tuple(a + b for a, b in zip(d1, d2))
                if not res.in_range(d):
                    continue
                p = c1 * c2
                out[d] = out[d] + p if d in out else p
        return self._like(out, mw, md)

    __rmul__ = __mul__

    def __eq__(self, other) -> bool:
        if not isinstance(other, Series):
            return NotImplemented
        if self.variables != other.variables or self.weights != other.weights:
            return False
        keys = set(self.coefficients) | set(other.coefficients)
        return all(self.coefficients.get(d, QFrac.zero()) == other.coefficients.get(d, QFrac.zero()) for d in keys)

    def truncate(self, max_weight: int | None = None, max_degrees=None) -> "Series":
        """Drop all terms of weight > max_weight (and outside the box ``max_degrees``)."""
        mw = self.max_weight if max_weight is None else min(self.max_weight, int(max_weight))
        md = self.max_degrees if max_degrees is None else self._joint_truncation(
            Series(self.variables, {}, self.weights, mw, max_degrees))[1]
        return self._like(self.coefficients, mw, md)

    def cancel(self) -> "Series":
        """Apply :meth:`QFrac.cancel` to every coefficient."""
        return self._like({d: c.cancel() for d, c in self.coefficients.items()})

    def map_coefficients(self, f) -> "Series":
        """Series with the same exponents and coefficients f(c); truncation data is kept."""
        return self._like({d: f(c) for d, c in self.coefficients.items()})

    # ------------------------------------------------------------------ substitution
    def substitute(self, class_map, new_variables: Sequence[str], new_weights: Sequence[int],
                   max_degrees=None) -> "Series":
        """Monomial substitution Q_e -> prod_i t_i^{c_{e,i}}.

        ``class_map``: either {old variable name: {new variable name: int}} or a matrix (one row
        per old variable, one column per new variable).  ``new_weights``: integer weights w_i of
        the new variables; they must satisfy w_e = sum_i c_{e,i} w_i for every old variable e
        (checked), so that the weight of every exponent is preserved and the truncation
        max_weight carries over unchanged.  ``max_degrees``: optional box bound for the result.
        """
        new_variables = tuple(new_variables)
        new_weights = tuple(int(w) for w in new_weights)
        if len(new_weights) != len(new_variables):
            raise ValueError("need one weight per new variable")
        if isinstance(class_map, Mapping):
            idx = {v: i for i, v in enumerate(new_variables)}
            rows = []
            for v in self.variables:
                m = class_map.get(v, {})
                row = [0] * len(new_variables)
                for nv, c in m.items():
                    if nv not in idx:
                        raise ValueError(f"unknown new variable {nv!r}")
                    row[idx[nv]] = int(c)
                rows.append(row)
        else:
            rows = [[int(c) for c in r] for r in class_map]
            if len(rows) != len(self.variables) or any(len(r) != len(new_variables) for r in rows):
                raise ValueError("substitution matrix has the wrong shape")
        for e, (v, row) in enumerate(zip(self.variables, rows)):
            we = sum(c * w for c, w in zip(row, new_weights))
            if we != self.weights[e]:
                raise ValueError(f"weights inconsistent for {v}: w = {self.weights[e]} but "
                                 f"sum_i c_i w_i = {we}")
        out: dict[Exp, QFrac] = {}
        for d, c in self.coefficients.items():
            nd = tuple(sum(d[e] * rows[e][i] for e in range(len(d))) for i in range(len(new_variables)))
            out[nd] = out[nd] + c if nd in out else c
        return Series(new_variables, out, new_weights, self.max_weight, max_degrees)

    # ------------------------------------------------------------------ log / exp
    def _graded_keys(self) -> list[Exp]:
        zero = (0,) * len(self.variables)
        keys = [d for d in self.coefficients if d != zero]
        for d in keys:
            if self.weight(d) <= 0:
                raise ValueError(f"exponent {d} has non-positive weight; log/exp need a positive grading")
        return sorted(keys, key=lambda d: (self.weight(d), d))

    def log(self) -> "Series":
        """F = log Z for Z with constant term 1, via the Euler-operator recursion
        |d| F_d = |d| Z_d - sum_{d'+d''=d; d',d''!=0} |d'| F_{d'} Z_{d''} (|d| = weight)."""
        zero = (0,) * len(self.variables)
        if self.constant_term() != 1:
            raise ValueError("log needs constant term 1")
        Z = self.coefficients
        zkeys = self._graded_keys()
        F: dict[Exp, QFrac] = {}
        # all exponents d that can receive a contribution: sums of Z-exponents within range
        targets = set(zkeys)
        frontier = list(zkeys)
        while frontier:
            new = []
            for a in frontier:
                for b in zkeys:
                    d = tuple(x + y for x, y in zip(a, b))
                    if d not in targets and self.in_range(d) and d != zero:
                        targets.add(d)
                        new.append(d)
            frontier = new
        order = sorted(targets, key=lambda d: (self.weight(d), d))
        for d in order:
            wd = self.weight(d)
            acc = QFrac.zero()
            for d1, f1 in F.items():
                d2 = tuple(x - y for x, y in zip(d, d1))
                z2 = Z.get(d2)
                if z2 is not None and d2 != zero:
                    acc = acc + f1 * z2 * self.weight(d1)
            val = Z.get(d, QFrac.zero()) - acc * Fraction(1, wd)
            val = val.cancel()
            if not val.is_zero():
                F[d] = val
        return self._like(F)

    def exp(self) -> "Series":
        """Z = exp F for F with zero constant term: |d| Z_d = sum_{d'+d''=d, d'!=0} |d'| F_{d'} Z_{d''}."""
        zero = (0,) * len(self.variables)
        if not self.constant_term().is_zero():
            raise ValueError("exp needs zero constant term")
        F = self.coefficients
        fkeys = self._graded_keys()
        targets = set(fkeys)
        frontier = list(fkeys)
        while frontier:
            new = []
            for a in frontier:
                for b in fkeys:
                    d = tuple(x + y for x, y in zip(a, b))
                    if d not in targets and self.in_range(d):
                        targets.add(d)
                        new.append(d)
            frontier = new
        Z: dict[Exp, QFrac] = {zero: QFrac.one()}
        for d in sorted(targets, key=lambda d: (self.weight(d), d)):
            acc = QFrac.zero()
            for d1, f1 in F.items():
                d2 = tuple(x - y for x, y in zip(d, d1))
                z2 = Z.get(d2)
                if z2 is not None:
                    acc = acc + f1 * z2 * self.weight(d1)
            val = (acc * Fraction(1, self.weight(d))).cancel()
            if not val.is_zero():
                Z[d] = val
        return self._like(Z)

    # ------------------------------------------------------------------ output
    def _monomial_str(self, d: Exp, pow_op: str = "^", mul: str = "*") -> str:
        parts = []
        for v, e in zip(self.variables, d):
            if e == 1:
                parts.append(v)
            elif e:
                parts.append(f"{v}{pow_op}{e}" if e > 0 else f"{v}{pow_op}({e})")
        return mul.join(parts)

    def to_sympy(self, q=None, symbols=None):
        """sympy expression sum_d S_d(q) prod_i t_i^{d_i} (coefficients via QFrac.to_sympy; symbols
        named after the variables unless ``symbols`` is given).  No O(...) term is added."""
        import sympy as sp
        if q is None:
            q = sp.Symbol("q", positive=True)
        if symbols is None:
            symbols = sp.symbols(self.variables) if self.variables else ()
            if len(self.variables) == 1:
                symbols = (symbols[0],) if isinstance(symbols, (tuple, list)) else (symbols,)
        expr = sp.Integer(0)
        for d, c in self.terms():
            mono = sp.Integer(1)
            for s, e in zip(symbols, d):
                mono *= s ** e
            expr += c.to_sympy(q) * mono
        return expr

    def to_mathematica(self) -> str:
        """Mathematica input string (variables as named, q for the vertex variable)."""
        out = []
        for d, c in self.terms():
            m = self._monomial_str(d, "^", "*")
            out.append(c.to_mathematica() + (f"*{m}" if m else ""))
        return " + ".join(out) if out else "0"

    def latex(self, max_terms: int | None = None) -> str:
        """LaTeX string; coefficients via QFrac.latex ([k] = q^{k/2} - q^{-k/2})."""
        out = []
        terms = self.terms()
        for d, c in terms[: max_terms if max_terms is not None else len(terms)]:
            mono = " ".join((f"{v}^{{{e}}}" if e != 1 else v) for v, e in zip(self.variables, d) if e)
            out.append(f"\\left({c.latex()}\\right) {mono}".strip())
        s = " + ".join(out) if out else "0"
        if max_terms is not None and len(terms) > max_terms:
            s += r" + \dots"
        return s

    def __str__(self) -> str:
        lines = []
        for d, c in self.terms():
            m = self._monomial_str(d)
            lines.append(f"  [{m or '1'}]  {c}")
        trunc = f"weight <= {self.max_weight}" + (f", degrees <= {self.max_degrees}" if self.max_degrees else "")
        head = f"Series in {', '.join(self.variables)} (weights {self.weights}; {trunc}), {len(self)} terms"
        return "\n".join([head] + lines)

    def __repr__(self) -> str:
        return f"<Series vars={self.variables} weights={self.weights} max_weight={self.max_weight} terms={len(self)}>"
