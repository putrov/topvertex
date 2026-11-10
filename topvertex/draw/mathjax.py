r"""LaTeX for the rendered (MathJax 3) output of the drawing page.

One display line per coefficient of a series, ``[t_{1}^{2}]\,Z = \frac{N(q)}{[1]^{4}[2]^{2}}``,
with the denominator in q-numbers [k] = q^{k/2} - q^{-k/2} as in :meth:`topvertex.qring.QFrac.latex`.
Unlike that method (which goes through sympy and writes negative powers as nested fractions), the
numerator is written here as a flat Laurent polynomial in q with exponents q^{-2}, q^{3/2}, and a
numerator with more than ``chunk`` terms is broken into an ``aligned`` block of ``chunk`` terms per
line, ``\frac{1}{den}\Big( ... \\ ... \Big)``.  MathJax 3 has no automatic line breaking of
display math (that arrived in MathJax 4), so the breaks are made here.  Only environments that
MathJax 3 supports are used (``array``, ``aligned``, ``\frac``, ``\tfrac``).
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Sequence

from ..qring import QFrac
from ..series import Series

__all__ = ["var_latex", "monomial_latex", "laurent_terms", "coefficient_latex", "series_latex",
           "gv_latex", "edges_latex", "CHUNK"]

CHUNK = 6


def var_latex(name: str) -> str:
    """'t1' -> 't_{1}', 'Q12' -> 'Q_{12}', 'QB' -> 'QB'."""
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    return f"{m.group(1)}_{{{m.group(2)}}}" if m else name


def monomial_latex(variables: Sequence[str], d) -> str:
    """prod_i v_i^{d_i} in LaTeX; the constant monomial is written v_1^{0} v_2^{0} ... (not '1',
    which would read as the q-number [1] inside the coefficient bracket)."""
    if not any(d):
        return " ".join(var_latex(v) + "^{0}" for v in variables) or "1"
    return " ".join(var_latex(v) + (f"^{{{e}}}" if e != 1 else "") for v, e in zip(variables, d) if e)


def _qpow(n: int) -> str:
    """q^{n/2} for the x-exponent n (x = q^{1/2})."""
    if n == 0:
        return ""
    if n % 2:
        return f"q^{{{n}/2}}"
    return "q" if n == 2 else f"q^{{{n // 2}}}"


def laurent_terms(coeffs: dict[int, Fraction]) -> list[str]:
    """Signed terms ('+3q^{2}', '-\\tfrac{1}{2}q^{-1/2}', ...) of sum_n c_n x^n, by increasing n."""
    out = []
    for n, c in sorted(coeffs.items()):
        c = Fraction(c)
        sign, a = ("-" if c < 0 else "+"), abs(c)
        mono = _qpow(n)
        if a.denominator != 1:
            num = rf"\tfrac{{{a.numerator}}}{{{a.denominator}}}"
        elif a == 1 and mono:
            num = ""
        else:
            num = str(a.numerator)
        out.append(f"{sign}{num}{mono}")
    return out


def _join(terms: list[str], first: bool, aligned: bool = False) -> str:
    """Signed terms -> 'a + b - c'.  First line: drop a leading '+'; a leading '-' after an
    alignment point is written {-} (unary spacing).  Continuation lines start with {} so that the
    sign gets binary spacing."""
    s = " ".join(terms)
    if not first:
        return "{}" + s
    if s.startswith("+"):
        return s[1:]
    return "{-}" + s[1:] if aligned and s.startswith("-") else s


def coefficient_latex(label: str, c: QFrac, chunk: int = CHUNK) -> str:
    """'label = c' with c = x^off P(x) / prod [k]^m; numerators longer than ``chunk`` terms are
    broken into lines of ``chunk`` terms (``aligned``)."""
    terms = laurent_terms(c.numerator_laurent()) or ["0"]
    den = "".join(f"[{k}]" + (f"^{{{m}}}" if m > 1 else "") for k, m in sorted(c.denominator.items()))
    if len(terms) <= chunk:
        num = _join(terms, True)
        return f"{label} = " + (rf"\frac{{{num}}}{{{den}}}" if den else num)
    lines = [_join(terms[i:i + chunk], i == 0, True) for i in range(0, len(terms), chunk)]
    body = r" \\ &".join(lines)
    # the label is braced: '\begin{aligned} [' would be read as aligned's optional argument
    if den:
        return rf"\begin{{aligned}} {{{label}}} = \frac{{1}}{{{den}}}\Big(&{body}\Big) \end{{aligned}}"
    return rf"\begin{{aligned}} {{{label}}} = {{}}&{body} \end{{aligned}}"


def series_latex(S: Series, symbol: str = "Z", chunk: int = CHUNK) -> list[str]:
    r"""One LaTeX line per coefficient: ``[monomial]\,symbol = coefficient``."""
    return [coefficient_latex(rf"[{monomial_latex(S.variables, d)}]\,{symbol}", c, chunk) for d, c in S.terms()]


def gv_latex(gv: dict, variables: Sequence[str]) -> str:
    """The table of :func:`topvertex.partition_function.gv_table` as a LaTeX ``array``:
    header (classes in the variables, then g = 0..gmax), one row per class beta."""
    if not gv:
        return r"\text{(no nonzero GV invariants)}"
    gmax = max(g for d in gv.values() for g in d)
    head = "(" + ",".join(var_latex(v) for v in variables) + ")"
    rows = [" & ".join([head] + [f"g={g}" for g in range(gmax + 1)])]
    for beta in sorted(gv, key=lambda b: (sum(b), b)):
        cls = str(beta[0]) if len(beta) == 1 else "(" + ",".join(map(str, beta)) + ")"
        rows.append(" & ".join([cls] + [str(gv[beta].get(g, 0)) for g in range(gmax + 1)]))
    cols = "c|" + "r" * (gmax + 1)
    return rf"\begin{{array}}{{{cols}}} {rows[0]} \\ \hline " + r" \\ ".join(rows[1:]) + r" \end{array}"


def edges_latex(edges: list[dict]) -> str:
    """Array of the internal edges (rows of :func:`topvertex.cli.geometry_report`): name, segment,
    framing n_e (at the tail), truncation (Kahler) weight w_e, H_2 class (basis coordinates).  The
    header takes two lines (words, then symbols) to keep the columns narrow."""
    if not edges:
        return r"\text{(no internal edges)}"
    pt = lambda p: f"({p[0]},{p[1]})"
    rows = [r"\text{edge} & \text{segment} & \text{framing} & \text{trunc. weight} & H_2\ \text{class} \\ "
            r"e & & n_e & w_e & "]
    for e in edges:
        cls = "(" + ",".join(map(str, e["class"])) + ")" if e["class"] is not None else "-"
        a, b = e["segment"]
        rows.append(f"{var_latex(e['name'])} & {pt(a)}\\text{{–}}{pt(b)} & {e['n']} & {e['weight']} & {cls}")
    return r"\begin{array}{lcccc} " + rows[0] + r" \\ \hline " + r" \\ ".join(rows[1:]) + r" \end{array}"
