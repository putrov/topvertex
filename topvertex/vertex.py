r"""The topological vertex (three-point function) and framing factors.

Konishi, math/0504188, Definition 2.2 (the same function as the vertex of AKMV hep-th/0305132 sec. 6, in the symmetric form of Okounkov-Reshetikhin-Vafa, up to
the convention q <-> q^{-1}; see CONVENTIONS.md):

    C_{l1,l2,l3}(q) = q^{kappa(l3)/2} s_{l2}(q^rho)
                      sum_{eta} s_{l1/eta}(q^{l2^t+rho}) s_{l3^t/eta}(q^{l2+rho}),

eta running over partitions contained in l1 cap l3^t.  C is cyclically symmetric.

Special cases (direct consequences of the definition, checked in tests/test_vertex.py):
    C_{l1,l2,()} = s_{l2}(q^rho) s_{l1}(q^{l2^t+rho}),
    C_{(),l2,l3} = q^{kappa(l3)/2} s_{l2}(q^rho) s_{l3^t}(q^{l2+rho}),
    C_{l1,(),l3} = q^{kappa(l3)/2} sum_eta s_{l1/eta}(q^rho) s_{l3^t/eta}(q^rho),
    C_{l,(),()}  = s_l(q^rho).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .partitions import Partition, conjugate, intersection, kappa, size, subpartitions
from .qring import QFrac
from .schur import schur_rho, skew_schur_at

#: Type of a (possibly refined, in the future) vertex: (l1, l2, l3) -> QFrac, partitions read
#: counterclockwise.  The default implementation is :func:`vertex`; partition-function code
#: should accept any callable of this type.
Vertex = Callable[[Partition, Partition, Partition], QFrac]


@lru_cache(maxsize=None)
def vertex(l1: Partition, l2: Partition, l3: Partition) -> QFrac:
    """Three-point function C_{l1,l2,l3}(q) of Konishi Def. 2.2:

    q^{kappa(l3)/2} s_{l2}(q^rho) sum_{eta subset l1 cap l3^t} s_{l1/eta}(q^{l2^t+rho}) s_{l3^t/eta}(q^{l2+rho}).

    Returned in reduced form (:meth:`QFrac.cancel`)."""
    l2t, l3t = conjugate(l2), conjugate(l3)
    total = QFrac.zero()
    for eta in subpartitions(intersection(l1, l3t)):
        a = skew_schur_at(l1, eta, l2t)
        if a.is_zero():
            continue
        b = skew_schur_at(l3t, eta, l2)
        if b.is_zero():
            continue
        total = total + a * b
    return (QFrac.x_pow(kappa(l3)) * schur_rho(l2) * total).cancel()


#: The default vertex implementation.
default_vertex: Vertex = vertex


def framing_factor(n: int, lam: Partition) -> QFrac:
    """Framing factor of an internal edge (KP eq. 4): (-1)^{|lam|(n+1)} q^{n kappa(lam)/2}."""
    sign = -1 if (size(lam) * (n + 1)) % 2 else 1
    return QFrac.x_pow(n * kappa(lam), sign)
