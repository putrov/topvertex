r"""Integer partitions, following the notation of Konishi (math/0504188, Sec. 2.2).

A partition is a tuple of positive ints in weakly decreasing order; the empty partition is ``()``.
Box coordinates (i, j) are 0-based: row i in range(len(lam)), column j in range(lam[i]).
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Iterator, Sequence

Partition = tuple[int, ...]


def is_partition(lam) -> bool:
    """True iff lam is a tuple of positive ints, weakly decreasing."""
    return (isinstance(lam, tuple) and all(isinstance(p, int) and p > 0 for p in lam)
            and all(lam[i] >= lam[i + 1] for i in range(len(lam) - 1)))


def size(lam: Partition) -> int:
    """|lam| = sum_i lam_i."""
    return sum(lam)


def length(lam: Partition) -> int:
    """l(lam) = number of nonzero parts."""
    return len(lam)


@lru_cache(maxsize=None)
def conjugate(lam: Partition) -> Partition:
    """Conjugate (transposed) partition: lam^t_j = #{i : lam_i >= j} (KP Sec. 2.2)."""
    if not lam:
        return ()
    return tuple(sum(1 for p in lam if p >= j) for j in range(1, lam[0] + 1))


@lru_cache(maxsize=None)
def kappa(lam: Partition) -> int:
    """kappa(lam) = sum_{i>=1} lam_i (lam_i - 2i + 1) = 2 * sum of contents (KP Sec. 2.2).
    Always even, kappa(lam^t) = -kappa(lam)."""
    return sum(p * (p - 2 * i + 1) for i, p in enumerate(lam, start=1))


def arm(lam: Partition, i: int, j: int) -> int:
    """Arm length of box (i, j) (0-based): number of boxes to its right, lam_i - j - 1."""
    return lam[i] - j - 1


def leg(lam: Partition, i: int, j: int) -> int:
    """Leg length of box (i, j) (0-based): number of boxes below it, lam^t_j - i - 1."""
    return conjugate(lam)[j] - i - 1


def hooks(lam: Partition) -> list[int]:
    """Hook lengths h(i,j) = arm + leg + 1 of all boxes, row by row."""
    lt = conjugate(lam)
    return [lam[i] - j + lt[j] - i - 1 for i in range(len(lam)) for j in range(lam[i])]


@lru_cache(maxsize=None)
def _partitions_bounded(d: int, maxpart: int) -> tuple[Partition, ...]:
    if d == 0:
        return ((),)
    out = []
    for first in range(min(d, maxpart), 0, -1):
        for rest in _partitions_bounded(d - first, first):
            out.append((first,) + rest)
    return tuple(out)


def partitions_of(d: int) -> list[Partition]:
    """All partitions of d in reverse lexicographic order: (d), (d-1,1), ..., (1,...,1)."""
    if d < 0:
        return []
    return list(_partitions_bounded(d, d))


def partitions_up_to(d: int) -> list[Partition]:
    """All partitions of size 0..d, ordered by size, then as in :func:`partitions_of`."""
    return [lam for n in range(d + 1) for lam in partitions_of(n)]


def contains(mu: Partition, nu: Partition) -> bool:
    """True iff nu is contained in mu as Young diagrams (nu_i <= mu_i for all i)."""
    return len(nu) <= len(mu) and all(a <= b for a, b in zip(nu, mu))


@lru_cache(maxsize=None)
def _subpartitions(mu: Partition) -> tuple[Partition, ...]:
    out: list[Partition] = []

    def rec(i: int, bound: int, cur: list[int]):
        out.append(tuple(cur))
        if i >= len(mu):
            return
        for p in range(1, min(mu[i], bound) + 1):
            cur.append(p)
            rec(i + 1, p, cur)
            cur.pop()

    rec(0, mu[0] if mu else 0, [])
    return tuple(sorted(out, key=lambda p: (sum(p), [-x for x in p])))


def subpartitions(mu: Partition) -> list[Partition]:
    """All partitions nu contained in mu (including () and mu), ordered by size."""
    return list(_subpartitions(mu))


def intersection(mu: Partition, nu: Partition) -> Partition:
    """Box-wise intersection mu cap nu: parts min(mu_i, nu_i)."""
    return tuple(min(a, b) for a, b in zip(mu, nu))


def multipartitions(weights: Sequence[int], max_weight: int) -> Iterator[tuple[Partition, ...]]:
    """Generate all tuples (lam_1, ..., lam_m) of partitions with sum_e weights[e] |lam_e| <=
    max_weight (weights positive ints).  Deterministic order: by size vector (lexicographic in
    sizes), then by the order of :func:`partitions_of` in each slot."""
    weights = list(weights)
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be positive")
    m = len(weights)

    def size_vectors(e: int, budget: int):
        if e == m:
            yield ()
            return
        for s in range(budget // weights[e] + 1):
            for rest in size_vectors(e + 1, budget - s * weights[e]):
                yield (s,) + rest

    for sizes in size_vectors(0, max_weight):
        yield from product(*(_partitions_bounded(s, s) for s in sizes))
