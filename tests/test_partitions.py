"""Tests for topvertex.partitions."""
from topvertex.partitions import (arm, conjugate, contains, hooks, intersection, is_partition,
                                  kappa, leg, length, multipartitions, partitions_of,
                                  partitions_up_to, size, subpartitions)

P_COUNTS = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]


def test_counts():
    for d, c in enumerate(P_COUNTS):
        ps = partitions_of(d)
        assert len(ps) == c
        assert len(set(ps)) == c
        assert all(is_partition(p) and size(p) == d for p in ps)
    assert len(partitions_up_to(10)) == sum(P_COUNTS)
    assert partitions_of(3) == [(3,), (2, 1), (1, 1, 1)]
    assert partitions_of(0) == [()]


def test_is_partition():
    assert is_partition(()) and is_partition((3, 3, 1))
    assert not is_partition((1, 2)) and not is_partition((2, 0)) and not is_partition([1])


def test_conjugate():
    assert conjugate(()) == ()
    assert conjugate((3, 1)) == (2, 1, 1)
    for lam in partitions_up_to(9):
        assert conjugate(conjugate(lam)) == lam
        assert size(conjugate(lam)) == size(lam)
        if lam:
            assert length(conjugate(lam)) == lam[0]


def test_kappa():
    assert kappa(()) == 0
    assert kappa((1,)) == 0
    assert kappa((2,)) == 2
    assert kappa((1, 1)) == -2
    assert kappa((3,)) == 6
    assert kappa((2, 1)) == 0
    for lam in partitions_up_to(9):
        assert kappa(conjugate(lam)) == -kappa(lam)
        assert kappa(lam) % 2 == 0
        contents = sum(j - i for i in range(len(lam)) for j in range(lam[i]))
        assert kappa(lam) == 2 * contents


def test_hooks_arm_leg():
    assert sorted(hooks((2, 1))) == [1, 1, 3]
    assert hooks((3,)) == [3, 2, 1]
    assert arm((3, 1), 0, 0) == 2 and leg((3, 1), 0, 0) == 1
    from math import factorial, prod
    for d in range(1, 8):
        # sum over lam of (d!/prod hooks)^2 = d!
        assert sum((factorial(d) // prod(hooks(lam))) ** 2 for lam in partitions_of(d)) == factorial(d)


def test_containment():
    assert contains((3, 2), (2, 2)) and not contains((3, 2), (1, 1, 1))
    assert contains((1,), ()) and contains((), ())
    assert intersection((3, 1), (2, 2)) == (2, 1)
    assert intersection((3,), (1, 1)) == (1,)
    subs = subpartitions((2, 1))
    assert set(subs) == {(), (1,), (2,), (1, 1), (2, 1)}
    for mu in partitions_up_to(6):
        subs = subpartitions(mu)
        assert len(set(subs)) == len(subs)
        assert set(subs) == {nu for nu in partitions_up_to(size(mu)) if contains(mu, nu)}


def test_multipartitions():
    assert list(multipartitions([1], 3)) == [(lam,) for lam in partitions_up_to(3)]
    # two slots of weight 1: number of pairs with |a|+|b| <= d
    for d in range(6):
        expect = sum(P_COUNTS[i] * P_COUNTS[j] for i in range(d + 1) for j in range(d + 1 - i))
        assert len(list(multipartitions([1, 1], d))) == expect
    # weights (1, 2), budget 4: sizes (s1, s2) with s1 + 2 s2 <= 4
    expect = sum(P_COUNTS[s1] * P_COUNTS[s2] for s1 in range(5) for s2 in range(3) if s1 + 2 * s2 <= 4)
    mps = list(multipartitions([1, 2], 4))
    assert len(mps) == expect and len(set(mps)) == expect
    assert all(size(a) + 2 * size(b) <= 4 for a, b in mps)
    assert list(multipartitions([], 3)) == [()]
