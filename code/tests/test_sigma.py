"""Checks of the Chapter 1 code against the examples worked out in the notes."""
from fractions import Fraction as Fr
from itertools import combinations

import numpy as np
import pytest

from probnotes.sigma import (
    atoms, atoms_from_matrix, bell, borel_generator_traces, check_axioms, closure_steps,
    count_algebras_bruteforce, count_sigma_algebras, generate, interval_complement, interval_union,
    is_algebra, is_sigma_algebra, membership_matrix, set_partitions, sigma_algebras, unions_of,
)

X3 = [1, 2, 3]
f = frozenset


def P(X):
    """The power set, for comparisons."""
    X = list(X)
    return {f(c) for r in range(len(X) + 1) for c in combinations(X, r)}


# ---------------------------------------------------------------------------- the notes' examples
def test_notes_example_three_points():
    # {X, {1}, {2}, {3}} is not an algebra; {X, emptyset, {1}, {2,3}} is.
    bad = check_axioms([X3, {1}, {2}, {3}], X3)
    assert bad["contains_X"] and bad["complements"] == f({1}) and bad["unions"] is not None
    assert not is_algebra([X3, {1}, {2}, {3}], X3)
    assert is_algebra([X3, set(), {1}, {2, 3}], X3)
    assert is_sigma_algebra([X3, set(), {1}, {2, 3}], X3)


def test_empty_family_is_not_an_algebra():
    r = check_axioms([], X3)
    assert not r["nonempty"] and not is_algebra([], X3)


def test_notable_sigma_algebras():
    assert is_sigma_algebra([set(), X3], X3)
    assert is_sigma_algebra(P(X3), X3)
    assert generate([], X3) == {f(), f(X3)}


def test_union_of_two_sigma_algebras_is_not_an_algebra():
    A1 = [X3, set(), {1}, {2, 3}]
    A2 = [X3, set(), {1, 2}, {3}]
    assert is_sigma_algebra(A1, X3) and is_sigma_algebra(A2, X3)
    union = {f(s) for s in A1 + A2}
    r = check_axioms(union, X3)
    assert r["unions"] is not None and f({1, 3}) not in union      # {1} u {3} is missing
    inter = {f(s) for s in A1} & {f(s) for s in A2}
    assert inter == {f(), f(X3)} and is_sigma_algebra(inter, X3)    # intersections are fine
    assert generate(union, X3) == P(X3)                              # the join A1 v A2


def test_generated_by_one_set_and_by_nothing():
    X = range(1, 6)
    assert generate([set()], X) == {f(), f(X)} == generate([X], X)   # sigma(emptyset) = sigma(X)
    A = {1, 4}
    assert generate([A], X) == {f(), f(X), f(A), f(X) - f(A)}


@pytest.mark.parametrize("A, B, size", [
    ({1, 2}, {2, 3}, 16),          # the four atoms are non-empty: 16 sets
    ({1, 2}, {3}, 8),              # A and B disjoint, (A u B)^c = {4} non-empty: 8 sets
    ({1, 2}, {3, 4}, 4),           # disjoint and covering X: only 4
    ({1, 2}, {1, 2, 3}, 8),        # nested: B \ A, A, (A u B)^c
    (set(), set(), 2),
])
def test_two_sets(A, B, size):
    X = [1, 2, 3, 4]
    S = generate([A, B], X)
    assert len(S) == size == 2 ** len(atoms([A, B], X))
    assert {f(A), f(B), f(X) - f(A), f(X) - f(B), f(A) | f(B), f(X) - (f(A) | f(B))} <= S


# ---------------------------------------------------------------------------- the algorithms
def test_atoms_partition_X_and_closure_agrees_with_atoms():
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(1, 9))
        m = int(rng.integers(0, 5))
        X = list(range(n))
        fam = [set(np.flatnonzero(rng.random(n) < 0.5)) for _ in range(m)]
        at = atoms(fam, X)
        assert sorted(x for a in at for x in a) == X                  # a partition of X
        S = generate(fam, X, "atoms")
        assert S == generate(fam, X, "closure")
        assert len(S) == 2 ** len(at) and is_sigma_algebra(S, X)
        assert all(f(E) in S for E in fam)


def test_closure_steps_grow_to_the_generated_sigma_algebra():
    steps = closure_steps([{1, 2}, {2, 3}], range(1, 7))
    assert [len(s) for s in steps] == [3, 7, 11, 14, 16]
    assert all(a <= b for a, b in zip(steps, steps[1:]))
    assert steps[-1] == generate([{1, 2}, {2, 3}], range(1, 7))


def test_atoms_of_a_matrix_and_membership_matrix():
    M = membership_matrix([{"a", "b"}, {"b", "c"}], ["a", "b", "c", "d"])
    assert M.tolist() == [[True, True, False, False], [False, True, True, False]]
    assert atoms_from_matrix(M, "abcd") == [f("a"), f("b"), f("c"), f("d")]
    assert atoms_from_matrix(np.zeros((0, 3), bool)) == [f({0, 1, 2})]


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        atoms([{4}], X3)
    with pytest.raises(ValueError):
        generate([{1}], X3, method="magic")


def test_same_generated_sigma_algebra():
    # E is contained in sigma(E') and E' in sigma(E): the two generate the same sigma-algebra.
    X = range(1, 7)
    E = [{1, 2}, {3, 4}, {5, 6}]
    Ep = [{1, 2, 3, 4}, {3, 4, 5, 6}, {1, 2, 5, 6}]
    assert all(f(e) in generate(Ep, X) for e in E) and all(f(e) in generate(E, X) for e in Ep)
    assert generate(E, X) == generate(Ep, X) and len(generate(E, X)) == 8


# ---------------------------------------------------------------------------- counting
def test_bell_numbers():
    assert [bell(n) for n in range(11)] == [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
    assert count_sigma_algebras(5) == 52


def test_brute_force_count_matches_bell():
    assert [count_algebras_bruteforce(n) for n in range(5)] == [1, 1, 2, 5, 15]


def test_partitions_give_all_sigma_algebras():
    for n in range(1, 7):
        X = list(range(n))
        parts = list(set_partitions(X))
        assert len(parts) == bell(n)
        algs = list(sigma_algebras(X))
        assert len(set(algs)) == bell(n)
        if n <= 4:
            assert all(is_sigma_algebra(S, X) for S in algs)


def test_unions_of_blocks():
    assert unions_of([{1}, {2, 3}]) == {f(), f({1}), f({2, 3}), f({1, 2, 3})}


# ---------------------------------------------------------------------------- Example: intervals [a, b)
def test_union_of_two_intervals_in_the_four_positions():
    # The four pictures on p. 5 of the notes: nested, disjoint, adjacent (b = c), overlapping.
    assert interval_union([(0.1, 0.9)], [(0.3, 0.6)]) == [(0.1, 0.9)]
    assert interval_union([(0.1, 0.3)], [(0.5, 0.8)]) == [(0.1, 0.3), (0.5, 0.8)]
    assert interval_union([(0.1, 0.5)], [(0.5, 0.8)]) == [(0.1, 0.8)]
    assert interval_union([(0.1, 0.6)], [(0.4, 0.8)]) == [(0.1, 0.8)]


def test_interval_complement_formula():
    A = [(Fr(1, 10), Fr(3, 10)), (Fr(1, 2), Fr(7, 10))]
    assert interval_complement(A) == [(0, Fr(1, 10)), (Fr(3, 10), Fr(1, 2)), (Fr(7, 10), 1)]
    assert interval_complement([]) == [(0, 1)]
    assert interval_complement([(0, 1)]) == []
    assert interval_union(A, interval_complement(A)) == [(0, 1)]


def test_increasing_intervals_never_reach_the_open_interval():
    # A_n = [1/n, 1): every finite union is [1/N, 1), which contains its left endpoint.
    for N in (1, 2, 5, 100):
        U = interval_union(*[[(Fr(1, n), 1)] for n in range(1, N + 1)])
        assert U == ([] if N == 1 else [(Fr(1, N), 1)])


# ---------------------------------------------------------------------------- Borel generators on a grid
def test_borel_traces_small_grid():
    grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ends = [0.0, 0.5, 1.0]
    assert borel_generator_traces(1, grid, ends).sum(axis=1).tolist() == [1, 3, 1]   # (0,.5),(0,1),(.5,1)
    assert borel_generator_traces(4, grid, ends).sum(axis=1).tolist() == [3, 5, 3]
    assert borel_generator_traces(6, grid, ends).sum(axis=1).tolist() == [1, 3, 5]
    assert borel_generator_traces(7, grid, ends).sum(axis=1).tolist() == [4, 2, 0]
    sizes = {i: len(atoms_from_matrix(borel_generator_traces(i, grid, ends))) for i in range(1, 9)}
    assert sizes == {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3}


def test_borel_traces_separate_points_when_endpoints_are_dense():
    # With all grid points (and one point beyond each end) as endpoints, every family
    # generates the power set of the grid.
    grid = np.arange(9) / 8
    ends = np.arange(-1, 10) / 8
    for i in range(1, 9):
        assert len(atoms_from_matrix(borel_generator_traces(i, grid, ends))) == 9


# ---------------------------------------------------------------------------- numbers quoted in the exercises
def test_closure_rounds_of_a_separating_family():
    E_b = [{1, 2, 3}, {3, 4, 5}, {5, 6, 1}]
    assert [len(s) for s in closure_steps(E_b, range(1, 7))] == [4, 11, 23, 38, 63, 64]
    assert len(atoms(E_b, range(1, 7))) == 6


def test_pairs_of_subsets_generating_the_power_set():
    # Ordered pairs (E1, E2) with sigma({E1, E2}) = P(X): the 4! bijections onto {0,1}^2 for 4 points,
    # none for 5 points (5 > 2^2).
    def count(n):
        subsets = sorted(P(range(n)), key=sorted)
        return sum(len(atoms([a, b], range(n))) == n for a in subsets for b in subsets), len(subsets) ** 2
    assert count(4) == (24, 256)
    assert count(5) == (0, 1024)


def test_sigma_algebras_by_number_of_atoms_are_stirling_numbers():
    table = {1: [1], 2: [1, 1], 3: [1, 3, 1], 4: [1, 7, 6, 1], 5: [1, 15, 25, 10, 1],
             6: [1, 31, 90, 65, 15, 1], 7: [1, 63, 301, 350, 140, 21, 1]}
    for n, row in table.items():
        counts = [0] * n
        for p in set_partitions(range(1, n + 1)):
            counts[len(p) - 1] += 1
        assert counts == row and sum(row) == bell(n)


def test_counts_of_sigma_algebras_containing_given_sets():
    for n in range(2, 9):
        parts = list(set_partitions(range(1, n + 1)))
        with_1 = sum(f({1}) in p for p in parts)
        with_12 = sum(all(b <= {1, 2} for b in p if b & {1, 2}) for p in parts)
        assert (with_1, with_12) == (bell(n - 1), 2 * bell(n - 2))


def test_random_families_birthday_formula():
    def exact(n, m):
        return float(np.prod([1 - k / 2**m for k in range(n)]))
    # exhaustive check of the formula for n = 3 points and m = 2 sets: 24 of the 64 families
    subsets = sorted(P(range(3)), key=sorted)
    hits = sum(len(atoms([a, b], range(3))) == 3 for a in subsets for b in subsets)
    assert hits / 64 == exact(3, 2) == 3 / 8
    assert round(exact(10, 6), 4) == 0.4768 and round(exact(10, 7), 4) == 0.6973
    smallest = {n: next(m for m in range(64) if exact(n, m) >= 0.5) for n in (10, 100, 1000)}
    assert smallest == {10: 7, 100: 13, 1000: 20}


def test_borel_generators_on_the_grid_of_the_exercise():
    grid = np.arange(13) / 12

    def counts(ends):
        return [len(atoms_from_matrix(borel_generator_traces(i, grid, ends))) for i in range(1, 9)]
    assert counts(np.arange(5) / 4) == [8, 5, 5, 7, 5, 5, 5, 5]
    assert counts(grid) == [12] + [13] * 7
    assert counts(np.arange(-1, 14) / 12) == [13] * 8
    # four groups of equal sigma-algebras: {1}, {4}, {2, 6, 7}, {3, 5, 8}
    ends = np.arange(5) / 4
    parts = {i: set(atoms_from_matrix(borel_generator_traces(i, grid, ends))) for i in range(1, 9)}
    assert parts[2] == parts[6] == parts[7] and parts[3] == parts[5] == parts[8]
    assert len({frozenset(v) for v in parts.values()}) == 4
