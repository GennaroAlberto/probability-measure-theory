"""Checks of the Chapter 3 code against the notes and the exercises."""
import math
from fractions import Fraction

import numpy as np
import pytest

from probnotes.measurable import (
    atoms,
    binary_value,
    canonical_form,
    common_refinement,
    complement,
    contains,
    cantor_function,
    cantor_limit_measure,
    cantor_point,
    cantor_stage,
    cantor_stage_measure,
    dyadic_approximation,
    dyadic_index,
    dyadic_levels,
    in_cantor_stage,
    indicator,
    intersection,
    is_measurable,
    is_sigma_algebra,
    lebesgue_measure,
    monte_carlo_measure,
    normalize_intervals,
    power_set,
    preimage,
    sigma_generated,
    sigma_of_function,
    union,
)

# ---------------------------------------------------------------- Lebesgue measure


def test_length_of_intervals_step_1():
    assert lebesgue_measure([(0, 1)]) == 1.0                  # m([0,1)) = 1
    assert lebesgue_measure([(0.25, 0.75)]) == 0.5            # m([a,b)) = b - a
    assert lebesgue_measure([(0.3, 0.3)]) == 0.0              # [a,a) is empty
    assert lebesgue_measure([]) == 0.0


def test_measure_does_not_depend_on_the_decomposition():
    # [0,1) split at random points in many ways: always measure 1 (Step 1).
    rng = np.random.default_rng(0)
    for _ in range(50):
        cuts = np.sort(np.concatenate([[0, 1], rng.random(rng.integers(1, 30))]))
        pieces = np.column_stack([cuts[:-1], cuts[1:]])
        assert lebesgue_measure(pieces) == pytest.approx(1.0, abs=1e-14)
        assert np.allclose(normalize_intervals(pieces), [[0, 1]])


def test_finite_additivity_and_inclusion_exclusion():
    rng = np.random.default_rng(1)
    for _ in range(100):
        A = np.sort(rng.random((5, 2)), axis=1)
        B = np.sort(rng.random((7, 2)), axis=1)
        mA, mB = lebesgue_measure(A), lebesgue_measure(B)
        assert lebesgue_measure(union(A, B)) + lebesgue_measure(intersection(A, B)) == pytest.approx(mA + mB)
        # complement inside [0, 1): m(A^c) = 1 - m(A)
        assert lebesgue_measure(complement(A)) == pytest.approx(1 - mA)
        # disjoint pieces A and B minus A
        assert lebesgue_measure(intersection(B, complement(A))) == pytest.approx(lebesgue_measure(union(A, B)) - mA)


def test_contains_uses_half_open_intervals():
    A = [(0, 0.5), (0.5, 1), (2, 3)]
    assert contains(A, [0, 0.5, 0.999, 1, 2.5, 3, -1]).tolist() == [True, True, True, False, True, False, False]


def test_step_3_limit_is_monotone():
    # A = union_k [k, k + 2^-k): m(A cap [-n, n)) increases to m(A) = 2 (Step 3).
    A = [(k, k + 2.0**-k) for k in range(60)]
    vals = [lebesgue_measure(intersection(A, [(-n, n)])) for n in range(1, 40)]
    assert np.all(np.diff(vals) >= 0)
    assert vals[-1] == pytest.approx(2.0, abs=1e-10)
    assert lebesgue_measure([(0, np.inf)]) == np.inf


def test_sin_reciprocal_set_measure():
    # {x in (0,1] : sin(1/x) > 0} = (1/pi, 1] cup U_k (1/((2k+1)pi), 1/(2k pi)), of measure 1 - ln2/pi.
    exact = 1 - math.log(2) / math.pi
    assert exact == pytest.approx(0.7793644, abs=1e-7)
    K = 100_000
    k = np.arange(1, K + 1)
    A = np.concatenate([[[1 / math.pi, 1]], np.column_stack([1 / ((2 * k + 1) * math.pi), 1 / (2 * k * math.pi)])])
    trunc = lebesgue_measure(A)
    # the tail beyond K has measure (1/pi) sum_{k>K} 1/(2k(2k+1)) ~ 1/(4 pi K)
    assert exact - trunc == pytest.approx(1 / (4 * math.pi * K), rel=1e-3)


# ---------------------------------------------------------------- Cantor sets


@pytest.mark.parametrize("k", [0, 1, 2, 5, 12])
def test_cantor_stages(k):
    I = cantor_stage(k)
    assert I.shape == (2**k, 2)
    assert np.allclose(I[:, 1] - I[:, 0], 3.0**-k)
    assert lebesgue_measure(I) == pytest.approx((2 / 3) ** k, rel=1e-12)
    assert cantor_stage_measure(k) == pytest.approx((2 / 3) ** k)


def test_cantor_stages_are_nested_and_negligible_in_the_limit():
    for k in range(10):
        inner, outer = cantor_stage(k + 1), cantor_stage(k)
        assert lebesgue_measure(intersection(inner, outer)) == pytest.approx(lebesgue_measure(inner))
    # m(C) <= m(C_k) = (2/3)^k -> 0; e.g. below 1e-6 from k = 35 on
    assert (2 / 3) ** 35 < 1e-6 < (2 / 3) ** 34


def test_points_of_the_cantor_set():
    x = np.array([0, 1 / 3, 2 / 3, 1, 1 / 4, 3 / 4, 1 / 9, 8 / 9, 1 / 2, 0.4])
    assert in_cantor_stage(x, 20).tolist() == [True] * 8 + [False, False]


def test_cantor_coding_is_onto_the_binary_numbers():
    # c(sum 2 b_j 3^-j) = sum b_j 2^-j: the Cantor set maps onto [0, 1], so it is uncountable.
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=(1000, 30))
    pts = cantor_point(bits)
    assert np.all(in_cantor_stage(pts, 18))
    assert np.allclose(cantor_function(pts), binary_value(bits), atol=1e-8)
    assert cantor_function(np.array([1 / 3, 1 / 4, 2 / 3, 1.0])) == pytest.approx([0.5, 1 / 3, 0.5, 1.0])
    grid = np.linspace(0, 1, 2001)
    assert np.all(np.diff(cantor_function(grid)) >= 0)


def test_fat_cantor_sets():
    assert cantor_limit_measure(1 / 3) == 0.0
    assert cantor_limit_measure(1 / 4) == pytest.approx(0.5)
    for r in (1 / 4, 1 / 5, 0.3):
        for k in (1, 3, 8):
            assert lebesgue_measure(cantor_stage(k, r)) == pytest.approx(cantor_stage_measure(k, r), rel=1e-12)
        assert cantor_stage_measure(60, r) == pytest.approx(cantor_limit_measure(r), abs=1e-12)
    with pytest.raises(ValueError):
        cantor_stage(3, 0.4)


# ---------------------------------------------------------------- dyadic approximation


def notes_definition(fx, n):
    """f_n literally as in the notes: (i-1)/2^n on {(i-1)/2^n <= f < i/2^n}, n on {f >= n}."""
    out = np.empty_like(fx, dtype=float)
    for j, v in enumerate(fx):
        if v >= n:
            out[j] = n
        else:
            for i in range(1, n * 2**n + 1):
                if (i - 1) / 2**n <= v < i / 2**n:
                    out[j] = (i - 1) / 2**n
                    break
    return out


def test_closed_form_matches_the_notes_definition():
    rng = np.random.default_rng(3)
    fx = np.concatenate([rng.exponential(1.5, 400), [0, 0.5, 1, 2, 3, 2.999, np.inf], dyadic_levels(4)])
    for n in (1, 2, 3, 4, 5):
        assert np.array_equal(dyadic_approximation(fx, n), notes_definition(fx, n))


def test_levels_of_f1_and_f2_as_in_the_figure():
    assert dyadic_levels(1).tolist() == [0, 0.5, 1]
    assert dyadic_levels(2).tolist() == [k / 4 for k in range(9)]       # steps of 1/4 up to 2
    for n in range(1, 8):
        assert len(dyadic_levels(n)) == n * 2**n + 1                   # n 2^n intervals, plus {f >= n}
    fx = np.linspace(0, 5, 10001)
    for n in (1, 2, 3):
        assert set(np.unique(dyadic_approximation(fx, n))) <= set(dyadic_levels(n))


def test_monotone_and_convergent():
    rng = np.random.default_rng(4)
    fx = np.concatenate([rng.exponential(3.0, 2000), [np.inf, 0.0]])
    prev = np.zeros_like(fx)
    for n in range(1, 25):
        fn = dyadic_approximation(fx, n)
        assert np.all(fn >= prev) and np.all(fn <= fx) and np.all(fn >= 0)
        below = fx < n
        assert np.all(fx[below] - fn[below] < 2.0**-n)
        assert np.all(fn[~below] == n)
        prev = fn
    assert dyadic_approximation([np.inf], 30)[0] == 30


def test_uniform_convergence_for_bounded_f():
    x = np.linspace(0, 1, 100_001)
    f = 12 * x * (1 - x)                       # bounded by M = 3
    errs = [np.max(f - dyadic_approximation(f, n)) for n in range(1, 12)]
    assert errs[0] == pytest.approx(2.0) and errs[1] == pytest.approx(1.0)   # sup f - n while n < 3
    for n in range(3, 12):
        assert errs[n - 1] <= 2.0**-n
        assert errs[n - 1] > 0.9 * 2.0**-n     # and the bound 2^-n is essentially attained


def l1_error_log(n):
    """Exact integral over (0,1) of f - f_n for f(x) = -log x (exercise on the convergence rate)."""
    h = 2.0**-n
    return math.exp(-n) + (1 - math.exp(-n)) * (1 - math.exp(-h) * (1 + h)) / (1 - math.exp(-h))


def test_l1_error_for_minus_log():
    for n in (1, 2, 4, 6):
        # midpoint rule in y = -log x, dx = e^-y dy, on a grid much finer than 2^-n
        dy = 2.0**-n / 64
        y = np.arange(0, 45, dy) + dy / 2
        num = np.sum((y - dyadic_approximation(y, n)) * np.exp(-y)) * dy
        assert num == pytest.approx(l1_error_log(n), rel=1e-4)
    assert l1_error_log(1) == pytest.approx(0.51279, abs=1e-5)
    assert l1_error_log(4) == pytest.approx(0.048674, abs=1e-6)
    assert l1_error_log(10) == pytest.approx(math.exp(-10) + 2.0**-11, rel=2e-3)


def test_canonical_form_and_common_partition():
    rng = np.random.default_rng(5)
    f = rng.choice([0.0, 1.5, np.inf], size=200)
    g = rng.choice([-1.0, 2.0], size=200)
    a, lab = canonical_form(f)
    assert np.array_equal(a[lab], f)                        # f = sum a_i 1_{A_i}
    b, labg = canonical_form(g)
    c = common_refinement(lab, labg)
    assert c.max() + 1 <= len(a) * len(b)
    for k in range(c.max() + 1):                            # f and g constant on each A_i cap B_j
        assert len(np.unique(f[c == k])) == 1 and len(np.unique(g[c == k])) == 1


# ---------------------------------------------------------------- finite spaces


def test_sigma_of_a_function_on_a_die():
    X = range(6)                                            # faces 1..6 stored as 0..5
    parity = [(x + 1) % 2 for x in X]
    s = sigma_of_function(parity)
    assert s == {frozenset(), frozenset({0, 2, 4}), frozenset({1, 3, 5}), frozenset(X)}
    f = [0, 1, 0, 2, 1, 2]
    sf = sigma_of_function(f)
    assert len(sf) == 2**3 and is_sigma_algebra(sf, 6)
    assert sf == sigma_generated([preimage(f, {v}) for v in (0, 1, 2)], 6)


def test_preimage_criterion_on_generators():
    # f: X -> Y = {0,1,2,3} with target sigma-algebra generated by G = {{0,1},{1,2}}.
    f = np.array([0, 3, 1, 1, 2, 0])
    G = [{0, 1}, {1, 2}]
    E = sigma_generated(G, 4)
    for A in ({frozenset(S) for S in power_set(range(6))}, sigma_generated([{0, 5}, {2, 3}, {4}], 6),
              sigma_generated([{0, 1, 5}], 6)):
        assert is_measurable(f, A, G) == is_measurable(f, A, E)


def test_indicator_measurable_iff_set_measurable():
    A1 = sigma_generated([{0}], 3)                          # {X, empty, {0}, {1,2}}
    assert A1 == {frozenset(), frozenset({0}), frozenset({1, 2}), frozenset({0, 1, 2})}
    borel_values = power_set([0, 1])
    for S in power_set(range(3)):
        assert is_measurable(indicator(S, 3), A1, borel_values) == (S in A1)


def test_atoms():
    assert sorted(map(sorted, atoms([{0, 1}, {1, 2}], 4))) == [[0], [1], [2], [3]]
    assert len(sigma_generated([{0, 1}, {2, 3}], 4)) == 4    # disjoint A, B covering X: 4 sets


def test_reciprocal_preimages():
    # phi(x) = 1/x, phi(0) = 0: {phi <= t} is [1/t, 0) for t < 0, (-inf, 0] for t = 0,
    # (-inf, 0] cup [1/t, inf) for t > 0 (worked exercise in the notes).
    x = np.concatenate([np.linspace(-5, 5, 20001), [0.0]])
    with np.errstate(divide="ignore"):
        phi = np.where(x != 0, 1 / np.where(x != 0, x, 1), 0.0)
    for t in (-2.0, -0.5, 0.0, 0.5, 3.0):
        lhs = phi <= t
        if t < 0:
            rhs = (x >= 1 / t) & (x < 0)
        elif t == 0:
            rhs = x <= 0
        else:
            rhs = (x <= 0) | (x >= 1 / t)
        assert np.array_equal(lhs, rhs)


# ---------------------------------------------------------------- Monte Carlo


def test_monte_carlo_measure():
    exact = 1 - math.log(2) / math.pi
    res = monte_carlo_measure(lambda u: np.sin(1 / u) > 0, 0.0, 1.0, 400_000, seed=2026)
    assert abs(res.estimate - exact) < 4 * res.std_error
    assert res.ci[0] < res.estimate < res.ci[1]
    assert res.hoeffding_halfwidth == pytest.approx(math.sqrt(math.log(40) / (2 * 400_000)))
    disk = monte_carlo_measure(lambda p: (p ** 2).sum(axis=1) < 1, [0, 0], [1, 1], 200_000, seed=7, batch=50_000)
    assert abs(disk.estimate - math.pi / 4) < 4 * disk.std_error


def test_monte_carlo_cannot_see_the_rationals():
    # Every float is a dyadic rational, so "x in Q" is true for every sample, although m(Q) = 0.
    u = np.random.default_rng(0).random(1000)
    assert all(Fraction(float(v)).denominator & (Fraction(float(v)).denominator - 1) == 0 for v in u)
