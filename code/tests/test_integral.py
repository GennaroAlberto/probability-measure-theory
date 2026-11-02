"""Checks of the Chapter 4 code against the notes and the exercises."""
import math

import numpy as np
import pytest

from probnotes.integral import (
    EXAMPLES,
    INF,
    SEQUENCES,
    StepFunction,
    alternating,
    bump_plus_fixed,
    combine,
    dyadic_as_indicator_sum,
    dyadic_integral,
    dyadic_integral_finite,
    dyadic_integrals,
    fatou_sides,
    increasing_plateau,
    indicator_step,
    integral_finite,
    integral_series,
    lebesgue_simple_integral,
    mul,
    periodic_fatou,
    simple_integral,
    simple_integral_on,
    sliding_bump,
    spike,
    spreading,
    step_function,
    tail_from_values,
    tail_infimum,
)
from probnotes.measurable import dyadic_approximation
from probnotes.measures import DiscreteMeasure, FinCofin, counting_measure, dirac

# ------------------------------------------------------- arithmetic and simple integrals


def test_zero_times_infinity_is_zero():
    assert mul(0, INF) == 0.0 and mul(INF, 0) == 0.0
    assert mul(2, INF) == INF and mul(INF, INF) == INF
    assert np.array_equal(mul([0, 1, INF], [INF, 3, 0]), [0, 3, 0])
    with pytest.raises(ValueError):
        mul(-1, 2)


def test_simple_integral_definition():
    # sum a_i mu(A_i) with 0 * inf = 0: a null set carries nothing even where f = +inf,
    # and the value 0 carries nothing even on a set of infinite measure.
    assert simple_integral([1.0, 2.0, 0.5], [2.0, 1.0, 4.0]) == 6.0
    assert simple_integral([INF, 3.0], [0.0, 1.0]) == 3.0
    assert simple_integral([0.0, 3.0], [INF, 1.0]) == 3.0
    assert simple_integral([INF], [0.5]) == INF          # (+inf) * mu(E) = inf iff mu(E) > 0


def random_simple(rng, n_points, n_values, allow_inf=True):
    values = rng.choice([0.0, 0.5, 1.0, 2.0, 3.5] + ([INF] if allow_inf else []), size=n_values)
    labels = rng.integers(0, n_values, size=n_points)
    return values[labels]


def test_independent_of_the_representation_finite_space():
    # Proposition: refine a partition at random; the integral does not change.
    rng = np.random.default_rng(0)
    for _ in range(200):
        N = int(rng.integers(1, 12))
        w = rng.choice([0.0, 0.25, 1.0, 2.0, INF], size=N)
        f = random_simple(rng, N, 4)
        canonical = np.unique(f)
        direct = simple_integral(canonical, [w[f == a].sum() for a in canonical])
        # a finer partition: split each level set into random pieces
        pieces = rng.integers(0, 3, size=N)
        coeffs, meas = [], []
        for a in canonical:
            for p in range(3):
                mask = (f == a) & (pieces == p)
                coeffs.append(a)
                meas.append(w[mask].sum())
        assert simple_integral(coeffs, meas) == direct == integral_finite(f, w)


def test_properties_of_the_simple_integral():
    # Properties (1)-(5) of the notes on random simple functions with values in [0, +inf].
    rng = np.random.default_rng(1)
    for _ in range(300):
        N = int(rng.integers(1, 10))
        w = rng.choice([0.0, 0.5, 1.0, 3.0, INF], size=N)
        f, g = random_simple(rng, N, 3), random_simple(rng, N, 3)
        A = rng.random(N) < 0.5
        E = rng.random(N) < 0.5
        F = E | (rng.random(N) < 0.5)
        assert integral_finite(A.astype(float), w) == w[A].sum()                     # (1)
        for alpha in (0.0, 0.5, 2.0, INF):                                           # (2)
            assert integral_finite(mul(alpha, f), w) == mul(alpha, integral_finite(f, w))
        assert integral_finite(f + g, w) == integral_finite(f, w) + integral_finite(g, w)  # (3)
        lo, hi = np.minimum(f, g), np.maximum(f, g)
        assert integral_finite(lo, w) <= integral_finite(hi, w)                      # (4)
        assert integral_finite(f, w, E) <= integral_finite(f, w, F)                  # (5)
        assert integral_finite(f, w, E) == integral_finite(mul(f, E), w)             # int_E f = int f 1_E


def test_simple_integral_on_a_countable_space():
    mu = DiscreteMeasure(lambda n: 2.0**-n, total=2.0)        # mu({n}) = 2^-n on N
    head, tail = FinCofin({0, 1, 2}), FinCofin({0, 1, 2}, cofinite=True)
    assert simple_integral_on([5.0, 1.0], [head, tail], mu) == pytest.approx(9.0)  # 5 * 7/4 + 1 * 1/4
    assert simple_integral_on([5.0, 1.0], [head, tail], mu, E=FinCofin({1, 7})) == pytest.approx(2.5 + 2**-7)
    nu = counting_measure()
    assert simple_integral_on([3.0, 0.0], [FinCofin({0, 1}), FinCofin({0, 1}, cofinite=True)], nu) == 6.0
    assert simple_integral_on([3.0, 1.0], [FinCofin({0, 1}), FinCofin({0, 1}, cofinite=True)], nu) == INF
    assert simple_integral_on([INF, 0.0], [{0}, {1}], dirac(1)) == 0.0             # (+inf) 1_{0}: null set


def test_integral_series_increases_to_the_integral():
    # f(n) = n on N with mu({n}) = 2^-n: sum n 2^-n = 2.
    S = integral_series(lambda n: n.astype(float), lambda n: 0.5**n, 80)
    assert np.all(np.diff(S) >= 0)
    assert S[-1] == pytest.approx(2.0, abs=1e-12)

# ------------------------------------------------------------------- step functions


def test_step_function_from_overlapping_sets():
    sets = [[(0, 1)], [(0.5, 2)], [(1.5, 3), (4, 5)]]
    f = step_function([2, 3, 1], sets)
    assert np.allclose(f.breaks, [0, 0.5, 1, 1.5, 2, 3, 4, 5])
    assert np.allclose(f.values, [2, 5, 3, 4, 1, 0, 1])
    assert f.integral() == lebesgue_simple_integral([2, 3, 1], sets) == 9.0
    a, level_sets = f.canonical()
    assert np.allclose(a, [1, 2, 3, 4, 5])
    assert simple_integral(a, [np.sum(S[:, 1] - S[:, 0]) for S in level_sets]) == 9.0
    assert f.integral([(0.25, 1.75)]) == pytest.approx(0.5 + 2.5 + 1.5 + 1.0)
    assert np.allclose(f([0, 0.5, 0.99, 2.5, 3.0, 4.2, 5.0, -1]), [2, 5, 5, 1, 0, 1, 0, 0])


def test_step_integral_is_well_defined_and_linear():
    rng = np.random.default_rng(2)
    for _ in range(100):
        k = int(rng.integers(1, 6))
        sets = []
        for _ in range(k):
            ends = np.sort(rng.uniform(-3, 3, size=2 * int(rng.integers(1, 4))))
            sets.append(ends.reshape(-1, 2))
        a = rng.choice([0.0, 0.5, 1.0, 2.0], size=k)
        f = step_function(a, sets)
        assert f.integral() == pytest.approx(lebesgue_simple_integral(a, sets), abs=1e-12)
        g = step_function(a[::-1], sets)
        s = combine([f, g], "sum")
        assert s.integral() == pytest.approx(f.integral() + g.integral(), abs=1e-12)
        assert combine([f, g], "min").integral() <= combine([f, g], "max").integral() + 1e-12
        assert f.scale(2.5).integral() == pytest.approx(2.5 * f.integral())
        E = [(-1.0, 0.5), (1.0, 2.0)]
        assert f.restrict(E).integral() == pytest.approx(f.integral(E), abs=1e-12)


def test_step_functions_with_infinite_measure_or_values():
    zero_on_R = step_function([0.0], [[(-INF, INF)]])
    assert zero_on_R.integral() == 0.0                    # 0 * m(R) = 0
    assert step_function([1.0], [[(0, INF)]]).integral() == INF
    assert StepFunction(np.array([0.0, 1.0, 2.0]), np.array([INF, 0.0])).integral() == INF
    assert StepFunction(np.array([0.0, 1.0, 2.0]), np.array([INF, 0.0])).integral([(1.5, 3)]) == 0.0

# ---------------------------------------------------------------- dyadic approximants


def test_dyadic_approximant_is_a_sum_of_indicators():
    rng = np.random.default_rng(3)
    v = np.concatenate([rng.exponential(2.0, 500), [0, 0.5, 1, 2.75, 3, 7, INF]])
    for n in range(1, 6):
        assert np.array_equal(dyadic_as_indicator_sum(v, n), dyadic_approximation(v, n))


def test_dyadic_integrals_of_x_are_exact():
    tail = EXAMPLES["x on [0, 1]"].tail
    for n in range(1, 16):
        assert dyadic_integral(tail, n) == pytest.approx(0.5 - 2.0 ** (-n - 1), abs=1e-14)


def test_dyadic_integrals_increase_to_the_integral():
    for name, ex in EXAMPLES.items():
        I = dyadic_integrals(ex.tail, 14)
        assert np.all(np.diff(I) > 0), name                   # monotonicity
        assert np.all(I <= ex.integral), name
    # the three error regimes of Exercise 4.4
    gap = lambda key, n: EXAMPLES[key].integral - dyadic_integral(EXAMPLES[key].tail, n)
    assert gap("x^2 on [0, 1]", 16) * 2**17 == pytest.approx(1.0, abs=2e-3)
    assert gap("x^(-1/2) on (0, 1)", 16) * 16 == pytest.approx(1.0, abs=2e-4)
    zeta_half = -1.4603545088095868
    assert gap("1/(1+x^2) on R", 18) * 2**9 == pytest.approx(-2 * zeta_half, abs=2e-4)


def test_dyadic_integrals_of_minus_log_match_chapter_3():
    tail = EXAMPLES["-log x on (0, 1)"].tail
    for n in (1, 2, 4, 6, 10):
        h = 2.0**-n
        E = math.exp(-n) + (1 - math.exp(-n)) * (1 - math.exp(-h) * (1 + h)) / (1 - math.exp(-h))
        assert 1 - dyadic_integral(tail, n) == pytest.approx(E, rel=1e-10)


def test_dyadic_integrals_of_one_over_x_diverge_like_log_n():
    tail = EXAMPLES["1/x on (0, 1)"].tail
    for n in (1, 3, 8, 12):
        harmonic = math.fsum(1 / k for k in range(2**n + 1, n * 2**n + 1))
        assert dyadic_integral(tail, n) == pytest.approx(1 + harmonic, rel=1e-12)
    assert dyadic_integral(tail, 16) == pytest.approx(1 + math.log(16), abs=1e-4)


def test_dyadic_integral_on_a_finite_space_matches_the_tail_formula():
    rng = np.random.default_rng(4)
    for _ in range(20):
        f = rng.exponential(1.5, 40)
        w = rng.random(40)
        tail = tail_from_values(f, w)
        for n in (1, 3, 5):
            assert dyadic_integral_finite(f, w, n) == pytest.approx(dyadic_integral(tail, n), rel=1e-12)

# ------------------------------------------------- sequences, monotonicity, Fatou


def test_sliding_bump_of_the_notes():
    for n in range(0, 20):
        assert sliding_bump(n).integral() == 1.0              # m([n, n+1)) = 1
    x = np.linspace(-2, 10, 97)
    assert np.all(np.array([sliding_bump(n)(x) for n in range(11, 15)]) == 0)  # f_n(x) = 0 once n > x
    assert sliding_bump(3)(3.5) == 1 and sliding_bump(4)(3.5) == 0             # not increasing


def test_fatou_sides_for_the_example_sequences():
    for name, (seq, liminf) in SEQUENCES.items():
        ints, infs = fatou_sides(seq, 6, 80)
        assert np.all(np.diff(infs) >= -1e-12), name           # g_n increases
        assert np.all(infs <= ints + 1e-12), name
        assert infs[0] >= liminf.integral() - 1e-12
    ints, infs = fatou_sides(alternating, 6, 40)
    assert np.all(ints == 1) and np.all(infs == 0)              # 0 < 1: strict
    ints, infs = fatou_sides(bump_plus_fixed, 6, 40)
    assert np.all(ints == 2) and np.all(infs == 1)              # 1 < 2: strict, both finite
    ints, infs = fatou_sides(spike, 5, 1000)
    assert np.allclose(infs, np.arange(1, 6) / 1000)            # -> 0 as K grows


def test_fatou_is_an_equality_for_increasing_sequences():
    for n, K in ((1, 5), (3, 9)):
        g = tail_infimum(increasing_plateau, n, K)
        assert g.integral() == increasing_plateau(n).integral() == n


def test_periodic_fatou_on_a_finite_space():
    rng = np.random.default_rng(5)
    strict = 0
    for _ in range(2000):
        h = rng.integers(0, 4, size=(3, 6)).astype(float)
        w = rng.random(6)
        lhs, rhs = periodic_fatou(h, w)
        assert lhs <= rhs + 1e-12
        strict += lhs < rhs - 1e-12
    assert strict > 1000
    lhs, rhs = periodic_fatou([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
    assert (lhs, rhs) == (0.0, 1.0)


def test_spreading_and_spike_have_integral_one():
    for n in (1, 2, 5, 40):
        assert spike(n).integral() == pytest.approx(1.0)
        assert spreading(n).integral() == pytest.approx(1.0)
    assert indicator_step(2.0, 2.0).integral() == 0.0

# ------------------------------------------------- numbers quoted in the exercise solutions


def test_dyadic_rates_table():
    # Exercise 4.4 (b): 2^(n+1) delta_n for x^2, n delta_n for x^(-1/2), 2^(n/2) delta_n for 1/(1+x^2).
    table = {1: (0.74755, 1.00000, 3.02867), 2: (0.81293, 1.16951, 2.97378),
             4: (0.90126, 1.11463, 2.93378), 8: (0.97434, 1.01536, 2.92152),
             12: (0.99352, 1.00145, 2.92076), 16: (0.99838, 1.00012, 2.92071),
             18: (0.99919, 1.00003, 2.92071)}
    for n, (a, b, c) in table.items():
        ex = [EXAMPLES[k] for k in ("x^2 on [0, 1]", "x^(-1/2) on (0, 1)", "1/(1+x^2) on R")]
        d = [e.integral - dyadic_integral(e.tail, n) for e in ex]
        assert d[0] * 2 ** (n + 1) == pytest.approx(a, abs=1e-5)
        assert d[1] * n == pytest.approx(b, abs=1e-5)
        assert d[2] * 2 ** (n / 2) == pytest.approx(c, abs=1e-5)
    # (c) at n = 10 for 1/(1+x^2): discarded small values and rounding
    n = 10
    small = 2 * math.atan((2**n - 1) ** -0.5)
    delta = math.pi - dyadic_integral(EXAMPLES["1/(1+x^2) on R"].tail, n)
    assert small == pytest.approx(0.06251, abs=1e-5) and delta - small == pytest.approx(0.02877, abs=1e-5)
    # (c) at n = 10 for x^2 (small values h^(3/2)/3, rounding 0.966 * 2^-11) and x^(-1/2) (cut 1/n)
    h = 2.0**-n
    small = h**1.5 / 3
    rounding = 1 / 3 - dyadic_integral(EXAMPLES["x^2 on [0, 1]"].tail, n) - small
    assert small == pytest.approx(1.02e-5, abs=5e-8)
    assert rounding == pytest.approx(4.718e-4, abs=5e-8) and rounding * 2**11 == pytest.approx(0.966, abs=5e-4)
    rounding = 2 - dyadic_integral(EXAMPLES["x^(-1/2) on (0, 1)"].tail, n) - 1 / n
    assert rounding == pytest.approx(4.83e-4, abs=5e-7)


def test_dyadic_rates_finite_space_check():
    rng = np.random.default_rng(2026)
    x = rng.random(1000)
    w = np.full(1000, 1e-3)
    for n, value in ((1, 0.16), (3, 0.297375), (6, 0.346171875)):
        assert dyadic_integral_finite(x**2, w, n) == pytest.approx(value, abs=1e-12)
        assert dyadic_integral(tail_from_values(x**2, w), n) == pytest.approx(value, abs=1e-12)


def test_decreasing_sequences_numbers():
    # Exercise 4.8 (c), (d)
    tail_f = lambda n: (lambda t: np.minimum(1 / n, 1 / np.asarray(t)))
    tail_g = lambda n: (lambda t: np.minimum(1 / n, np.asarray(t) ** -2.0))
    for n in (1, 2, 4):
        for k in (1, 2, 3, 4, 6, 9, 12):
            I = dyadic_integral(tail_f(n), k)
            exact = k / n if k <= n else 1 + math.fsum(1 / j for j in range(n * 2**k + 1, k * 2**k + 1))
            assert I == pytest.approx(exact, rel=1e-12)
        assert dyadic_integral(tail_f(n), 20) == pytest.approx(1 + math.log(20 / n), abs=1e-6)
    assert dyadic_integrals(tail_f(1), 4)[[0, 1, 3]] == pytest.approx([1.0, 1.6345, 2.3632], abs=1e-4)
    for n in (1, 4, 16, 64):
        short = 2 / math.sqrt(n) - dyadic_integral(tail_g(n), 20)
        assert 0 < short - 1 / 20 < 2.0**-20 / n


def test_fatou_strict_fraction_seed_4():
    # Exercise 4.9 (c): 10^4 random periodic sequences, seed 4.
    rng = np.random.default_rng(4)
    strict = 0
    for t in range(10_000):
        h = rng.integers(0, 4, size=(3, 6)).astype(float)
        w = rng.random(6)
        lhs, rhs = periodic_fatou(h, w)
        if t == 0:
            assert (round(lhs, 3), round(rhs, 3)) == (3.338, 4.743)
        some_row_is_min = any(np.all(h[j] == h.min(axis=0)) for j in range(3))
        assert (lhs < rhs - 1e-12) == (not some_row_is_min)
        strict += lhs < rhs - 1e-12
    assert strict == 9716
    from fractions import Fraction
    exact = 1 - (3 * Fraction(30, 64) ** 6 - 3 * Fraction(10, 64) ** 6 + Fraction(4, 64) ** 6)
    assert exact == Fraction(129952095, 134217728) and float(exact) == pytest.approx(0.968219, abs=1e-6)
    assert abs(strict / 10_000 - float(exact)) < 2 * math.sqrt(float(exact) * (1 - float(exact)) / 10_000)
    assert periodic_fatou([[2, 0, 1], [0, 2, 1], [1, 1, 0]], [1, 1, 1]) == (0.0, 2.0)
    assert combine([alternating(k) for k in range(5, 50)], "max").integral() == 2.0


def test_layer_cake_power():
    # Exercise 4.3 (e): int_(0,1) x^-p dm = 1/(1-p) for p < 1, through the tail min(1, t^(-1/p)).
    for p in (0.25, 0.5, 0.75):
        tail = lambda t, p=p: np.minimum(1.0, np.asarray(t) ** (-1 / p))
        n = 16
        cut = p / (1 - p) * n ** (1 - 1 / p)          # integral of (x^-p - n)^+
        assert dyadic_integral(tail, n) + cut == pytest.approx(1 / (1 - p), abs=2.0**-n)
