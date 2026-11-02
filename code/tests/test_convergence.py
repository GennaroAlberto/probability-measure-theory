"""Checks of the Chapter 15 code against the notes and the exercises."""
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from probnotes.convergence import (
    LAWS,
    bernoulli_mean_tail,
    bernstein_polynomial,
    chebyshev_bound,
    chebyshev_sample_size,
    deviation_probability,
    first_n_below,
    last_n_above,
    mode_quantities,
    prob_metric,
    sample_means,
    spike,
    square_floor,
    square_gap_variance,
    tail_sup_deviation,
    typewriter,
    typewriter_hits,
    typewriter_index,
    typewriter_interval,
)

# ---------------------------------------------------------------- the typewriter sequence


def test_typewriter_first_terms_as_in_the_notes():
    # X1 = Y11, X2 = Y21, X3 = Y22, X4 = Y31, X5 = Y32, X6 = Y33
    k, j = typewriter_index(np.arange(1, 7))
    assert list(zip(k, j)) == [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]
    a, b = typewriter_interval(np.arange(1, 7))
    assert np.allclose(a, [0, 0, 0.5, 0, 1 / 3, 2 / 3]) and np.allclose(b, [1, 0.5, 1, 1 / 3, 2 / 3, 1])
    with pytest.raises(ValueError):
        typewriter_index(0)


def test_typewriter_index_round_trip_and_large_n():
    n = np.concatenate([np.arange(1, 5000), np.random.default_rng(0).integers(1, 10 ** 12, 2000)])
    k, j = typewriter_index(n)
    assert np.all((1 <= j) & (j <= k))
    assert np.array_equal(k * (k - 1) // 2 + j, n)
    assert np.array_equal(k[:4999], np.ceil((np.sqrt(8.0 * n[:4999] + 1) - 1) / 2).astype(int))
    k6, j6 = typewriter_index(10 ** 6)                          # exercise: (1414, 1009)
    assert (k6, j6) == (1414, 1009)


def test_typewriter_values_and_exact_endpoints():
    n = np.arange(1, 11)
    assert np.array_equal(typewriter(n, 0.3), [1, 1, 0, 1, 0, 0, 0, 1, 0, 0])   # the figure
    # omega = 1/3 is an endpoint of the third block: both X_4 and X_5 equal 1 there
    assert np.array_equal(typewriter(n, Fraction(1, 3)), [1, 1, 0, 1, 1, 0, 0, 1, 0, 0])
    assert typewriter(1, 0) == 1 and typewriter(1, Fraction(1)) == 1


def test_every_point_is_hit_in_every_block_and_missed_from_block_three():
    # the a.s. argument of the notes: liminf X_n(omega) = 0 and limsup X_n(omega) = 1
    rng = np.random.default_rng(1)
    w = np.concatenate([rng.random(200), [0.0, 0.5, 1.0]])
    for k in range(1, 40):
        n = k * (k - 1) // 2 + np.arange(1, k + 1)
        vals = typewriter(n[:, None], w[None, :])
        assert np.all(vals.max(axis=0) == 1)
        if k >= 3:
            assert np.all(vals.min(axis=0) == 0)


def test_typewriter_hits_counts_of_the_exercise():
    assert len(typewriter_hits(2 ** -0.5, 10 ** 6)) == 1414      # one per block
    assert len(typewriter_hits(Fraction(1, 3), 10 ** 6)) == 1885  # 1413 + 471 + 1
    assert len(typewriter_hits(Fraction(7, 10), 10 ** 6)) == 1555  # 1413 + 141 + 1
    assert len(typewriter_hits(0.7, 10 ** 6)) == 1526              # floating point misses 29 endpoints
    assert list(typewriter_hits(Fraction(1, 3), 21)) == [1, 2, 4, 5, 8, 12, 17, 18]


# ---------------------------------------------------------------- the spike and the three modes


def test_spike_values():
    assert np.array_equal(spike(np.arange(1, 6), 0.3), [2, 4, 8, 0, 0])
    assert spike(5, 0.0) == 0 and spike(5, 0.2) == 0 and spike(5, 0.19) == 32
    assert np.isinf(spike(1100, 1e-5))


def test_mode_quantities_exact_values():
    t = mode_quantities("typewriter", 10, p=2, eps=0.5)            # k = 4
    assert t["prob"] == t["moment"] == 0.25 and t["metric"] == 0.125
    assert mode_quantities("typewriter", 10 ** 4, eps=0.5)["prob"] == pytest.approx(1 / 141)
    assert mode_quantities("typewriter", 10, eps=1.0)["prob"] == 0.0
    s = mode_quantities("spike", 10, p=1, eps=0.5)
    assert s["prob"] == 0.1 and s["moment"] == pytest.approx(102.4)
    assert s["metric"] == pytest.approx(1024 / 1025 / 10)
    assert mode_quantities("spike", 10, p=2)["moment"] == pytest.approx(4 ** 10 / 10)
    assert mode_quantities("spike", 1, eps=3.0)["prob"] == 0.0     # 2^1 <= 3
    # log10 of the moments quoted in the solution table
    for n, l1, l2 in ((100, 28.10, 58.21), (1000, 298.03, 599.06), (10 ** 4, 3006.3, 6016.6)):
        assert mode_quantities("spike", n, 1)["log2_moment"] * math.log10(2) == pytest.approx(l1, abs=0.01)
        assert mode_quantities("spike", n, 2)["log2_moment"] * math.log10(2) == pytest.approx(l2, abs=0.01)
    with pytest.raises(ValueError):
        mode_quantities("other", 3)


def test_modes_match_the_counterexamples():
    n = np.array([10, 100, 1000, 10 ** 4])
    t, s = mode_quantities("typewriter", n), mode_quantities("spike", n)
    assert np.all(np.diff(t["prob"]) < 0) and np.all(np.diff(t["moment"]) < 0)  # P and L^p
    assert np.all(np.diff(s["prob"]) < 0) and np.all(np.diff(s["metric"]) < 0)  # P ...
    assert np.all(np.diff(s["log2_moment"]) > 0)                               # ... not L^p


def test_spike_monte_carlo_of_the_exercise():
    w = np.random.default_rng(2026).random(10 ** 5)
    out = {}
    for n in (10, 100):
        x = spike(n, w)
        r = x / (1 + x)
        exact = mode_quantities("spike", n, eps=0.5)
        se = r.std(ddof=1) / math.sqrt(w.size)
        assert abs(r.mean() - exact["metric"]) < 3 * se
        assert abs(np.mean(x > 0.5) - exact["prob"]) < 3 * math.sqrt(exact["prob"] / w.size)
        out[n] = (np.mean(x > 0.5), r.mean())
    assert out[10] == pytest.approx((0.10065, 0.100552), abs=1e-6)
    assert out[100] == pytest.approx((0.01043, 0.01043), abs=1e-6)


# ---------------------------------------------------------------- the metric


def test_prob_metric_is_a_metric_and_matches_the_spike():
    rng = np.random.default_rng(3)
    x, y, z = rng.standard_cauchy((3, 1000))
    assert prob_metric(x, z) <= prob_metric(x, y) + prob_metric(y, z) + 1e-15
    assert prob_metric(x, x) == 0 and prob_metric(x, y) == prob_metric(y, x)
    assert prob_metric([0.0, 1.0, np.inf]) == pytest.approx(0.5)
    grid = (np.arange(10 ** 6) + 0.5) / 10 ** 6                  # Lebesgue measure on [0, 1]
    assert prob_metric(spike(10, grid)) == pytest.approx(mode_quantities("spike", 10)["metric"], abs=1e-5)
    assert prob_metric(typewriter(10, grid), 0, weights=np.full(grid.size, 2.0)) == pytest.approx(0.125, abs=1e-5)


def test_metric_characterisation_inequalities():
    # eps/(1+eps) P(|X|>eps) <= d(X, 0) <= P(|X|>eps) + eps, as in the proof of the notes
    x = np.random.default_rng(4).exponential(size=10 ** 5)
    for eps in (0.1, 0.5, 2.0):
        p = np.mean(x > eps)
        assert eps / (1 + eps) * p <= prob_metric(x) <= p + eps


# ---------------------------------------------------------------- sample means and Chebyshev


def test_chebyshev_sample_sizes():
    assert chebyshev_sample_size(0.25, 0.05, 0.05) == 2000
    assert chebyshev_sample_size(0.09, 0.05, 0.05) == 720
    assert chebyshev_bound(0.25, 2000, 0.05) == pytest.approx(0.05)
    assert math.ceil(math.log(2 / 0.05) / (2 * 0.05 ** 2)) == 738          # Hoeffding, exercise
    assert math.ceil(math.log(2 / 1e-6) / (2 * 0.05 ** 2)) == 2902
    with pytest.raises(ValueError):
        chebyshev_sample_size(0.25, 0.0, 0.05)


def test_bernoulli_tail_against_exact_fractions():
    for n in range(1, 61):
        p = Fraction(1, 2)
        strict = sum(Fraction(math.comb(n, k), 2 ** n) for k in range(n + 1)
                     if abs(Fraction(k, n) - p) > Fraction(1, 20))
        loose = sum(Fraction(math.comb(n, k), 2 ** n) for k in range(n + 1)
                    if abs(Fraction(k, n) - p) >= Fraction(1, 20))
        assert bernoulli_mean_tail(n, 0.5, 0.05) == pytest.approx(float(strict), abs=1e-12)
        assert bernoulli_mean_tail(n, 0.5, 0.05, strict=False) == pytest.approx(float(loose), abs=1e-12)
    assert bernoulli_mean_tail(100, 0.5, 0.05) == pytest.approx(0.2712530240738354, rel=1e-9)
    assert bernoulli_mean_tail(100, 0.5, 0.05, strict=False) == pytest.approx(0.36820161732669615, rel=1e-9)


def test_sample_sizes_quoted_in_the_notes_and_exercises():
    f = lambda n: bernoulli_mean_tail(n, 0.5, 0.05)
    assert first_n_below(f, 0.05, 4000) == 371
    assert last_n_above(f, 0.05, 4000) + 1 == 399                # "below 0.05 for every n >= 399"
    assert f(398) == pytest.approx(0.050459, abs=1e-6) and f(399) == pytest.approx(0.045095, abs=1e-6)
    assert f(2000) == pytest.approx(6.856e-6, rel=1e-3)
    g = lambda n: bernoulli_mean_tail(n, 0.1, 0.05)
    assert first_n_below(g, 0.05, 4000) == 120 and last_n_above(g, 0.05, 4000) + 1 == 147
    n = np.arange(1, 3001)                                        # Chebyshev is a bound
    assert np.all(f(n) <= chebyshev_bound(0.25, n, 0.05) + 1e-12)
    s = np.random.default_rng(2026).binomial(399, 0.5, size=10 ** 5)
    assert np.mean(np.abs(s / 399 - 0.5) > 0.05) == pytest.approx(0.04572)


def test_sample_means_and_deviation_probability():
    m = sample_means("bernoulli", 400, 300, seed=5)
    assert m.shape == (300, 400)
    assert np.all((m >= 0) & (m <= 1))
    x = np.full((1, 20), 0.5)
    x[0, 19] = 11 / 20                                            # |Xbar_20 - 1/2| = eps exactly
    assert deviation_probability(x, 0.5, 0.05)[19] == 0.0
    est = deviation_probability(m, 0.5, 0.05)
    exact = bernoulli_mean_tail(np.arange(1, 401), 0.5, 0.05)
    assert np.max(np.abs(est - exact)) < 5 * math.sqrt(0.25 / 300)
    with pytest.raises(ValueError):
        sample_means("normal", 10 ** 6, 100)


def test_tail_sup_deviation_by_hand():
    m = np.array([[0.9, 0.2, 0.6, 0.45, 0.5]])
    assert np.allclose(tail_sup_deviation(m, 0.5), [[0.4, 0.3, 0.1, 0.05, 0.0]])


def test_cauchy_means_do_not_settle():
    m = sample_means("cauchy", 10 ** 4, 1000, seed=2026)
    est = [np.mean(np.abs(m[:, n - 1]) > 1) for n in (10, 100, 1000, 10 ** 4)]
    assert est == [0.479, 0.495, 0.507, 0.519]                   # the numbers of the solution
    assert all(abs(e - 0.5) < 3 * math.sqrt(0.25 / 1000) for e in est)
    assert np.mean(np.abs(m[:, 9999] - m[:, 4999]) > 1) == pytest.approx(0.512)
    del m
    z = sample_means("normal", 10 ** 4, 1000, seed=2026)
    assert np.mean(np.abs(z[:, 9]) > 1) == 0.0
    assert 2 * stats.norm.sf(math.sqrt(10)) == pytest.approx(0.0016, abs=5e-5)
    assert math.isnan(LAWS["cauchy"].mean)


# ---------------------------------------------------------------- the proof of the strong law


def test_square_floor_exact():
    m = np.array([1, 2, 3, 4, 8, 9, 10, 15, 16, 10 ** 12 - 1, 10 ** 12, 10 ** 12 + 1])
    k = square_floor(m)
    assert np.all((k * k <= m) & (m < (k + 1) * (k + 1)))
    assert list(k[:9]) == [1, 1, 1, 2, 2, 3, 3, 3, 4]


def test_interpolation_between_squares():
    m = np.arange(1, 10 ** 6 + 1)
    k = square_floor(m)
    assert np.all(m - k * k <= 2 * k)                             # m - k^2 < 2k + 1
    exact, bound = square_gap_variance(m, sigma2=2.0)
    assert np.all(exact <= bound)
    assert exact[19] == pytest.approx(4 * 2.0 / 400)              # m = 20, k = 4
    assert np.sum(exact) < np.sum(bound) < 3 * 2.0 * 2.6124       # 3 sigma^2 zeta(3/2)
    # the identity (1/m) sum_{k^2 < i <= m} X_i = Xbar_m - (k^2/m) Xbar_{k^2}
    x = np.random.default_rng(6).standard_normal(50)
    mm = 37
    kk = int(square_floor(mm))
    lhs = x[kk * kk:mm].sum() / mm
    assert lhs == pytest.approx(x[:mm].mean() - kk * kk / mm * x[:kk * kk].mean())


def test_variance_of_the_gap_by_simulation():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((10 ** 5, 20))                        # m = 20, k_m = 4, sigma^2 = 1
    gap = x[:, :20].mean(axis=1) - 16 / 20 * x[:, :16].mean(axis=1)
    assert np.mean(gap ** 2) == pytest.approx(0.01, rel=0.03)


# ---------------------------------------------------------------- numbers of the other exercises


def test_records_numbers():
    for n, H, V in ((10 ** 4, 9.787606036, 8.142771964), (10 ** 6, 14.392726723, 12.747793656)):
        i = np.arange(1, n + 1, dtype=float)
        assert np.sum(1 / i) == pytest.approx(H, abs=1e-8)
        assert np.sum(1 / i - 1 / i ** 2) == pytest.approx(V, abs=1e-8)
    assert 12.747793656 / (14.392726723 / 2) ** 2 == pytest.approx(0.2462, abs=1e-4)
    # exact law of R_n for n = 10^4: convolution of the independent b(1/i)
    n = 10 ** 4
    pmf = np.zeros(80)
    pmf[0] = 1.0
    for i in range(1, n + 1):
        pmf[1:] = pmf[1:] * (1 - 1 / i) + pmf[:-1] / i
        pmf[0] *= 1 - 1 / i
    k = np.arange(80)
    assert pmf.sum() == pytest.approx(1.0) and k @ pmf == pytest.approx(9.787606036)
    tail = pmf[np.abs(k - 9.787606036) > 9.787606036 / 2].sum()
    assert tail == pytest.approx(0.0773, abs=5e-5)
    assert (0.089 - tail) / math.sqrt(tail * (1 - tail) / 2000) == pytest.approx(1.96, abs=0.01)


def test_bernstein_polynomials():
    x = np.linspace(0, 1, 11)
    assert np.allclose(bernstein_polynomial(lambda s: 3 * s - 1, 7, x), 3 * x - 1)
    assert np.allclose(bernstein_polynomial(lambda s: s ** 2, 10, x), x ** 2 + x * (1 - x) / 10)
    f = lambda s: np.abs(s - 0.5)
    assert bernstein_polynomial(f, 10, 0.5) == pytest.approx(63 / 512)
    for n in (10, 100, 1000):
        e = bernstein_polynomial(f, n, 0.5)
        assert e == pytest.approx(math.comb(n, n // 2) / 2 ** (n + 1))
        assert e <= 0.5 / math.sqrt(n)
        assert e == pytest.approx(1 / math.sqrt(2 * math.pi * n), rel=0.03)


def test_cantelli_numbers():
    assert 500 / 600 == pytest.approx(0.8333, abs=1e-4)
    assert 2500 / (2500 + 100 ** 2) == 0.2
    assert stats.norm.sf(2) == pytest.approx(0.02275, abs=1e-5)


def test_uncorrelated_cosines():
    u = (np.arange(4096) + 0.5) / 4096                            # exact for these trigonometric sums
    X = np.sqrt(2) * np.cos(2 * np.pi * np.arange(1, 9)[:, None] * u[None, :])
    gram = X @ X.T / u.size
    assert np.allclose(gram, np.eye(8), atol=1e-12) and np.allclose(X.mean(axis=1), 0, atol=1e-12)
    assert np.allclose(X[1], np.sqrt(2) * (X[0] ** 2 - 1))        # X_2 is a function of X_1
    uu = np.random.default_rng(8).random(500)
    N = np.arange(1, 301)
    partial = np.cumsum(np.cos(2 * np.pi * N[:, None] * uu[None, :]), axis=0)
    assert np.all(np.abs(partial) <= 1 / np.abs(np.sin(np.pi * uu)) + 1e-9)


def test_records_simulation_of_the_solution():
    rng = np.random.default_rng(2026)
    R = []
    for _ in range(10):                                           # 2000 paths of 10^4 uniforms, in chunks
        u = rng.random((200, 10 ** 4))
        R.append((u == np.maximum.accumulate(u, axis=1)).sum(axis=1))
    R = np.concatenate(R)
    H = 9.787606036044382
    assert R.mean() == pytest.approx(9.7925) and np.mean(np.abs(R - H) > H / 2) == pytest.approx(0.089)
