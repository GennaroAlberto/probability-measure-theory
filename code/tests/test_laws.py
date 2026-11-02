"""Checks of the Chapter 6 code against the notes and the exercises."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

from probnotes.laws import (
    DiscreteLaw,
    MixedLaw,
    are_equivalent,
    are_singular,
    binomial_law,
    chebyshev_sample_size,
    conjugate_exponent,
    density_measure,
    dkw_epsilon,
    dkw_sample_size,
    ecdf,
    ess_sup,
    ess_sup_by_definition,
    expectation_density,
    image_measure,
    integral,
    is_absolutely_continuous,
    kolmogorov_tail,
    ks_distance,
    lp_norm,
    mass_above,
    mc_expectation,
    poisson_law,
    power_integral,
    power_lp_norm,
    preimage_mass,
    quantile,
    radon_nikodym,
)

# ---------------------------------------------------------------- L^p norms and ess sup


def test_lp_norms_of_sequences_are_ell_p_norms():
    a = [3.0, -4.0]                                    # counting measure on {0, 1}
    assert lp_norm(a, p=1) == pytest.approx(7.0)
    assert lp_norm(a, p=2) == pytest.approx(5.0)
    assert lp_norm(a, p=math.inf) == 4.0
    assert lp_norm([0, 0, 0], p=3) == 0.0


def test_lp_norm_ignores_null_points_and_handles_large_p():
    f = np.array([1.0, 2.0, np.inf, 50.0])
    w = np.array([0.5, 0.5, 0.0, 0.0])               # inf and 50 sit on null points (0 * inf = 0)
    assert lp_norm(f, w, 1) == pytest.approx(1.5)
    assert lp_norm(f, w, math.inf) == 2.0
    assert lp_norm(f, w, 5000) == pytest.approx(2.0 * 0.5 ** (1 / 5000), rel=1e-12)   # no overflow
    assert lp_norm([1.0, np.inf], [1.0, 1e-9], 2) == math.inf
    # a zero value on a point of infinite weight contributes 0 * inf = 0
    assert lp_norm([0.0, 1.0], [np.inf, 1.0], 2) == 1.0
    assert lp_norm([0.0, 0.0], [np.inf, 1.0], 1) == 0.0
    assert lp_norm([2.0, 1.0], [np.inf, 1.0], 2) == math.inf
    assert integral([0.0, 3.0], [np.inf, 0.5]) == 1.5


def test_ess_sup_matches_the_definition_of_the_notes():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(1, 9)
        f = rng.choice([-3.0, -1.0, 0.0, 0.5, 2.0, np.inf], size=n)
        w = rng.choice([0.0, 0.0, 0.25, 1.0], size=n)
        assert ess_sup(f, w) == ess_sup_by_definition(f, w)
    # f = 0 a.e.: every M > 0 qualifies and the infimum is 0
    assert ess_sup_by_definition([5.0, 0.0], [0.0, 1.0]) == 0.0
    assert ess_sup([5.0, 0.0], [0.0, 1.0]) == 0.0
    assert ess_sup([7.0], [0.0]) == 0.0                # the zero measure: everything is null
    # an infinite value on a set of positive measure: the set of M is empty, ess sup = +inf
    assert ess_sup_by_definition([np.inf, 1.0], [1.0, 1.0]) == math.inf


def test_notes_example_of_an_essentially_bounded_unbounded_function():
    # Discrete analogue of the example on [0, 1]: f = n at the points 1/n, which are null,
    # and f = 1 on the points that carry the measure.
    n = np.arange(1, 51)
    grid = np.linspace(0, 1, 1001)[1:-1]
    grid = grid[~np.isin(grid, 1 / n)]
    f = np.concatenate([n.astype(float), np.ones(grid.size)])
    w = np.concatenate([np.zeros(n.size), np.full(grid.size, 1 / grid.size)])
    assert f.max() == 50 and ess_sup(f, w) == 1.0
    assert mass_above(f, w, 1.0) == 0.0 and mass_above(f, w, 0.99) == pytest.approx(1.0)
    for p in (1, 2, 7.5):
        assert lp_norm(f, w, p) == pytest.approx(1.0)  # ||f||_p = 1 for every p (exercise)


def test_holder_and_cauchy_schwarz():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = rng.integers(1, 12)
        f, g, w = rng.normal(size=n), rng.normal(size=n), rng.random(n)
        p = rng.choice([1.0, 1.5, 2.0, 3.0, 7.0, math.inf])
        q = conjugate_exponent(p)
        assert lp_norm(f * g, w, 1) <= lp_norm(f, w, p) * lp_norm(g, w, q) * (1 + 1e-12) + 1e-15
        assert abs(integral(f * g, w)) <= lp_norm(f, w, 2) * lp_norm(g, w, 2) * (1 + 1e-12)
    # equality in Hoelder when |g|^q is proportional to |f|^p
    f, w, p = np.array([1.0, 2.0, 0.5]), np.array([0.2, 0.3, 0.5]), 3.0
    q = conjugate_exponent(p)
    g = np.abs(f) ** (p / q)
    assert lp_norm(f * g, w, 1) == pytest.approx(lp_norm(f, w, p) * lp_norm(g, w, q))
    assert conjugate_exponent(1) == math.inf and conjugate_exponent(math.inf) == 1.0


def test_inclusions_on_finite_measure_spaces():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = rng.integers(1, 10)
        f, w = rng.normal(size=n) * 3, rng.random(n) * 2
        total = w.sum()
        p, q = np.sort(rng.uniform(1, 8, size=2))
        # ||f||_p <= mu(X)^(1/p - 1/q) ||f||_q, and ||f||_p <= mu(X)^(1/p) ||f||_inf
        assert lp_norm(f, w, p) <= total ** (1 / p - 1 / q) * lp_norm(f, w, q) * (1 + 1e-12)
        assert lp_norm(f, w, p) <= total ** (1 / p) * ess_sup(f, w) * (1 + 1e-12)
        # on a probability space p -> ||f||_p is non-decreasing and tends to ||f||_inf
        P = w / total
        norms = [lp_norm(f, P, r) for r in (1, 2, 4, 16, 256, 4096)]
        assert np.all(np.diff(norms) >= -1e-12)
        assert norms[-1] == pytest.approx(ess_sup(f, P), rel=2e-3)


def test_power_functions_and_failure_of_inclusions():
    # x^(-1/3) on (0, 1): in L^p iff p < 3; on (1, inf): iff p > 3 (Exercise on powers)
    assert power_lp_norm(1 / 3, 2, 0, 1) == pytest.approx(math.sqrt(3))
    assert power_lp_norm(1 / 3, 2.9, 0, 1) == pytest.approx(30 ** (1 / 2.9))
    assert power_lp_norm(1 / 3, 3, 0, 1) == math.inf
    assert power_lp_norm(1 / 3, 4, 1, math.inf) == pytest.approx(3 ** 0.25)
    assert power_lp_norm(1 / 3, 3, 1, math.inf) == math.inf
    assert power_lp_norm(1 / 3, 2, 1, math.inf) == math.inf
    assert power_lp_norm(1 / 3, math.inf, 0, 1) == math.inf
    assert power_lp_norm(1 / 3, math.inf, 1, math.inf) == 1.0
    for b, lo, hi in ((0.5, 0.0, 2.0), (2.0, 1.0, math.inf), (1.0, 0.5, 3.0), (-1.0, 0.0, 1.0)):
        assert power_integral(b, lo, hi) == pytest.approx(integrate.quad(lambda x: x ** -b, lo, hi)[0], rel=1e-7)
    assert power_integral(1.0, 0.0, 1.0) == math.inf


# ---------------------------------------------------------------- image measures


def test_image_measure_of_two_dice_under_the_sum():
    i, j = np.meshgrid(np.arange(1, 7), np.arange(1, 7))
    w = np.full(36, 1 / 36)
    support, masses = image_measure(w, (i + j).ravel())
    assert support.tolist() == list(range(2, 13))
    assert np.allclose(masses * 36, [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    assert masses.sum() == pytest.approx(1.0)          # the image of a probability is one
    assert preimage_mass(w, (i + j).ravel(), {7, 11}) == pytest.approx(8 / 36)


def test_image_measure_definition_and_integral_on_random_finite_spaces():
    rng = np.random.default_rng(3)
    for _ in range(100):
        n = rng.integers(1, 20)
        w = rng.random(n) * rng.integers(0, 2, size=n)        # some null points
        phi = rng.integers(0, 5, size=n)
        support, masses = image_measure(w, phi)
        B = set(rng.choice(5, size=rng.integers(0, 6)).tolist())
        assert preimage_mass(w, phi, B) == pytest.approx(sum(m for y, m in zip(support, masses) if y in B))
        g = lambda y: np.cos(y) + y**2                         # int g d(mu_phi) = int g(phi) d mu
        assert integral(g(support), masses) == pytest.approx(integral(g(phi), w))
        psi = lambda y: y % 2                                  # (mu_phi)_psi = mu_(psi o phi)
        s1, m1 = image_measure(masses, psi(support))
        s2, m2 = image_measure(w, psi(phi))
        assert np.array_equal(s1, s2) and np.allclose(m1, m2)


# ---------------------------------------------------------------- discrete laws


def test_discrete_law_of_a_die():
    X = DiscreteLaw(np.arange(1, 7), np.full(6, 1 / 6))
    assert X.cdf(3.5) == pytest.approx(0.5)
    assert X.cdf(3.0) == pytest.approx(0.5) and X.cdf_left(3.0) == pytest.approx(1 / 3)
    assert X.cdf(0.99) == 0.0 and X.cdf(6) == pytest.approx(1.0)
    assert X.pmf([3, 3.5, 7]).tolist() == pytest.approx([1 / 6, 0, 0])
    assert X.cdf(4) - X.cdf_left(4) == pytest.approx(X.pmf(4))          # P(X = x) = F(x) - F(x-)
    assert X.mean() == pytest.approx(3.5)
    assert X.expectation(lambda x: x**2) == pytest.approx(91 / 6)
    assert X.prob(lambda x: x % 2 == 0) == pytest.approx(0.5)
    s = X.sample(200_000, rng=4)
    assert abs(s.mean() - 3.5) < 4 * math.sqrt(35 / 12 / 200_000)


def test_discrete_law_merges_repeated_points_and_checks_mass():
    Y = DiscreteLaw([1, 1, 2], [0.25, 0.25, 0.5])
    assert Y.support.tolist() == [1, 2] and Y.masses.tolist() == [0.5, 0.5]
    with pytest.raises(ValueError):
        DiscreteLaw([0, 1], [0.5, 0.6])


def test_poisson_and_binomial_laws():
    P = poisson_law(3.0)
    assert P.masses.sum() == pytest.approx(1.0)
    assert P.mean() == pytest.approx(3.0) and P.var() == pytest.approx(3.0)
    assert P.cdf(2) == pytest.approx(stats.poisson.cdf(2, 3.0))
    B = binomial_law(10, 0.3)
    assert B.mean() == pytest.approx(3.0) and B.var() == pytest.approx(2.1)


def test_newsvendor_exercise_numbers():
    D = poisson_law(20.0)
    s, c = 3.0, 1.0
    profit = lambda k: s * D.expectation(lambda x: np.minimum(x, k)) - c * k
    k = 22
    assert D.expectation(lambda x: np.minimum(x, k)) == pytest.approx(sum(1 - D.cdf(j) for j in range(k)))
    kstar = min(k for k in range(60) if D.cdf(k) >= 1 - c / s)
    assert kstar == 22
    assert D.cdf(21) == pytest.approx(0.6437, abs=1e-4) and D.cdf(22) == pytest.approx(0.7206, abs=1e-4)
    assert profit(22) == pytest.approx(35.06, abs=0.01) and profit(20) == pytest.approx(34.67, abs=0.01)
    assert profit(22) == max(profit(k) for k in range(10, 35))
    r = 0.5                                                          # salvage value
    profit_r = lambda k: D.expectation(lambda x: s * np.minimum(x, k) + r * np.maximum(k - x, 0)) - c * k
    assert min(k for k in range(60) if D.cdf(k) >= 1 - (c - r) / (s - r)) == 24
    assert profit_r(24) == pytest.approx(36.78, abs=0.01)
    assert profit_r(24) == max(profit_r(k) for k in range(10, 35))


# ---------------------------------------------------------------- empirical distribution functions


def test_ecdf_by_hand_and_ks_distance_against_a_fine_grid():
    s = [3.0, 1.0, 2.0, 2.0]
    assert ecdf(s, [0.5, 1, 1.5, 2, 2.5, 3]).tolist() == [0, 0.25, 0.25, 0.75, 0.75, 1.0]
    rng = np.random.default_rng(5)
    F = lambda x: 1 - np.exp(-np.clip(x, 0, None))
    x = rng.exponential(size=30)
    grid = np.sort(np.concatenate([np.linspace(-1, 8, 200_001), x, x - 1e-12]))
    assert ks_distance(x, F) == pytest.approx(np.max(np.abs(ecdf(x, grid) - F(grid))), abs=1e-9)
    # a law with atoms: the left limits matter
    P = poisson_law(2.0)
    y = P.sample(40, rng)
    pts = np.arange(-1, 15)
    brute = max(np.max(np.abs(ecdf(y, pts) - P.cdf(pts))),
                np.max(np.abs(ecdf(y, pts - 1e-9) - P.cdf_left(pts))))
    assert ks_distance(y, P.cdf, P.cdf_left) == pytest.approx(brute)


def test_sample_sizes_for_the_empirical_cdf_exercise():
    assert dkw_sample_size(0.01, 0.05) == 18445
    assert chebyshev_sample_size(0.01, 0.05) == 50000
    assert dkw_epsilon(18445, 0.05) <= 0.01
    assert 2 * math.exp(-2 * 18445 * 1e-4) <= 0.05
    assert kolmogorov_tail(0.01 * math.sqrt(18445)) == pytest.approx(0.05, abs=1e-4)


def test_dkw_coverage_simulation_of_the_exercise():
    rng = np.random.default_rng(2026)
    F = lambda x: 1 - np.exp(-np.clip(x, 0, None))
    samples = [rng.exponential(1.0, 18445) for _ in range(500)]
    D = np.array([ks_distance(x, F) for x in samples])
    assert (D > 0.01).mean() == pytest.approx(0.054)
    assert D.mean() == pytest.approx(0.00650, abs=5e-6)
    assert np.quantile(D, 0.95) == pytest.approx(0.01006, abs=5e-6)
    assert D.max() == pytest.approx(0.01325, abs=5e-6)
    # pointwise at the median ln 2, on the same samples: 4 of 500
    assert sum(abs(ecdf(x, math.log(2)) - 0.5) >= 0.01 for x in samples) == 4
    P = poisson_law(3.0)
    rng = np.random.default_rng(2026)
    Dp = np.array([ks_distance(P.sample(18445, rng), P.cdf, P.cdf_left) for _ in range(500)])
    assert (Dp > 0.01).mean() == pytest.approx(0.014)
    assert Dp.mean() == pytest.approx(0.00443, abs=5e-6)
    assert np.quantile(Dp, 0.95) == pytest.approx(0.00766, abs=5e-6)
    assert Dp.max() == pytest.approx(0.01180, abs=5e-6)


# ---------------------------------------------------------------- absolute continuity and Radon-Nikodym


def test_radon_nikodym_on_a_finite_space():
    k = np.arange(0, 41)
    nu = stats.binom.pmf(k, 10, 0.3)
    mu = stats.poisson.pmf(k, 3)
    assert is_absolutely_continuous(nu, mu) and not is_absolutely_continuous(mu, nu)
    assert not are_equivalent(nu, mu)
    f = radon_nikodym(nu, mu)
    assert f[0] == pytest.approx(0.7**10 * math.exp(3)) and f[0] == pytest.approx(0.5674, abs=1e-4)
    assert np.argmax(f) in (3, 4) and f.max() == pytest.approx(1.191, abs=1e-3)
    assert np.allclose(density_measure(f, mu), nu)
    for g in (k, k**2, np.exp(k / 2)):
        assert integral(g, nu) == pytest.approx(integral(g * f, mu))
    assert integral(k**2, nu) == pytest.approx(11.1)
    assert integral(np.exp(k / 2), nu) == pytest.approx((0.7 + 0.3 * math.exp(0.5)) ** 10)
    with pytest.raises(ValueError):
        radon_nikodym(mu, nu)


def test_singular_and_equivalent_measures():
    even = np.array([0.5, 0, 0.5, 0])
    odd = np.array([0, 0.3, 0, 0.7])
    assert are_singular(even, odd) and not is_absolutely_continuous(even, odd)
    a, b = np.array([0.1, 0.2, 0.7]), np.array([0.5, 0.25, 0.25])
    assert are_equivalent(a, b)
    f = radon_nikodym(a, b)
    assert np.allclose(radon_nikodym(b, a), 1 / f)              # d mu / d nu = 1 / (d nu / d mu)
    zero = np.zeros(3)
    assert is_absolutely_continuous(zero, a) and are_singular(zero, a)


def test_importance_sampling_with_a_radon_nikodym_weight():
    exact = stats.binom.sf(24, 100, 0.1)
    assert exact == pytest.approx(1.3073e-5, rel=1e-4)
    rng = np.random.default_rng(2026)
    K = rng.binomial(100, 0.25, 10_000)
    w = 0.4**K * 1.2 ** (100.0 - K)                               # d Bin(100,.1) / d Bin(100,.25)
    est = mc_expectation(lambda k: w * (k >= 25), K)
    assert est.estimate == pytest.approx(1.2846e-5, rel=1e-3)
    assert abs(est.estimate - exact) < 2 * est.stderr
    assert (np.random.default_rng(2026).binomial(100, 0.1, 10_000) >= 25).sum() == 0


# ---------------------------------------------------------------- laws on the line, Monte Carlo


def test_quantile_transform_gives_the_law():
    u = np.array([0.1, 0.5, 0.9])
    F = lambda x: 1 - np.exp(-np.clip(x, 0, None))
    assert quantile(F, u, 0, 50) == pytest.approx(-np.log(1 - u), abs=1e-9)
    die = DiscreteLaw(np.arange(1, 7), np.full(6, 1 / 6))
    v = np.array([0.01, 1 / 6, 0.17, 0.5, 0.99])
    assert quantile(die.cdf, v, 0, 7) == pytest.approx(np.ceil(6 * v - 1e-12), abs=1e-9)
    assert die.quantile(v).tolist() == [1, 1, 2, 3, 6]
    rng = np.random.default_rng(6)
    x = quantile(die.cdf, rng.random(60_000), 0, 7)              # the atoms, up to the tolerance
    assert np.max(np.abs(x - np.round(x))) < 1e-9
    assert ks_distance(np.round(x), die.cdf, die.cdf_left) < 0.01


def test_expectation_of_a_function_three_ways():
    # E[X^2] for X ~ Exp(1): integral against the density, Monte Carlo, and the law of X^2.
    f = lambda x: np.exp(-x)
    assert expectation_density(lambda x: x**2, f, 0, math.inf) == pytest.approx(2.0)
    r = mc_expectation(lambda x: x**2, np.random.default_rng(7).exponential(size=400_000))
    assert abs(r.estimate - 2.0) < 4 * r.stderr
    fY = lambda y: np.exp(-np.sqrt(y)) / (2 * np.sqrt(y))          # density of Y = X^2
    assert expectation_density(lambda y: y, fY, 0, math.inf) == pytest.approx(2.0, rel=1e-6)


def test_tile_exercise_numbers():
    fS = lambda s: np.where(np.abs(s - 10) < 1, 0.75 * (1 - (s - 10) ** 2), 0.0)
    assert expectation_density(lambda s: s**2, fS, 9, 11) == pytest.approx(100.2)
    assert expectation_density(lambda s: s**4, fS, 9, 11) - 100.2**2 == pytest.approx(80.0457, abs=1e-4)
    FT = lambda t: 0.5 + 0.75 * (t - t**3 / 3)
    assert 1 - FT(math.sqrt(110) - 10) == pytest.approx(0.16300, abs=1e-5)
    rng = np.random.default_rng(2026)
    S = 10 + np.median(rng.uniform(-1, 1, size=(10**6, 3)), axis=1)
    r = mc_expectation(lambda s: s**2, S)
    assert r.estimate == pytest.approx(100.1919, abs=1e-4) and r.ci[0] < 100.2 < r.ci[1]
    assert ks_distance(S, lambda s: FT(np.clip(s - 10, -1, 1))) < 0.002


def test_insurance_exercise_numbers():
    lam, d, u = 1 / 1000, 500.0, 2500.0
    p0, pu = 1 - math.exp(-lam * d), math.exp(-lam * (d + u))
    dens = lambda y: np.where((y > 0) & (y < u), lam * np.exp(-lam * (y + d)), 0.0)
    Y = MixedLaw([0.0, u], [p0, pu], dens, 0.0, u)
    EY = 1000 * (math.exp(-0.5) - math.exp(-3))
    assert EY == pytest.approx(556.74, abs=0.01)
    assert Y.expectation() == pytest.approx(EY)
    h = lambda l: np.minimum(np.maximum(l - d, 0), u)
    assert expectation_density(h, lambda l: lam * np.exp(-lam * l), 0, math.inf, points=[d, d + u]) == pytest.approx(EY)
    EY2 = 2 * math.exp(-lam * d) * (1 - math.exp(-lam * u) * (1 + lam * u)) / lam**2
    assert Y.expectation(lambda y: y**2) == pytest.approx(EY2)
    assert math.sqrt(EY2 - EY**2) == pytest.approx(744.71, abs=0.01)
    assert Y.cdf(0.0)[0] == pytest.approx(p0) and Y.cdf_left(u)[0] == pytest.approx(1 - pu)
    L = np.random.default_rng(2026).exponential(1000, 10**6)
    r = mc_expectation(h, L)
    assert r.estimate == pytest.approx(555.49, abs=0.01) and r.ci[0] < EY < r.ci[1]
    Ys = h(L)
    assert (Ys == 0).mean() == pytest.approx(0.394388) and (Ys == u).mean() == pytest.approx(0.049315)
    se = math.sqrt(0.049315 * (1 - 0.049315) / 1e6)
    assert 0.049315 + 1.96 * se < pu < 0.049315 + 2.576 * se       # 95% interval misses, 99% covers
    assert ks_distance(Y.sample(20_000, rng=8), Y.cdf, Y.cdf_left) < 0.02
