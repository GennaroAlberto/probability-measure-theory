"""Checks of the Chapter 16 code against the notes and the exercises."""
import itertools
import math

import numpy as np
import pytest
from scipy import stats

from probnotes.inlaw import (
    bernoulli_poisson_coupling,
    binomial_charfun,
    binomial_poisson_distances,
    binomial_poisson_pmfs,
    ecdf,
    empirical_charfun,
    kolmogorov_distance_pmf,
    kolmogorov_distance_sample,
    le_cam_bound,
    matching_pmf,
    normal_charfun,
    one_over_n_cdf,
    poisson_approximation,
    poisson_binomial_pmf,
    poisson_charfun,
    relaxed_mean_moments,
    relaxed_mean_paths,
    sample_relaxed_mean,
    sample_scaled_max,
    scaled_max_cdf,
    scaled_max_kolmogorov,
    sum_counterexample,
    total_variation_pmf,
)

DELTA0 = (lambda x: (np.asarray(x) >= 0).astype(float), lambda x: (np.asarray(x) > 0).astype(float))
EXP1 = lambda t: np.where(t < 0, 0.0, -np.expm1(-np.maximum(t, 0.0)))

# ---------------------------------------------------------------- distances between laws


def test_ecdf_by_hand():
    assert np.allclose(ecdf([3, 1, 2, 2], [0, 1, 1.5, 2, 2.5, 3, 4]), [0, 0.25, 0.25, 0.75, 0.75, 1, 1])
    with pytest.raises(ValueError):
        ecdf([], 0.0)


def test_kolmogorov_distance_sample_matches_scipy_and_handles_atoms():
    x = np.random.default_rng(1).standard_normal(500)
    assert kolmogorov_distance_sample(x, stats.norm.cdf) == pytest.approx(stats.kstest(x, "norm").statistic, abs=1e-12)
    # by hand: sample {0, 1} against U(0, 1): F_N = 1/2 on [0, 1), sup |1/2 - x| = 1/2 at x = 0 and x -> 1
    assert kolmogorov_distance_sample([0.0, 1.0], lambda t: np.clip(t, 0, 1)) == pytest.approx(0.5)
    # a sample of zeros against the point mass at 0 (with its left limit), and a symmetric sample against it
    assert kolmogorov_distance_sample(np.zeros(10), *DELTA0) == 0.0
    assert kolmogorov_distance_sample([-1.0, 1.0], *DELTA0) == pytest.approx(0.5)


def test_pmf_distances_by_hand():
    p, q = [0.5, 0.5], [0.0, 0.5, 0.5]                          # b(1/2) and 1 + b(1/2)
    assert total_variation_pmf(p, q) == pytest.approx(0.5)
    assert kolmogorov_distance_pmf(p, q) == pytest.approx(0.5)
    assert total_variation_pmf(q, q) == 0.0


def test_one_over_n_example():
    # F_n(0) = 0 for every n, F(0) = 1, and F_n(x) -> G(x) = 1{x > 0} at every x
    x = np.array([-0.5, 0.0, 1e-3, 0.5])
    assert np.array_equal(one_over_n_cdf(4, x), [0, 0, 0, 1])
    assert np.array_equal(one_over_n_cdf(np.inf, x), [0, 1, 1, 1])
    assert np.array_equal(one_over_n_cdf(10 ** 4, x), [0, 0, 1, 1])


def test_empirical_charfun_of_a_gaussian_sample():
    x = np.random.default_rng(2).standard_normal(20000)
    t = np.linspace(-3, 3, 13)
    assert np.max(np.abs(empirical_charfun(x, t) - normal_charfun(1.0, t))) < 0.03


# ---------------------------------------------------------------- binomial and Poisson


def test_worked_exercise_b_and_its_numbers():
    exact, approx = poisson_approximation(10000, 2.0, 2)
    assert approx == pytest.approx(5 * math.exp(-2), rel=1e-12)        # 0.676676...
    assert exact == pytest.approx(0.676676417085, abs=1e-11)
    assert exact - approx == pytest.approx(9.024e-10, rel=1e-3)          # second order: k = lambda
    with pytest.raises(ValueError):
        poisson_approximation(1, 2.0, 2)                                 # B(1, 2) does not exist


def test_densities_converge_pointwise():
    for n in (100, 10 ** 4, 10 ** 6):
        k, b, p = binomial_poisson_pmfs(n, 2.0, kmax=10)
        assert np.max(np.abs(b - p)) < 2.0 / n


def test_distances_against_brute_force_and_le_cam():
    n, lam = 50, 2.0
    k = np.arange(400)
    b, p = stats.binom.pmf(k, n, lam / n), stats.poisson.pmf(k, lam)
    tv, ks = binomial_poisson_distances(n, lam)
    assert tv == pytest.approx(0.5 * np.sum(np.abs(b - p)), abs=1e-15)
    assert ks == pytest.approx(np.max(np.abs(np.cumsum(b) - np.cumsum(p))), abs=1e-15)
    ns = np.array([10, 100, 1000, 10000])
    tv, ks = binomial_poisson_distances(ns, lam)
    assert np.all(tv <= le_cam_bound(ns, lam)) and np.all(ks <= tv)
    assert np.allclose(tv, [5.219887e-02, 4.572242e-03, 4.517200e-04, 4.511778e-05], rtol=1e-6)
    assert np.allclose(ks, [3.019621e-02, 2.734139e-03, 2.709416e-04, 2.706976e-05], rtol=1e-6)
    # n d_TV -> (1/4) sum_k pi_k |(k - lam)^2 - k|,  n d_K -> (lam/2) max_k |k - lam| pi_k = 2 e^{-2}
    kk = np.arange(80)
    pi = stats.poisson.pmf(kk, lam)
    assert 1e4 * tv[-1] == pytest.approx(0.25 * np.sum(pi * np.abs((kk - lam) ** 2 - kk)), abs=1e-4)
    assert 1e4 * ks[-1] == pytest.approx(2 * math.exp(-2), abs=1e-4)


def test_charfun_convergence_with_the_bound_of_the_notes():
    lam, t = 2.0, np.linspace(-10, 10, 2001)
    for n in (8, 100, 10 ** 4):                                          # n >= 4 lambda
        gap = np.max(np.abs(binomial_charfun(n, lam / n, t) - poisson_charfun(lam, t)))
        assert gap <= 8 * lam ** 2 / n


def test_poisson_binomial_and_matching():
    p = [0.1, 0.5, 0.7]
    brute = np.zeros(4)
    for bits in itertools.product([0, 1], repeat=3):
        brute[sum(bits)] += np.prod([q if b else 1 - q for q, b in zip(p, bits)])
    assert np.allclose(poisson_binomial_pmf(p), brute)
    assert np.allclose(poisson_binomial_pmf([0.3] * 7), stats.binom.pmf(np.arange(8), 7, 0.3))
    # fixed points of permutations of 4 objects: 9, 8, 6, 0, 1 permutations out of 24
    assert np.allclose(matching_pmf(4), np.array([9, 8, 6, 0, 1]) / 24)
    m = matching_pmf(10)
    k = np.arange(11)
    assert m.sum() == pytest.approx(1) and np.sum(k * m) == pytest.approx(1) and np.sum(k ** 2 * m) == pytest.approx(2)
    tv = 0.5 * (np.sum(np.abs(m - stats.poisson.pmf(k, 1))) + stats.poisson.sf(10, 1))
    assert tv == pytest.approx(2.1954e-05, rel=1e-3)


def test_le_cam_coupling():
    u = np.random.default_rng(2026).random(10 ** 6)
    for p in (0.1, 0.5):
        ind, N = bernoulli_poisson_coupling(p, u)
        assert ind.mean() == pytest.approx(p, abs=3e-3) and N.mean() == pytest.approx(p, abs=3e-3)
        assert np.mean(ind != N) == pytest.approx(p * (1 - math.exp(-p)), abs=2e-3)
    # the proofreader of the exercise: p_i = i / 50000, i = 1, ..., 250
    pp = np.arange(1, 251) / 50000
    pmf = poisson_binomial_pmf(pp)
    lam = pp.sum()
    assert lam == pytest.approx(0.6275) and np.sum(pp ** 2) == pytest.approx(0.0020959, abs=1e-7)
    assert 1 - pmf[0] - pmf[1] == pytest.approx(0.130827, abs=1e-6)
    k = np.arange(pmf.size)
    tv = 0.5 * (np.sum(np.abs(pmf - stats.poisson.pmf(k, lam))) + stats.poisson.sf(k[-1], lam))
    assert tv == pytest.approx(8.0203e-04, rel=1e-3) and tv <= np.sum(pp ** 2)


# ---------------------------------------------------------------- the maximum of uniforms


def test_scaled_max_cdf_formula_and_simulation():
    x = np.array([-1.0, 0.0, 0.5, 2.0, 9.9, 10.0, 12.0])
    assert np.allclose(scaled_max_cdf(10, x), [0, 0, 1 - 0.95 ** 10, 1 - 0.8 ** 10, 1 - 0.01 ** 10, 1, 1])
    assert np.allclose(scaled_max_cdf(np.inf, x[x >= 0]), 1 - np.exp(-x[x >= 0]))
    z = sample_scaled_max(5, 20000, seed=3, direct=True)               # really the max of 5 uniforms
    assert kolmogorov_distance_sample(z, lambda t: scaled_max_cdf(5, t)) < 0.015
    z = sample_scaled_max(10 ** 6, 20000, seed=4)                       # the shortcut V^{1/n}
    assert np.all(z > 0) and kolmogorov_distance_sample(z, EXP1) < 0.015


def test_scaled_max_kolmogorov_exact_rate_and_bound():
    assert scaled_max_kolmogorov(1) == pytest.approx(math.exp(-1))
    for n in (2, 5, 10, 100):
        x = np.linspace(0, n + 5, 400001)
        brute = np.max(np.abs(scaled_max_cdf(n, x) - (1 - np.exp(-x))))
        assert scaled_max_kolmogorov(n) == pytest.approx(brute, abs=1e-9)
    d, xs = scaled_max_kolmogorov(np.array([10, 100, 1000]), return_argmax=True)
    assert np.allclose(d, [2.8000080e-02, 2.7157582e-03, 2.7076082e-04], rtol=1e-6)
    n = np.array([10, 100, 1000])
    assert np.all(d <= 2 * math.exp(-2) * n / (n - 1) ** 2)              # the bound of the exercise
    assert 1e6 * scaled_max_kolmogorov(10 ** 6) == pytest.approx(2 * math.exp(-2), abs=1e-6)
    assert xs[-1] == pytest.approx(2, abs=1e-3)


def test_exercise_simulation_of_the_maximum():
    z = sample_scaled_max(10, 20000, seed=2026)
    assert kolmogorov_distance_sample(z, EXP1) == pytest.approx(0.03275, abs=5e-5)
    assert kolmogorov_distance_sample(z, lambda t: scaled_max_cdf(10, t)) == pytest.approx(0.00569, abs=5e-5)


# ---------------------------------------------------------------- relaxed law of large numbers, sums


def test_relaxed_mean_variance_by_enumeration():
    p = 0.3
    for n in (1, 2, 3, 5):
        vals, probs = [], []
        for bits in itertools.product([0, 1], repeat=n + 1):
            x = np.array(bits)
            vals.append(np.mean(x[:-1] + x[1:]))
            probs.append(p ** x.sum() * (1 - p) ** (n + 1 - x.sum()))
        vals, probs = np.array(vals), np.array(probs)
        mean, var = relaxed_mean_moments(n, p)
        assert np.sum(probs * vals) == pytest.approx(mean)
        assert np.sum(probs * (vals - mean) ** 2) == pytest.approx(var)
    assert relaxed_mean_moments(1, 0.5)[1] == pytest.approx(2 * 0.25)        # T_1 = Y_1 ~ B(2, p)


def test_relaxed_mean_simulations_agree():
    p, n = 0.3, 100
    t = sample_relaxed_mean(n, p, 10 ** 5, seed=2026)
    mean, var = relaxed_mean_moments(n, p)
    assert t.mean() == pytest.approx(mean, abs=4 * math.sqrt(var / 1e5))
    assert t.var(ddof=1) == pytest.approx(var, rel=0.02)
    paths = relaxed_mean_paths(n, p, paths=20000, seed=5)[:, -1]
    assert kolmogorov_distance_pmf(np.bincount(np.round(paths * n).astype(int), minlength=2 * n + 1) / 20000,
                                   np.bincount(np.round(t * n).astype(int), minlength=2 * n + 1) / 1e5) < 0.03
    assert np.all(relaxed_mean_paths(5, p, paths=3, seed=0)[:, 0] % 1 == 0)  # T_1 = Y_1 is an integer


def test_sum_counterexample():
    s, z = sum_counterexample(10 ** 5, seed=2026)
    assert np.all(z == 0)
    assert kolmogorov_distance_sample(s, stats.norm(scale=math.sqrt(2)).cdf) < math.sqrt(math.log(200) / 2e5)
    assert kolmogorov_distance_sample(s, *DELTA0) == pytest.approx(0.5, abs=0.005)
    s, _ = sum_counterexample(10 ** 4, rho=-1.0, seed=1)
    assert np.allclose(s, 0)
    with pytest.raises(ValueError):
        sum_counterexample(10, rho=2.0)


# ---------------------------------------------------------------- numbers quoted in the solutions


def test_pinned_poisson_simulation_and_sums():
    x = np.random.default_rng(2026).binomial(10000, 2e-4, 10 ** 5)              # Exercise 16.2(d)
    assert np.mean(x <= 2) == pytest.approx(0.67841, abs=1e-12)
    s, _ = sum_counterexample(10 ** 5, seed=2026)                               # Exercise 16.4(a)
    assert kolmogorov_distance_sample(s, stats.norm(scale=math.sqrt(2)).cdf) == pytest.approx(0.002269, abs=1e-6)
    assert kolmogorov_distance_sample(s, *DELTA0) == pytest.approx(0.50068, abs=1e-12)
    table = {1: (1.993107, 0.50068, 0.72838, 0.72910), 10: (0.199876, 0.50071, 0.35228, 0.35288),
             100: (0.0199963, 0.50030, 0.11276, 0.11284), 1000: (0.00199980, 0.50056, 0.035666, 0.035683)}
    for n, (var, d0, emin, exact) in table.items():                              # Exercise 16.4(d)
        s, _ = sum_counterexample(10 ** 5, rho=-1 + 1 / n, seed=2026)
        sd = math.sqrt(2 / n)
        formula = sd * math.sqrt(2 / math.pi) * (1 - math.exp(-1 / (2 * sd * sd))) + 2 * stats.norm.sf(1 / sd)
        assert s.var() == pytest.approx(var, rel=5e-5)
        assert kolmogorov_distance_sample(s, *DELTA0) == pytest.approx(d0, abs=1e-12)
        assert np.mean(np.minimum(np.abs(s), 1)) == pytest.approx(emin, abs=5e-6)
        assert formula == pytest.approx(exact, abs=5e-6)


def test_pinned_matching_distances_in_exact_arithmetic():
    from decimal import Decimal, getcontext
    from fractions import Fraction
    getcontext().prec = 50
    e_inv = 1 / Decimal(1).exp()
    for n, tv_exact, tv_binom in ((5, 3.440832e-02, 6.258084e-02), (20, 1.880765e-14, 1.421124e-02)):
        pm = [sum(Fraction((-1) ** j, math.factorial(j)) for j in range(n - k + 1)) / math.factorial(k)
              for k in range(n + 1)]
        tv, mass = Decimal(0), Decimal(0)
        for k in range(n + 1):
            pk = e_inv / math.factorial(k)
            mass += pk
            tv += abs(Decimal(pm[k].numerator) / Decimal(pm[k].denominator) - pk)
        assert float((tv + 1 - mass) / 2) == pytest.approx(tv_exact, rel=1e-6)      # Exercise 16.7(d)
        assert np.allclose(matching_pmf(n), [float(q) for q in pm], atol=1e-15)
        assert binomial_poisson_distances(n, 1.0)[0] == pytest.approx(tv_binom, rel=1e-6)


def test_pinned_coupling_and_relaxed_tables():
    u = np.random.default_rng(2026).random(10 ** 6)                              # Exercise 16.8(a)
    for p, freq in ((0.1, 0.009431), (0.5, 0.196939)):
        ind, N = bernoulli_poisson_coupling(p, u)
        assert np.mean(ind != N) == pytest.approx(freq, abs=1e-12)
    for n, var, freq in ((10, 7.95935e-02, 0.84289), (100, 8.34395e-03, 0.58424),
                         (1000, 8.34891e-04, 0.08311)):                         # Exercise 16.9(d)
        t = sample_relaxed_mean(n, 0.3, 10 ** 5, seed=2026)
        assert t.var(ddof=1) == pytest.approx(var, rel=1e-5)
        assert np.mean(np.abs(t - 0.6) > 0.05) == pytest.approx(freq, abs=1e-12)
