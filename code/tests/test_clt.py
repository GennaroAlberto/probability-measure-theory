"""Checks of the Chapter 17 code against the notes and the exercises."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

from probnotes.clt import (
    LAW_NAMES,
    RAMANUJAN_CONSTANT,
    STABILISERS,
    berry_esseen_bound,
    binomial_normal_density,
    binomial_tail,
    delta_sample,
    delta_variance,
    edgeworth_constant,
    empirical_kolmogorov,
    exam_abs_moment,
    exam_cf,
    exam_sample,
    gaussian_abs_moment,
    kolmogorov_distance,
    make_law,
    matching_shift,
    max_cdf_error,
    normal_rule,
    normal_tail,
    poisson_partial_sum,
    poll_numbers,
    sample_means,
    second_order_cdf,
    second_order_sample,
    simulate_standardised,
    stabilising_transform,
    standardise,
    standardised_cdf,
    standardised_cf,
    variance_counterexample_sample,
    variance_counterexample_var,
)

# ---------------------------------------------------------------- laws


@pytest.mark.parametrize("name", LAW_NAMES)
def test_laws_moments_and_cf_against_simulation(name):
    L = make_law(name)
    rng = np.random.default_rng(1)
    x = L.sample(rng, 400_000)
    se = L.sd / math.sqrt(x.size)
    assert abs(x.mean() - L.mean) < 5 * se
    assert x.var() == pytest.approx(L.var, rel=0.02)
    assert np.mean(np.abs(x - L.mean) ** 3) == pytest.approx(L.rho, rel=0.03)
    for t in (0.3, 1.0, 2.5):
        assert abs(np.mean(np.exp(1j * t * x)) - L.cf(t)) < 0.01
    assert L.cf(0.0) == pytest.approx(1.0)


def test_law_constants_by_hand():
    E = make_law("exponential")
    assert E.rho == pytest.approx(12 / math.e - 2, rel=1e-9)          # E|X - 1|^3 for Exp(1)
    assert E.skew == 2
    B = make_law("bernoulli", p=0.5)
    assert B.rho == pytest.approx(1 / 8) and B.skew == 0
    assert make_law("uniform").rho == pytest.approx(1 / 32)
    assert make_law("gamma", shape=4, rate=2).mean == 2
    with pytest.raises(ValueError):
        make_law("cauchy")
    with pytest.raises(ValueError):
        make_law("bernoulli", p=1.0)


# ---------------------------------------------------------------- standardised sums


def test_standardise_and_exact_cdf():
    assert standardise(60, 100, 0.4, 0.24) == pytest.approx(20 / math.sqrt(24))
    # B(100, 0.4): P(S* <= x) at an atom
    x = (50 - 40) / math.sqrt(24)
    assert standardised_cdf(make_law("bernoulli", p=0.4), 100, x) == pytest.approx(stats.binom.cdf(50, 100, 0.4))
    # exponential: S_n ~ Gamma(n, 1)
    assert standardised_cdf("exponential", 9, 0.5) == pytest.approx(stats.gamma.cdf(9 + 0.5 * 3, 9))
    with pytest.raises(ValueError):
        standardised_cdf("uniform", 5, 0.0)


def test_kolmogorov_distance_matches_brute_force():
    for law, n in ((make_law("bernoulli", p=0.1), 30), (make_law("poisson", lam=2), 7), ("exponential", 5)):
        x = np.linspace(-6, 6, 400_001)
        brute = np.max(np.abs(standardised_cdf(law, n, x) - stats.norm.cdf(x)))
        d = kolmogorov_distance(law, n)
        assert d >= brute - 1e-12
        assert d == pytest.approx(brute, abs=2e-5)


def test_kolmogorov_rate_one_over_sqrt_n():
    # exponential law: sqrt(n) D_n -> 2 / (6 sqrt(2 pi)) = 0.13298 (Edgeworth), D_1 = Phi(-1)
    E = make_law("exponential")
    assert kolmogorov_distance(E, 1) == pytest.approx(stats.norm.cdf(-1), abs=1e-9)
    assert edgeworth_constant(2.0) == pytest.approx(0.1329808, abs=1e-7)
    assert math.sqrt(10_000) * kolmogorov_distance(E, 10_000) == pytest.approx(0.13298, abs=1e-4)
    # fair coin: sqrt(n) D_n -> 1 / sqrt(2 pi), half the central jump
    assert math.sqrt(10_000) * kolmogorov_distance(make_law("bernoulli"), 10_000) == pytest.approx(
        1 / math.sqrt(2 * math.pi), abs=1e-3)
    # Berry--Esseen holds
    for law in ("exponential", make_law("bernoulli", p=0.1), "poisson"):
        for n in (1, 10, 100, 1000):
            assert kolmogorov_distance(law, n) <= berry_esseen_bound(law, n)


def test_simulated_standardised_sums_and_empirical_distance():
    z = simulate_standardised("exponential", 50, 200_000, seed=2026)
    assert abs(z.mean()) < 0.01 and z.var() == pytest.approx(1, rel=0.02)
    d_emp = empirical_kolmogorov(z)
    assert abs(d_emp - kolmogorov_distance("exponential", 50)) < 0.006
    u = simulate_standardised("uniform", 12, 100_000, seed=3)        # no closed form: summed draws
    assert abs(u.mean()) < 0.02 and u.var() == pytest.approx(1, rel=0.03)
    assert empirical_kolmogorov(stats.norm.ppf((np.arange(1, 1001) - 0.5) / 1000)) == pytest.approx(0.0005)


def test_characteristic_function_of_standardised_sum():
    # the proof: Phi_Y(t / sqrt n)^n -> exp(-t^2 / 2)
    t = np.linspace(-3, 3, 13)
    for law in ("exponential", "bernoulli", "uniform", "poisson"):
        err = [np.max(np.abs(standardised_cf(law, t, n) - np.exp(-t ** 2 / 2))) for n in (10, 100, 1000)]
        assert err[0] > err[1] > err[2] and err[2] < 0.015      # O(1/sqrt n) for skewed laws
    # exponential: S_n ~ Gamma(n, 1), so the cf of S_n^* is exact
    n, tt = 25, 1.3
    exact = np.exp(-1j * tt * math.sqrt(n)) * (1 - 1j * tt / math.sqrt(n)) ** (-n)
    assert standardised_cf("exponential", tt, n) == pytest.approx(exact)


# ---------------------------------------------------------------- the election poll


def test_poll_numbers_of_the_notes():
    d = poll_numbers()
    assert d["exact"] == pytest.approx(0.016762, abs=1e-6)
    # the notes read Phi from a table at z cut to two decimals, 2.24, 2.04, 2.14: 0.0125, 0.0207, 0.0162
    assert round(d["eps1_table"], 4) == 0.0125 and math.floor(100 * d["eps1_z"]) == 224
    assert round(d["eps0_table"], 4) == 0.0207 and math.floor(100 * d["eps0_z"]) == 204
    assert round(d["eps_half_table"], 4) == 0.0162 and math.floor(100 * d["eps_half_z"]) == 214
    # unrounded values
    assert d["eps1"] == pytest.approx(0.01237, abs=1e-5)
    assert d["eps0"] == pytest.approx(0.02061, abs=1e-5)
    assert d["eps_half"] == pytest.approx(0.01604, abs=1e-5)
    # the continuity correction is the best of the three
    errs = {k: abs(d[k] - d["exact"]) for k in ("eps1", "eps0", "eps_half")}
    assert min(errs, key=errs.get) == "eps_half"
    assert matching_shift(50, 100, 0.4) == pytest.approx(0.4140, abs=1e-4)
    assert normal_tail(50, 100, 0.4, matching_shift(50, 100, 0.4)) == pytest.approx(d["exact"])
    assert normal_rule(100, 0.4) and not normal_rule(20, 0.1) and normal_rule(100, 0.1)
    assert normal_rule(50, 0.1, threshold=5) and not normal_rule(50, 0.1)


def test_histogram_example_and_correction_gain():
    # B(20, 1/2): P(X >= 12) = P(X > 11) = P(X > 11.5)
    assert binomial_tail(11, 20, 0.5) == pytest.approx(0.2517223, abs=1e-7)
    approx = [float(normal_tail(11, 20, 0.5, e)) for e in (0.0, 0.5, 1.0)]
    assert approx == pytest.approx([0.32736, 0.25117, 0.18555], abs=1e-5)
    assert binomial_normal_density(10, 20, 0.5) == pytest.approx(1 / math.sqrt(2 * math.pi * 5))
    # with the correction the error is O(1/n) for p = 1/2, O(1/sqrt n) for p = 0.1
    e = {n: (max_cdf_error(n, 0.5, 0.0), max_cdf_error(n, 0.5, 0.5)) for n in (25, 100, 400)}
    assert e[100][0] / e[400][0] == pytest.approx(2, rel=0.05)
    assert e[100][1] / e[400][1] == pytest.approx(4, rel=0.1)
    assert max_cdf_error(100, 0.1, 0.5) < max_cdf_error(100, 0.1, 0.0) / 4


# ---------------------------------------------------------------- the delta method


def test_delta_method_sqrt_poisson():
    # sqrt(n)(sqrt(Xbar_n) - sqrt(lambda)) -> N(0, 1/4) whatever lambda (variance stabilisation)
    g = STABILISERS["poisson"][1]
    for lam in (0.5, 4.0):
        assert delta_variance(1 / (2 * math.sqrt(lam)), lam) == pytest.approx(0.25)
        z = delta_sample(make_law("poisson", lam=lam), g, 2000, 200_000, seed=2026)
        assert z.var() == pytest.approx(0.25, rel=0.03)
        assert abs(z.mean()) < 0.01


def test_stabilising_transforms():
    sq = stabilising_transform(lambda t: math.sqrt(t), 1.0)
    assert sq(4.0) == pytest.approx(2.0)                                   # 2 sqrt(4) - 2 sqrt(1)
    arc = stabilising_transform(lambda t: math.sqrt(t * (1 - t)), 0.5)
    th = np.array([0.1, 0.3, 0.9])
    assert arc(th) == pytest.approx(2 * np.arcsin(np.sqrt(th)) - 2 * np.arcsin(math.sqrt(0.5)), abs=1e-8)
    lg = stabilising_transform(lambda t: t, 1.0)
    assert lg(math.e) == pytest.approx(1.0)
    # each g satisfies g'(theta)^2 sigma^2(theta) = limit variance
    for name, (law, g, v) in STABILISERS.items():
        L = make_law(law, p=0.3) if law == "bernoulli" else make_law(law, lam=3.0) if law == "poisson" else make_law(law, rate=2.0)
        h = 1e-6
        dg = (g(L.mean + h) - g(L.mean - h)) / (2 * h)
        assert delta_variance(dg, L.var) == pytest.approx(v, rel=1e-6)
        z = delta_sample(L, g, 3000, 100_000, seed=7)
        assert z.var() == pytest.approx(v, rel=0.04)


def test_second_order_bernoulli_variance_estimator():
    # g(x) = x (1 - x) at theta = 1/2: g' = 0, g'' = -2, sigma^2 = 1/4: limit -(1/4) chi^2(1)
    g = lambda x: x * (1 - x)
    w = second_order_sample(make_law("bernoulli", p=0.5), g, 400, 200_000, seed=2026)
    assert np.all(w <= 1e-12)
    # exactly -(1/4) (S_n^*)^2
    x = np.array([-1.0, -0.5, -0.1])
    lim = second_order_cdf(x, 0.25, -2.0)
    assert lim == pytest.approx(stats.chi2.sf(-4 * x, 1))
    emp = np.array([np.mean(w <= v) for v in x])
    assert np.max(np.abs(emp - lim)) < 0.06            # lattice effects at n = 400
    assert second_order_cdf(0.3, 1.0, 2.0) == pytest.approx(stats.gamma.cdf(0.3, 0.5, scale=2.0))  # Gamma(1/2, 1/(sigma^2 g''))
    assert second_order_cdf(-0.1, 0.0, 1.0) == 0 and second_order_cdf(0.1, 0.0, 1.0) == 1


def test_sample_means():
    m = sample_means("uniform", 30, 50_000, seed=1)
    assert abs(m.mean() - 0.5) < 0.002 and m.var() == pytest.approx(1 / 360, rel=0.03)


# ---------------------------------------------------------------- worked exercises


def test_poisson_partial_sum_and_ramanujan():
    n = np.array([10, 100, 1000, 10_000])
    a = poisson_partial_sum(n)
    direct = math.exp(-10) * sum(10 ** k / math.factorial(k) for k in range(11))
    assert a[0] == pytest.approx(direct, rel=1e-12)
    assert np.all(np.diff(a) < 0) and a[-1] == pytest.approx(0.5, abs=3e-3)
    assert np.sqrt(n[-1]) * (a[-1] - 0.5) == pytest.approx(RAMANUJAN_CONSTANT, rel=1e-3)
    assert RAMANUJAN_CONSTANT == pytest.approx(0.2659615, abs=1e-7)
    assert poisson_partial_sum(10_000, 1.0) == pytest.approx(stats.norm.cdf(1.0), abs=5e-3)


def test_variance_counterexample():
    assert variance_counterexample_var(np.array([1, 2, 10])) == pytest.approx([1, 1.5, 1.9])
    z = variance_counterexample_sample(100, 400_000, seed=2026)
    assert z.var() == pytest.approx(2 - 1 / 100, rel=0.03)
    # yet Z_n is close to N(0,1) in law: the spike sqrt(n) Y_n is nonzero with probability 1/n
    assert np.mean(np.abs(z) <= 1.96) == pytest.approx(0.95 - 0.01, abs=0.015)


def test_exam_problem():
    t = np.linspace(-5, 5, 11)
    assert exam_cf(t, 0.0) == pytest.approx(np.ones_like(t))
    for alpha in (1.0, 0.1, 0.01):
        x = exam_sample(alpha, 400_000, seed=5)
        for tt in (0.5, 2.0):
            assert abs(np.mean(np.cos(tt * x)) - exam_cf(tt, alpha)) < 0.005
    assert np.max(np.abs(exam_cf(t, 1e-4) - 1)) < 1e-6
    # moments: E|Y| = sqrt(2/pi), E Y^2 = 1, E|Y|^3 = 2 sqrt(2/pi)
    assert gaussian_abs_moment(np.array([1, 2, 3, 4])) == pytest.approx(
        [math.sqrt(2 / math.pi), 1, 2 * math.sqrt(2 / math.pi), 3])
    x = exam_sample(0.5, 400_000, seed=6)
    assert np.mean(x ** 2) == pytest.approx(exam_abs_moment(0.5, 2), rel=0.02)
    assert exam_abs_moment(0.5, 2) == pytest.approx(0.125)
    # half of the draws are exactly 0, the others are alpha Y
    assert np.mean(x == 0) == pytest.approx(0.5, abs=0.005)
    assert integrate.quad(lambda y: abs(y) ** 3 * stats.norm.pdf(y), -np.inf, np.inf)[0] == pytest.approx(
        gaussian_abs_moment(3))
