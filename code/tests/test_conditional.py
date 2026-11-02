"""Checks of the Chapter 11 code against the notes and the exercises."""
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import integrate, stats

from probnotes.conditional import (
    SUM_PAIRS,
    binned_conditional_mean,
    cell_average,
    check_conditional_expectation,
    conditional_density,
    conditional_expectation_given,
    conditional_mean,
    conditional_mean_from_density,
    conditional_pmf,
    defining_property_mc,
    factor_through,
    midpoints,
    partition_expectation,
    poisson_sum_conditional,
    poisson_sum_table,
    sample_exponential_sum,
    sample_poisson_sum,
    sample_sum,
    sigma_from_partition,
    two_dice,
    uniform_renewal_count,
)
from probnotes.sigma import is_sigma_algebra

# ---------------------------------------------------------------- discrete tables


def test_conditional_pmf_normalises_columns_and_rows():
    f = np.array([[0.1, 0.0, 0.2], [0.3, 0.0, 0.1], [0.1, 0.0, 0.2]])
    c = conditional_pmf(f, "y")
    assert np.allclose(c[:, [0, 2]].sum(axis=0), 1.0)
    assert np.isnan(c[:, 1]).all()                          # f_Y(y) = 0: left arbitrary
    assert np.allclose(c[:, 0], [0.2, 0.6, 0.2])
    r = conditional_pmf(f, "x", fill=0.0)
    assert np.allclose(r.sum(axis=1), 1.0)
    assert np.allclose(r[0], [1 / 3, 0, 2 / 3])
    with pytest.raises(ValueError):
        conditional_pmf([[-0.1, 1.1]])
    with pytest.raises(ValueError):
        conditional_pmf(f, "z")


def test_conditional_mean_by_hand():
    # X in {0, 1, 2}, Y in {0, 1, 2}: h(0) = (0*.1 + 1*.3 + 2*.1)/.5 = 1, h(2) = (.1 + .4)/.5 = 1
    f = np.array([[0.1, 0.0, 0.2], [0.3, 0.0, 0.1], [0.1, 0.0, 0.2]])
    h = conditional_mean(f, [0, 1, 2], "y")
    assert h[0] == pytest.approx(1.0) and h[2] == pytest.approx(1.0) and np.isnan(h[1])
    g = conditional_mean(f, [0, 1, 2], "x")                 # E[Y | X = x_i]
    assert g == pytest.approx([4 / 3, 0.5, 4 / 3])


@pytest.mark.parametrize("lam, mu", [(2.0, 3.0), (0.5, 4.0), (7.0, 1.0)])
def test_poisson_given_sum_is_binomial(lam, mu):
    # worked exercise of the notes: X | Z = m ~ B(m, lam / (lam + mu)), E[X | Z] = Z lam/(lam+mu)
    p = lam / (lam + mu)
    for m in (0, 1, 6, 15):
        assert np.allclose(poisson_sum_conditional(lam, mu, m), stats.binom.pmf(np.arange(m + 1), m, p),
                           atol=1e-13)
    t = poisson_sum_table(lam, mu, 60)
    assert np.allclose(t.sum(axis=0)[:41], stats.poisson.pmf(np.arange(41), lam + mu))
    h = conditional_mean(t, np.arange(61), "y")
    assert np.allclose(h[:41], p * np.arange(41), atol=1e-10)


# ---------------------------------------------------------------- finite probability spaces


def test_dice_exact_conditional_expectations():
    p, d1, d2 = two_dice()
    M, S = np.maximum(d1, d2), d1 + d2
    hM, eM = conditional_expectation_given(d1, M, p)
    assert hM == {m: Fraction(m * (3 * m - 1), 2 * (2 * m - 1)) for m in range(1, 7)}
    assert hM[6] == Fraction(51, 11)
    r = check_conditional_expectation(d1, eM, p, M)
    assert r == {"measurable": True, "max_gap": 0, "n_events": 64}
    hS, eS = conditional_expectation_given(d1, S, p)
    assert hS == {s: Fraction(s, 2) for s in range(2, 13)}
    assert check_conditional_expectation(d1, eS, p, S)["max_gap"] == 0
    assert check_conditional_expectation(d1, eS, p, S)["n_events"] == 2048
    # S/2 satisfies property 2) on sigma(M) but is not sigma(M)-measurable (Exercise ce-partition)
    half = np.array([Fraction(int(s), 2) for s in S], dtype=object)
    r = check_conditional_expectation(d1, half, p, M)
    assert r["measurable"] is False and r["max_gap"] == 0
    # a wrong candidate fails property 2)
    assert check_conditional_expectation(d1, eM + Fraction(1, 7), p, M)["max_gap"] == Fraction(1, 7)


def test_dice_variances_increase_with_information():
    p, d1, d2 = two_dice()
    M, N, S = np.maximum(d1, d2), np.minimum(d1, d2), d1 + d2

    def var(v):
        m = sum(v * p)
        return sum((v - m) ** 2 * p)

    eM = conditional_expectation_given(d1, M, p)[1]
    eS = conditional_expectation_given(d1, S, p)[1]
    eMN = conditional_expectation_given(d1, np.column_stack([M, N]), p)[1]
    assert var(eM) == Fraction(134587, 124740)
    assert var(eS) == Fraction(35, 24) and list(eMN) == list(eS)
    assert var(np.array([Fraction(int(a)) for a in d1], dtype=object)) == Fraction(35, 12)


def test_sigma_from_partition_is_a_sigma_algebra():
    labels = [0, 0, 1, 2, 2, 2, 3]
    G = sigma_from_partition(labels)
    assert len(G) == 16
    assert is_sigma_algebra(G, range(7))
    assert frozenset({3, 4, 5}) in G and frozenset({0, 3}) not in G


def test_factor_through_is_doob_dynkin_on_a_finite_space():
    p, d1, d2 = two_dice()
    S, T = d1 + d2, np.abs(d1 - d2)
    h = factor_through((d1 + d2) ** 2, S)
    assert h == {s: s * s for s in range(2, 13)}
    with pytest.raises(ValueError, match=r"Y = 4"):
        factor_through(S, d1 * d2)
    h = factor_through(np.maximum(d1, d2), np.column_stack([S, T]))
    assert len(h) == 21 and all(v == (k[0] + k[1]) // 2 for k, v in h.items())
    with pytest.raises(ValueError, match=r"\(3, 1\)"):
        factor_through(d1, np.column_stack([S, T]))


def test_partition_expectation_float_matches_exact_cell_averages():
    # Omega = (0,1) with Lebesgue measure, X = 1 + sin(3 pi w) + w (the figure of the notes)
    n = 40_000
    w = midpoints(0, 1, n)
    x = 1 + np.sin(3 * np.pi * w) + w
    xhat = partition_expectation(x, np.full(n, 1 / n), np.floor(w * 4))
    for j in range(4):
        a, b = j / 4, (j + 1) / 4
        exact = 4 * integrate.quad(lambda t: 1 + np.sin(3 * np.pi * t) + t, a, b)[0]
        assert xhat[(w >= a) & (w < b)] == pytest.approx(exact, abs=1e-8)
    # exact path agrees with the float path
    p, d1, d2 = two_dice()
    M = np.maximum(d1, d2)
    ex = partition_expectation(d1, p, M)
    fl = partition_expectation(d1.astype(float), np.full(36, 1 / 36), M)
    assert np.allclose(fl, [float(v) for v in ex])


# ---------------------------------------------------------------- densities on a grid


def test_conditional_means_from_densities():
    f = lambda x, y: np.where((0 < x) & (x < y) & (y < 1), 8 * x * y, 0.0)     # Exercise ce-joint-density
    ys = np.array([0.2, 0.5, 0.9])
    assert conditional_mean_from_density(f, ys, 0, 1, 20_000) == pytest.approx(2 * ys / 3, abs=2e-4)
    g = lambda y, x: f(x, y)                                                   # roles swapped
    xs = np.array([0.1, 0.5, 0.8])
    assert conditional_mean_from_density(g, xs, 0, 1, 20_000) == pytest.approx(
        2 / 3 * (1 + xs + xs ** 2) / (1 + xs), abs=2e-4)
    # P_{X|Y=1/2}((-inf, 1/4]) = 1/4 and P_{Y|X=1/2}([3/4, inf)) = 7/12
    x, c = conditional_density(f, 0.5, 0, 1, 20_000)
    assert (c[x <= 0.25, 0].sum() / 20_000) == pytest.approx(0.25, abs=1e-3)
    y, c = conditional_density(g, 0.5, 0, 1, 20_000)
    assert (c[y >= 0.75, 0].sum() / 20_000) == pytest.approx(7 / 12, abs=1e-3)
    # exponential pair: f_{X|Z=z} = 1/z on (0, z), and nothing where f_Z = 0
    fe = lambda x, z: np.where((x > 0) & (z > x), np.exp(-x) * np.exp(-(z - x)), 0.0)
    x, c = conditional_density(fe, [2.0, -1.0], 0, 4, 8000)
    assert np.allclose(c[(x > 0.01) & (x < 1.99), 0], 0.5)
    assert np.isnan(c[:, 1]).all()
    assert conditional_mean_from_density(fe, [0.5, 3.0], 0, 4, 8000) == pytest.approx([0.25, 1.5], abs=1e-3)


# ---------------------------------------------------------------- Monte Carlo


def test_binned_exponential_means_and_defining_property():
    # Exercise ce-binning: n from the variance bound, exact cell targets E[Z 1{Z in I}] / (2 P(Z in I))
    edges = np.round(np.arange(0, 3.0001, 0.2), 10)
    P = stats.gamma.cdf(edges[1:], 2) - stats.gamma.cdf(edges[:-1], 2)
    n = math.ceil(np.max(25 * np.arange(1, 16) ** 2 / P))
    assert n == 176_167
    x, z = sample_exponential_sum(n, 1.0, seed=2026)
    b = binned_conditional_mean(z, x, edges)
    G = lambda t: -(t * t + 2 * t + 2) * np.exp(-t)
    target = (G(edges[1:]) - G(edges[:-1])) / P / 2
    assert target[0] == pytest.approx(0.065541, abs=1e-6)
    assert np.max(np.abs(b.mean_x - target) / b.se_x) < 4.0
    assert np.all(b.se_x <= 0.02)
    assert abs(b.mean_x[0] - b.centres[0] / 2) / b.se_x[0] > 10       # binning bias in the first bin
    ca = cell_average(z, x, edges)
    k = np.searchsorted(edges, z[:5], side="right") - 1
    assert np.allclose(ca[:5][k < 15], b.mean_x[k[k < 15]])
    good = defining_property_mc(x, z / 2, z, [(-1, 1), (1, 2), (2, np.inf)])
    bad = defining_property_mc(x, z / 3, z, [(-1, 1), (1, 2), (2, np.inf)])
    assert np.all(np.abs(good.mean) < 4 * good.se)
    exact_bad = [(2 - 5 / math.e) / 6, (5 / math.e - 10 / math.e ** 2) / 6, 10 / (6 * math.e ** 2)]
    assert np.allclose(exact_bad, [0.026767, 0.081007, 0.225559], atol=1e-6)
    assert np.all(np.abs(bad.mean - exact_bad) < 4 * bad.se) and np.all(bad.mean > 20 * bad.se)


def test_gamma_pair_exact_bin_targets():
    # Exercise ce-binning (e): X ~ Gamma(2,1), Y ~ Exp(1), Z ~ Gamma(3,1), E[X | Z] = 2Z/3
    edges = np.round(np.arange(0, 3.0001, 0.2), 10)
    target = 2 / 3 * 3 * np.diff(stats.gamma.cdf(edges, 4)) / np.diff(stats.gamma.cdf(edges, 3))
    assert target[0] == pytest.approx(0.098983, abs=1e-6) and target[-1] == pytest.approx(1.9326, abs=1e-4)
    x, z = sample_sum("gamma", 176_167, seed=2026)
    b = binned_conditional_mean(z, x, edges)
    assert b.count[0] == 195
    assert np.max(np.abs(b.mean_x - target) / b.se_x) == pytest.approx(1.644, abs=1e-3)
    assert np.all(b.se_x <= 0.02)

@pytest.mark.parametrize("kind", sorted(SUM_PAIRS))
def test_sum_pairs_have_the_stated_conditional_expectation(kind):
    x, z = sample_sum(kind, 200_000, seed=11)
    h = SUM_PAIRS[kind][1]
    lo, hi = np.quantile(z, [0.02, 0.98])
    b = binned_conditional_mean(z, x, np.linspace(lo, hi, 13))
    assert np.max(np.abs(b.mean_x - h(b.mean_z)) / b.se_x) < 4.5     # h is linear
    with pytest.raises(ValueError):
        sample_sum("cauchy", 10)


def test_poisson_simulation_numbers():
    # Exercise ce-poisson-simulation
    x, z = sample_poisson_sum(10**6, 2.0, 3.0, seed=2026)
    sel = x[z == 5]
    assert sel.size == 176_054
    th = stats.binom.pmf(np.arange(6), 5, 0.4)
    emp = np.bincount(sel, minlength=6)[:6] / sel.size
    assert np.max(np.abs(emp - th) / np.sqrt(th * (1 - th) / sel.size)) < 4
    exact = 0.4 * 5 * stats.poisson.cdf(3, 5)
    assert exact == pytest.approx(0.530052, abs=1e-6)
    est = np.mean(x * (z <= 4))
    assert abs(est - exact) < 4 * np.std(x * (z <= 4)) / 1e3
    assert np.allclose([z[x == k].mean() for k in range(6)], np.arange(6) + 3.0, atol=0.03)


def test_renewal_count_and_sample_size():
    # Exercise ce-renewal: E[N(x)] = e^x, Var N(1) = 3e - e^2, n for half-width 0.002
    v = 3 * math.e - math.e ** 2
    assert v == pytest.approx(0.765789, abs=1e-6)
    n = math.ceil((1.959964 / 0.002) ** 2 * v)
    assert n == 735_438
    N1 = uniform_renewal_count(n, 1.0, seed=2026)
    se = N1.std(ddof=1) / math.sqrt(n)
    assert N1.mean() == pytest.approx(2.719154, abs=1e-6)
    assert abs(N1.mean() - math.e) < 1.96 * se
    assert N1.min() >= 2                                  # S_1 = U_1 <= 1 always
    N5 = uniform_renewal_count(10**5, 0.5, seed=2026)
    assert abs(N5.mean() - math.exp(0.5)) < 4 * N5.std(ddof=1) / math.sqrt(1e5)
    assert math.fsum((2 * k + 1) / math.factorial(k) for k in range(30)) == pytest.approx(3 * math.e)


def test_triangular_density_simulation():
    # Exercise ce-joint-density (d): Y = U^(1/4), X = Y sqrt(V)
    rng = np.random.default_rng(2026)
    n = 10**6
    U, V = rng.random(n), rng.random(n)
    Y = U ** 0.25
    X = Y * np.sqrt(V)
    xs = np.sort(X)
    F = 2 * xs ** 2 - xs ** 4
    ks = max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))
    assert ks == pytest.approx(0.00062, abs=1e-5)
    assert ks < math.sqrt(math.log(2 / 0.01) / (2 * n))
    edges = np.arange(0, 1.0001, 0.05)
    b = binned_conditional_mean(X, Y, edges)
    num = [integrate.quad(lambda t: 8 / 3 * t * (1 - t ** 3), a, c)[0] for a, c in zip(edges[:-1], edges[1:])]
    den = [integrate.quad(lambda t: 4 * t * (1 - t ** 2), a, c)[0] for a, c in zip(edges[:-1], edges[1:])]
    assert np.max(np.abs(b.mean_x - np.array(num) / np.array(den)) / b.se_x) < 4
