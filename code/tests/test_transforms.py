"""Checks of the Chapter 8 code against the notes and the exercises."""
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import integrate, special, stats

from probnotes.transforms import (
    Branch,
    affine_branch,
    affine_density,
    annulus_marginal_density,
    conditional_survival,
    discrete_image_law,
    erlang_cdf,
    gamma_cdf,
    gamma_moment,
    gamma_moments,
    gamma_pdf,
    gamma_survival,
    hazard_rate,
    histogram_check,
    ks_distance,
    monotone_branch,
    multinomial_coefficient,
    multinomial_marginal,
    multinomial_pmf,
    multinomial_support,
    poisson_arrivals,
    poisson_conditionals_constant,
    poisson_conditionals_table,
    poisson_counts,
    poisson_path,
    poisson_pmf,
    sample_annulus,
    sample_multinomial,
    square_branches,
    square_density,
    transformed_density,
)

# ------------------------------------------------------------------ the multinomial law

URN = [0.5, 0.2, 0.2]          # black, red, white; green has probability 0.1


def test_multinomial_coefficient_is_the_telescoping_product():
    assert multinomial_coefficient([5, 3, 1, 1]) == 5040
    assert multinomial_coefficient([5, 3, 2]) == 2520
    assert multinomial_coefficient([2] * 6) == math.factorial(12) // 2**6
    # C(n, n1) C(n - n1, n2) ... = n! / (n1! ... nk!)
    for counts in ([3, 0, 4, 2], [1, 1, 1, 1, 1], [7]):
        n = sum(counts)
        assert multinomial_coefficient(counts) == math.factorial(n) // math.prod(math.factorial(c) for c in counts)


def test_urn_numbers_of_the_notes():
    # (a) 5 black, 3 red, 1 white (hence 1 green) in 10 draws with replacement
    assert multinomial_pmf([5, 3, 1], 10, URN) == pytest.approx(0.0252, abs=1e-12)
    # (b) 5 black and 3 red: the sum left to the reader, and the merged trinomial
    parts = multinomial_pmf([[5, 3, 0], [5, 3, 1], [5, 3, 2]], 10, URN)
    assert parts == pytest.approx([0.0063, 0.0252, 0.0252], abs=1e-12)
    assert parts.sum() == pytest.approx(0.0567, abs=1e-12)
    assert multinomial_pmf([5, 3], 10, [0.5, 0.2]) == pytest.approx(0.0567, abs=1e-12)
    # (c) 5 black: the binomial marginal
    assert multinomial_pmf([5], 10, [0.5]) == pytest.approx(252 / 1024, abs=1e-14)
    assert round(252 / 1024, 4) == 0.2461
    # Exercise: P(X_2 = 3 | X_1 = 5) = 0.0567 / 0.2461... = B(5, 0.4)(3) = 0.2304
    assert parts.sum() / (252 / 1024) == pytest.approx(stats.binom.pmf(3, 5, 0.4), rel=1e-12)
    assert stats.binom.pmf(3, 5, 0.4) == pytest.approx(0.2304, abs=1e-12)


def test_multinomial_is_a_law_with_binomial_marginals():
    n, p = 7, [0.1, 0.25, 0.4]
    pts = multinomial_support(n, 3)
    assert len(pts) == math.comb(n + 3, 3)
    assert multinomial_pmf(pts, n, p).sum() == pytest.approx(1.0, abs=1e-12)
    for i in range(3):
        assert np.allclose(multinomial_marginal(n, p, i), stats.binom.pmf(np.arange(n + 1), n, p[i]), atol=1e-13)
    # outside X(Omega) the density vanishes
    assert multinomial_pmf([[4, 4, 0], [-1, 2, 2], [1.5, 1, 1]], n, p).tolist() == [0.0, 0.0, 0.0]
    # N = 1 is the binomial law; p_{N+1} = 0 uses 0^0 = 1
    assert np.allclose(multinomial_pmf(np.arange(n + 1)[:, None], n, [0.3]), stats.binom.pmf(np.arange(n + 1), n, 0.3))
    assert multinomial_pmf([2, 5], 7, [0.5, 0.5]) == pytest.approx(math.comb(7, 2) / 2**7)
    with pytest.raises(ValueError):
        multinomial_pmf([1, 1], 3, [0.7, 0.6])


def test_sampled_multinomial_matches_the_density():
    x = sample_multinomial(10, URN, 200_000, rng=2026)
    assert x.shape == (200_000, 3)
    freq = np.mean((x == [5, 3, 1]).all(axis=1))
    assert abs(freq - 0.0252) < 5 * math.sqrt(0.0252 * 0.9748 / 200_000)
    # covariance -n p_i p_j (exercise)
    assert np.cov(x[:, 0], x[:, 1])[0, 1] == pytest.approx(-10 * 0.5 * 0.2, abs=0.02)


def test_dice_numbers_of_the_exercise():
    each_twice = Fraction(multinomial_coefficient([2] * 6), 6**12)
    assert each_twice == Fraction(1925, 559872)
    assert float(each_twice) == pytest.approx(0.003438, abs=5e-7)
    eq = sum(multinomial_pmf([k, k], 12, [1 / 6, 1 / 6]) for k in range(7))
    assert (1 - eq) / 2 == pytest.approx(0.4001594, abs=1e-7)

# ------------------------------------------------------------------ the Gamma law


def test_gamma_function_values_of_the_notes():
    assert special.gamma(1) == 1
    for a in (0.3, 1.7, 4.2):
        assert special.gamma(a + 1) == pytest.approx(a * special.gamma(a))
    for n in range(1, 8):
        assert special.gamma(n) == pytest.approx(math.factorial(n - 1))
    assert special.gamma(0.5) == pytest.approx(math.sqrt(math.pi))
    assert special.gamma(1.5) == pytest.approx(math.sqrt(math.pi) / 2)
    assert special.gamma(2.5) == pytest.approx(3 * math.sqrt(math.pi) / 4)
    for n in range(6):   # exercise: Gamma(n + 1/2) = (2n)! sqrt(pi) / (4^n n!)
        assert special.gamma(n + 0.5) == pytest.approx(math.factorial(2 * n) * math.sqrt(math.pi) / (4**n * math.factorial(n)))


@pytest.mark.parametrize("alpha,lam", [(0.5, 1.0), (1.0, 2.0), (2.0, 0.7), (3.5, 3.0)])
def test_gamma_density_constant_and_moments(alpha, lam):
    tot = integrate.quad(lambda x: float(gamma_pdf(x, alpha, lam)), 0, math.inf)[0]
    assert tot == pytest.approx(1.0, abs=1e-7)
    m1 = integrate.quad(lambda x: x * float(gamma_pdf(x, alpha, lam)), 0, math.inf)[0]
    m2 = integrate.quad(lambda x: x * x * float(gamma_pdf(x, alpha, lam)), 0, math.inf)[0]
    mean, var = gamma_moments(alpha, lam)
    assert mean == alpha / lam and var == alpha / lam**2
    assert m1 == pytest.approx(mean, rel=1e-7)
    assert m2 == pytest.approx((alpha + 1) * alpha / lam**2, rel=1e-7)
    assert m2 - m1**2 == pytest.approx(var, rel=1e-6)
    assert gamma_moment(1, alpha, lam) == pytest.approx(mean)
    assert gamma_moment(2, alpha, lam) == pytest.approx((alpha + 1) * alpha / lam**2)
    assert gamma_moment(-alpha, alpha, lam) == math.inf
    # density agrees with scipy (lam is a rate: scipy's scale is 1 / lam)
    x = np.array([-1.0, 0.0, 0.3, 1.0, 4.0])
    assert np.allclose(gamma_pdf(x, alpha, lam), np.where(x > 0, stats.gamma.pdf(x, alpha, scale=1 / lam), 0.0))
    assert np.allclose(gamma_cdf(x, alpha, lam), stats.gamma.cdf(x, alpha, scale=1 / lam))
    assert np.allclose(gamma_survival(x, alpha, lam), 1 - gamma_cdf(x, alpha, lam))


def test_gamma_special_cases_and_scaling():
    x = np.linspace(0.01, 5, 50)
    assert np.allclose(gamma_pdf(x, 1.0, 2.0), 2.0 * np.exp(-2.0 * x))           # Exp(lam)
    assert np.allclose(gamma_pdf(x, 0.5, 0.5), np.exp(-x / 2) / np.sqrt(2 * np.pi * x))   # chi^2(1)
    # c X ~ Gamma(alpha, lam / c): lam is a rate, 1 / lam a scale (footnote and exercise)
    c, alpha, lam = 2.5, 1.7, 0.8
    assert np.allclose(affine_density(x, lambda t: gamma_pdf(t, alpha, lam), c, 0.0), gamma_pdf(x, alpha, lam / c))
    assert gamma_moment(-1, 3.0, 2.0) == pytest.approx(2.0 / (3.0 - 1))


def test_ageing_remark():
    s = np.array([0.5, 1.0, 5.0])
    t = 1.0
    assert np.allclose(conditional_survival(s, t, 1.0, 1.3), math.exp(-1.3))      # memoryless
    assert (conditional_survival(s, t, 3.0, 1.0) < gamma_survival(t, 3.0, 1.0)).all()   # wear-out
    assert (conditional_survival(s, t, 0.5, 1.0) > gamma_survival(t, 0.5, 1.0)).all()   # rejuvenation
    # numbers of the exercise: P(X > s + 1 | X > s), lam = 1, s = 0, 1, 5
    expect = {0.5: [0.1573, 0.2893, 0.3399], 1.0: [0.3679] * 3, 3.0: [0.9197, 0.7358, 0.4971]}
    for alpha, vals in expect.items():
        got = conditional_survival(np.array([0.0, 1.0, 5.0]), 1.0, alpha, 1.0)
        assert np.allclose(got, vals, atol=5e-5)
    # hazard rate: monotone, by 1 / h(x) = int_0^inf (1 + u/x)^(alpha - 1) e^(-lam u) du
    xs = np.linspace(0.1, 20, 80)
    assert (np.diff(hazard_rate(xs, 3.0, 1.0)) > 0).all()
    assert (np.diff(hazard_rate(xs, 0.5, 1.0)) < 0).all()
    assert np.allclose(hazard_rate(xs, 1.0, 1.7), 1.7)
    for alpha in (0.5, 3.0):
        for x0 in (0.5, 2.0):
            inv = integrate.quad(lambda u: (1 + u / x0) ** (alpha - 1) * math.exp(-u), 0, math.inf)[0]
            assert float(hazard_rate(np.array(x0), alpha, 1.0)) == pytest.approx(1 / inv, rel=1e-9)

# ------------------------------------------------------------------ Erlang and Poisson


@pytest.mark.parametrize("n,lam", [(1, 1.0), (3, 2.0), (5, 2.0), (10, 6.0)])
def test_erlang_cdf_is_the_gamma_cdf_and_a_poisson_tail(n, lam):
    t = np.array([-1.0, 0.0, 0.2, 1.0, 2.0, 7.0])
    F = erlang_cdf(t, n, lam)
    assert np.allclose(F, gamma_cdf(t, n, lam), atol=1e-14)
    assert np.allclose(F[t > 0], 1 - stats.poisson.cdf(n - 1, lam * t[t > 0]))    # P(N_t >= n)
    # the recursion F_n = F_{n-1} - (lam t)^{n-1} e^{-lam t} / (n-1)! of the integration by parts
    if n > 1:
        assert np.allclose(F, erlang_cdf(t, n - 1, lam) - np.where(t > 0, poisson_pmf(n - 1, lam * np.maximum(t, 0)), 0))
    assert np.allclose(erlang_cdf(t, 1, lam), np.where(t > 0, 1 - np.exp(-lam * t), 0.0))


def test_arrival_numbers_of_the_exercise():
    assert float(erlang_cdf(1.0, 10, 6.0)) == pytest.approx(0.083924, abs=5e-7)
    q = stats.gamma.ppf(0.9, 10, scale=1 / 6)
    assert q == pytest.approx(2.3677, abs=5e-5)
    assert float(erlang_cdf(2.0, 5, 2.0)) == pytest.approx(0.371163, abs=5e-7)
    lam, mu = 6.0, 4.0
    num = integrate.quad(lambda t: math.exp(-lam * t) * mu**2 * t * math.exp(-mu * t), 0, math.inf)[0]
    assert 1 - num == pytest.approx(0.84, abs=1e-12)


def test_poisson_process_simulation():
    rng = np.random.default_rng(2026)
    T = poisson_arrivals(2.0, 5, 100_000, rng)
    assert (np.diff(T, axis=1) > 0).all()
    d = ks_distance(T[:, 4], lambda t: erlang_cdf(t, 5, 2.0))
    assert d == pytest.approx(0.0038604, abs=1e-6)          # number quoted in the solutions
    assert d < 1.36 / math.sqrt(100_000)
    N = poisson_counts(2.0, 3.0, 100_000, rng)
    freq = np.bincount(N, minlength=30)[:30] / 100_000
    assert np.max(np.abs(freq - poisson_pmf(np.arange(30), 6.0))) == pytest.approx(0.0016669, abs=1e-6)
    path = poisson_path(3.0, 10.0, rng=1)
    assert (np.diff(path) > 0).all() and path.max() <= 10.0


def test_poisson_counts_edge_cases():
    # N_0 = 0, and large means are counted correctly
    N = poisson_counts(1.0, 0.0, 5, rng=3)
    assert (N == 0).all()
    assert poisson_counts(50.0, 1.0, 2000, rng=4).mean() == pytest.approx(50.0, abs=1.0)

# ------------------------------------------------------------------ the worked exercises


def test_joint_density_with_poisson_marginals():
    nu, mu = 2.0, 3.0
    c, f = poisson_conditionals_table(nu, mu, 1.0, 60, 60)
    assert c == pytest.approx(math.exp(-(nu + mu)), rel=1e-12)
    assert np.allclose(f, np.outer(stats.poisson.pmf(np.arange(61), nu), stats.poisson.pmf(np.arange(61), mu)))
    # alpha < 1: marginals c nu^n / n! e^{mu alpha^n}, conditional law P(nu alpha^m)
    alpha = 0.5
    c, f = poisson_conditionals_table(nu, mu, alpha, 80, 80)
    assert f.sum() == pytest.approx(1.0, abs=1e-12)
    n = np.arange(81)
    assert np.allclose(f.sum(axis=1), c * nu**n / special.factorial(n) * np.exp(mu * alpha**n))
    assert np.allclose(f.sum(axis=0), c * mu**n / special.factorial(n) * np.exp(nu * alpha**n))
    for m in (0, 1, 4):
        cond = f[:, m] / f[:, m].sum()
        assert np.allclose(cond, stats.poisson.pmf(n, nu * alpha**m), atol=1e-14)
    # not independent: the table is not of rank one
    assert f[0, 0] * f[1, 1] == pytest.approx(alpha * f[0, 1] * f[1, 0])
    # numbers of the exercise (nu = mu = 2)
    rho = {}
    for a in (0.75, 0.5, 0.25):
        c, F = poisson_conditionals_table(2.0, 2.0, a, 80, 80)
        fx = F.sum(axis=1)
        EX, VX = n @ fx, (n**2) @ fx - (n @ fx) ** 2
        rho[a] = (n @ F @ n - EX**2) / VX
    assert rho[0.5] == pytest.approx(-0.49875, abs=5e-6)
    assert poisson_conditionals_constant(2.0, 2.0, 0.5) == pytest.approx(0.0526314, abs=5e-8)
    with pytest.raises(ValueError):
        poisson_conditionals_constant(1.0, 1.0, 1.5)


def test_annulus_marginal_and_moments():
    x = np.array([-4.0, -3.0, -2.0, -0.5, 0.0, 1.0, 2.5, 3.0])
    expect = np.where(np.abs(x) > 3, 0.0,
                      np.where(np.abs(x) <= 1, (np.sqrt(np.clip(9 - x**2, 0, None)) - np.sqrt(np.clip(1 - x**2, 0, None))) / (4 * np.pi),
                               np.sqrt(np.clip(9 - x**2, 0, None)) / (4 * np.pi)))
    assert np.allclose(annulus_marginal_density(x), expect)
    assert integrate.quad(lambda t: float(annulus_marginal_density(t)), -3, 3, points=[-1, 1])[0] == pytest.approx(1.0)
    m2 = integrate.quad(lambda t: t * t * float(annulus_marginal_density(t)), -3, 3, points=[-1, 1])[0]
    assert m2 == pytest.approx(2.5)
    pts = sample_annulus(400_000, rng=2026)
    r2 = (pts**2).sum(axis=1)
    assert ((r2 >= 1) & (r2 <= 9)).all()
    X, Y = pts[:, 0], pts[:, 1]
    assert abs(np.mean(X * Y)) < 0.02                          # uncorrelated
    assert np.cov(X**2, Y**2)[0, 1] == pytest.approx(-59 / 24, abs=0.05)   # but dependent
    assert np.mean((X > 2.5) & (Y > 2.5)) == 0.0 and np.mean(X > 2.5) > 0

# ------------------------------------------------------------------ transformations


def test_discrete_example_of_the_notes():
    ys, fy = discrete_image_law([-2, -1, 0, 1], [0.25] * 4, lambda x: x**2)
    assert ys.tolist() == [0, 1, 4] and fy.tolist() == [0.25, 0.5, 0.25]


def test_square_of_a_gaussian_is_chi_square_one():
    y = np.array([-1.0, 0.0, 0.01, 0.5, 1.0, 3.0, 9.0])
    assert np.allclose(square_density(y, stats.norm.pdf), gamma_pdf(y, 0.5, 0.5))
    assert np.allclose(square_density(y, stats.norm.pdf), np.where(y > 0, stats.chi2.pdf(y, 1), 0.0))
    # Gamma(1/2) = sqrt(pi) from the comparison of the constants
    assert 0.5**0.5 / special.gamma(0.5) == pytest.approx(1 / math.sqrt(2 * math.pi))
    # simulation (Exercise): KS distance and histogram, seed 2026
    Z = np.random.default_rng(2026).standard_normal(1_000_000)
    d = ks_distance(Z**2, lambda t: gamma_cdf(t, 0.5, 0.5))
    assert d == pytest.approx(0.00076607, abs=1e-7) and d < 1.36e-3
    h, e, se = histogram_check(Z**2, lambda t: gamma_cdf(t, 0.5, 0.5), np.linspace(0, 4, 41))
    assert np.max(np.abs(h - e) / se) < 4


def test_affine_gaussian():
    y = np.linspace(-5, 5, 41)
    for sigma, mu in ((2.0, 1.0), (-0.5, 3.0)):
        assert np.allclose(affine_density(y, stats.norm.pdf, sigma, mu), stats.norm.pdf(y, mu, abs(sigma)))
    with pytest.raises(ValueError):
        affine_branch(0.0, 1.0)


def test_piecewise_monotone_examples():
    fx = lambda x: np.where((x > -1) & (x < 2), 1 / 3, 0.0)
    br = [monotone_branch(lambda x: x**2, -1.0, 0.0), monotone_branch(lambda x: x**2, 0.0, 2.0)]
    y = np.array([0.04, 0.25, 0.81, 1.5, 3.9, 4.5])
    closed = np.where(y < 1, 1 / (3 * np.sqrt(y)), np.where(y < 4, 1 / (6 * np.sqrt(y)), 0.0))
    assert np.allclose(transformed_density(y, fx, br), closed, rtol=1e-6)
    # closed-form branches give the same thing as the numerical inversion
    assert np.allclose(transformed_density(y, fx, square_branches()), closed)
    # sin of a uniform angle on (0, 2 pi): the arcsine law, three pieces
    fu = lambda x: np.where((x > 0) & (x < 2 * math.pi), 1 / (2 * math.pi), 0.0)
    brs = [monotone_branch(np.sin, 0, math.pi / 2, dg=np.cos), monotone_branch(np.sin, math.pi / 2, 3 * math.pi / 2),
           monotone_branch(np.sin, 3 * math.pi / 2, 2 * math.pi)]
    y = np.array([-0.99, -0.3, 0.2, 0.7])
    assert np.allclose(transformed_density(y, fu, brs), 1 / (math.pi * np.sqrt(1 - y**2)), rtol=1e-6)
    # tan of a uniform angle on (-pi, pi): Cauchy, three pieces with images (0, inf), R, (-inf, 0)
    ft = lambda x: np.where(np.abs(x) < math.pi, 1 / (2 * math.pi), 0.0)
    dat = lambda y: 1 / (1 + y**2)
    brt = [Branch(lambda y: np.arctan(y) - math.pi, dat, 0.0, math.inf),
           Branch(np.arctan, dat, -math.inf, math.inf),
           Branch(lambda y: np.arctan(y) + math.pi, dat, -math.inf, 0.0)]
    y = np.array([-3.0, -0.2, 0.5, 10.0])
    assert np.allclose(transformed_density(y, ft, brt), 1 / (math.pi * (1 + y**2)))


def test_ks_distance_exact():
    # one point at 0.5 against the uniform law: sup |F_1 - F| = 0.5
    assert ks_distance([0.5], lambda x: np.clip(x, 0, 1)) == pytest.approx(0.5)
    assert ks_distance([0.25, 0.75], lambda x: np.clip(x, 0, 1)) == pytest.approx(0.25)
