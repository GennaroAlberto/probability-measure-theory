"""Checks of the Chapter 9 code against the notes and the exercises."""
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import integrate, stats

from probnotes.vectors import (
    beta_function,
    beta_mean_var,
    beta_pdf,
    binomial_pmf,
    convolution,
    convolve_densities_grid,
    convolve_pmfs,
    convolve_power,
    dkw_sample_size,
    empirical_cdf,
    fractional_part_cdf,
    fractional_part_pdf,
    gamma_beta_density,
    gamma_pdf,
    geometric_pair_pmf,
    geometric_pmf,
    half_integer_part_pmf,
    integer_part_pmf,
    joint_cdf_gap,
    ks_distance,
    laplace_pdf,
    laplace_uniform_sum_pdf,
    law_of_function,
    linear_transformed_density,
    midpoints,
    min_diff_pmf,
    min_max_cdf,
    numerical_jacobian_det,
    pascal_pmf,
    piecewise_transformed_density,
    poisson_bernoulli_moments,
    poisson_bernoulli_pmf,
    poisson_pmf,
    sample_correlation,
    sample_gamma_beta,
    sample_worked,
    sqrt_abs_laplace_pdf,
    sum_density,
    sum_difference_pdf,
    transformed_density,
    uniform_product_cdf,
    uniform_product_pdf,
    uniform_sum_cdf,
    uniform_sum_pdf,
)

# ------------------------------------------------------------------ discrete laws and convolution


def test_pmfs_match_scipy_and_sum_to_one():
    k = np.arange(0, 200)
    assert np.allclose(poisson_pmf(k, 3.7), stats.poisson.pmf(k, 3.7))
    assert np.allclose(binomial_pmf(k, 12, 0.35), stats.binom.pmf(k, 12, 0.35))
    assert np.allclose(geometric_pmf(k, 0.3), stats.geom.pmf(k, 0.3))            # G(p) on N*
    # Pascal(n, p): trial of the n-th success = n + number of failures (negative binomial)
    assert np.allclose(pascal_pmf(k, 4, 0.3), stats.nbinom.pmf(k - 4, 4, 0.3))
    assert pascal_pmf(np.arange(4, 400), 4, 0.3).sum() == pytest.approx(1.0)
    assert np.array_equal(pascal_pmf(np.arange(4), 4, 0.3), np.zeros(4))
    assert np.allclose(pascal_pmf(k, 1, 0.3), geometric_pmf(k, 0.3))             # Pascal(1, p) = G(p)
    assert poisson_pmf(-1, 2.0) == 0 and poisson_pmf(1.5, 2.0) == 0


def test_notable_discrete_sums():
    # Proposition rv-notable-discrete, 1)-4)
    k = np.arange(61)
    r, r0 = convolve_pmfs(poisson_pmf(k, 2.0), poisson_pmf(k, 3.5))
    assert r0 == 0 and np.allclose(r[:61], poisson_pmf(k, 5.5), atol=1e-15)
    r, _ = convolve_pmfs(binomial_pmf(np.arange(8), 7, 0.3), binomial_pmf(np.arange(5), 4, 0.3))
    assert np.allclose(r, binomial_pmf(np.arange(12), 11, 0.3), atol=1e-15)
    g = geometric_pmf(np.arange(1, 200), 0.3)
    r, r0 = convolve_pmfs(g, g, 1, 1)
    assert r0 == 2 and np.allclose(r[:150], pascal_pmf(np.arange(2, 152), 2, 0.3), atol=1e-15)
    p3, p2 = pascal_pmf(np.arange(3, 300), 3, 0.4), pascal_pmf(np.arange(2, 300), 2, 0.4)
    r, r0 = convolve_pmfs(p3, p2, 3, 2)
    assert r0 == 5 and np.allclose(r[:200], pascal_pmf(np.arange(5, 205), 5, 0.4), atol=1e-15)


def test_truncated_poisson_convolution_is_exact_up_to_the_truncation():
    # Exercise rv-convolution (a)
    k = np.arange(31)
    r, _ = convolve_pmfs(poisson_pmf(k, 2.0), poisson_pmf(k, 3.0))
    assert np.max(np.abs(r[:31] - poisson_pmf(k, 5.0))) < 1e-15
    assert r[40] < poisson_pmf(40, 5.0)                      # terms are missing beyond 30


def test_ten_dice():
    # Remark rv-code-discrete and Exercise rv-convolution (b): P(sum = 35) = 4395456 / 6^10
    r, r0 = convolve_power(np.full(6, 1 / 6), 10, 1)
    assert r0 == 10 and len(r) == 51 and r.sum() == pytest.approx(1.0)
    assert r[35 - r0] == pytest.approx(4_395_456 / 6 ** 10, rel=1e-12)
    assert Fraction(4_395_456, 6 ** 10) == Fraction(7631, 104976)
    assert int(np.argmax(r)) + r0 == 35
    assert np.allclose(r, r[::-1])                            # symmetric about 35
    # repeated squaring agrees with n - 1 plain convolutions
    q, q0 = np.full(6, 1 / 6), 1
    for _ in range(9):
        q, q0 = convolve_pmfs(q, np.full(6, 1 / 6), q0, 1)
    assert q0 == r0 and np.allclose(q, r)


def test_law_of_function_is_the_preimage_sum():
    # Proposition rv-discrete on two dice: the sum, the maximum, and the two-point law of the parity
    xs = ys = np.arange(1, 7)
    table = np.full((6, 6), 1 / 36)
    ts, ft = law_of_function(table, xs, ys, lambda x, y: x + y)
    r, r0 = convolve_power(np.full(6, 1 / 6), 2, 1)
    assert np.array_equal(ts, np.arange(2, 13)) and np.allclose(ft, r)
    ts, ft = law_of_function(table, xs, ys, np.maximum)
    assert np.allclose(ft, (2 * ts - 1) / 36)
    ts, ft = law_of_function(table, xs, ys, lambda x, y: (x + y) % 2)
    assert np.allclose(ft, [0.5, 0.5])


def test_worked_geometric_pair():
    # Worked exercise exa:rv-geometric-pair, with the joint density truncated far out
    a, M = 0.4, 80
    i = np.arange(M)
    table = geometric_pair_pmf(i[:, None], i[None, :], a)
    fX = table.sum(axis=1)
    assert np.allclose(fX[:20], (1 - a) * a ** i[:20])
    assert np.allclose(table, np.outer((1 - a) * a ** i, (1 - a) * a ** i))     # X and Y independent
    ts, fs = law_of_function(table, i, i, lambda x, y: x + y)
    assert np.allclose(fs[:40], (1 - a) ** 2 * a ** ts[:40] * (ts[:40] + 1))
    # (U, V) = (min, X - Y): f = (1-a)^2 a^(2k + |r|), marginals, independence
    U, V = np.minimum.outer(i, i), np.subtract.outer(i, i)
    for k in range(5):
        for r in range(-5, 6):
            mass = table[(U == k) & (V == r)].sum()
            assert mass == pytest.approx(float(min_diff_pmf(k, r, a)), rel=1e-12)
    r = np.arange(-30, 31)
    k = np.arange(30)
    fU = (1 - a ** 2) * a ** (2 * k)
    fV = (1 - a) / (1 + a) * a ** np.abs(r)
    assert np.allclose(min_diff_pmf(k[:, None], r[None, :], a).sum(axis=1), fU, rtol=1e-12)
    assert np.allclose(min_diff_pmf(k[:, None], r[None, :], a).sum(axis=0), fV, rtol=1e-10)
    assert np.allclose(min_diff_pmf(k[:, None], r[None, :], a), np.outer(fU, fV))
    assert fU.sum() == pytest.approx(1.0) and fV.sum() == pytest.approx(1.0)


def test_worked_poisson_bernoulli():
    lam, p = 2.0, 0.3
    k = np.arange(80)
    f = poisson_bernoulli_pmf(k, lam, p)
    assert f.sum() == pytest.approx(1.0)
    assert f[0] == pytest.approx(1 - p + p * math.exp(-lam))
    mean, var, cov, rho = poisson_bernoulli_moments(lam, p)
    assert np.dot(k, f) == pytest.approx(mean) and mean == pytest.approx(lam * p)
    assert np.dot(k ** 2, f) - mean ** 2 == pytest.approx(var)
    assert var == pytest.approx(lam * p + lam ** 2 * p - lam ** 2 * p ** 2)
    # cov(X, U) = E[X^2 Y] - E[X] E[U] = lam p, and rho = sqrt(p / (1 + lam (1 - p)))
    joint = np.outer(poisson_pmf(k, lam), [1 - p, p])
    exu = sum(joint[x, y] * x * x * y for x in k for y in (0, 1))
    assert exu - lam * mean == pytest.approx(cov) and cov == pytest.approx(lam * p)
    assert rho == pytest.approx(math.sqrt(p / (1 + lam * (1 - p))))
    x, u = sample_worked("poisson-bernoulli", 400_000, seed=1, lam=lam, p=p)
    assert np.mean(u == 0) == pytest.approx(f[0], abs=0.004)
    assert np.corrcoef(x, u)[0, 1] == pytest.approx(rho, abs=0.01)


# ------------------------------------------------------------------ the Jacobian formula


def test_numerical_jacobian():
    A = np.array([[2.0, 1.0], [-0.5, 3.0]])
    Ainv = np.linalg.inv(A)
    ginv = lambda u, v: (Ainv[0, 0] * u + Ainv[0, 1] * v, Ainv[1, 0] * u + Ainv[1, 1] * v)
    u, v = np.linspace(-3, 3, 7), np.linspace(1, 2, 7)
    assert np.allclose(numerical_jacobian_det(ginv, u, v), 1 / np.linalg.det(A))
    # g^{-1}(s, t) = (s t, s (1 - t)) of Proposition rv-gamma-beta: det J = -s
    s, t = np.linspace(0.5, 5, 6), np.linspace(0.1, 0.9, 6)
    assert np.allclose(numerical_jacobian_det(lambda s, t: (s * t, s * (1 - t)), s, t), -s, rtol=1e-7)
    # polar coordinates: det J = r
    r, th = np.linspace(0.2, 3, 5), np.linspace(0.3, 6, 5)
    assert np.allclose(numerical_jacobian_det(lambda r, th: (r * np.cos(th), r * np.sin(th)), r, th), r, rtol=1e-7)


def test_linear_transformation_of_a_gaussian_vector():
    # Example cor:rv-linear: W = A Z + b with Z standard Gaussian is N(b, A A^T)
    A, b = np.array([[1.0, 0.5], [-1.0, 2.0]]), np.array([0.3, -1.0])
    fZ = lambda x, y: np.exp(-(x * x + y * y) / 2) / (2 * np.pi)
    fW = linear_transformed_density(fZ, A, b)
    rng = np.random.default_rng(0)
    w = rng.normal(size=(20, 2)) * 2
    assert np.allclose(fW(w[:, 0], w[:, 1]), stats.multivariate_normal(b, A @ A.T).pdf(w))
    with pytest.raises(ValueError):
        linear_transformed_density(fZ, [[1, 2], [2, 4]])


def test_gamma_beta_by_the_jacobian_formula():
    # Proposition rv-gamma-beta: f_(S,T) = Gamma(a + b, lam)(s) Beta(a, b)(t), from Theorem rv-piecewise
    a, b, lam = 2.0, 3.0, 1.5
    fZ = lambda x, y: gamma_pdf(x, a, lam) * gamma_pdf(y, b, lam)
    fU = transformed_density(fZ, lambda s, t: (s * t, s * (1 - t)),
                             image=lambda s, t: (s > 0) & (t > 0) & (t < 1))
    S, T = np.meshgrid(np.linspace(0.1, 6, 9), np.linspace(0.05, 0.95, 9))
    assert np.allclose(fU(S, T), gamma_beta_density(S, T, a, b, lam), rtol=1e-6)
    exact = transformed_density(fZ, lambda s, t: (s * t, s * (1 - t)), jac_det=lambda s, t: -s,
                                image=lambda s, t: (s > 0) & (t > 0) & (t < 1))
    assert np.allclose(exact(S, T), gamma_beta_density(S, T, a, b, lam), rtol=1e-12)
    assert exact(-1.0, 0.5) == 0 and exact(1.0, 1.5) == 0


def test_piecewise_order_statistics():
    # Exercise rv-piecewise (b): (min, max) of i.i.d. Exp(1) has density 2 f(u) f(v) 1_{u<v}
    f = lambda x: gamma_pdf(x, 1, 1.0)
    fZ = lambda x, y: f(x) * f(y)
    above = lambda u, v: u < v
    fW = piecewise_transformed_density(fZ, [(lambda u, v: (u, v), lambda u, v: np.ones_like(u), above),
                                            (lambda u, v: (v, u), lambda u, v: -np.ones_like(u), above)])
    U, V = np.meshgrid(np.linspace(0.1, 3, 8), np.linspace(0.1, 3, 8))
    assert np.allclose(fW(U, V), 2 * f(U) * f(V) * (U < V))


def test_piecewise_squares_of_gaussians():
    # Exercise rv-piecewise (c): (X^2, Y^2) has the product of two chi^2(1) densities
    fZ = lambda x, y: np.exp(-(x * x + y * y) / 2) / (2 * np.pi)
    quad = (lambda u, v: (u > 0) & (v > 0))
    pieces = [(lambda u, v, sx=sx, sy=sy: (sx * np.sqrt(u), sy * np.sqrt(v)), None, quad)
              for sx in (1, -1) for sy in (1, -1)]
    fW = piecewise_transformed_density(fZ, pieces)
    U, V = np.meshgrid(np.linspace(0.2, 4, 6), np.linspace(0.3, 5, 6))
    chi = lambda x: gamma_pdf(x, 0.5, 0.5)
    assert np.allclose(fW(U, V), chi(U) * chi(V), rtol=1e-6)


def test_polar_coordinates_of_a_gaussian_pair():
    # Exercise rv-polar (b): (R, Theta) has density r e^{-r^2/2} / (2 pi)
    fZ = lambda x, y: np.exp(-(x * x + y * y) / 2) / (2 * np.pi)
    f = transformed_density(fZ, lambda r, th: (r * np.cos(th), r * np.sin(th)), jac_det=lambda r, th: r,
                            image=lambda r, th: (r > 0) & (th > 0) & (th < 2 * np.pi))
    r, th = np.meshgrid(np.linspace(0.1, 4, 7), np.linspace(0.1, 6.2, 7))
    assert np.allclose(f(r, th), r * np.exp(-r * r / 2) / (2 * np.pi))


# ------------------------------------------------------------------ densities of sums


def test_sum_density_and_convolution():
    # Proposition rv-sum-density: Exp + Exp = Gamma(2), and the Gaussian sum of Proposition rv-notable-sums a)
    f = lambda x, y: gamma_pdf(x, 1, 2.0) * gamma_pdf(y, 1, 2.0)
    s = np.array([0.3, 1.0, 2.5])
    assert np.allclose(sum_density(f, s, (0, 3), 30_000), gamma_pdf(s, 2, 2.0), rtol=1e-3)
    nx = lambda x: stats.norm.pdf(x, 1.0, 2.0)
    ny = lambda y: stats.norm.pdf(y, -0.5, 1.5)
    s = np.linspace(-6, 6, 13)
    assert np.allclose(convolution(nx, ny, s, (-20, 20), 8000), stats.norm.pdf(s, 0.5, 2.5), atol=1e-10)
    # Gamma(alpha) + Gamma(beta) = Gamma(alpha + beta) (Proposition rv-notable-sums c)
    s = np.linspace(0.5, 12, 10)
    g = convolution(lambda x: gamma_pdf(x, 2.5, 1.3), lambda y: gamma_pdf(y, 1.5, 1.3), s, (0, 12), 40_000)
    assert np.allclose(g, gamma_pdf(s, 4.0, 1.3), atol=1e-6)


def test_grid_convolution_rates():
    # Exercise rv-convolution (c): order h^2 for smooth densities, exact for Exp + Exp
    errs = []
    for h in (0.01, 0.005):
        x = midpoints(0, 40, int(round(40 / h)))
        s, fs = convolve_densities_grid(gamma_pdf(x, 2, 1.0), gamma_pdf(x, 3, 1.0), h)
        keep = s <= 30
        errs.append(np.max(np.abs(fs[keep] - gamma_pdf(s[keep], 5, 1.0))))
    assert errs[0] == pytest.approx(1.128e-6, rel=2e-3) and errs[0] / errs[1] == pytest.approx(4.0, rel=0.01)
    x = midpoints(0, 20, 2000)
    s, fs = convolve_densities_grid(gamma_pdf(x, 1, 1.0), gamma_pdf(x, 1, 1.0), 0.01)
    assert np.max(np.abs(fs[s < 15] - s[s < 15] * np.exp(-s[s < 15]))) < 1e-14


def test_uniform_sums():
    # Exercise rv-uniform-sums: Irwin-Hall densities and P(S_n <= 1) = 1/n!
    s = np.linspace(0, 3, 301)
    f3 = np.where(s <= 1, s ** 2 / 2, np.where(s <= 2, (-2 * s ** 2 + 6 * s - 3) / 2, (3 - s) ** 2 / 2))
    assert np.allclose(uniform_sum_pdf(s, 3), f3)
    tri = lambda x: uniform_sum_pdf(x, 2)
    unif = lambda y: ((y > 0) & (y < 1)).astype(float)
    assert np.allclose(convolution(tri, unif, s[1:-1], (0, 2), 20_000), f3[1:-1], atol=1e-4)
    for n in range(1, 7):
        assert uniform_sum_cdf(1.0, n) == pytest.approx(1 / math.factorial(n))
        assert integrate.quad(lambda t: uniform_sum_pdf(t, n), 0, n, limit=200)[0] == pytest.approx(1.0)
    assert sum(1 / math.factorial(n) for n in range(30)) == pytest.approx(math.e)   # E[N] = e


# ------------------------------------------------------------------ minimum and maximum


def test_min_max():
    lam, mu = 0.7, 1.9
    FX = lambda t: np.where(t > 0, -np.expm1(-lam * np.maximum(t, 0)), 0.0)
    FY = lambda t: np.where(t > 0, -np.expm1(-mu * np.maximum(t, 0)), 0.0)
    t = np.linspace(-1, 5, 25)
    Fmin, Fmax = min_max_cdf(FX, FY, t)
    assert np.allclose(Fmin, np.where(t > 0, 1 - np.exp(-(lam + mu) * np.maximum(t, 0)), 0))  # series: Exp(lam + mu)
    assert np.allclose(Fmax, FX(t) * FY(t))
    # comonotone pair X = Y: F_(X,Y)(t, t) = F_X(t), so min and max both have law F_X
    Fmin, Fmax = min_max_cdf(FX, FX, t, F_joint=lambda s, u: FX(np.minimum(s, u)))
    assert np.allclose(Fmin, FX(t)) and np.allclose(Fmax, FX(t))
    # parallel mean 1/lam + 1/mu - 1/(lam + mu); P(X < Y) = lam / (lam + mu)
    mean = integrate.quad(lambda x: 1 - FX(x) * FY(x), 0, np.inf)[0]
    assert mean == pytest.approx(1 / lam + 1 / mu - 1 / (lam + mu))
    pxy = integrate.quad(lambda x: lam * math.exp(-lam * x) * math.exp(-mu * x), 0, np.inf)[0]
    assert pxy == pytest.approx(lam / (lam + mu))
    # max and min of n uniforms are Beta(n, 1) and Beta(1, n)
    u = np.linspace(0.01, 0.99, 9)
    assert np.allclose(beta_pdf(u, 4, 1), 4 * u ** 3) and np.allclose(beta_pdf(u, 1, 4), 4 * (1 - u) ** 3)


# ------------------------------------------------------------------ Gamma and Beta


def test_beta_function_and_moments():
    for a, b in ((0.5, 0.5), (1.0, 3.0), (1.8, 1.5), (2.0, 3.0), (2.5, 1.5)):
        assert beta_function(a, b) == pytest.approx(integrate.quad(lambda t: t ** (a - 1) * (1 - t) ** (b - 1), 0, 1)[0],
                                                    rel=1e-7)
        m, v = beta_mean_var(a, b)
        assert (m, v) == pytest.approx(stats.beta(a, b).stats("mv"))
        assert integrate.quad(lambda t: beta_pdf(t, a, b), 0, 1)[0] == pytest.approx(1.0, rel=1e-6)
        # Exercise rv-beta-moments (b): E[T^k] = prod (a + j) / (a + b + j)
        for k in (1, 2, 3):
            assert stats.beta(a, b).moment(k) == pytest.approx(np.prod([(a + j) / (a + b + j) for j in range(k)]))
    assert np.allclose(beta_pdf(np.linspace(0.01, 0.99, 7), 1, 1), 1.0)            # Beta(1, 1) = U(0, 1)
    assert beta_pdf(1e-9, 1.0, 3.0) == pytest.approx(3.0)                           # value beta at 0
    assert beta_pdf(0.0, 2, 2) == 0 and beta_pdf(1.0, 2, 2) == 0


def test_gamma_pdf():
    x = np.linspace(0.01, 20, 50)
    assert np.allclose(gamma_pdf(x, 2.5, 1.7), stats.gamma.pdf(x, 2.5, scale=1 / 1.7))
    assert gamma_pdf(-1.0, 2.0, 1.0) == 0
    # chi^2(n) = Gamma(n/2, 1/2)
    assert np.allclose(gamma_pdf(x, 1.5, 0.5), stats.chi2.pdf(x, 3))


def test_gamma_beta_simulation():
    # Proposition rv-gamma-beta by simulation: marginal laws, and joint CDF = product of marginal CDFs
    s, t = sample_gamma_beta(2.0, 3.0, 1.0, 200_000, seed=11)
    assert ks_distance(t, lambda x: stats.beta.cdf(x, 2, 3)) < 0.006
    assert ks_distance(s, lambda x: stats.gamma.cdf(x, 5)) < 0.006
    assert joint_cdf_gap(s, t) < 0.006
    # i.i.d. uniforms: uncorrelated but dependent (Remark rv-code-gamma-beta, Exercise rv-gamma-beta-sim)
    rng = np.random.default_rng(2026)
    x, y = rng.random(100_000), rng.random(100_000)
    s2, t2 = x + y, x / (x + y)
    assert abs(sample_correlation(s2, t2)) < 0.01
    assert 0.045 < joint_cdf_gap(s2, t2) < 0.056
    assert not np.any((s2 > 1.9) & (t2 < 0.1))
    assert np.mean(s2 > 1.9) == pytest.approx(0.005, abs=0.001)
    assert np.mean(t2 < 0.1) == pytest.approx(1 / 18, abs=0.003)
    # different rates: dependent and correlated
    s3, t3 = sample_gamma_beta(2.0, 3.0, 1.0, 100_000, seed=np.random.default_rng(2026), lam2=3.0)
    assert sample_correlation(s3, t3) > 0.35 and joint_cdf_gap(s3, t3) > 0.06


def test_joint_cdf_gap_on_a_product_and_on_a_diagonal():
    rng = np.random.default_rng(3)
    x, y = rng.random(50_000), rng.random(50_000)
    assert joint_cdf_gap(x, y) < 0.01
    # X = Y: F(s, s) = s and F_X F_Y = s^2, gap 1/4 at s = 1/2
    assert joint_cdf_gap(x, x) == pytest.approx(0.25, abs=0.01)


# ------------------------------------------------------------------ the worked exercises


def test_integer_and_fractional_parts():
    lam = 0.7
    k = np.arange(400)
    f = integer_part_pmf(k, lam)
    assert f.sum() == pytest.approx(1.0)
    assert np.allclose(f, (1 - math.exp(-lam)) * math.exp(-lam) ** k)
    assert np.dot(k, f) == pytest.approx(1 / (math.exp(lam) - 1))               # E[Y] = 1/(e^lam - 1)
    # Exercise rv-worked-details (a): P(Z = k/2) and E[[X]] <= E[Z] <= E[X]
    h = half_integer_part_pmf(k, 1.0)
    assert np.allclose(h, (1 - math.exp(-0.5)) * np.exp(-0.5 * k))
    ez = np.dot(k / 2, h)
    assert ez == pytest.approx(1 / (2 * (math.exp(0.5) - 1)))
    assert 1 / (math.e - 1) <= ez <= 1.0
    assert ez == pytest.approx(0.7707470, abs=1e-6)
    # c): density lam e^{-lam t} / (1 - e^{-lam}) on [0, 1)
    assert integrate.quad(lambda t: fractional_part_pdf(t, lam), 0, 1)[0] == pytest.approx(1.0)
    t = np.linspace(0, 0.99, 12)
    assert np.allclose(fractional_part_cdf(t, lam), [integrate.quad(lambda s: fractional_part_pdf(s, lam), 0, x)[0] for x in t])
    y, frac = sample_worked("integer-part", 200_000, seed=4, lam=lam)
    assert ks_distance(frac, lambda x: fractional_part_cdf(x, lam)) < 0.005
    assert np.mean(y) == pytest.approx(1 / (math.exp(lam) - 1), abs=0.01)


def test_laplace_and_uniform():
    # Worked exercise exa:rv-laplace-uniform
    assert integrate.quad(laplace_pdf, -np.inf, np.inf)[0] == pytest.approx(1.0)
    assert integrate.quad(sqrt_abs_laplace_pdf, 0, np.inf)[0] == pytest.approx(1.0)
    assert integrate.quad(laplace_uniform_sum_pdf, 0, 50, points=[1])[0] == pytest.approx(1.0)
    # f_V = f_U * f_Y
    v = np.linspace(0.05, 3.5, 12)
    conv = convolution(sqrt_abs_laplace_pdf, lambda y: ((y > 0) & (y < 1)).astype(float), v, (0, 4), 40_000)
    assert np.allclose(conv, laplace_uniform_sum_pdf(v), atol=2e-4)
    # (S, T) = (X + Y, X - Y): the linear-transformation formula, and the integral over the strip D
    fZ = lambda x, y: laplace_pdf(x) * ((y > 0) & (y < 1))
    fW = linear_transformed_density(fZ, [[1, 1], [1, -1]])
    S, T = np.meshgrid(np.linspace(-3, 4, 15), np.linspace(-4, 3, 15))
    assert np.allclose(fW(S, T), sum_difference_pdf(S, T))
    total = integrate.dblquad(lambda t, s: sum_difference_pdf(s, t), -40, 40, lambda s: s - 2, lambda s: s)[0]
    assert total == pytest.approx(1.0, abs=1e-6)
    x, y = sample_worked("laplace-uniform", 200_000, seed=5)
    u = np.sqrt(np.abs(x))
    assert ks_distance(u, lambda q: -np.expm1(-np.maximum(q, 0) ** 2)) < 0.005     # F_U(u) = 1 - e^{-u^2}
    grid = np.linspace(0, 4, 400)
    Fv = np.cumsum(laplace_uniform_sum_pdf(midpoints(0, 4, 4000))) * 0.001
    assert np.max(np.abs(empirical_cdf(u + y, grid) - np.interp(grid, midpoints(0, 4, 4000) + 0.0005, Fv))) < 0.006


def test_products_of_uniforms():
    # Worked exercise exa:rv-uniform-products and Exercise rv-products-uniforms
    for n in range(1, 7):
        assert integrate.quad(lambda u: uniform_product_pdf(u, n), 0, 1)[0] == pytest.approx(1.0)
        for u0 in (0.05, 0.3, 0.8):
            assert uniform_product_cdf(u0, n) == pytest.approx(integrate.quad(lambda u: uniform_product_pdf(u, n), 0, u0)[0])
            # Erlang tail: P(T_n >= -log u) with T_n ~ Gamma(n, 1)
            assert uniform_product_cdf(u0, n) == pytest.approx(stats.gamma.sf(-math.log(u0), n))
    u = np.linspace(0.01, 0.99, 9)
    assert np.allclose(uniform_product_pdf(u, 2), -np.log(u))
    assert np.allclose(uniform_product_cdf(u, 2), u - u * np.log(u))               # Exercise (b)
    # Exercise (a): (U, V) = (XY, X) has density 1/v on 0 < u < v < 1; integrate out v
    fUV = transformed_density(lambda x, y: ((x > 0) & (x < 1) & (y > 0) & (y < 1)).astype(float),
                              lambda uu, vv: (vv, uu / vv), image=lambda uu, vv: (0 < uu) & (uu < vv) & (vv < 1))
    for u0 in (0.1, 0.5):
        assert integrate.quad(lambda vv: fUV(u0, vv), u0, 1)[0] == pytest.approx(-math.log(u0), rel=1e-6)
    N = dkw_sample_size(0.005, 0.01)
    assert N == 105_967
    sample = sample_worked("uniform-product", N, seed=0, m=3)
    assert ks_distance(sample, lambda x: uniform_product_cdf(x, 3)) == pytest.approx(0.00252, abs=2e-5)
    assert ks_distance(sample, lambda x: uniform_product_cdf(x, 2)) == pytest.approx(0.2726, abs=2e-4)


def test_empirical_cdf_and_ks():
    s = np.array([0.2, 0.4, 0.4, 0.9])
    assert np.allclose(empirical_cdf(s, [0.1, 0.2, 0.4, 0.5, 1.0]), [0, 0.25, 0.75, 0.75, 1.0])
    # KS distance of {0.5} from U(0, 1) is 1/2
    assert ks_distance([0.5], lambda x: np.clip(x, 0, 1)) == pytest.approx(0.5)
