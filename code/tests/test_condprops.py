"""Checks of the Chapter 12 code against the notes and the exercises."""
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import integrate, special, stats

from probnotes.condprops import (
    affine_predictor,
    beta_binomial_pmf,
    beta_posterior,
    bin_index,
    bin_regression,
    binned_means,
    conditional_independence_gap,
    conditional_moments,
    die_coins_moments,
    gamma_posterior,
    indicator_features,
    is_independent,
    is_mean_independent,
    l2_projection,
    mse,
    negbin_pmf,
    nested_variance,
    poisson_gamma_pmf_quad,
    random_sum_moments,
    sample_die_coins,
    sample_die_coins_nested,
    sample_gamma_conditional,
    sample_poisson_gamma,
    sample_random_sum,
    sample_signal,
    sample_thinning,
    signal_density,
    signal_mse,
    signal_posterior_mean,
    square_example,
    thinning_table,
    total_variance,
)

# ---------------------------------------------------------------- discrete tables


def test_conditional_moments_by_hand():
    # X in {0, 1, 2}; column 0: f = (.1, .3, .1) -> mean 1, E[X^2] = (.3 + .4)/.5 = 1.4, var .4
    f = np.array([[0.1, 0.0, 0.2], [0.3, 0.0, 0.1], [0.1, 0.0, 0.2]])
    fY, h, v = conditional_moments(f, [0, 1, 2])
    assert fY == pytest.approx([0.5, 0.0, 0.5])
    assert h[0] == pytest.approx(1.0) and h[2] == pytest.approx(1.0) and np.isnan(h[1])
    assert v[0] == pytest.approx(0.4) and v[2] == pytest.approx(0.8) and np.isnan(v[1])
    with pytest.raises(ValueError):
        conditional_moments([[0.5, 0.6]], [0])
    with pytest.raises(ValueError):
        conditional_moments(f, [0, 1])


def test_square_example_mean_independent_not_independent():
    # Remark: X uniform on {-2,-1,1,2}, Y = X^2: E[X|Y] = 0 = E[X], but X, Y dependent
    f, xv, yv = square_example()
    assert f.sum() == pytest.approx(1.0)
    fY, h, v = conditional_moments(f, xv)
    assert fY == pytest.approx([0.5, 0.5]) and h == pytest.approx([0.0, 0.0])
    assert v == pytest.approx([1.0, 4.0])                       # X|Y=1 ~ U{-1,1}, X|Y=4 ~ U{-2,2}
    assert is_mean_independent(f, xv) and not is_independent(f)
    # P(X = 1, Y = 4) = 0 != 1/8
    assert f[2, 1] == 0 and f.sum(axis=1)[2] * f.sum(axis=0)[1] == pytest.approx(1 / 8)
    tv = total_variance(f, xv)                                   # computational remark: 0 + 5/2
    assert (tv.mean, tv.var, tv.var_of_mean, tv.mean_of_var) == pytest.approx((0, 2.5, 0, 2.5))


def test_law_of_total_variance_on_random_tables():
    rng = np.random.default_rng(0)
    for _ in range(20):
        f = rng.random((5, 4)) ** 3
        f[:, 1] = 0.0                                             # an empty column is allowed
        f /= f.sum()
        tv = total_variance(f, rng.normal(size=5))
        assert abs(tv.gap) < 1e-12


def test_total_variance_poisson_given_sum():
    # Example: X ~ P(lam), Z = X + Y, X|Z=m ~ B(m, p): var(E[X|Z]) = p^2 (lam+mu), E[var] = p(1-p)(lam+mu)
    lam, mu, M = 2.0, 3.0, 60
    p = lam / (lam + mu)
    k = np.arange(M + 1)
    f = stats.poisson.pmf(k, lam)[:, None] * stats.poisson.pmf(k[None, :] - k[:, None], mu)
    f /= f.sum()
    tv = total_variance(f, k)
    assert tv.mean == pytest.approx(lam, abs=1e-9) and tv.var == pytest.approx(lam, abs=1e-9)
    assert tv.var_of_mean == pytest.approx(p * p * (lam + mu), abs=1e-9)
    assert tv.mean_of_var == pytest.approx(p * (1 - p) * (lam + mu), abs=1e-9)


def test_total_variance_exponential_example():
    # Example: var(X) = var(Z)/3 + E[Z]^2/12 with Z ~ Gamma(2, lam)
    for lam in (0.5, 1.0, 3.0):
        EZ, VZ = 2 / lam, 2 / lam ** 2
        assert VZ / 4 + (VZ + EZ ** 2) / 12 == pytest.approx(1 / lam ** 2)
        assert VZ / 3 + EZ ** 2 / 12 == pytest.approx(1 / lam ** 2)


def test_conditional_independence_gap():
    # two fair coins and their sum: independent, not conditionally independent; gap 1/4 at Y3 = 1
    f = np.zeros((2, 2, 3))
    for a in (0, 1):
        for b in (0, 1):
            f[a, b, a + b] = 0.25
    assert conditional_independence_gap(f) == pytest.approx(0.25)
    assert is_independent(f.sum(axis=2))
    # a table built as f3(k) g1(i|k) g2(j|k) is conditionally independent but not independent
    rng = np.random.default_rng(1)
    f3 = np.array([0.2, 0.5, 0.3])
    g1, g2 = rng.dirichlet(np.ones(4), 3), rng.dirichlet(np.ones(3), 3)
    t = np.einsum("k,ki,kj->ijk", f3, g1, g2)
    assert conditional_independence_gap(t) < 1e-15
    assert not is_independent(t.sum(axis=2))


# ---------------------------------------------------------------- two-stage experiments


def test_die_coins_exact():
    tv = die_coins_moments()
    assert (tv.mean, tv.var, tv.var_of_mean, tv.mean_of_var) == pytest.approx((7 / 4, 77 / 48, 35 / 48, 7 / 8))
    # the law of X, P(X = 0) = 21/128, E[N | X = 0] = 40/21, mu_4 = 5495/768 (exercise)
    law = [sum(Fraction(1, 6) * Fraction(math.comb(n, k), 2 ** n) for n in range(max(k, 1), 7)) for k in range(7)]
    assert law == [Fraction(21, 128), Fraction(5, 16), Fraction(33, 128), Fraction(1, 6),
                   Fraction(29, 384), Fraction(1, 48), Fraction(1, 384)]
    m = sum(k * q for k, q in enumerate(law))
    assert sum((k - m) ** 2 * q for k, q in enumerate(law)) == Fraction(77, 48)
    assert sum((k - m) ** 4 * q for k, q in enumerate(law)) == Fraction(5495, 768)
    w = [Fraction(1, 2 ** n) for n in range(1, 7)]
    assert sum(n * x for n, x in zip(range(1, 7), w)) / sum(w) == Fraction(40, 21)
    # the same decomposition from the joint table of (X, N)
    f = np.array([[stats.binom.pmf(k, n, 0.5) / 6 for n in range(1, 7)] for k in range(7)])
    t = total_variance(f, np.arange(7))
    assert (t.var_of_mean, t.mean_of_var) == pytest.approx((35 / 48, 7 / 8))


def test_die_coins_simulation_numbers():
    N, X = sample_die_coins(10 ** 6, seed=2026)
    assert set(np.unique(N)) == set(range(1, 7)) and (X <= N).all()
    assert X.mean() == pytest.approx(1.748926, abs=1e-6)
    assert X.var(ddof=1) == pytest.approx(1.603457, abs=1e-6)


def test_nested_variance():
    nv = nested_variance(sample_die_coins_nested(10_000, 10, seed=2026))
    assert nv.within == pytest.approx(0.8816, abs=1e-4) and nv.within_se == pytest.approx(0.0060, abs=1e-4)
    assert nv.between_raw == pytest.approx(0.8104, abs=1e-4)
    assert nv.between == pytest.approx(0.7223, abs=1e-4) and nv.total == pytest.approx(1.6038, abs=1e-4)
    assert abs(nv.within - 7 / 8) < 3 * nv.within_se
    assert abs(nv.between_raw - (35 / 48 + 7 / 80)) < 3 * nv.between_raw_se
    # with one draw of the first stage per row and constant rows, the within term is 0
    g = np.repeat(np.arange(5.0)[:, None], 3, axis=1)
    assert nested_variance(g).within == 0 and nested_variance(g).between == pytest.approx(2.5)
    with pytest.raises(ValueError):
        nested_variance(np.ones((3, 1)))


def test_random_sums():
    s = sample_random_sum([0, 2, 1, 0], lambda rng, m: np.arange(1.0, m + 1), seed=0)
    assert s.tolist() == [0.0, 3.0, 3.0, 0.0]                    # summands 1, 2 | 3
    assert random_sum_moments(10, 10, 2, 2) == (20, 60)          # claims: Poisson(10) x Gamma(2, 1)
    N = np.random.default_rng(2026).poisson(10, 200_000)
    S = sample_random_sum(N, lambda rng, m: rng.gamma(2.0, 1.0, m), seed=2027)
    assert S.mean() == pytest.approx(20, abs=4 * math.sqrt(60 / 2e5))
    assert S.var() == pytest.approx(60, rel=0.02)
    # geometric number of exponentials: Exp(p theta)
    p, th = 0.3, 2.0
    Ng = np.random.default_rng(5).geometric(p, 100_000)
    Sg = sample_random_sum(Ng, lambda rng, m: rng.exponential(1 / th, m), seed=6)
    assert stats.kstest(Sg, "expon", args=(0, 1 / (p * th))).pvalue > 1e-3
    assert random_sum_moments(1 / p, (1 - p) / p ** 2, 1 / th, 1 / th ** 2)[1] == pytest.approx(1 / (p * th) ** 2)
    with pytest.raises(ValueError):
        sample_random_sum([-1], lambda rng, m: np.zeros(m))


# ---------------------------------------------------------------- thinning and mixtures


def test_thinning_table():
    lam, p = 3.0, 0.4
    t = thinning_table(lam, p, 80)
    assert t.sum() == pytest.approx(1.0)
    k = np.arange(20)
    assert t.sum(axis=0)[:20] == pytest.approx(stats.poisson.pmf(k, lam * p), abs=1e-14)     # Y ~ P(lam p)
    for kk in (0, 3, 7):
        cond = t[:, kk] / t[:, kk].sum()                          # law of X given Y = k
        n = np.arange(t.shape[0])
        assert cond[kk:kk + 15] == pytest.approx(stats.poisson.pmf(np.arange(15), lam * (1 - p)), abs=1e-12)
        assert (n * cond).sum() == pytest.approx(lam * (1 - p) + kk)                         # E[X|Y=k]


def test_sample_thinning():
    X, Y = sample_thinning(200_000, 3.0, 0.4, seed=1)
    assert (Y <= X).all()
    assert Y.mean() == pytest.approx(1.2, abs=0.01) and Y.var() == pytest.approx(1.2, abs=0.02)
    Z = X - Y                                                     # independent of Y
    assert abs(np.corrcoef(Z, Y)[0, 1]) < 0.01
    for k in (0, 2, 4):
        assert X[Y == k].mean() == pytest.approx(1.8 + k, abs=0.03)


def test_negbin_against_quadrature_and_scipy():
    # computational remark: for the figure's alpha = 2, lam = 1/2 they differ by at most 1.2e-11 (k <= 20)
    for alpha, lam, tol in ((2.0, 0.5, 1.2e-11), (2.5, 1.3, 1e-11), (0.7, 2.0, 3e-11)):
        k = np.arange(21)
        q = negbin_pmf(k, alpha, lam)
        assert q == pytest.approx([poisson_gamma_pmf_quad(int(j), alpha, lam) for j in k], rel=0, abs=tol)
        assert q == pytest.approx(stats.nbinom.pmf(k, alpha, lam / (lam + 1)), abs=1e-14)
        kk = np.arange(400)
        r = negbin_pmf(kk, alpha, lam)
        assert r.sum() == pytest.approx(1.0)
        assert (kk * r).sum() == pytest.approx(alpha / lam)                                # E[Y] = E[X]
        assert ((kk - alpha / lam) ** 2 * r).sum() == pytest.approx(alpha / lam + alpha / lam ** 2)
    # alpha = 2, lam = 1/2: the values of the figure, BN(2, 1/3)
    assert negbin_pmf([0, 1, 2, 3], 2.0, 0.5) == pytest.approx([1 / 9, 4 / 27, 4 / 27, 32 / 243])
    # Erlang case: Pascal form C(m+k-1, k) p^m (1-p)^k
    m, lam = 3, 0.8
    p = lam / (lam + 1)
    k = np.arange(8)
    assert negbin_pmf(k, m, lam) == pytest.approx([math.comb(m + j - 1, j) * p ** m * (1 - p) ** j for j in k])
    assert negbin_pmf(-1, 2.0, 0.5) == 0.0


def test_gamma_posterior():
    alpha, lam = 2.0, 0.5
    x = np.linspace(0.01, 20, 3000)
    for k in (0, 4, 10):
        shape, rate = gamma_posterior(k, alpha, lam)
        assert (shape, rate) == (alpha + k, lam + 1)
        bayes = stats.poisson.pmf(k, x) * stats.gamma.pdf(x, alpha, scale=1 / lam) / negbin_pmf(k, alpha, lam)
        assert bayes == pytest.approx(stats.gamma.pdf(x, shape, scale=1 / rate), abs=1e-10)
        assert shape / rate == pytest.approx((alpha + k) / (lam + 1))


def test_sample_poisson_gamma():
    X, Y = sample_poisson_gamma(200_000, 2.0, 0.5, seed=2026)
    assert Y.mean() == pytest.approx(4.0, abs=0.04) and Y.var() == pytest.approx(12.0, rel=0.03)
    for k in (0, 4, 10):
        assert X[Y == k].mean() == pytest.approx((2 + k) / 1.5, rel=0.03)


def test_beta_binomial():
    n = 10
    k = np.arange(n + 1)
    for a, b in ((1.0, 1.0), (2.0, 3.0), (0.5, 0.5)):
        q = beta_binomial_pmf(k, n, a, b)
        assert q == pytest.approx(stats.betabinom.pmf(k, n, a, b), abs=1e-14)
        EX = (k * q).sum()
        VX = ((k - EX) ** 2 * q).sum()
        assert EX == pytest.approx(n * a / (a + b))
        assert VX == pytest.approx(n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1)))
        # E[P|X] = (a + X)/(a + b + n) is the best affine predictor
        A, B = beta_posterior(k, n, a, b)
        post = A / (A + B)
        EP, VP = a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1))
        slope = n * VP / VX
        assert post == pytest.approx(EP + slope * (k - EX))
    assert beta_binomial_pmf(k, n, 1.0, 1.0) == pytest.approx(np.full(n + 1, 1 / (n + 1)))
    assert beta_binomial_pmf([-1, n + 1], n, 1.0, 1.0).tolist() == [0.0, 0.0]


def test_poisson_gamma_posterior_mean_is_best_affine():
    alpha, lam = 2.0, 0.5
    VX, VY = alpha / lam ** 2, alpha / lam + alpha / lam ** 2
    slope = VX / VY                                               # cov(X, Y) = var(X)
    assert slope == pytest.approx(1 / (lam + 1))
    assert alpha / lam - slope * alpha / lam == pytest.approx(alpha / (lam + 1))


# ---------------------------------------------------------------- L2 projection


def test_indicator_regression_is_bin_average_and_orthogonal():
    rng = np.random.default_rng(3)
    y = rng.normal(size=400)
    x = np.sin(y) + 0.3 * rng.normal(size=400)
    edges = np.array([-np.inf, -1.0, 0.0, 0.5, 1.0, np.inf])
    D = indicator_features(y, edges)
    assert (D.sum(axis=1) == 1).all()
    coef, fit = l2_projection(x, D)
    j = bin_index(y, edges)
    assert coef == pytest.approx([x[j == i].mean() for i in range(5)])
    assert coef == pytest.approx(bin_regression(x, y, edges))
    assert np.abs(D.T @ (x - fit)).max() < 1e-10                   # residual orthogonal to every indicator
    # empty bins get 0
    assert bin_regression([1.0, 3.0], [0.1, 0.2], [-np.inf, 0.0, 1.0, 2.0, np.inf]).tolist() == [0, 2, 0, 0]
    # the best affine predictor: normal equations
    a, b = affine_predictor(x, y)
    r = x - a - b * y
    assert abs(r.mean()) < 1e-12 and abs((r * y).mean()) < 1e-12


def test_signal_posterior_mean_by_quadrature():
    for s in (0.5, 1.0, 2.0):
        for y in (-3.0, -0.5, 0.3, 1.0, 4.0, 8.0):
            w = lambda x: np.exp(-x) * stats.norm.pdf(y - x, scale=s)
            num = integrate.quad(lambda x: x * w(x), 0, np.inf)[0]
            den = integrate.quad(w, 0, np.inf)[0]
            assert signal_posterior_mean(y, s) == pytest.approx(num / den, rel=1e-7)
            assert signal_density(y, s) == pytest.approx(den, rel=1e-7)
        assert integrate.quad(lambda y: signal_density(y, s), -np.inf, np.inf)[0] == pytest.approx(1.0)
    assert signal_posterior_mean(1.0, 1.0) == pytest.approx(math.sqrt(2 / math.pi))       # m = 0
    assert np.isfinite(signal_posterior_mean(-60.0, 1.0)) and signal_posterior_mean(-60.0, 1.0) > 0


def test_signal_mse_numbers():
    # notes and figure: 0.4118 for sigma = 1, against 1/2 for the best affine predictor
    assert signal_mse(1.0) == pytest.approx(0.41178, abs=1e-5)
    for s in (0.5, 1.0, 2.0):
        assert signal_mse(s) < s ** 2 / (1 + s ** 2)
    X, Y = sample_signal(10 ** 6, 1.0, seed=2026)
    d1 = (X - signal_posterior_mean(Y, 1.0)) ** 2
    d2 = (X - 0.5 - 0.5 * Y) ** 2
    assert d1.mean() == pytest.approx(0.41136, abs=1e-5) and d2.mean() == pytest.approx(0.49928, abs=1e-5)
    assert (d2 - d1).mean() == pytest.approx(0.08791, abs=1e-5)
    assert affine_predictor(X, Y) == pytest.approx((0.4997, 0.4991), abs=1e-4)


def test_bin_regression_numbers():
    xf, yf = sample_signal(100_000, 1.0, seed=2026)
    xt, yt = sample_signal(10 ** 6, 1.0, seed=2027)
    out = {}
    for K in (1, 64, 256):
        e = np.quantile(yf[:2000], np.linspace(0, 1, K + 1))
        e[0], e[-1] = -np.inf, np.inf
        c = bin_regression(xf[:2000], yf[:2000], e)
        out[K] = (mse(xf[:2000], c[bin_index(yf[:2000], e)]), mse(xt, c[bin_index(yt, e)]))
    assert out[64][1] == pytest.approx(0.4425, abs=1e-4) and out[256][1] == pytest.approx(0.4672, abs=1e-4)
    assert out[256][0] == pytest.approx(0.3733, abs=1e-4) and out[1][1] == pytest.approx(1.0003, abs=1e-4)


# ---------------------------------------------------------------- the heavy-tailed pair


def test_gamma_conditional_pair():
    X, Y = sample_gamma_conditional(10 ** 6, seed=2026)
    assert (Y >= 1).all()
    assert (Y > 2).mean() == pytest.approx(0.500688) and (Y > 10).mean() == pytest.approx(0.10006)
    W = X * (Y - 1)
    assert stats.kstest(W, "expon").pvalue > 1e-3 and abs(np.corrcoef(W, X)[0, 1]) < 0.005
    edges = np.arange(0.5, 3.0001, 0.25)
    b = binned_means(X, Y, edges)
    exact = 1 + (special.exp1(edges[:-1]) - special.exp1(edges[1:])) / np.diff(-np.exp(-edges))
    assert exact[0] == pytest.approx(2.6356, abs=1e-4) and exact[-1] == pytest.approx(1.3487, abs=1e-4)
    assert np.abs((b.mean - exact) / b.se).max() == pytest.approx(2.0594, abs=1e-3)
    assert 1 + (special.exp1(0.05) - special.exp1(0.1)) / (np.exp(-0.05) - np.exp(-0.1)) == pytest.approx(14.903, abs=1e-3)
    # E[Y 1{X > t}] = e^{-t} + E_1(t), by quadrature of (1 + 1/x) e^{-x}
    t = 1.0
    assert integrate.quad(lambda x: (1 + 1 / x) * np.exp(-x), t, np.inf)[0] == pytest.approx(np.exp(-t) + special.exp1(t))
    assert np.exp(-1) + special.exp1(1.0) == pytest.approx(0.58726, abs=1e-5)


def test_binned_means_by_hand():
    b = binned_means([0.1, 0.2, 0.7, 1.5], [1.0, 3.0, 5.0, 9.0], [0.0, 0.5, 1.0])
    assert b.count.tolist() == [2, 1] and b.mean.tolist() == [2.0, 5.0]
    assert b.se[0] == pytest.approx(1.0) and np.isnan(b.se[1])
    assert b.centres.tolist() == [0.25, 0.75]


def test_mean_independence_simulation_numbers():
    rng = np.random.default_rng(2026)
    n = 10 ** 6
    Y = rng.uniform(-1, 1, n)
    X = rng.choice([-1.0, 1.0], n) * Y
    b = binned_means(Y, X, np.linspace(-1, 1, 21))
    dev = np.abs(b.mean) / b.se
    assert dev.max() == pytest.approx(3.489, abs=1e-3) and int(np.argmax(dev)) == 18 and (dev > 2).sum() == 1
    rng = np.random.default_rng(2026)
    n = 250_000
    Y = rng.uniform(-1, 1, n)
    X = rng.choice([-1.0, 1.0], n) * Y
    assert ((X <= -0.5) & (Y <= -0.5)).mean() == pytest.approx(0.126508)
