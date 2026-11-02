"""Checks of the Chapter 14 code against the notes and the exercises."""
import numpy as np
import pytest
from scipy import stats

from probnotes.gaussian_conditioning import (
    affine_image,
    bivariate_conditional,
    conditional_independence_graph,
    conditional_law,
    correlation_matrix,
    pair_conditional_cov,
    partial_correlations,
    partial_sums_cov,
    partial_sums_precision,
    precision_matrix,
    regression_fit,
    regression_matrix,
    sample_gaussian,
    schur_complement,
    slab_mask,
    slab_moments,
    squares_cov,
)


def random_spd(n, seed):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    return B @ B.T + n * np.eye(n)


# ---------------------------------------------------------------- the bivariate case


def test_bivariate_matches_formulas_in_correlation_form():
    mx, my, sx, sy, rho = 1.0, -2.0, 1.5, 0.8, -0.6
    y = np.array([-3.0, -2.0, 0.5])
    m, s2 = bivariate_conditional(y, mx, my, sx ** 2, sy ** 2, rho * sx * sy)
    assert m == pytest.approx(mx - rho * sx / sy * (my - y))
    assert s2 == pytest.approx(sx ** 2 * (1 - rho ** 2))
    # Z = X - a Y is uncorrelated with Y: cov(X, Y) - a var(Y) = 0
    a = rho * sx * sy / sy ** 2
    assert rho * sx * sy - a * sy ** 2 == pytest.approx(0.0)
    with pytest.raises(ValueError):
        bivariate_conditional(0.0, 0, 0, 1, 0, 0)          # sigma_Y = 0


def test_bivariate_uncorrelated_gives_the_marginal():
    m, s2 = bivariate_conditional(np.linspace(-2, 2, 5), 3.0, 1.0, 2.0, 5.0, 0.0)
    assert np.allclose(m, 3.0) and s2 == pytest.approx(2.0)


def test_signal_example():
    # (X, Y) ~ N_2(0, [[1, 1], [1, 2]]): E[X | Y = y] = y / 2, variance 1 / 2
    m, s2 = bivariate_conditional(np.array([-1.0, 0.0, 3.0]), 0, 0, 1, 2, 1)
    assert np.allclose(m, [-0.5, 0.0, 1.5]) and s2 == pytest.approx(0.5)


def test_bivariate_agrees_with_block_formula():
    C = np.array([[2.0, 0.7], [0.7, 1.3]])
    mu = np.array([0.4, -1.1])
    y = np.array([[-1.0], [0.0], [2.5]])
    r = conditional_law(mu, C, [1], y)
    m, s2 = bivariate_conditional(y.ravel(), mu[0], mu[1], C[0, 0], C[1, 1], C[0, 1])
    assert np.allclose(r.mean.ravel(), m) and r.cov[0, 0] == pytest.approx(s2)


# ---------------------------------------------------------------- worked exercise: N_2(0, [[1, 1], [1, 4]])


def test_worked_exercise_correlations():
    C = np.array([[1.0, 1.0], [1.0, 4.0]])
    for alpha in (-1.0, 0.0, 1.0, 2.5):                     # Z = Y - alpha X, var = alpha^2 - 2 alpha + 4
        _, V = affine_image([0, 0], C, [[-alpha, 1.0]])
        assert V[0, 0] == pytest.approx(alpha ** 2 - 2 * alpha + 4)
    p = stats.norm.cdf(1 / np.sqrt(3))
    assert p == pytest.approx(0.71815, abs=5e-6)            # P(Z < 1) for alpha = 1
    assert stats.norm.cdf(0.58) == pytest.approx(0.7190, abs=5e-5)   # the notes' table value
    assert regression_matrix(C, [0], [1])[0, 0] == pytest.approx(1.0)  # alpha = 1: Y - X indep. X
    assert correlation_matrix(C)[0, 1] == pytest.approx(0.5)
    S = squares_cov([0, 0], C)
    assert np.allclose(S, [[2.0, 2.0], [2.0, 32.0]])         # var X^2 = 2, cov = 2, var Y^2 = 32
    assert S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]) == pytest.approx(0.25)


def test_square_of_N04_is_gamma_half_eighth():
    # Y ~ N(0, 4) => Y^2 ~ Gamma(1/2, 1/8) (rate 1/8, scale 8): mean 4, variance 32
    g = stats.gamma(a=0.5, scale=8.0)
    assert g.mean() == pytest.approx(4.0) and g.var() == pytest.approx(32.0)
    x = np.linspace(0.1, 30, 50)
    assert np.allclose(g.pdf(x), stats.chi2(1).pdf(x / 4) / 4)   # c X ~ Gamma(alpha, lambda / c)


def test_squares_cov_by_monte_carlo():
    mu, C = np.array([0.5, -1.0]), np.array([[1.0, 0.6], [0.6, 2.0]])
    s = sample_gaussian(mu, C, 400_000, seed=11)
    emp = np.cov(s ** 2, rowvar=False)
    assert np.allclose(emp, squares_cov(mu, C), rtol=0.05, atol=0.05)


# ---------------------------------------------------------------- worked exercise: N_2(0, [[1, 2], [2, 8]])


def test_worked_exercise_two_combinations():
    C = np.array([[1.0, 2.0], [2.0, 8.0]])
    for alpha in (0.0, 1.0, 22 / 5, 7.0):
        m, V = affine_image([0, 0], C, [[1.0, -3.0], [-alpha, 1.0]])
        assert np.allclose(m, 0)
        assert np.allclose(V, [[61, 5 * alpha - 22], [5 * alpha - 22, alpha ** 2 - 4 * alpha + 8]])
    _, V = affine_image([0, 0], C, [[1.0, -3.0], [-22 / 5, 1.0]])
    assert V[0, 1] == pytest.approx(0.0, abs=1e-12)          # W indep. Z iff alpha = 22/5
    assert regression_matrix(C, [0], [1])[0, 0] == pytest.approx(2.0)   # X indep. Z iff alpha = 2
    r = conditional_law([0, 0], C, [0], [[-1.0], [0.0], [1.5]])
    assert np.allclose(r.mean.ravel(), [-2.0, 0.0, 3.0]) and r.cov[0, 0] == pytest.approx(4.0)


# ---------------------------------------------------------------- block formulas


def test_block_formula_makes_residual_independent():
    C = random_spd(5, 1)
    A = regression_matrix(C, [3, 4], [0, 1, 2])
    # cov(X_1 - A X_2, X_2) = C_12 - A C_22 = 0
    assert np.allclose(C[np.ix_([0, 1, 2], [3, 4])] - A @ C[np.ix_([3, 4], [3, 4])], 0)
    S = schur_complement(C, [3, 4])
    M = np.block([[np.eye(3), -A], [np.zeros((2, 3)), np.eye(2)]])
    V = M @ C @ M.T                                          # covariance of (Z, X_2)
    assert np.allclose(V[:3, :3], S) and np.allclose(V[:3, 3:], 0)
    assert np.linalg.det(C) == pytest.approx(np.linalg.det(S) * np.linalg.det(C[3:, 3:]))


def test_block_formula_both_directions_and_permutations():
    C, mu = random_spd(4, 2), np.array([1.0, -1.0, 0.5, 2.0])
    x = np.array([0.3, -0.2])
    r1 = conditional_law(mu, C, [0, 1], x)                   # X_2 | X_1, the second bullet
    C11, C12, C21, C22 = C[:2, :2], C[:2, 2:], C[2:, :2], C[2:, 2:]
    assert np.allclose(r1.mean, mu[2:] - C21 @ np.linalg.solve(C11, mu[:2] - x))
    assert np.allclose(r1.cov, C22 - C21 @ np.linalg.solve(C11, C12))
    r2 = conditional_law(mu, C, [1, 0], x[::-1], target=[3, 2])    # same, components permuted
    assert np.allclose(r2.mean, r1.mean[::-1]) and np.allclose(r2.cov, r1.cov[::-1, ::-1])


def test_conditional_law_vectorised_and_errors():
    C = random_spd(3, 3)
    xs = np.array([[0.0], [1.0], [-2.0]])
    r = conditional_law(None, C, [2], xs)
    assert r.mean.shape == (3, 2)
    for k in range(3):
        assert np.allclose(r.mean[k], conditional_law(None, C, [2], xs[k]).mean)
    with pytest.raises(ValueError):
        conditional_law(None, C, [2], [0.0, 1.0])            # wrong shape
    with pytest.raises(ValueError):
        conditional_law(None, C, [2], [0.0], target=[2])     # overlap
    with pytest.raises(ValueError):
        conditional_law(None, [[1.0, 1.0], [1.0, 1.0]], [0, 1], [0.0, 0.0], target=[])
    with pytest.raises(ValueError):
        regression_matrix(np.diag([1.0, 0.0, 2.0]), [1])     # C_22 = 0 is not invertible
    with pytest.raises(ValueError):
        conditional_law(None, [[1.0, 2.0], [0.0, 1.0]], [1], [0.0])   # not symmetric


def test_singular_C_with_invertible_C22():
    # X = 2 Y + 1: C singular, conditional variance 0 and mean 2 y + 1
    C = np.array([[4.0, 2.0], [2.0, 1.0]])
    r = conditional_law([1.0, 0.0], C, [1], [[0.5], [-1.0]])
    assert np.allclose(r.mean.ravel(), [2.0, -1.0]) and r.cov[0, 0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- worked exercise: partial sums


def test_partial_sums_worked_exercise():
    C = partial_sums_cov(3)
    assert np.allclose(C, [[1, 1, 1], [1, 2, 2], [1, 2, 3]])
    A = np.tril(np.ones((3, 3)))
    assert np.allclose(affine_image(None, np.eye(3), A)[1], C)
    a = conditional_law(None, C, [2], [[3.0], [-1.5]])        # (Y1, Y2) | Y3 = y3
    assert np.allclose(a.mean, [[1.0, 2.0], [-0.5, -1.0]])    # (y3/3, 2 y3/3)
    assert np.allclose(a.cov, [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
    b = conditional_law(None, C, [1, 2], [[1.0, 7.0], [-2.0, 0.0]])   # Y1 | (Y2, Y3)
    assert np.allclose(b.A, [[0.5, 0.0]])
    assert np.allclose(b.mean.ravel(), [0.5, -1.0]) and b.cov[0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize("N", [1, 2, 3, 6, 10])
def test_partial_sums_precision_is_tridiagonal(N):
    assert np.allclose(precision_matrix(partial_sums_cov(N)), partial_sums_precision(N))
    assert np.allclose(partial_sums_cov(N, 2.5) @ partial_sums_precision(N, 2.5), np.eye(N))


def test_bridge_law():
    N, s = 8, 2.0
    r = conditional_law(None, partial_sums_cov(N), [N - 1], [s])
    k = np.arange(1, N)
    assert np.allclose(r.mean, k * s / N)
    assert np.allclose(r.cov, np.minimum.outer(k, k) - np.outer(k, k) / N)
    one = conditional_law(None, partial_sums_cov(N), [2, 4], [1.0, -1.0], target=[3])
    assert one.mean[0] == pytest.approx(0.0) and one.cov[0, 0] == pytest.approx(0.5)


# ---------------------------------------------------------------- precision matrix and conditional independence


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_block_of_inverse_is_inverse_of_schur_complement(seed):
    C = random_spd(6, seed)
    P = precision_matrix(C)
    for i, j in [(0, 1), (2, 5), (4, 3)]:
        Sig = pair_conditional_cov(C, i, j)
        assert np.allclose(np.linalg.inv(Sig), P[np.ix_([i, j], [i, j])])
        R = partial_correlations(C)
        assert R[i, j] == pytest.approx(Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1]))


def test_zeros_of_precision_give_the_graph():
    edges = [(0, 1), (1, 2), (2, 3), (1, 4)]
    P = np.eye(5)
    for i, j in edges:
        P[i, j] = P[j, i] = -0.4
    C = np.linalg.inv(P)
    assert np.all(np.abs(C) > 1e-3)                          # every pair is correlated ...
    assert conditional_independence_graph(C) == sorted(edges)   # ... but the graph is a tree
    assert pair_conditional_cov(C, 0, 2)[0, 1] == pytest.approx(0.0, abs=1e-12)
    # separation: X_0 and X_3 are independent given X_1 alone
    S = schur_complement(C, [1], [0, 3])
    assert S[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert pair_conditional_cov(np.array([[2.0, 0.5], [0.5, 1.0]]), 0, 1)[0, 1] == 0.5   # N = 2
    with pytest.raises(ValueError):
        pair_conditional_cov(C, 1, 1)


# ---------------------------------------------------------------- Monte Carlo


def test_sample_gaussian_moments_and_degenerate():
    mu, C = np.array([1.0, -2.0, 0.0]), random_spd(3, 7)
    s = sample_gaussian(mu, C, 200_000, seed=2026)
    assert np.allclose(s.mean(axis=0), mu, atol=4 * np.sqrt(np.diag(C) / 2e5).max())
    assert np.allclose(np.cov(s, rowvar=False), C, rtol=0.03, atol=0.05)
    D = np.array([[1.0, 2.0], [2.0, 4.0]])                    # Y = 2 X
    d = sample_gaussian([0, 1], D, 1000, seed=1)
    assert np.allclose(d[:, 1] - 1, 2 * d[:, 0])
    assert np.array_equal(sample_gaussian(mu, C, 5, seed=3), sample_gaussian(mu, C, 5, seed=3))


def test_slab_conditioning_signal_example():
    C = np.array([[1.0, 1.0], [1.0, 2.0]])
    s = sample_gaussian(None, C, 1_000_000, seed=2026)
    for y0 in (-2.0, 0.0, 1.5):
        e = slab_moments(s, [1], [y0], 0.02)
        # the exact mean over the slab is E[X | Y in slab] = (slab average of Y) / 2
        assert abs(e.mean[0] - e.given_mean[0] / 2) < 4 * e.se[0]
        assert abs(e.mean[0] - y0 / 2) < 4 * e.se[0] + 0.01
        assert e.cov[0, 0] == pytest.approx(0.5, rel=0.1)
    m = slab_mask(s, [1], [0.0], 0.02)
    assert m.sum() == slab_moments(s, [1], [0.0], 0.02).n
    with pytest.raises(ValueError):
        slab_moments(s[:10], [1], [5.0], 0.01)                # empty slab


def test_regression_recovers_block_formula():
    mu = np.array([1.0, 0.0, -1.0, 2.0])
    C = np.array([[4.0, 2.0, 1.0, 0.5], [2.0, 3.0, 1.0, 0.0], [1.0, 1.0, 2.0, 0.5], [0.5, 0.0, 0.5, 1.0]])
    s = sample_gaussian(mu, C, 100_000, seed=2026)
    f = regression_fit(s, [1, 2, 3], [0])
    r = conditional_law(mu, C, [1, 2, 3], mu[1:])
    assert np.all(np.abs(f.slope - r.A) < 4 * f.slope_se)
    assert f.intercept[0] == pytest.approx(mu[0] - r.A[0] @ mu[1:], abs=0.03)
    assert f.residual_cov[0, 0] == pytest.approx(r.cov[0, 0], rel=0.02)


# ---------------------------------------------------------------- numbers quoted in captions and solutions


def test_figure_precision_caption_numbers():
    s = sample_gaussian(None, partial_sums_cov(3), 1_000_000, seed=2026)
    m = slab_mask(s, [1], [1.0], 0.05)
    assert m.sum() == 22057
    assert round(np.corrcoef(s[m, 0], s[m, 2])[0, 1], 3) == -0.003
    assert np.corrcoef(s[:, 0], s[:, 2])[0, 1] == pytest.approx(1 / np.sqrt(3), abs=2e-3)


def test_regression_exercise_exact_numbers():
    mu = np.array([1.0, 0.0, -1.0, 2.0])
    C = np.array([[4, 2, 1, 0.5], [2, 3, 1, 0], [1, 1, 2, 0.5], [0.5, 0, 0.5, 1]])
    r = conditional_law(mu, C, [1, 2, 3], mu[1:])
    assert np.allclose(r.A, np.array([[11, 1, 8]]) / 17)
    assert mu[0] - r.A[0] @ mu[1:] == pytest.approx(2 / 17)
    assert r.cov[0, 0] == pytest.approx(41 / 17)
    assert np.allclose(17 * np.linalg.inv(C[1:, 1:]), [[7, -4, 2], [-4, 12, -6], [2, -6, 20]])
    n = int(np.ceil(r.cov[0, 0] * np.linalg.inv(C[1:, 1:]).diagonal().max() / 0.005 ** 2))
    assert n == 113495


def test_heights_exercise_numbers():
    c = 0.5 * 7 * 7.5
    m, s2 = bivariate_conditional(190, 177, 175, 7.5 ** 2, 7 ** 2, c)      # Y | X = 190
    assert round(float(m), 2) == 185.04 and s2 == pytest.approx(42.1875)
    assert round(stats.norm.sf((190 - m) / np.sqrt(s2)), 4) == 0.2223
    mx, _ = bivariate_conditional(190, 175, 177, 7 ** 2, 7.5 ** 2, c)     # X | Y = 190
    assert round(float(mx), 2) == 181.07
    k = 15 / 7
    assert round(177 + 0.5 * 7.5 * stats.norm.pdf(k) / stats.norm.sf(k), 2) == 186.38
    assert round(175 + 7 * stats.norm.pdf(k) / stats.norm.sf(k), 2) == 192.50


def test_graphical_exercise_numbers():
    P = np.eye(5)
    for k, l in [(0, 1), (1, 2), (2, 3), (1, 4)]:
        P[k, l] = P[l, k] = -0.4
    assert round(np.linalg.eigvalsh(P).min(), 4) == 0.2609
    assert np.linalg.eigvalsh(P).min() == pytest.approx(1 - 0.4 * np.sqrt(2 + np.sqrt(2)))
    C = np.linalg.inv(P)
    assert round(schur_complement(C, [4], [0, 3])[0, 1], 4) == 0.1173
    assert round(schur_complement(C, [4], [0, 2])[0, 1], 4) == 0.2933
    assert np.allclose(partial_correlations(C)[[0, 1, 2, 1], [1, 2, 3, 4]], 0.4)
