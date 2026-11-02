"""Checks of the Chapter 13 code (Gaussian vectors) against the notes and the exercises."""
import math

import numpy as np
import pytest
from scipy import special, stats

from probnotes.gaussian import (affine_image, bivariate_cov, bivariate_density, block_split,
                                blocks_uncorrelated, cf_factorisation_defect, concat_independent,
                                distance_to_support, factor, gaussian_cf, gaussian_density,
                                is_covariance, is_diagonal, ks_distance, mahalanobis_sq, marginal,
                                psd_cholesky, psd_inv_sqrt, psd_sqrt, random_sign_sum_cdf, rank,
                                sample_affine, sample_gaussian, sample_mean_cov,
                                sample_random_sign_pair, sample_sign_flip_pair, sign_flip_covariance,
                                sign_flip_root, standard_density, support_basis, whiten)

Phi = special.ndtr
C_NOTES = np.array([[2.0, 1, -1], [1, 2, 1], [-1, 1, 1]])      # the handwritten example: not PSD
C_AFF = np.array([[2.0, 1, -1], [1, 2, 1], [-1, 1, 3]])       # the corrected example
A_AFF = np.array([[1.0, -1, 0], [1, 0, 1]])
C_SING = np.array([[2.0, 1, -1], [1, 2, 1], [-1, 1, 2]])      # singular, kernel (1, -1, 1)


# --- covariance matrices and square roots -------------------------------------------------

def test_handwritten_matrix_is_not_a_covariance():
    assert np.linalg.det(C_NOTES) == pytest.approx(-3.0)
    a = np.array([1.0, -1, 1])
    assert a @ C_NOTES @ a == pytest.approx(-1.0)                # "var(X1 - X2 + X3) = -1"
    assert not is_covariance(C_NOTES)
    with pytest.raises(ValueError):
        psd_cholesky(C_NOTES)
    for c, det in ((1, -3.0), (2, 0.0), (3, 3.0)):                # det C_c = 3c - 6
        Cc = C_AFF.copy()
        Cc[2, 2] = c
        assert np.linalg.det(Cc) == pytest.approx(det, abs=1e-12)
        assert is_covariance(Cc) == (c >= 2)


def test_equicorrelation_range():
    for N in (2, 3, 5):
        for rho, ok in ((-1 / (N - 1), True), (-1 / (N - 1) - 0.01, False), (0.4, True), (1.0, True),
                        (1.01, False)):
            R = np.full((N, N), rho)
            np.fill_diagonal(R, 1.0)
            assert is_covariance(R) == ok


@pytest.mark.parametrize("C", [C_AFF, C_SING, np.array([[2.0, 4], [4, 8]]), np.diag([1.0, 0, 3]),
                               np.array([[1.0]]), np.zeros((2, 2))])
def test_square_roots(C):
    A = psd_sqrt(C)
    assert np.allclose(A, A.T) and np.linalg.eigvalsh(A).min() > -1e-12
    assert np.allclose(A @ A, C, atol=1e-12)
    L = psd_cholesky(C)
    assert np.allclose(L, np.tril(L))
    assert np.allclose(L @ L.T, C, atol=1e-12)
    for method in ("sqrt", "cholesky", "eigen"):
        F = factor(C, method)
        assert np.allclose(F @ F.T, C, atol=1e-12)
    assert rank(C) == np.linalg.matrix_rank(C)


def test_cholesky_of_singular_matrices():
    # numpy refuses, the semidefinite algorithm does not
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(C_SING)
    L = psd_cholesky(C_SING)
    s2, s15 = math.sqrt(2), math.sqrt(1.5)
    assert np.allclose(L, [[s2, 0, 0], [1 / s2, s15, 0], [-1 / s2, s15, 0]])
    L2 = psd_cholesky([[2.0, 4], [4, 8]])
    assert np.allclose(L2, [[s2, 0], [2 * s2, 0]])


def test_inverse_square_root():
    B = psd_inv_sqrt(C_AFF)
    assert np.allclose(B @ C_AFF @ B, np.eye(3))
    with pytest.raises(ValueError):
        psd_inv_sqrt(C_SING)


# --- laws: affine images, the notes' examples ------------------------------------------------

def test_affine_example_of_the_notes():
    # Example: Z = (X1 - X2, X1 + X3) for X ~ N_3(0, C) (with c_33 = 3, corrected)
    m, S = affine_image(np.zeros(3), C_AFF, A_AFF)
    assert np.allclose(m, 0)
    assert np.allclose(A_AFF @ C_AFF, [[1, -1, -2], [1, 2, 2]])
    assert np.allclose(S, [[2, -1], [-1, 3]])
    # the handwritten matrix would give [[2, -1], [-1, 1]]
    assert np.allclose(A_AFF @ C_NOTES @ A_AFF.T, [[2, -1], [-1, 1]])
    # composition: B (A X + mu) + gamma = (BA) X + (B mu + gamma)
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(3, 4)), rng.normal(size=(2, 3))
    mu, gam = rng.normal(size=3), rng.normal(size=2)
    m2, S2 = affine_image(mu, A @ A.T, B, gam)
    assert np.allclose(m2, B @ mu + gam) and np.allclose(S2, (B @ A) @ (B @ A).T)


def test_characteristic_function():
    rng = np.random.default_rng(1)
    t = rng.normal(size=(7, 3))
    mu = np.array([1.0, -2, 0.5])
    expected = np.exp(1j * t @ mu - 0.5 * np.einsum("ki,ij,kj->k", t, C_AFF, t))
    assert np.allclose(gaussian_cf(t, mu, C_AFF), expected)
    # standard vector: exp(-|t|^2/2); marginals: t e_j gives N(mu_j, c_jj)
    assert np.allclose(gaussian_cf(t, np.zeros(3), np.eye(3)), np.exp(-0.5 * np.sum(t * t, axis=1)))
    s = np.linspace(-2, 2, 9)
    for j in range(3):
        tj = np.zeros((9, 3))
        tj[:, j] = s
        assert np.allclose(gaussian_cf(tj, mu, C_AFF), np.exp(1j * s * mu[j] - 0.5 * C_AFF[j, j] * s**2))


def test_density_examples_of_the_notes():
    x = np.array([[0.3, -0.7], [1.2, 0.4], [0.0, 0.0], [-2.0, 1.5]])
    x1, x2 = x[:, 0], x[:, 1]
    # Example 1: C = [[2,1],[1,1]], C^{-1} = [[1,-1],[-1,2]]
    C1 = np.array([[2.0, 1], [1, 1]])
    assert np.allclose(np.linalg.inv(C1), [[1, -1], [-1, 2]])
    f1 = np.exp(-0.5 * (x1**2 - 2 * x1 * x2 + 2 * x2**2)) / (2 * math.pi)
    assert np.allclose(gaussian_density(x, [0, 0], C1), f1)
    # Example 2: density exp(-(x1^2 + x1 x2 + x2^2)/3) / (2 pi sqrt 3) is N_2(0, [[2,-1],[-1,2]])
    C2 = np.array([[2.0, -1], [-1, 2]])
    assert np.allclose(np.linalg.inv(C2), np.array([[2, 1], [1, 2]]) / 3)
    assert np.linalg.det(C2) == pytest.approx(3.0)
    f2 = np.exp(-(x1**2 + x1 * x2 + x2**2) / 3) / (2 * math.pi * math.sqrt(3))
    assert np.allclose(gaussian_density(x, [0, 0], C2), f2)
    # against scipy, with a mean
    mu = np.array([1.0, -2, 0.5])
    y = np.random.default_rng(2).normal(size=(6, 3))
    assert np.allclose(gaussian_density(y, mu, C_AFF), stats.multivariate_normal(mu, C_AFF).pdf(y))
    # standard density
    assert np.allclose(gaussian_density(y, np.zeros(3), np.eye(3)), standard_density(y))


def test_bivariate_form_matches_general_density():
    mx, my, sx, sy, rho = 70.0, 65.0, 10.0, 12.0, 0.6
    C = bivariate_cov(sx, sy, rho)
    assert np.allclose(C, [[100, 72], [72, 144]])
    pts = np.array([[70.0, 65], [80, 60], [55, 90]])
    assert np.allclose(bivariate_density(pts[:, 0], pts[:, 1], mx, my, sx, sy, rho),
                       gaussian_density(pts, [mx, my], C))


def test_bivariate_exercise_numbers():
    # Ross-style exam scores: X ~ (70, 10^2), Y ~ (65, 12^2), rho = 0.6
    m, S = affine_image([70, 65], bivariate_cov(10, 12, 0.6), [[1, 1], [-1, 1]])
    assert np.allclose(m, [135, -5]) and np.allclose(np.diag(S), [388, 100])
    assert 1 - Phi(15 / math.sqrt(388)) == pytest.approx(0.2232, abs=5e-5)
    assert 1 - Phi(0.5) == pytest.approx(0.3085, abs=5e-5)
    # X + aY independent of X - Y iff a = 7/18
    _, S = affine_image([70, 65], bivariate_cov(10, 12, 0.6), [[1, 7 / 18], [1, -1]])
    assert S[0, 1] == pytest.approx(0, abs=1e-12)
    assert 2 * math.log(10) == pytest.approx(4.6052, abs=5e-5)


def test_degenerate_example_of_the_notes():
    A = np.array([[1.0, 1], [2, 2]])
    C = A @ A.T
    assert np.allclose(C, [[2, 4], [4, 8]]) and rank(C) == 1
    with pytest.raises(ValueError):
        gaussian_density([0, 0], [0, 0], C)
    U = support_basis(C)
    assert np.allclose(np.abs(U[:, 0]), np.array([1, 2]) / math.sqrt(5))
    assert (U.T @ C @ U)[0, 0] == pytest.approx(10.0)
    Y = sample_affine(A, [0, 0], 5000, seed=3)
    assert np.allclose(Y[:, 1], 2 * Y[:, 0])                   # Y2 = 2 Y1: the line y2 = 2 y1
    assert np.max(distance_to_support(Y, [0, 0], C)) < 1e-12
    # marginals N(0, 2) and N(0, 8)
    assert np.allclose(marginal([0, 0], C, [0])[1], [[2]]) and np.allclose(marginal([0, 0], C, [1])[1], [[8]])


def test_singular_sampling_stays_on_the_plane():
    a = np.array([1.0, -1, 1])
    assert np.allclose(C_SING @ a, 0)
    for method in ("sqrt", "cholesky", "eigen"):
        Y = sample_gaussian([1.0, 2, 3], C_SING, 20_000, method=method, seed=4)
        assert np.max(np.abs((Y - [1, 2, 3]) @ a)) < 1e-12
        assert np.max(distance_to_support(Y, [1, 2, 3], C_SING)) < 1e-12
        _, S = sample_mean_cov(Y)
        assert np.max(np.abs(S - C_SING)) < 0.1


# --- sampling and the density theorem ---------------------------------------------------------

@pytest.mark.parametrize("method", ["sqrt", "cholesky", "eigen"])
def test_sampling_moments_and_chi_square(method):
    mu = np.array([1.0, -2, 0.5])
    Y = sample_gaussian(mu, C_AFF, 100_000, method=method, seed=2026)
    m, S = sample_mean_cov(Y)
    assert np.max(np.abs(m - mu)) < 0.03          # standard errors <= sqrt(3/1e5) ~ 0.0055
    assert np.max(np.abs(S - C_AFF)) < 0.06
    q = mahalanobis_sq(Y, mu, C_AFF)              # chi^2(3), via X = A^{-1}(Y - mu)
    for r in (1.0, 3.0, 7.815):
        assert np.mean(q <= r) == pytest.approx(stats.chi2.cdf(r, 3), abs=0.006)


def test_sampling_numbers_quoted_in_the_solutions():
    Y = sample_gaussian(np.zeros(3), C_AFF, 100_000, method="sqrt", seed=2026)
    q = mahalanobis_sq(Y, np.zeros(3), C_AFF)
    assert np.mean(q <= 1.0) == pytest.approx(0.1996, abs=1e-12)
    assert np.mean(q <= 7.815) == pytest.approx(0.95022, abs=1e-12)
    _, S = sample_mean_cov(Y @ A_AFF.T)
    assert np.allclose(S, [[1.9933, -0.9911], [-0.9911, 2.9962]], atol=5e-5)


def test_same_law_for_different_factors():
    # the law depends on A only through A A^T: compare the two factors of C = [[2,4],[4,8]]
    A1 = np.array([[1.0, 1], [2, 2]])                       # K = 2
    A2 = np.array([[math.sqrt(2)], [2 * math.sqrt(2)]])     # K = 1
    assert np.allclose(A1 @ A1.T, A2 @ A2.T)
    Y1 = sample_affine(A1, [0, 0], 50_000, seed=5)[:, 0]
    Y2 = sample_affine(A2, [0, 0], 50_000, seed=6)[:, 0]
    assert ks_distance(Y1, lambda x: Phi(x / math.sqrt(2))) < 0.01
    assert ks_distance(Y2, lambda x: Phi(x / math.sqrt(2))) < 0.01


# --- independence and blocks ------------------------------------------------------------------

def test_independence_and_blocks():
    assert is_diagonal(np.diag([1.0, 2, 3])) and not is_diagonal(C_AFF)
    mu, C = concat_independent([1.0, 2], [[2, 1], [1, 1]], [3.0], [[4.0]])
    assert np.allclose(mu, [1, 2, 3])
    assert np.allclose(C, [[2, 1, 0], [1, 1, 0], [0, 0, 4]])
    assert blocks_uncorrelated(C, 2) and not is_diagonal(C)
    m1, m2, C11, C12, C21, C22 = block_split(mu, C, 2)
    assert np.allclose(C11, [[2, 1], [1, 1]]) and np.allclose(C22, [[4]])
    assert np.allclose(C12, 0) and np.allclose(C21, C12.T)
    assert not blocks_uncorrelated(C_AFF, 2)


def test_cf_factorisation_defect():
    rng = np.random.default_rng(7)
    Y = sample_gaussian([0, 0], [[1, 0], [0, 4]], 40_000, rng=rng)
    assert cf_factorisation_defect(Y[:, 0], Y[:, 1]) < 0.02         # independent: noise ~ n^{-1/2}
    Y = sample_gaussian([0, 0], [[1, 0.5], [0.5, 1]], 40_000, rng=rng)
    assert cf_factorisation_defect(Y[:, 0], Y[:, 1]) > 0.1          # correlated
    X, Z = sample_random_sign_pair(40_000, rng=rng)
    assert cf_factorisation_defect(X, Z) > 0.4                      # uncorrelated, dependent


# --- Gaussian marginals that are not jointly Gaussian -------------------------------------------

def test_random_sign_pair():
    X, Y = sample_random_sign_pair(100_000, seed=2026)
    assert np.allclose(np.abs(X), np.abs(Y))
    S = X + Y
    p = np.mean(S == 0)
    assert p == pytest.approx(0.50214, abs=1e-12)                  # the number in the solutions
    assert abs(p - 0.5) < 1.96 * math.sqrt(0.25 / 1e5)
    left = lambda x: random_sign_sum_cdf(x, left=True)
    assert ks_distance(S, random_sign_sum_cdf, left) == pytest.approx(0.0030728, abs=1e-7)
    assert ks_distance(Y, Phi) == pytest.approx(0.0026151, abs=1e-7)
    assert ks_distance(S, random_sign_sum_cdf, left) < math.sqrt(math.log(40) / 2e5)   # DKW 95%
    assert np.corrcoef(X**2, Y**2)[0, 1] == pytest.approx(1.0)
    assert random_sign_sum_cdf(0.0) - random_sign_sum_cdf(0.0, left=True) == pytest.approx(0.5)


def test_ks_distance_by_brute_force():
    x = np.array([0.0, 0, 0, 1.2, -0.4, 0, 2.0])
    F, Fl = random_sign_sum_cdf, lambda z: random_sign_sum_cdf(z, left=True)
    grid = np.concatenate([np.linspace(-4, 4, 40001), np.unique(x)])
    Fn = np.searchsorted(np.sort(x), grid, side="right") / x.size
    brute = np.max(np.abs(Fn - F(grid)))
    # left limits at the sample points
    Fn_l = np.searchsorted(np.sort(x), np.unique(x), side="left") / x.size
    brute = max(brute, np.max(np.abs(Fn_l - Fl(np.unique(x)))))
    assert ks_distance(x, F, Fl) == pytest.approx(brute, abs=1e-4)


def test_sign_flip_pair():
    c = sign_flip_root()
    assert c == pytest.approx(1.538172, abs=1e-6)
    assert sign_flip_covariance(0.0) == pytest.approx(-1.0)
    assert sign_flip_covariance(50.0) == pytest.approx(1.0)
    cs = np.linspace(0, 6, 200)
    assert np.all(np.diff(sign_flip_covariance(cs)) > 0)            # derivative 4 c^2 phi(c)
    assert 2 * (1 - Phi(c)) == pytest.approx(0.124007, abs=1e-6)
    X, Y = sample_sign_flip_pair(200_000, 1.0, seed=8)
    assert ks_distance(Y, Phi) < 0.006                              # Y ~ N(0, 1)
    assert np.mean(X * Y) == pytest.approx(sign_flip_covariance(1.0), abs=4 * math.sqrt(3 / 2e5))
    X, Y = sample_sign_flip_pair(10**6, c, seed=2026)
    assert np.corrcoef(X, Y)[0, 1] == pytest.approx(-0.003364, abs=1e-6)
    assert np.mean(X + Y == 0) == pytest.approx(0.124507, abs=1e-9)
    assert np.mean((X > c) & (Y > c)) == 0.0


# --- whitening ------------------------------------------------------------------------------

def test_whitening():
    mu = np.array([1.0, -2, 0.5])
    Y = sample_gaussian(mu, C_AFF, 2000, seed=2026)
    for method in ("zca", "cholesky"):
        W = whiten(Y, method=method)                      # estimated parameters: exact identity
        m, S = sample_mean_cov(W)
        assert np.allclose(m, 0, atol=1e-12) and np.allclose(S, np.eye(3), atol=1e-12)
        Wt = whiten(Y, mu, C_AFF, method=method)          # true parameters: approximately
        assert np.max(np.abs(sample_mean_cov(Wt)[1] - np.eye(3))) < 0.1
    # ZCA and Cholesky whitening differ by the orthogonal matrix Q = L^{-1} C^{1/2}
    L = np.linalg.cholesky(C_AFF)
    Q = np.linalg.solve(L, psd_sqrt(C_AFF))
    assert np.allclose(Q @ Q.T, np.eye(3))
    assert np.allclose(whiten(Y, mu, C_AFF, "cholesky"), whiten(Y, mu, C_AFF, "zca") @ Q.T)
    S = sample_mean_cov(whiten(Y, mu, C_AFF, "zca"))[1]
    assert np.max(np.abs(S - np.diag(np.diag(S)))) == pytest.approx(0.048933, abs=1e-6)


# --- numbers quoted in the solutions ------------------------------------------------------------

def test_solution_numbers_random_sign_against_normal():
    X, Y = sample_random_sign_pair(100_000, seed=2026)
    assert ks_distance(X + Y, lambda x: Phi(x / math.sqrt(2))) == pytest.approx(0.2523075, abs=1e-6)


def test_solution_numbers_sample_mean_exponential():
    # Exercise 13.7 (d): cov(Xbar, S^2) = 1 for two Exp(1) variables; simulation 0.993
    E = np.random.default_rng(2026).exponential(size=(10**6, 2))
    xbar, s2 = E.mean(axis=1), (E[:, 0] - E[:, 1]) ** 2 / 2
    assert np.cov(xbar, s2)[0, 1] == pytest.approx(0.99336, abs=1e-5)


def test_solution_numbers_sample_variance_size():
    # Exercise 13.8 (d): n = 80001 makes the sd of the sample variance (true variance 2) equal 0.01
    n = 80001
    assert 2 * math.sqrt(2 / (n - 1)) == pytest.approx(0.01)
    g = np.random.default_rng(2026)
    v = np.array([(math.sqrt(2) * g.standard_normal(n)).var(ddof=1) for _ in range(1000)])
    assert v.std(ddof=1) == pytest.approx(0.010127, abs=1e-6)
    assert v.mean() == pytest.approx(2.000507, abs=1e-6)
    assert np.mean(np.abs(v - 2) <= 0.0196) == pytest.approx(0.942)


def test_solution_numbers_whitening():
    mu = np.array([1.0, -2, 0.5])
    Y = sample_gaussian(mu, C_AFF, 2000, seed=2026)
    S = sample_mean_cov(whiten(Y, mu, C_AFF, "cholesky"))[1]
    assert np.max(np.abs(S - np.diag(np.diag(S)))) == pytest.approx(0.043010, abs=1e-6)
    Q = np.linalg.solve(np.linalg.cholesky(C_AFF), psd_sqrt(C_AFF))
    assert np.allclose(Q, [[0.9010, 0.3237, -0.2887], [-0.1464, 0.8536, 0.5], [0.4082, -0.4082, 0.8165]],
                       atol=5e-5)
    # Exercise 13.9 (d): 100 repetitions with n = 38416
    assert math.ceil((1.96 / 0.01) ** 2) == 38416
    g = np.random.default_rng(2026)
    C12 = np.array([[2.0, 1], [1, 2]])
    offs = np.array([np.cov(whiten(sample_gaussian([1, -2], C12, 38416, rng=g), [1, -2], C12),
                            rowvar=False)[0, 1] for _ in range(100)])
    assert np.sum(np.abs(offs) <= 0.01) == 96
    assert offs.std(ddof=1) == pytest.approx(0.0047697, abs=1e-6)
