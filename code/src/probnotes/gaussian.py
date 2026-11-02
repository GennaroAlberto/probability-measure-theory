"""Gaussian vectors (Chapter 13).

Five groups of tools, in the order of the chapter:

* **Covariance matrices.** :func:`is_covariance` checks that a matrix is symmetric and positive
  semidefinite (the lemma of Section 13.5); :func:`psd_sqrt` computes its symmetric square root
  ``A = A^T >= 0`` with ``A^2 = C`` (used in the proof of the density theorem), and
  :func:`psd_cholesky` a lower-triangular ``L`` with ``L L^T = C``. Both also work for a
  *singular* ``C``, where ``numpy.linalg.cholesky`` fails.
* **Laws.** :func:`affine_image` maps ``(mu, C)`` to ``(B mu + gamma, B C B^T)`` (closure under
  affine maps), :func:`marginal` and :func:`block_split` extract components and blocks,
  :func:`concat_independent` puts two independent Gaussian vectors together, and
  :func:`gaussian_cf` is the characteristic function ``exp(i t^T mu - t^T C t / 2)``.
* **Sampling and densities.** :func:`sample_affine` samples ``Y = A X + mu`` with ``X`` standard,
  as in the definition; :func:`sample_gaussian` samples ``N_N(mu, C)`` through a square root of
  ``C`` (symmetric, Cholesky or spectral). :func:`gaussian_density` evaluates the density of
  ``N_N(mu, C)`` for invertible ``C``; :func:`mahalanobis_sq` the quadratic form
  ``(y - mu)^T C^{-1} (y - mu)``.
* **The degenerate case.** :func:`support_basis` gives an orthonormal basis of ``Im C``, so that
  the law of ``N_N(mu, C)`` lives on the affine subspace ``mu + Im C``, and
  :func:`distance_to_support` measures how far sample points are from it.
* **Independence and counterexamples.** :func:`is_diagonal` and :func:`blocks_uncorrelated`
  (independence of Gaussian components or blocks), :func:`cf_factorisation_defect` (an empirical
  check of the independence criterion), the pair ``(X, ZX)`` of the notes
  (:func:`sample_random_sign_pair`, :func:`random_sign_sum_cdf`) and the sign-flip pair of the
  exercises (:func:`sample_sign_flip_pair`, :func:`sign_flip_covariance`,
  :func:`sign_flip_root`). :func:`whiten` turns a sample into an (approximately) standard one,
  and :func:`bivariate_cov`, :func:`bivariate_density` give the classical bivariate form.

Vectors are rows of arrays: a sample of ``n`` points of ``R^N`` has shape ``(n, N)``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import linalg, optimize, special

_CHUNK = 2_000_000          # largest number of complex entries formed at once


# ---------------------------------------------------------------------------
# Covariance matrices
# ---------------------------------------------------------------------------


def _as_matrix(C) -> np.ndarray:
    C = np.atleast_2d(np.asarray(C, dtype=float))
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {C.shape}")
    return C


def _scale(C: np.ndarray) -> float:
    return max(float(np.max(np.abs(C))), 1e-300)


def is_covariance(C, tol: float = 1e-10) -> bool:
    """True if ``C`` is symmetric and positive semidefinite (up to a relative tolerance).

    These are exactly the covariance matrices of random vectors with components in ``L^2``:
    the lemma of the notes gives one direction, and ``N_N(0, C)`` the other.
    """
    C = _as_matrix(C)
    s = _scale(C)
    if not np.allclose(C, C.T, atol=tol * s, rtol=0):
        return False
    return bool(np.linalg.eigvalsh(0.5 * (C + C.T)).min() >= -tol * s)


def _check_covariance(C, tol: float = 1e-10) -> np.ndarray:
    C = _as_matrix(C)
    if not is_covariance(C, tol):
        raise ValueError("C must be symmetric and positive semidefinite")
    return 0.5 * (C + C.T)


def psd_sqrt(C, tol: float = 1e-10) -> np.ndarray:
    """The symmetric positive semidefinite square root of ``C``: ``A = A^T``, ``A >= 0``, ``A @ A = C``.

    With ``C = Q diag(lam) Q^T`` (spectral theorem), ``A = Q diag(sqrt(lam)) Q^T``. Eigenvalues
    that are negative by rounding only are set to zero, so singular ``C`` is allowed; ``A`` is
    invertible exactly when ``C`` is.
    """
    C = _check_covariance(C, tol)
    lam, Q = np.linalg.eigh(C)
    A = (Q * np.sqrt(np.clip(lam, 0.0, None))) @ Q.T
    return 0.5 * (A + A.T)


def psd_inv_sqrt(C, tol: float = 1e-10) -> np.ndarray:
    """``C^{-1/2}``, the inverse of :func:`psd_sqrt`, for a positive definite ``C``."""
    C = _check_covariance(C, tol)
    lam, Q = np.linalg.eigh(C)
    if lam.min() <= tol * _scale(C):
        raise ValueError("C is singular: it has no inverse square root")
    B = (Q / np.sqrt(lam)) @ Q.T
    return 0.5 * (B + B.T)


def psd_cholesky(C, tol: float = 1e-10) -> np.ndarray:
    """A lower-triangular ``L`` with ``L @ L.T = C``, for any positive semidefinite ``C``.

    This is the outer-product Cholesky algorithm. When a pivot vanishes (``C`` singular), positive
    semidefiniteness forces the rest of its column to vanish too, and the column of ``L`` is set
    to zero; ``numpy.linalg.cholesky`` would fail instead. A negative pivot means ``C`` is not
    positive semidefinite, and raises ``ValueError``.
    """
    C = _as_matrix(C)
    if not np.allclose(C, C.T, atol=tol * _scale(C), rtol=0):
        raise ValueError("C must be symmetric")
    s = _scale(C)
    S = 0.5 * (C + C.T)
    n = S.shape[0]
    L = np.zeros_like(S)
    for k in range(n):
        d = S[k, k]
        if d > tol * s:
            L[k:, k] = S[k:, k] / math.sqrt(d)
            S[k:, k:] -= np.outer(L[k:, k], L[k:, k])
        elif d < -tol * s or np.max(np.abs(S[k:, k])) > math.sqrt(tol) * s:
            raise ValueError("C is not positive semidefinite")
    return L


def rank(C, tol: float = 1e-10) -> int:
    """Rank of the covariance matrix ``C``: the dimension of the subspace ``Im C``."""
    C = _check_covariance(C, tol)
    return int(np.sum(np.linalg.eigvalsh(C) > tol * _scale(C)))


# ---------------------------------------------------------------------------
# Laws: affine images, marginals, blocks, characteristic function
# ---------------------------------------------------------------------------


def affine_image(mu, C, B, gamma=None):
    """Parameters of ``U = B Y + gamma`` for ``Y ~ N_N(mu, C)``: ``(B mu + gamma, B C B^T)``."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    C = _as_matrix(C)
    B = np.atleast_2d(np.asarray(B, dtype=float))
    m = B @ mu
    if gamma is not None:
        m = m + np.atleast_1d(np.asarray(gamma, dtype=float))
    S = B @ C @ B.T
    return m, 0.5 * (S + S.T)


def marginal(mu, C, idx):
    """Mean and covariance of the sub-vector ``(Y_i)_{i in idx}``."""
    idx = np.atleast_1d(np.asarray(idx, dtype=int))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    C = _as_matrix(C)
    return mu[idx], C[np.ix_(idx, idx)]


def block_split(mu, C, n1: int):
    """Split ``(mu, C)`` for the blocks ``Y_1 = (Y_1..Y_n1)`` and ``Y_2 = (Y_{n1+1}..Y_N)``.

    Returns ``(mu1, mu2, C11, C12, C21, C22)``.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    C = _as_matrix(C)
    return mu[:n1], mu[n1:], C[:n1, :n1], C[:n1, n1:], C[n1:, :n1], C[n1:, n1:]


def concat_independent(mu1, C1, mu2, C2):
    """Parameters of ``(Y_1, Y_2)`` for independent ``Y_i ~ N(mu_i, C_i)``: block-diagonal covariance."""
    mu = np.concatenate([np.atleast_1d(np.asarray(mu1, dtype=float)),
                         np.atleast_1d(np.asarray(mu2, dtype=float))])
    return mu, linalg.block_diag(_as_matrix(C1), _as_matrix(C2))


def gaussian_cf(t, mu, C) -> np.ndarray:
    """Characteristic function ``exp(i t^T mu - t^T C t / 2)`` of ``N_N(mu, C)``.

    ``t`` has shape ``(..., N)``; the result has shape ``(...)``.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    C = _as_matrix(C)
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        t = t[None]
    quad = np.einsum("...i,ij,...j->...", t, C, t)
    return np.exp(1j * (t @ mu) - 0.5 * quad)


# ---------------------------------------------------------------------------
# Sampling and densities
# ---------------------------------------------------------------------------


def _rng(seed=None, rng=None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng(seed)


def sample_affine(A, mu, n: int, seed=None, rng=None) -> np.ndarray:
    """``n`` draws of ``Y = A X + mu`` with ``X ~ N_K(0, I_K)``, ``A`` of shape ``(N, K)``: the definition.

    Returns an array of shape ``(n, N)``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    X = _rng(seed, rng).standard_normal((n, A.shape[1]))
    return X @ A.T + mu


def factor(C, method: str = "sqrt", tol: float = 1e-10) -> np.ndarray:
    """A matrix ``A`` with ``A A^T = C``: ``"sqrt"`` (symmetric square root), ``"cholesky"``
    (lower triangular) or ``"eigen"`` (``Q diag(sqrt(lam))``, columns along the eigenvectors)."""
    if method == "sqrt":
        return psd_sqrt(C, tol)
    if method == "cholesky":
        return psd_cholesky(C, tol)
    if method == "eigen":
        lam, Q = np.linalg.eigh(_check_covariance(C, tol))
        return Q * np.sqrt(np.clip(lam, 0.0, None))
    raise ValueError(f"unknown method {method!r}")


def sample_gaussian(mu, C, n: int, method: str = "sqrt", seed=None, rng=None) -> np.ndarray:
    """``n`` draws of ``N_N(mu, C)``, as ``A X + mu`` with ``A A^T = C`` (see :func:`factor`).

    All three methods give the same law (the characteristic function depends on ``A`` only
    through ``A A^T``) but different maps from ``X`` to ``Y``. Singular ``C`` is allowed.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    return sample_affine(factor(C, method), mu, n, seed=seed, rng=rng)


def mahalanobis_sq(y, mu, C) -> np.ndarray:
    """``(y - mu)^T C^{-1} (y - mu)`` for invertible ``C``; ``y`` of shape ``(..., N)``.

    For ``Y ~ N_N(mu, C)`` it equals ``|X|^2`` with ``X = A^{-1}(Y - mu)`` standard, so it has the
    law ``chi^2(N)``.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    C = _check_covariance(C)
    if np.linalg.eigvalsh(C).min() <= 1e-12 * _scale(C):
        raise ValueError("C is singular")
    L = np.linalg.cholesky(C)
    d = np.asarray(y, dtype=float) - mu
    shape = d.shape
    z = linalg.solve_triangular(L, d.reshape(-1, mu.size).T, lower=True)
    return np.sum(z * z, axis=0).reshape(shape[:-1])


def gaussian_logpdf(y, mu, C) -> np.ndarray:
    """Logarithm of the density of ``N_N(mu, C)`` at ``y`` (shape ``(..., N)``), for invertible ``C``.

    A singular ``C`` raises ``ValueError``: the law is then degenerate and has no density.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    C = _check_covariance(C)
    lam = np.linalg.eigvalsh(C)
    if lam.min() <= 1e-12 * _scale(C):
        raise ValueError("C is singular: N_N(mu, C) is degenerate and has no density")
    logdet = float(np.sum(np.log(lam)))
    return -0.5 * (mu.size * math.log(2 * math.pi) + logdet) - 0.5 * mahalanobis_sq(y, mu, C)


def gaussian_density(y, mu, C) -> np.ndarray:
    """Density ``(2 pi)^{-N/2} det(C)^{-1/2} exp(-(y - mu)^T C^{-1} (y - mu) / 2)`` of ``N_N(mu, C)``."""
    return np.exp(gaussian_logpdf(y, mu, C))


def standard_density(x) -> np.ndarray:
    """Density ``(2 pi)^{-K/2} exp(-|x|^2 / 2)`` of the standard Gaussian vector; ``x`` of shape ``(..., K)``."""
    x = np.asarray(x, dtype=float)
    K = x.shape[-1]
    return (2 * math.pi) ** (-K / 2) * np.exp(-0.5 * np.sum(x * x, axis=-1))


def bivariate_cov(sx: float, sy: float, rho: float) -> np.ndarray:
    """Covariance matrix ``[[sx^2, rho sx sy], [rho sx sy, sy^2]]`` of a bivariate Gaussian."""
    return np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]])


def bivariate_density(x, y, mx: float, my: float, sx: float, sy: float, rho: float) -> np.ndarray:
    """The classical form of the bivariate Gaussian density (``sx, sy > 0``, ``|rho| < 1``)::

        exp(-(u^2 - 2 rho u v + v^2) / (2 (1 - rho^2))) / (2 pi sx sy sqrt(1 - rho^2)),

    with ``u = (x - mx) / sx`` and ``v = (y - my) / sy``.
    """
    u = (np.asarray(x, dtype=float) - mx) / sx
    v = (np.asarray(y, dtype=float) - my) / sy
    q = (u * u - 2 * rho * u * v + v * v) / (1 - rho * rho)
    return np.exp(-0.5 * q) / (2 * math.pi * sx * sy * math.sqrt(1 - rho * rho))


# ---------------------------------------------------------------------------
# The degenerate case
# ---------------------------------------------------------------------------


def support_basis(C, tol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis of ``Im C`` (columns), from the eigenvectors with positive eigenvalue.

    ``N_N(mu, C)`` is concentrated on the affine subspace ``mu + Im C``, of dimension
    ``rank C``; it is a proper subspace exactly when ``C`` is singular.
    """
    C = _check_covariance(C, tol)
    lam, Q = np.linalg.eigh(C)
    return Q[:, lam > tol * _scale(C)]


def distance_to_support(y, mu, C, tol: float = 1e-10) -> np.ndarray:
    """Euclidean distance of the points ``y`` (shape ``(..., N)``) from ``mu + Im C``."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    U = support_basis(C, tol)
    d = np.asarray(y, dtype=float) - mu
    r = d - (d @ U) @ U.T
    return np.sqrt(np.sum(r * r, axis=-1))


# ---------------------------------------------------------------------------
# Independence
# ---------------------------------------------------------------------------


def is_diagonal(C, tol: float = 1e-12) -> bool:
    """True if every off-diagonal entry of ``C`` is zero (relative tolerance): for a Gaussian vector,
    this is equivalent to the independence of its components."""
    C = _as_matrix(C)
    off = C - np.diag(np.diag(C))
    return bool(np.max(np.abs(off), initial=0.0) <= tol * _scale(C))


def blocks_uncorrelated(C, n1: int, tol: float = 1e-12) -> bool:
    """True if ``C_12 = 0``: for a Gaussian vector, the blocks of sizes ``n1`` and ``N - n1`` are independent."""
    C = _as_matrix(C)
    return bool(np.max(np.abs(C[:n1, n1:]), initial=0.0) <= tol * _scale(C))


def _ecf_matrix(x: np.ndarray, s: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``(1/n) sum_k exp(i s_a x_k) exp(i t_b y_k)`` for all grid points, by chunks."""
    n = x.size
    out = np.zeros((s.size, t.size), dtype=complex)
    step = max(1, _CHUNK // max(s.size, t.size))
    for a in range(0, n, step):
        ex = np.exp(1j * np.outer(s, x[a:a + step]))
        ey = np.exp(1j * np.outer(y[a:a + step], t))
        out += ex @ ey
    return out / n


def cf_factorisation_defect(x, y, s=None, t=None) -> float:
    """``max |phi_n(s, t) - phi_n(s, 0) phi_n(0, t)|`` over a grid, with ``phi_n`` the empirical
    characteristic function of the pairs ``(x_k, y_k)``.

    If ``X`` and ``Y`` are independent it tends to ``0``, like ``n^{-1/2}``, by the independence
    criterion of Chapter 10; a value that stays away from ``0`` as ``n`` grows therefore proves that
    they are dependent. The converse fails on a finite grid: dependent variables can factorise at the
    grid points.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    s = np.linspace(-3, 3, 25) if s is None else np.asarray(s, dtype=float).ravel()
    t = s if t is None else np.asarray(t, dtype=float).ravel()
    J = _ecf_matrix(x, np.append(s, 0.0), y, np.append(t, 0.0))   # last row/column: marginals
    joint, phx, phy = J[:-1, :-1], J[:-1, -1], J[-1, :-1]
    return float(np.max(np.abs(joint - np.outer(phx, phy))))


# ---------------------------------------------------------------------------
# Gaussian marginals that are not jointly Gaussian
# ---------------------------------------------------------------------------


def sample_random_sign_pair(n: int, seed=None, rng=None):
    """The pair of the notes: ``X ~ N(0, 1)``, ``Z = +-1`` with probability ``1/2`` independent of
    ``X``, and ``Y = Z X``. Returns ``(X, Y)``; ``Y ~ N(0, 1)``, but ``X + Y = 0`` with probability ``1/2``."""
    g = _rng(seed, rng)
    X = g.standard_normal(n)
    Z = np.where(g.random(n) < 0.5, -1.0, 1.0)
    return X, Z * X


def random_sign_sum_cdf(x, left: bool = False) -> np.ndarray:
    """Distribution function of ``X + ZX``: the mixture ``(1/2) delta_0 + (1/2) N(0, 4)``.

    With ``left=True``, the left limit ``F(x-)``, which differs from ``F(x)`` only at the atom ``0``.
    """
    x = np.asarray(x, dtype=float)
    jump = (x > 0) if left else (x >= 0)
    return 0.5 * jump + 0.5 * special.ndtr(x / 2)


def ks_distance(sample, cdf, cdf_left=None) -> float:
    """``sup_x |F_n(x) - F(x)|``, with ``F_n`` the empirical distribution function of ``sample``.

    ``F_n`` is constant between sample points and ``F`` is non-decreasing, so the supremum is
    reached at a sample point ``u``, comparing ``F_n(u)`` with ``F(u)`` or ``F_n(u-)`` with ``F(u-)``.
    Pass ``cdf_left(x) = F(x-)`` when ``F`` has atoms; by default ``F`` is taken continuous. Ties in
    the sample (which a law with atoms produces) are handled.
    """
    xs = np.sort(np.asarray(sample, dtype=float).ravel())
    n = xs.size
    u = np.unique(xs)
    right = np.searchsorted(xs, u, side="right") / n
    left = np.searchsorted(xs, u, side="left") / n
    F_left = cdf(u) if cdf_left is None else cdf_left(u)
    return float(max(np.max(np.abs(right - cdf(u))), np.max(np.abs(left - F_left))))


def sample_sign_flip_pair(n: int, c: float, seed=None, rng=None):
    """``X ~ N(0, 1)`` and ``Y = X`` where ``|X| <= c``, ``Y = -X`` where ``|X| > c``. Returns ``(X, Y)``."""
    X = _rng(seed, rng).standard_normal(n)
    return X, np.where(np.abs(X) <= c, X, -X)


def sign_flip_covariance(c) -> np.ndarray:
    """``cov(X, Y) = 2 E[X^2 1{|X| <= c}] - 1 = 2 (2 Phi(c) - 1 - 2 c phi(c)) - 1`` for the sign-flip pair."""
    c = np.asarray(c, dtype=float)
    phi = np.exp(-0.5 * c * c) / math.sqrt(2 * math.pi)
    return 2.0 * (2.0 * special.ndtr(c) - 1.0 - 2.0 * c * phi) - 1.0


def sign_flip_root() -> float:
    """The unique ``c > 0`` with ``cov(X, Y) = 0`` for the sign-flip pair (about ``1.5381``)."""
    return float(optimize.brentq(lambda c: float(sign_flip_covariance(c)), 0.1, 5.0, xtol=1e-14))


# ---------------------------------------------------------------------------
# Whitening and sample moments
# ---------------------------------------------------------------------------


def sample_mean_cov(Y):
    """Sample mean and sample covariance matrix (divisor ``n - 1``) of the rows of ``Y``."""
    Y = np.asarray(Y, dtype=float)
    return Y.mean(axis=0), np.atleast_2d(np.cov(Y, rowvar=False, ddof=1))


def whiten(Y, mu=None, C=None, method: str = "zca") -> np.ndarray:
    """Whitened sample ``W_k = M (Y_k - mu)`` with ``M C M^T = I``.

    ``method="zca"`` takes ``M = C^{-1/2}`` (the inverse of the symmetric square root, as in the
    proof of the density theorem); ``method="cholesky"`` takes ``M = L^{-1}`` with ``L L^T = C``.
    If ``mu`` or ``C`` is omitted, the sample mean or covariance is used, and then the sample
    covariance of the result is exactly the identity (up to rounding).
    """
    Y = np.asarray(Y, dtype=float)
    m, S = sample_mean_cov(Y)
    mu = m if mu is None else np.atleast_1d(np.asarray(mu, dtype=float))
    C = S if C is None else _as_matrix(C)
    D = Y - mu
    if method == "zca":
        return D @ psd_inv_sqrt(C)          # psd_inv_sqrt is symmetric
    if method == "cholesky":
        L = np.linalg.cholesky(_check_covariance(C))
        return linalg.solve_triangular(L, D.T, lower=True).T
    raise ValueError(f"unknown method {method!r}")
