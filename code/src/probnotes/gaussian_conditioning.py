"""Conditioning Gaussian vectors (Chapter 14).

Four groups of tools, in the order of the chapter:

* **The bivariate case.** :func:`bivariate_conditional` returns the mean ``m(y)`` and the variance
  ``sigma^2`` of ``X`` given ``Y = y`` for ``(X, Y) ~ N_2(mu, C)``, from the coefficient
  ``a = c_XY / sigma_Y^2`` that makes ``Z = X - a Y`` independent of ``Y``.
* **Block formulas.** For ``X ~ N_N(mu, C)`` split into the components ``target`` and ``given``,
  :func:`regression_matrix` is ``C_12 C_22^{-1}``, :func:`schur_complement` is
  ``C_11 - C_12 C_22^{-1} C_21``, and :func:`conditional_law` returns the conditional mean
  ``mu_1 - C_12 C_22^{-1} (mu_2 - x_2)`` (for one or many values ``x_2``) and the conditional
  covariance. :func:`affine_image` is the law ``N(A mu + b, A C A^T)`` of ``A X + b``.
* **Precision matrices.** :func:`precision_matrix` is ``C^{-1}``; :func:`pair_conditional_cov`
  is the covariance of ``(X_i, X_j)`` given the other components, whose inverse is the
  ``{i, j}`` block of ``C^{-1}``; :func:`partial_correlations` and
  :func:`conditional_independence_graph` read off which pairs are conditionally independent given
  the rest (the zeros of ``C^{-1}``). :func:`partial_sums_cov` and :func:`partial_sums_precision`
  are the matrices of the partial sums of i.i.d. ``N(0, 1)`` variables (tridiagonal precision).
* **Monte Carlo.** :func:`sample_gaussian` draws from ``N_N(mu, C)`` (also degenerate);
  :func:`slab_moments` conditions a sample on a thin slab ``|X_given - x_given| < h`` and returns
  the mean and covariance of the other components with standard errors; :func:`regression_fit`
  fits a least-squares regression of some components on the others, whose coefficients estimate
  the block formula. :func:`squares_cov` is the covariance matrix of ``(X^2, Y^2)`` for a Gaussian
  pair.

Everything is plain NumPy/SciPy; the module does not import the Chapter 13 code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg

__all__ = [
    "affine_image", "correlation_matrix", "bivariate_conditional", "regression_matrix",
    "schur_complement", "ConditionalGaussian", "conditional_law", "precision_matrix",
    "pair_conditional_cov", "partial_correlations", "conditional_independence_graph",
    "partial_sums_cov", "partial_sums_precision", "sample_gaussian", "slab_mask", "SlabEstimate",
    "slab_moments", "RegressionFit", "regression_fit", "squares_cov",
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cov(C, psd_tol: float = 1e-10) -> np.ndarray:
    """Check that ``C`` is a symmetric positive semi-definite matrix and return it as floats."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("a covariance matrix is square")
    if not np.all(np.isfinite(C)):
        raise ValueError("a covariance matrix has finite entries")
    scale = max(1.0, float(np.max(np.abs(C))))
    if not np.allclose(C, C.T, atol=1e-12 * scale, rtol=0):
        raise ValueError("a covariance matrix is symmetric")
    if np.linalg.eigvalsh(C).min() < -psd_tol * scale:
        raise ValueError("a covariance matrix is positive semi-definite")
    return (C + C.T) / 2


def _mean(mu, n: int) -> np.ndarray:
    """The mean vector: ``None`` means ``0``, and a scalar is repeated in every component."""
    mu = np.zeros(n) if mu is None else np.asarray(mu, dtype=float).ravel()
    if mu.size == 1:
        mu = np.full(n, mu[0])
    if mu.shape != (n,):
        raise ValueError(f"the mean vector must have {n} components")
    return mu


def _split(n: int, given, target=None) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays ``(target, given)``; ``target`` defaults to the components not in ``given``."""
    g = np.atleast_1d(np.asarray(given, dtype=int))
    t = (np.setdiff1d(np.arange(n), g) if target is None
         else np.atleast_1d(np.asarray(target, dtype=int)))
    for idx in (g, t):
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise ValueError(f"indices must lie in 0, ..., {n - 1}")
        if np.unique(idx).size != idx.size:
            raise ValueError("indices must not repeat")
    if np.intersect1d(g, t).size:
        raise ValueError("the target and given components must be disjoint")
    if g.size == 0 or t.size == 0:
        raise ValueError("both the target and the given components must be non-empty")
    return t, g


def _cho(C22: np.ndarray):
    """Cholesky factor of the block we condition on; it must be invertible (positive definite)."""
    try:
        return linalg.cho_factor(C22, lower=True, check_finite=True)
    except linalg.LinAlgError as exc:
        raise ValueError("the covariance matrix of the conditioning components must be "
                         "invertible") from exc


# ---------------------------------------------------------------------------
# affine maps and the bivariate case
# ---------------------------------------------------------------------------


def affine_image(mu, C, A, b=None) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of ``A X + b`` for ``X`` with mean ``mu`` and covariance ``C``.

    For a Gaussian ``X ~ N_N(mu, C)`` this is the law ``N_M(A mu + b, A C A^T)`` of the affine image.
    """
    C = _cov(C)
    mu = _mean(mu, C.shape[0])
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[1] != C.shape[0]:
        raise ValueError("A must have one column per component of X")
    b = np.zeros(A.shape[0]) if b is None else np.asarray(b, dtype=float).ravel()
    return A @ mu + b, A @ C @ A.T


def correlation_matrix(C) -> np.ndarray:
    """``rho_ij = c_ij / sqrt(c_ii c_jj)``; components with variance ``0`` give ``nan``."""
    C = _cov(C)
    s = np.sqrt(np.diag(C))
    with np.errstate(invalid="ignore", divide="ignore"):
        R = C / np.outer(s, s)
    return R


def bivariate_conditional(y, mu_x: float, mu_y: float, var_x: float, var_y: float, cov_xy: float):
    """Mean ``m(y)`` and variance ``sigma^2`` of ``X`` given ``Y = y`` for a Gaussian pair.

    With ``a = cov_xy / var_y``, ``m(y) = mu_x - a (mu_y - y)`` and
    ``sigma^2 = var_x - cov_xy^2 / var_y = var_x (1 - rho^2)``. ``y`` may be an array. Requires
    ``var_y > 0``; returns ``(m, sigma2)``.
    """
    _cov([[var_x, cov_xy], [cov_xy, var_y]])
    if var_y <= 0:
        raise ValueError("the conditioning variable must have positive variance")
    a = cov_xy / var_y
    m = mu_x - a * (mu_y - np.asarray(y, dtype=float))
    return m, max(var_x - cov_xy ** 2 / var_y, 0.0)


# ---------------------------------------------------------------------------
# block formulas
# ---------------------------------------------------------------------------


def regression_matrix(C, given, target=None) -> np.ndarray:
    """``C_12 C_22^{-1}``, with ``1`` = ``target`` (default: all other components), ``2`` = ``given``.

    It is the matrix ``A`` such that ``X_1 - A X_2`` is uncorrelated with (for a Gaussian vector,
    independent of) ``X_2``. Requires ``C_22`` invertible.
    """
    C = _cov(C)
    t, g = _split(C.shape[0], given, target)
    return linalg.cho_solve(_cho(C[np.ix_(g, g)]), C[np.ix_(g, t)]).T


def schur_complement(C, given, target=None) -> np.ndarray:
    """``C_11 - C_12 C_22^{-1} C_21``: the conditional covariance of ``X_1`` given ``X_2``."""
    C = _cov(C)
    t, g = _split(C.shape[0], given, target)
    S = C[np.ix_(t, t)] - C[np.ix_(t, g)] @ linalg.cho_solve(_cho(C[np.ix_(g, g)]), C[np.ix_(g, t)])
    return (S + S.T) / 2


@dataclass(frozen=True)
class ConditionalGaussian:
    """Conditional law ``N(mean, cov)`` of ``X[target]`` given ``X[given] = x_given``.

    ``mean`` has shape ``(len(target),)`` for one conditioning value and ``(k, len(target))`` for
    ``k`` values; ``cov`` does not depend on the value. ``A`` is the regression matrix
    ``C_12 C_22^{-1}``, so that ``mean = mu_1 - A (mu_2 - x_given)``.
    """
    mean: np.ndarray
    cov: np.ndarray
    A: np.ndarray
    target: np.ndarray
    given: np.ndarray


def conditional_law(mu, C, given, x_given, target=None) -> ConditionalGaussian:
    """Block formulas: ``X_1 | X_2 = x_2 ~ N(mu_1 - C_12 C_22^{-1}(mu_2 - x_2), C_11 - C_12 C_22^{-1} C_21)``.

    ``given`` lists the components conditioned on, ``target`` those whose law is wanted (default:
    all the others). ``x_given`` has shape ``(len(given),)`` or ``(k, len(given))``. Requires
    ``C_22`` invertible (``C`` itself may be singular; then the conditional covariance is).
    """
    C = _cov(C)
    n = C.shape[0]
    mu = _mean(mu, n)
    t, g = _split(n, given, target)
    x = np.asarray(x_given, dtype=float)
    if x.shape[-1:] != (g.size,) or x.ndim > 2:
        raise ValueError("x_given must have shape (len(given),) or (k, len(given))")
    A = regression_matrix(C, g, t)
    mean = mu[t] + (x - mu[g]) @ A.T
    return ConditionalGaussian(mean, schur_complement(C, g, t), A, t, g)


# ---------------------------------------------------------------------------
# precision matrices and conditional independence
# ---------------------------------------------------------------------------


def precision_matrix(C) -> np.ndarray:
    """``C^{-1}`` for an invertible covariance matrix (via its Cholesky factor)."""
    C = _cov(C)
    P = linalg.cho_solve(_cho(C), np.eye(C.shape[0]))
    return (P + P.T) / 2


def pair_conditional_cov(C, i: int, j: int) -> np.ndarray:
    """Covariance ``Sigma`` (2 x 2) of ``(X_i, X_j)`` given the other components.

    Its inverse is the ``{i, j}`` block of ``C^{-1}``, so ``Sigma`` is diagonal (``X_i`` and ``X_j``
    are conditionally independent given the rest) if and only if ``(C^{-1})_{ij} = 0``. For
    ``N = 2`` there is nothing to condition on and ``Sigma`` is the ``{i, j}`` block of ``C``.
    """
    C = _cov(C)
    n = C.shape[0]
    if i == j:
        raise ValueError("i and j must differ")
    rest = [k for k in range(n) if k not in (i, j)]
    if not rest:
        return C[np.ix_([i, j], [i, j])].copy()
    return schur_complement(C, rest, [i, j])


def partial_correlations(C) -> np.ndarray:
    """Matrix of the correlations of ``(X_i, X_j)`` given the other components.

    Equals ``-P_ij / sqrt(P_ii P_jj)`` with ``P = C^{-1}``, and ``1`` on the diagonal.
    """
    P = precision_matrix(C)
    d = np.sqrt(np.diag(P))
    R = -P / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def conditional_independence_graph(C, tol: float = 1e-9) -> list[tuple[int, int]]:
    """Edges ``(i, j)``, ``i < j``, with ``(C^{-1})_{ij} != 0`` (relative to ``tol``).

    A missing edge means that ``X_i`` and ``X_j`` are conditionally independent given the other
    components: this is the Gaussian graphical model of ``C``.
    """
    R = partial_correlations(C)
    n = R.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if abs(R[i, j]) > tol]


def partial_sums_cov(N: int, var: float = 1.0) -> np.ndarray:
    """Covariance ``var * min(k, l)`` of ``S_k = X_1 + ... + X_k``, ``k = 1..N``, ``X_i`` i.i.d.

    ``S = A X`` with ``A`` lower triangular of ones, so this is ``var * A A^T``.
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    k = np.arange(1, N + 1)
    return var * np.minimum.outer(k, k).astype(float)


def partial_sums_precision(N: int, var: float = 1.0) -> np.ndarray:
    """Inverse of :func:`partial_sums_cov`: tridiagonal, ``2`` on the diagonal except a final
    ``1``, ``-1`` next to the diagonal (all divided by ``var``)."""
    if N < 1:
        raise ValueError("N must be at least 1")
    P = 2.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    P[-1, -1] = 1.0
    return P / var


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


def sample_gaussian(mu, C, n: int, seed=None) -> np.ndarray:
    """``n`` draws of ``N_N(mu, C)``, one per row, as ``mu + A X`` with ``X ~ N_N(0, I)``.

    ``A`` is the Cholesky factor of ``C`` if ``C`` is invertible and ``Q diag(sqrt(lambda))`` from
    the spectral decomposition otherwise, so degenerate laws are allowed (``A A^T = C``).
    """
    C = _cov(C)
    d = C.shape[0]
    mu = _mean(mu, d)
    try:
        A = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        lam, Q = np.linalg.eigh(C)
        A = Q * np.sqrt(np.clip(lam, 0.0, None))
    rng = np.random.default_rng(seed)
    return mu + rng.standard_normal((int(n), d)) @ A.T


def slab_mask(samples, given, x_given, half_width: float) -> np.ndarray:
    """Rows of ``samples`` with ``|X_k - x_k| < half_width`` for every ``k`` in ``given``."""
    s = np.asarray(samples, dtype=float)
    g = np.atleast_1d(np.asarray(given, dtype=int))
    x = np.atleast_1d(np.asarray(x_given, dtype=float))
    if x.shape != g.shape:
        raise ValueError("give one conditioning value per given component")
    if half_width <= 0:
        raise ValueError("the half-width of the slab must be positive")
    return np.all(np.abs(s[:, g] - x) < half_width, axis=1)


@dataclass(frozen=True)
class SlabEstimate:
    """Sample moments of ``X[target]`` over the draws in a slab around ``X[given] = x_given``.

    ``n`` is the number of draws in the slab, ``mean`` and ``cov`` their sample mean and covariance
    (``ddof = 1``), ``se`` the standard errors of the means, and ``given_mean`` the average of
    ``X[given]`` over the slab (which differs slightly from ``x_given``).
    """
    n: int
    mean: np.ndarray
    cov: np.ndarray
    se: np.ndarray
    given_mean: np.ndarray


def slab_moments(samples, given, x_given, half_width: float, target=None) -> SlabEstimate:
    """Condition a Gaussian sample on a thin slab and estimate the conditional mean and covariance.

    Since ``P(X_2 = x_2) = 0``, the event ``{X_2 = x_2}`` is replaced by
    ``{|X_2 - x_2| < h}``; as ``h -> 0`` (and the sample grows) the moments tend to those of
    :func:`conditional_law`.
    """
    s = np.asarray(samples, dtype=float)
    t, g = _split(s.shape[1], given, target)
    sel = s[slab_mask(s, g, x_given, half_width)]
    k = sel.shape[0]
    if k < 2:
        raise ValueError(f"only {k} draws in the slab: widen it or draw more")
    y = sel[:, t]
    cov = np.atleast_2d(np.cov(y, rowvar=False, ddof=1))
    return SlabEstimate(k, y.mean(axis=0), cov, np.sqrt(np.diag(cov) / k), sel[:, g].mean(axis=0))


@dataclass(frozen=True)
class RegressionFit:
    """Least-squares fit ``X_1 ~ b + B X_2`` on a sample.

    ``intercept`` estimates ``mu_1 - A mu_2``, ``slope`` (shape ``(len(target), len(given))``)
    estimates ``A = C_12 C_22^{-1}``, ``residual_cov`` estimates ``C_11 - C_12 C_22^{-1} C_21``,
    and ``slope_se`` gives the usual standard errors of the slopes.
    """
    intercept: np.ndarray
    slope: np.ndarray
    residual_cov: np.ndarray
    slope_se: np.ndarray


def regression_fit(samples, given, target=None) -> RegressionFit:
    """Regress the ``target`` components of a sample on the ``given`` ones, with an intercept."""
    s = np.asarray(samples, dtype=float)
    t, g = _split(s.shape[1], given, target)
    n = s.shape[0]
    if n <= g.size + 1:
        raise ValueError("not enough draws for the regression")
    D = np.column_stack([np.ones(n), s[:, g]])
    coef, *_ = np.linalg.lstsq(D, s[:, t], rcond=None)
    res = s[:, t] - D @ coef
    rcov = np.atleast_2d(res.T @ res / (n - g.size - 1))
    G = np.linalg.inv(D.T @ D)[1:, 1:]
    se = np.sqrt(np.outer(np.diag(rcov), np.diag(G)))
    return RegressionFit(coef[0], coef[1:].T, rcov, se)


def squares_cov(mu, C) -> np.ndarray:
    """Covariance matrix of ``(X^2, Y^2)`` for a Gaussian pair with mean ``mu``, covariance ``C``.

    ``cov(X^2, Y^2) = 2 c_XY^2 + 4 mu_X mu_Y c_XY`` and ``var(X^2) = 2 sigma_X^4 + 4 mu_X^2 sigma_X^2``.
    """
    C = _cov(C)
    if C.shape != (2, 2):
        raise ValueError("a pair has a 2 x 2 covariance matrix")
    mu = _mean(mu, 2)
    return 2 * C ** 2 + 4 * np.outer(mu, mu) * C
