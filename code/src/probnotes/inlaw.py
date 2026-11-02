"""Convergence in law (Chapter 16).

Four groups of tools, in the order of the chapter:

* **Distribution functions and distances between laws.** :func:`ecdf` is the empirical
  distribution function ``F_N(x) = #{k : X_k <= x} / N`` of a sample, and
  :func:`kolmogorov_distance_sample` the exact distance ``sup_x |F_N(x) - F(x)|`` to a
  distribution function ``F`` (continuous, or with atoms if its left limits are given).
  For two laws on ``{0, 1, 2, ...}`` given by their discrete densities,
  :func:`kolmogorov_distance_pmf` is ``sup_x |F(x) - G(x)|`` and :func:`total_variation_pmf` is
  ``sup_A |P(A) - Q(A)| = (1/2) sum_k |p_k - q_k|``. Convergence in law is convergence of the
  distribution functions at the continuity points of the limit; the total variation distance
  measures the stronger convergence ``P_{X_n}(A) -> P_X(A)`` for every Borel set ``A``.
  :func:`one_over_n_cdf` is the distribution function of the constant ``1/n`` of the notes'
  example. :func:`empirical_charfun` estimates a characteristic function from a sample.
* **The binomial and the Poisson law** (worked exercise 3). :func:`binomial_poisson_pmfs`
  returns the discrete densities of ``B(n, lam/n)`` and ``P(lam)``,
  :func:`binomial_poisson_distances` the exact total variation and Kolmogorov distances between
  them, :func:`le_cam_bound` the bound ``n p^2 = lam^2 / n`` for the first, and
  :func:`poisson_approximation` the probability ``P(X_n <= x)`` exactly and by the Poisson
  approximation. For the exercises, :func:`poisson_binomial_pmf` is the law of a sum of
  independent Bernoulli variables with different parameters, :func:`bernoulli_poisson_coupling`
  is Le Cam's coupling of ``b(p)`` and ``P(p)``, and :func:`matching_pmf` is the law of the
  number of fixed points of a random permutation. :func:`binomial_charfun`, :func:`poisson_charfun` and :func:`normal_charfun`
  are characteristic functions.
* **The maximum of uniforms** (worked exercise 2). :func:`scaled_max_cdf` is the exact
  distribution function of ``Z_n = n (1 - max(U_1, ..., U_n))``, :func:`scaled_max_kolmogorov`
  the exact distance ``sup_x |F_{Z_n}(x) - (1 - e^{-x})|`` to the exponential law, with its
  maximiser, and :func:`sample_scaled_max` simulates ``Z_n`` without drawing ``n`` uniforms.
* **The relaxed law of large numbers** (worked exercise 1) and **sums.**
  :func:`relaxed_mean_moments` gives the mean ``2p`` and the variance ``(4n - 2) p (1 - p) / n^2``
  of ``T_n = (1/n) sum_{i <= n} (X_i + X_{i+1})``, :func:`sample_relaxed_mean` simulates it
  exactly, and :func:`relaxed_mean_paths` simulates whole paths ``n -> T_n``.
  :func:`sum_counterexample` simulates the notes' counterexample for sums, ``X_n + Y_n`` against
  ``X + Y``, with an optional correlation between ``X_n`` and ``Y_n``.

All functions are vectorised with NumPy where it matters.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import optimize, stats

__all__ = [
    "ecdf", "kolmogorov_distance_sample", "kolmogorov_distance_pmf", "total_variation_pmf",
    "one_over_n_cdf", "empirical_charfun",
    "binomial_poisson_pmfs", "binomial_poisson_distances", "le_cam_bound", "poisson_approximation",
    "poisson_binomial_pmf", "matching_pmf", "bernoulli_poisson_coupling",
    "binomial_charfun", "poisson_charfun", "normal_charfun",
    "scaled_max_cdf", "scaled_max_kolmogorov", "sample_scaled_max",
    "relaxed_mean_moments", "sample_relaxed_mean", "relaxed_mean_paths", "sum_counterexample",
]

# ---------------------------------------------------------------------------
# Distribution functions and distances between laws
# ---------------------------------------------------------------------------


def ecdf(sample, x) -> np.ndarray:
    """Empirical distribution function ``F_N(x) = #{k : X_k <= x} / N`` (vectorised in ``x``)."""
    s = np.sort(np.asarray(sample, dtype=float).ravel())
    if s.size == 0:
        raise ValueError("empty sample")
    return np.searchsorted(s, np.asarray(x, dtype=float), side="right") / s.size


def kolmogorov_distance_sample(sample, cdf: Callable, cdf_left: Callable | None = None) -> float:
    """Exact ``sup_x |F_N(x) - F(x)|`` between the empirical distribution function and ``F``.

    ``cdf`` is ``F`` (right-continuous, vectorised); ``cdf_left`` is ``x -> F(x-) = P(X < x)``,
    which equals ``cdf`` when ``F`` is continuous (the default). Between two consecutive
    distinct sample points ``F_N`` is constant and ``F`` is monotone, so the supremum is
    attained, or approached, at the sample points: it is the largest of
    ``|F_N(x_i) - F(x_i)|`` and ``|F_N(x_i-) - F(x_i-)|``. Ties are allowed.
    """
    s = np.sort(np.asarray(sample, dtype=float).ravel())
    if s.size == 0:
        raise ValueError("empty sample")
    u, counts = np.unique(s, return_counts=True)
    right = np.cumsum(counts) / s.size                # F_N(u_i)
    left = right - counts / s.size                    # F_N(u_i-)
    F = np.asarray(cdf(u), dtype=float)
    Fl = F if cdf_left is None else np.asarray(cdf_left(u), dtype=float)
    return float(max(np.max(np.abs(right - F)), np.max(np.abs(left - Fl))))


def _pad(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = max(p.size, q.size)
    return np.pad(p, (0, m - p.size)), np.pad(q, (0, m - q.size))


def kolmogorov_distance_pmf(p, q) -> float:
    """``sup_x |F(x) - G(x)|`` for two laws on ``{0, 1, ..., K}`` with discrete densities ``p``, ``q``.

    Both distribution functions are constant on every ``[k, k+1)``, so the supremum is the
    largest ``|F(k) - G(k)|``. Missing entries count as ``0``.
    """
    p, q = _pad(p, q)
    return float(np.max(np.abs(np.cumsum(p - q))))


def total_variation_pmf(p, q) -> float:
    """Total variation distance ``sup_A |P(A) - Q(A)| = (1/2) sum_k |p_k - q_k|`` of two laws on ``N``."""
    p, q = _pad(p, q)
    return float(0.5 * np.sum(np.abs(p - q)))


def one_over_n_cdf(n, x) -> np.ndarray:
    """Distribution function of the constant random variable ``X_n = 1/n``: ``1`` if ``x >= 1/n``, else ``0``.

    ``n = inf`` gives the limit ``X = 0``. Broadcasts ``n`` against ``x``.
    """
    n = np.asarray(n, dtype=float)
    x = np.asarray(x, dtype=float)
    return (x >= 1.0 / n).astype(float)


def empirical_charfun(sample, t) -> np.ndarray:
    """Empirical characteristic function ``(1/N) sum_k exp(i t X_k)`` (vectorised in ``t``)."""
    s = np.asarray(sample, dtype=float).ravel()
    t = np.asarray(t, dtype=float)
    out = np.empty(t.shape, dtype=complex)
    flat = t.ravel()
    for start in range(0, flat.size, 64):            # keep the (t, sample) matrix small
        tt = flat[start:start + 64, None]
        out.ravel()[start:start + 64] = np.exp(1j * tt * s[None, :]).mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# The binomial and the Poisson law
# ---------------------------------------------------------------------------


def _check_n_lam(n: int, lam: float) -> float:
    if lam <= 0:
        raise ValueError("lam must be positive")
    if n < lam:
        raise ValueError("B(n, lam/n) needs n >= lam")
    return lam / n


def binomial_poisson_pmfs(n: int, lam: float, kmax: int | None = None):
    """Discrete densities of ``B(n, lam/n)`` and ``P(lam)`` on ``k = 0, ..., kmax``.

    The default ``kmax`` is ``n``. Returns ``(k, binomial, poisson)``.
    """
    p = _check_n_lam(n, lam)
    kmax = n if kmax is None else int(kmax)
    k = np.arange(kmax + 1)
    return k, stats.binom.pmf(k, n, p), stats.poisson.pmf(k, lam)


def binomial_poisson_distances(n, lam: float):
    """Exact total variation and Kolmogorov distances between ``B(n, lam/n)`` and ``P(lam)``.

    Vectorised in ``n`` (each ``n >= lam``). The sums run over ``k <= K``, where both laws have
    mass below ``1e-17`` beyond ``K``; the mass beyond ``K`` is added to the total variation
    distance, which is therefore exact to within about ``1e-17``. Returns ``(tv, kolmogorov)``.
    """
    ns = np.atleast_1d(np.asarray(n))
    tv = np.empty(ns.shape, dtype=float)
    ks = np.empty(ns.shape, dtype=float)
    kcut = int(math.ceil(lam))
    while stats.poisson.logsf(kcut, lam) > math.log(1e-17):     # P(Y > kcut) < 1e-17
        kcut += max(1, int(math.sqrt(lam)))
    for i, nn in enumerate(ns.ravel()):
        nn = int(nn)
        p = _check_n_lam(nn, lam)
        K = min(nn, kcut)
        k = np.arange(K + 1)
        b, q = stats.binom.pmf(k, nn, p), stats.poisson.pmf(k, lam)
        tail_b, tail_q = stats.binom.sf(K, nn, p), stats.poisson.sf(K, lam)
        tv.ravel()[i] = 0.5 * (np.sum(np.abs(b - q)) + tail_b + tail_q)
        Fb, Fq = stats.binom.cdf(k, nn, p), stats.poisson.cdf(k, lam)
        ks.ravel()[i] = max(np.max(np.abs(Fb - Fq)), tail_b, tail_q)
    if np.ndim(n) == 0:
        return float(tv[0]), float(ks[0])
    return tv, ks


def le_cam_bound(n, lam: float) -> np.ndarray:
    """Le Cam's bound ``n p^2 = lam^2 / n`` on the total variation distance between ``B(n, p)`` and ``P(np)``."""
    return lam ** 2 / np.asarray(n, dtype=float)


def poisson_approximation(n: int, lam: float, x: float):
    """``(P(X_n <= x), P(Y <= x))`` with ``X_n ~ B(n, lam/n)`` and ``Y ~ P(lam)``."""
    p = _check_n_lam(n, lam)
    return float(stats.binom.cdf(x, n, p)), float(stats.poisson.cdf(x, lam))


def poisson_binomial_pmf(probs) -> np.ndarray:
    """Discrete density of ``S = I_1 + ... + I_n`` with independent ``I_i ~ b(p_i)``, on ``k = 0, ..., n``.

    Computed by adding one Bernoulli variable at a time (a discrete convolution).
    """
    probs = np.asarray(probs, dtype=float).ravel()
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    pmf = np.zeros(probs.size + 1)
    pmf[0] = 1.0
    for i, p in enumerate(probs):
        pmf[1:i + 2] = pmf[1:i + 2] * (1 - p) + pmf[0:i + 1] * p
        pmf[0] *= 1 - p
    return pmf


def matching_pmf(n: int) -> np.ndarray:
    """Law of the number of fixed points of a uniform random permutation of ``n`` objects.

    ``P(M_n = k) = (1/k!) sum_{j=0}^{n-k} (-1)^j / j!`` for ``k = 0, ..., n`` (the matching
    problem). Returns the array of these probabilities.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    j = np.arange(n + 1)
    partial = np.cumsum((-1.0) ** j / np.array([math.factorial(int(i)) for i in j], dtype=float))
    k = np.arange(n + 1)
    return np.array([partial[n - kk] / math.factorial(int(kk)) for kk in k])


def bernoulli_poisson_coupling(p: float, u) -> tuple:
    """Le Cam's coupling: ``(I, N)`` with ``I ~ b(p)``, ``N ~ P(p)``, both functions of ``U ~ U(0, 1)``.

    ``I = 1{U >= 1 - p}`` and ``N = F^{-1}(U)`` for the Poisson distribution function ``F``.
    They differ exactly when ``1 - p <= U < e^{-p}`` or ``U >= (1 + p) e^{-p}``, an event of
    probability ``p (1 - e^{-p}) <= p^2``.
    """
    u = np.asarray(u, dtype=float)
    ind = (u >= 1 - p).astype(int)
    N = stats.poisson.ppf(u, p).astype(int)        # smallest k with F(k) >= u
    return ind, N


def binomial_charfun(n: int, p: float, t) -> np.ndarray:
    """Characteristic function ``(1 - p + p e^{it})^n`` of ``B(n, p)``."""
    t = np.asarray(t, dtype=float)
    return (1 - p + p * np.exp(1j * t)) ** n


def poisson_charfun(lam: float, t) -> np.ndarray:
    """Characteristic function ``exp(lam (e^{it} - 1))`` of ``P(lam)``."""
    t = np.asarray(t, dtype=float)
    return np.exp(lam * (np.exp(1j * t) - 1))


def normal_charfun(var: float, t, mean: float = 0.0) -> np.ndarray:
    """Characteristic function ``exp(i mean t - var t^2 / 2)`` of ``N(mean, var)``."""
    t = np.asarray(t, dtype=float)
    return np.exp(1j * mean * t - 0.5 * var * t ** 2)


# ---------------------------------------------------------------------------
# The maximum of uniforms
# ---------------------------------------------------------------------------


def scaled_max_cdf(n, x) -> np.ndarray:
    """Exact distribution function of ``Z_n = n (1 - Y_n)``, ``Y_n = max(U_1, ..., U_n)``, ``U_i ~ U(0, 1)`` i.i.d.

    ``F_n(x) = 1 - (1 - x/n)^n`` for ``0 <= x <= n``, ``0`` for ``x < 0`` and ``1`` for
    ``x >= n``; ``n = inf`` gives the limit ``1 - e^{-x}`` (``x >= 0``). Broadcasts ``n``
    against ``x``.
    """
    n = np.asarray(n, dtype=float)
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = np.where(np.isinf(n), -x, n * np.log1p(-np.clip(x / n, 0.0, 1.0)))
        F = -np.expm1(inner)                                  # 1 - (1 - x/n)^n
    return np.where(x < 0, 0.0, np.where(x >= n, 1.0, F))


def scaled_max_kolmogorov(n, return_argmax: bool = False):
    """Exact ``sup_x |F_{Z_n}(x) - (1 - e^{-x})|`` for ``Z_n = n (1 - max(U_1, ..., U_n))``.

    On ``[0, n]`` the gap ``g(x) = e^{-x} - (1 - x/n)^n`` is non-negative; ``g'`` has the sign
    of ``h(x) = (n-1) log(1 - x/n) + x``, which increases on ``[0, 1]`` and decreases on
    ``[1, n)``, so ``g`` has a single maximum at the zero ``x*`` of ``h`` in ``(1, n)`` (``x* = 1``
    for ``n = 1``). Beyond ``n`` the gap is ``e^{-x} <= e^{-n}``. As ``n -> inf``, ``x* -> 2``
    and ``n * sup -> 2 e^{-2}``. Vectorised in ``n``; with ``return_argmax`` also returns ``x*``.
    """
    ns = np.atleast_1d(np.asarray(n))
    sup = np.empty(ns.shape, dtype=float)
    arg = np.empty(ns.shape, dtype=float)
    for i, nn in enumerate(ns.ravel()):
        nn = int(nn)
        if nn < 1:
            raise ValueError("n must be a positive integer")
        if nn == 1:
            xs = 1.0
        else:
            h = lambda x: (nn - 1) * math.log1p(-x / nn) + x
            hi = nn * (1 - 1e-15)
            xs = optimize.brentq(h, 1.0, hi, xtol=1e-14) if h(hi) < 0 else hi
        g = math.exp(-xs) - (0.0 if xs >= nn else math.exp(nn * math.log1p(-xs / nn)))
        sup.ravel()[i] = max(g, math.exp(-nn))
        arg.ravel()[i] = xs if g >= math.exp(-nn) else float(nn)
    if np.ndim(n) == 0:
        sup, arg = float(sup[0]), float(arg[0])
    return (sup, arg) if return_argmax else sup


def sample_scaled_max(n: int, size: int, seed=None, direct: bool = False) -> np.ndarray:
    """Draw ``size`` values of ``Z_n = n (1 - max(U_1, ..., U_n))``.

    By default ``Y_n = max U_i`` is drawn as ``V^{1/n}`` with ``V ~ U(0, 1)``, which has the same
    distribution function ``y^n`` on ``(0, 1)``; ``1 - V^{1/n}`` is computed as
    ``-expm1(log(V) / n)`` to keep its precision for large ``n``. With ``direct=True`` the ``n``
    uniforms are drawn and their maximum taken (only for small ``n * size``).
    """
    rng = np.random.default_rng(seed)
    if direct:
        if n * size > 10 ** 7:
            raise ValueError("n * size too large for direct simulation")
        return n * (1.0 - rng.random((size, n)).max(axis=1))
    v = rng.random(size)
    return -n * np.expm1(np.log(v) / n)


# ---------------------------------------------------------------------------
# The relaxed law of large numbers, and sums
# ---------------------------------------------------------------------------


def relaxed_mean_moments(n, p: float):
    """Mean ``2p`` and variance ``(4n - 2) p (1 - p) / n^2`` of ``T_n = (1/n) sum_{i=1}^n (X_i + X_{i+1})``.

    Here ``X_i ~ b(p)`` i.i.d.; vectorised in ``n``.
    """
    n = np.asarray(n, dtype=float)
    return 2 * p + 0 * n, (4 * n - 2) * p * (1 - p) / n ** 2


def sample_relaxed_mean(n: int, p: float, size: int, seed=None) -> np.ndarray:
    """Draw ``size`` values of ``T_n`` exactly, as ``(X_1 + X_{n+1} + 2 B) / n`` with ``B ~ B(n - 1, p)``.

    Indeed ``sum_{i=1}^n (X_i + X_{i+1}) = X_1 + 2 (X_2 + ... + X_n) + X_{n+1}``, and the three
    pieces are independent.
    """
    rng = np.random.default_rng(seed)
    ends = rng.binomial(2, p, size)                              # X_1 + X_{n+1}
    middle = rng.binomial(n - 1, p, size) if n > 1 else np.zeros(size, dtype=int)
    return (ends + 2 * middle) / n


def relaxed_mean_paths(N: int, p: float, paths: int = 1, seed=None) -> np.ndarray:
    """Paths ``n -> T_n``, ``n = 1, ..., N``, from one sequence ``X_1, ..., X_{N+1}`` per path.

    Returns an array of shape ``(paths, N)``; keep ``paths * N`` below about ``10^7``.
    """
    if paths * (N + 1) > 2 * 10 ** 7:
        raise ValueError("paths * N too large")
    rng = np.random.default_rng(seed)
    X = (rng.random((paths, N + 1)) < p).astype(np.int32)
    Y = X[:, :-1] + X[:, 1:]                                     # Y_i = X_i + X_{i+1}
    return np.cumsum(Y, axis=1) / np.arange(1, N + 1)


def sum_counterexample(size: int, rho: float = 0.0, seed=None):
    """The notes' counterexample for sums: samples of ``X_n + Y_n`` and of ``X + Y``.

    ``X_n, Y_n ~ N(0, 1)`` with correlation ``rho`` (``rho = 0``: independent, as in the notes),
    ``X ~ N(0, 1)`` and ``Y = -X``. Each marginal has law ``N(0, 1)`` whatever ``rho`` is, but
    ``X_n + Y_n ~ N(0, 2 + 2 rho)`` while ``X + Y = 0``. Returns ``(sums_n, sums_limit)``.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    rng = np.random.default_rng(seed)
    xn = rng.standard_normal(size)
    yn = rho * xn + math.sqrt(1.0 - rho ** 2) * rng.standard_normal(size)
    x = rng.standard_normal(size)
    return xn + yn, x + (-x)
