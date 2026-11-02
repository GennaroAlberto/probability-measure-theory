"""Convergence of random variables and the laws of large numbers (Chapter 15).

Four groups of tools, in the order of the chapter:

* **The two counterexamples** on ``Omega = [0, 1]`` with Lebesgue measure. The *typewriter*
  sequence (Counterexample 1) runs through the indicators ``Y_{k,j} = 1_[(j-1)/k, j/k]``,
  ``j = 1, ..., k``, block after block: :func:`typewriter_index` maps ``n`` to ``(k, j)``,
  :func:`typewriter_interval` gives the interval ``A_n`` and :func:`typewriter` evaluates
  ``X_n(omega)``. The *spike* ``X_n = 2^n 1_(0, 1/n)`` (Counterexample 2) is :func:`spike`.
  :func:`mode_quantities` returns, exactly, the three numbers that decide the three modes of
  convergence to ``0``: ``P(|X_n| > eps)``, ``E|X_n|^p`` and the metric ``E[|X_n| / (1 + |X_n|)]``
  of the characterisation of convergence in probability. :func:`typewriter_hits` lists the
  ``n`` with ``X_n(omega) = 1`` at a fixed point, exactly when ``omega`` is a
  :class:`~fractions.Fraction`.
* **The metric of convergence in probability**, ``d(X, Y) = E[|X - Y| / (1 + |X - Y|)]``, for
  random variables given as samples or as values on a grid of ``Omega`` (:func:`prob_metric`).
* **Sample means.** :data:`LAWS` lists a few laws with their mean and variance (the Cauchy law
  has neither); :func:`sample_means` simulates paths ``n -> Xbar_n``, :func:`deviation_probability`
  estimates ``P(|Xbar_n - mu| > eps)`` from them, and :func:`tail_sup_deviation` computes
  ``sup_{n <= m <= N} |Xbar_m - mu|`` along each path, the quantity that the strong law controls.
  :func:`chebyshev_bound` is the bound ``sigma^2 / (n eps^2)`` of the proof of the weak law,
  :func:`chebyshev_sample_size` the sample size it guarantees, and :func:`bernoulli_mean_tail`
  the exact ``P(|Xbar_n - p| > eps)`` for Bernoulli variables, to compare with.
  :func:`first_n_below` and :func:`last_n_above` find sample sizes from a tail function.
* **The proof of the strong law.** :func:`square_floor` is ``k_m = floor(sqrt(m))``, and
  :func:`square_gap_variance` the exact ``E[(Xbar_m - (k_m^2 / m) Xbar_{k_m^2})^2]`` for centred
  variables, with the bound ``3 sigma^2 / m^{3/2}`` of the notes. :func:`bernstein_polynomial`
  evaluates the Bernstein polynomials of an exercise.

All functions are vectorised with NumPy where it matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import numpy as np
from scipy import stats

__all__ = [
    "typewriter_index", "typewriter_interval", "typewriter", "spike", "mode_quantities",
    "typewriter_hits", "prob_metric", "Law", "LAWS", "sample_means", "deviation_probability",
    "tail_sup_deviation", "chebyshev_bound", "chebyshev_sample_size", "bernoulli_mean_tail",
    "first_n_below", "last_n_above", "square_floor", "square_gap_variance",
    "bernstein_polynomial",
]

# ---------------------------------------------------------------------------
# The two counterexamples
# ---------------------------------------------------------------------------


def _positive_int(n) -> np.ndarray:
    n = np.asarray(n)
    if not np.issubdtype(n.dtype, np.integer):
        if not np.all(np.equal(np.mod(n, 1), 0)):
            raise ValueError("n must be a positive integer")
        n = n.astype(np.int64)
    if np.any(n < 1):
        raise ValueError("n must be a positive integer")
    return n.astype(np.int64)


def typewriter_index(n):
    """Block ``k`` and position ``j`` of the ``n``-th term of the typewriter sequence.

    ``X_n = Y_{k,j}`` with ``n = k(k-1)/2 + j`` and ``1 <= j <= k``: the first block (``k = 1``)
    has one term, the second two, and so on. Vectorised; ``n >= 1``.
    """
    n = _positive_int(n)
    k = np.floor((np.sqrt(8.0 * n + 1.0) - 1.0) / 2.0).astype(np.int64)
    k = k + (k * (k + 1) // 2 < n)                # correct the float estimate: k(k+1)/2 >= n
    k = k - ((k - 1) * k // 2 >= n)               # ... and (k-1)k/2 < n
    j = n - k * (k - 1) // 2
    return k, j


def typewriter_interval(n):
    """The interval ``A_n = [(j-1)/k, j/k]`` with ``X_n = 1_{A_n}``, as arrays ``(a, b)``."""
    k, j = typewriter_index(n)
    return (j - 1) / k, j / k


def typewriter(n, omega):
    """``X_n(omega)`` for the typewriter sequence (broadcasting ``n`` against ``omega``).

    ``omega`` may be a :class:`~fractions.Fraction` (or an integer), in which case the test
    ``(j-1)/k <= omega <= j/k`` is done in exact arithmetic; this matters at the endpoints
    ``j/k``, which belong to two consecutive intervals of the same block.
    """
    k, j = typewriter_index(n)
    if isinstance(omega, (Fraction, int)):
        w = Fraction(omega)
        p, q = w.numerator, w.denominator
        pk = p * k                                   # omega * k = p k / q
        return (((j - 1) * q <= pk) & (pk <= j * q)).astype(float)
    omega = np.asarray(omega, dtype=float)
    wk = omega * k
    return ((wk >= j - 1) & (wk <= j)).astype(float)


def spike(n, omega):
    """``X_n(omega) = 2^n 1_(0, 1/n)(omega)`` (Counterexample 2); ``inf`` once ``2^n`` overflows."""
    n = _positive_int(n)
    omega = np.asarray(omega, dtype=float)
    with np.errstate(over="ignore"):
        height = np.ldexp(1.0, n)
    return np.where((omega > 0) & (omega * n < 1), height, 0.0)


def mode_quantities(kind: str, n, p: float = 1.0, eps: float = 0.5) -> dict:
    """Exact ``P(|X_n| > eps)``, ``E|X_n|^p`` and ``d(X_n, 0) = E[|X_n| / (1 + |X_n|)]``.

    ``kind`` is ``"typewriter"`` or ``"spike"``. For the typewriter, ``X_n = 1_{A_n}`` with
    ``m(A_n) = 1/k``: all three numbers tend to ``0``. For the spike, ``P(X_n = 2^n) = 1/n``:
    the first and third tend to ``0`` but ``E|X_n|^p = 2^{np} / n`` tends to ``+inf``. Also returns
    ``log2_moment``, the base-2 logarithm of ``E|X_n|^p``, which does not overflow.
    """
    if p < 1 or eps <= 0:
        raise ValueError("need p >= 1 and eps > 0")
    n = _positive_int(n)
    if kind == "typewriter":
        k, _ = typewriter_index(n)
        a = 1.0 / k
        prob = np.where(eps < 1, a, 0.0)
        moment = a
        log2_moment = -np.log2(k)
        metric = a / 2
    elif kind == "spike":
        a = 1.0 / n
        prob = np.where(n * math.log(2) > math.log(eps), a, 0.0)     # 2^n > eps
        log2_moment = n * p - np.log2(n)
        with np.errstate(over="ignore"):
            moment = np.exp2(log2_moment)
        metric = a / (1.0 + np.exp2(-n.astype(float)))                # 2^n / (1 + 2^n) * (1/n)
    else:
        raise ValueError("kind must be 'typewriter' or 'spike'")
    return {"prob": np.asarray(prob, dtype=float)[()], "moment": np.asarray(moment, dtype=float)[()],
            "log2_moment": np.asarray(log2_moment, dtype=float)[()],
            "metric": np.asarray(metric, dtype=float)[()]}


def typewriter_hits(omega, n_max: int) -> np.ndarray:
    """The indices ``n <= n_max`` with ``X_n(omega) = 1`` for the typewriter sequence.

    Each block ``k`` contributes one index, or two when ``omega`` is an interior endpoint
    ``j/k`` (``0 < omega < 1``). Pass ``omega`` as a :class:`~fractions.Fraction` for exact
    results at rational points.
    """
    n = np.arange(1, int(n_max) + 1)
    return n[typewriter(n, omega) == 1.0]


# ---------------------------------------------------------------------------
# The metric of convergence in probability
# ---------------------------------------------------------------------------


def prob_metric(x, y=0.0, weights=None) -> float:
    """``d(X, Y) = E[|X - Y| / (1 + |X - Y|)]`` for ``X``, ``Y`` given on the same points.

    The points are a sample (equal weights) or a grid of ``Omega`` with probabilities
    ``weights``. Infinite differences contribute ``1``. ``d`` is a metric on random variables
    modulo equality a.s., and ``d(X_n, X) -> 0`` if and only if ``X_n -> X`` in probability.
    """
    z = np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))
    with np.errstate(invalid="ignore"):
        r = np.where(np.isinf(z), 1.0, z / (1.0 + z))
    if weights is None:
        return float(np.mean(r))
    w = np.broadcast_to(np.asarray(weights, dtype=float), r.shape)
    return float(np.sum(r * w) / np.sum(w))


# ---------------------------------------------------------------------------
# Sample means
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Law:
    """A law to sample from: ``sample(rng, size)``, its mean and variance (``nan`` if none)."""
    name: str
    sample: Callable
    mean: float
    var: float


LAWS = {
    "bernoulli": Law("Bernoulli b(1/2)", lambda rng, size: rng.binomial(1, 0.5, size).astype(float),
                     0.5, 0.25),
    "uniform": Law("uniform U(0,1)", lambda rng, size: rng.random(size), 0.5, 1 / 12),
    "exponential": Law("exponential Exp(1)", lambda rng, size: rng.exponential(1.0, size), 1.0, 1.0),
    "normal": Law("normal N(0,1)", lambda rng, size: rng.standard_normal(size), 0.0, 1.0),
    "cauchy": Law("Cauchy", lambda rng, size: rng.standard_cauchy(size), math.nan, math.nan),
}


def sample_means(law, n: int, paths: int = 1, seed=None) -> np.ndarray:
    """Simulated paths of the sample mean: array ``m`` of shape ``(paths, n)`` with
    ``m[r, i-1] = (X_1 + ... + X_i) / i`` along path ``r``.

    ``law`` is a key of :data:`LAWS`, a :class:`Law`, or a function ``sample(rng, size)``.
    Keep ``paths * n`` below about ``10^7``.
    """
    if paths * n > 2 * 10 ** 7:
        raise ValueError("paths * n is too large for this machine; use fewer paths or a shorter n")
    if isinstance(law, str):
        law = LAWS[law]
    sampler = law.sample if isinstance(law, Law) else law
    rng = np.random.default_rng(seed)
    x = np.asarray(sampler(rng, (paths, n)), dtype=float)
    return np.cumsum(x, axis=1) / np.arange(1, n + 1)


def deviation_probability(means, mu: float, eps: float, tol: float = 1e-9) -> np.ndarray:
    """Fraction of paths with ``|Xbar_n - mu| > eps``, for each ``n`` (column of ``means``).

    The comparison is ``|Xbar_n - mu| > eps + tol``: for lattice laws such as the Bernoulli law,
    ``|Xbar_n - mu| = eps`` happens with positive probability, and rounding (``11/20 - 0.5`` is
    ``0.05000000000000004`` in floating point) would otherwise count it.
    """
    return np.mean(np.abs(np.asarray(means) - mu) > eps + tol, axis=0)


def tail_sup_deviation(means, mu: float) -> np.ndarray:
    """``s[r, n-1] = max_{n <= m <= N} |Xbar_m - mu|`` along each path (``N`` = path length).

    By the strong law, ``P(sup_{m >= n} |Xbar_m - mu| > eps) -> 0``: the deviations stop for
    good. Here the supremum is truncated at the simulated horizon ``N``.
    """
    d = np.abs(np.asarray(means, dtype=float) - mu)
    return np.flip(np.maximum.accumulate(np.flip(d, axis=-1), axis=-1), axis=-1)


def chebyshev_bound(var: float, n, eps: float) -> np.ndarray:
    """Chebyshev's bound ``min(1, var / (n eps^2))`` on ``P(|Xbar_n - mu| > eps)``."""
    n = np.asarray(n, dtype=float)
    return np.minimum(1.0, var / (n * eps ** 2))


def chebyshev_sample_size(var: float, eps: float, delta: float) -> int:
    """Smallest ``n`` with ``var / (n eps^2) <= delta``: Chebyshev then guarantees
    ``P(|Xbar_n - mu| > eps) <= delta``."""
    if var < 0 or eps <= 0 or not 0 < delta <= 1:
        raise ValueError("need var >= 0, eps > 0 and 0 < delta <= 1")
    x = Fraction(var).limit_denominator(10 ** 12) / (Fraction(eps).limit_denominator(10 ** 12) ** 2
                                                     * Fraction(delta).limit_denominator(10 ** 12))
    return max(1, math.ceil(x))


def bernoulli_mean_tail(n, p: float, eps: float, strict: bool = True) -> np.ndarray:
    """Exact ``P(|S_n / n - p| > eps)`` for ``S_n ~ B(n, p)`` (``>=`` if ``strict=False``).

    The thresholds are found with a tolerance, so that ``S_n = n(p + eps)`` exactly, when it is an
    integer, is not counted by the strict inequality. Vectorised over ``n``.
    """
    n = _positive_int(n)
    tol = 1e-7
    up, lo = n * (p + eps), n * (p - eps)
    if strict:
        k_hi = np.floor(up + tol) + 1               # S_n >= k_hi  <=>  S_n > n(p + eps)
        k_lo = np.ceil(lo - tol) - 1                # S_n <= k_lo  <=>  S_n < n(p - eps)
    else:
        k_hi = np.ceil(up - tol)                    # S_n >= n(p + eps)
        k_lo = np.floor(lo + tol)                   # S_n <= n(p - eps)
    upper = stats.binom.sf(k_hi - 1, n, p)
    lower = np.where(k_lo >= 0, stats.binom.cdf(k_lo, n, p), 0.0)
    return np.asarray(upper + lower)[()]


def first_n_below(tail: Callable, delta: float, n_max: int) -> int | None:
    """Smallest ``n <= n_max`` with ``tail(n) <= delta`` (``tail`` vectorised), or ``None``."""
    n = np.arange(1, n_max + 1)
    ok = np.flatnonzero(np.asarray(tail(n)) <= delta)
    return int(n[ok[0]]) if ok.size else None


def last_n_above(tail: Callable, delta: float, n_max: int) -> int:
    """Largest ``n <= n_max`` with ``tail(n) > delta`` (``0`` if none): from the next ``n`` on,
    up to ``n_max``, the tail stays below ``delta``. The exact Bernoulli tail is not monotone in
    ``n``, so this can differ from :func:`first_n_below`."""
    n = np.arange(1, n_max + 1)
    bad = np.flatnonzero(np.asarray(tail(n)) > delta)
    return int(n[bad[-1]]) if bad.size else 0


# ---------------------------------------------------------------------------
# The proof of the strong law
# ---------------------------------------------------------------------------


def square_floor(m):
    """``k_m = floor(sqrt(m))``, the integer with ``k_m^2 <= m < (k_m + 1)^2`` (exact)."""
    m = _positive_int(m)
    k = np.floor(np.sqrt(m.astype(float))).astype(np.int64)
    k = k - (k * k > m)
    k = k + ((k + 1) * (k + 1) <= m)
    return k


def square_gap_variance(m, sigma2: float = 1.0):
    """For centred i.i.d. variables with variance ``sigma2``: the exact value
    ``E[(Xbar_m - (k_m^2/m) Xbar_{k_m^2})^2] = (m - k_m^2) sigma2 / m^2`` and the bound
    ``3 sigma2 / m^{3/2}`` used in the proof of the strong law, as a pair of arrays."""
    m = _positive_int(m)
    k = square_floor(m)
    exact = (m - k * k) * sigma2 / m.astype(float) ** 2
    bound = 3.0 * sigma2 / m.astype(float) ** 1.5
    return exact, bound


def bernstein_polynomial(f: Callable, n: int, x) -> np.ndarray:
    """``B_n f(x) = sum_k f(k/n) C(n, k) x^k (1-x)^{n-k} = E[f(S_n / n)]``, ``S_n ~ B(n, x)``."""
    shape = np.shape(x)
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    k = np.arange(n + 1)
    w = stats.binom.pmf(k[None, :], n, x[:, None])
    return (w @ np.asarray(f(k / n), dtype=float)).reshape(shape)
