"""The central limit theorem and the delta method (Chapter 17).

Five groups of tools, in the order of the chapter:

* **Laws.** :func:`make_law` builds a :class:`Law` (Bernoulli, Poisson, exponential, Gamma,
  uniform) with its mean, variance, skewness, third absolute central moment, characteristic
  function, a sampler and, when there is one in closed form, the law of the sum ``S_n`` of ``n``
  i.i.d. copies (binomial, Poisson, Gamma).
* **Standardised sums.** :func:`standardised_cdf` is the exact distribution function of
  ``S_n^* = (S_n - n mu) / sqrt(n sigma^2)``; :func:`kolmogorov_distance` is
  ``sup_x |F_{S_n^*}(x) - Phi(x)|``, computed exactly (at the atoms for lattice laws, by a grid
  search refined by a local maximisation for continuous ones); :func:`simulate_standardised`
  draws ``S_n^*``; :func:`empirical_kolmogorov` is the Kolmogorov distance of a sample from
  ``Phi``; :func:`standardised_cf` is ``Phi_Y(t / sqrt n)^n``, the characteristic function of
  ``S_n^*`` that appears in the proof of the theorem; :func:`berry_esseen_bound` and
  :func:`edgeworth_constant` give the two classical descriptions of the rate ``1 / sqrt n``.
* **Normal approximation of the binomial.** :func:`normal_tail` approximates ``P(Y > k)`` for
  ``Y ~ B(n, p)`` by ``1 - Phi((k + eps - n p) / sqrt(n p (1 - p)))``, where ``eps = 1/2`` is the
  continuity correction; :func:`poll_numbers` reproduces the election poll of the notes,
  :func:`matching_shift` finds the ``eps`` that makes the approximation exact,
  :func:`max_cdf_error` is the largest error over ``k`` and :func:`normal_rule` the rule of thumb
  ``n p >= 10``, ``n (1 - p) >= 10``.
* **The delta method.** :func:`sample_means` draws ``Xbar_n``; :func:`delta_sample` and
  :func:`second_order_sample` return ``sqrt(n) (g(Xbar_n) - g(theta))`` and
  ``n (g(Xbar_n) - g(theta))``; :func:`delta_variance` is ``g'(theta)^2 sigma^2(theta)`` and
  :func:`second_order_cdf` the distribution function of ``(sigma^2 g''(theta) / 2) chi^2(1)``.
  :data:`STABILISERS` lists the variance-stabilising transforms of the chapter, and
  :func:`stabilising_transform` computes ``g(theta) = int_{theta_0}^theta dt / sigma(t)`` for any
  ``sigma``.
* **The worked exercises.** :func:`poisson_partial_sum` is ``e^{-n} sum_{k <= n + x sqrt n} n^k / k!``
  (whose limit is ``Phi(x)``; ``1/2`` for ``x = 0``) and :data:`RAMANUJAN_CONSTANT` the constant of
  its rate; :func:`variance_counterexample_sample` and :func:`variance_counterexample_var` are the
  sequence ``Z_n = sqrt(n) Y_n + X_n`` whose variances tend to ``2``; :func:`exam_cf`,
  :func:`exam_sample` and :func:`exam_abs_moment` are the exam problem ``X_n = Y Z_n``.

All functions are vectorised with NumPy where it matters. Simulations draw at most a few million
numbers per call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate, optimize, special, stats

__all__ = [
    "Law", "make_law", "LAW_NAMES", "standardise", "standardised_cdf", "kolmogorov_distance",
    "simulate_standardised", "empirical_kolmogorov", "standardised_cf", "berry_esseen_bound",
    "edgeworth_constant", "normal_tail", "binomial_tail", "poll_numbers", "matching_shift",
    "max_cdf_error", "normal_rule", "binomial_normal_density", "sample_means", "delta_variance",
    "delta_sample", "second_order_sample", "second_order_cdf", "STABILISERS",
    "stabilising_transform", "poisson_partial_sum", "RAMANUJAN_CONSTANT",
    "variance_counterexample_sample", "variance_counterexample_var", "exam_cf", "exam_sample",
    "gaussian_abs_moment", "exam_abs_moment",
]

MAX_DRAWS = 20_000_000          # cap on the numbers drawn by one simulation call

# ---------------------------------------------------------------------------
# Laws
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Law:
    """A law on ``R`` with what the chapter needs of it.

    ``cf(t)`` is the characteristic function, ``sample(rng, size)`` a sampler, and
    ``sum_law(n)`` the law of ``S_n = X_1 + ... + X_n`` for i.i.d. ``X_i`` as a frozen
    :mod:`scipy.stats` distribution (``None`` if there is no closed form). ``lattice`` is true for
    integer-valued laws, ``rho`` is ``E|X - mu|^3``.
    """
    name: str
    mean: float
    var: float
    skew: float
    rho: float
    cf: Callable
    sample: Callable
    sum_law: Callable | None
    lattice: bool

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)


def _abs_third(dist, mean) -> float:
    """``E|X - mean|^3`` for a frozen scipy distribution, by quadrature or summation."""
    if hasattr(dist, "pmf"):
        hi = int(dist.ppf(1 - 1e-16)) + 2
        k = np.arange(0, hi + 1)
        return float(np.sum(np.abs(k - mean) ** 3 * dist.pmf(k)))
    lo, hi = dist.support()
    f = lambda x: abs(x - mean) ** 3 * dist.pdf(x)
    pts = [mean] if lo < mean < hi else None
    return float(integrate.quad(f, lo, hi, points=pts, limit=200)[0]) if np.isfinite(hi) else \
        float(integrate.quad(f, lo, mean)[0] + integrate.quad(f, mean, np.inf, limit=200)[0])


def make_law(name: str, **params) -> Law:
    """Build one of the laws ``bernoulli(p)``, ``poisson(lam)``, ``exponential(rate)``,
    ``gamma(shape, rate)`` and ``uniform(a, b)``; defaults ``p = 1/2``, ``lam = 1``, ``rate = 1``,
    ``shape = 2``, ``(a, b) = (0, 1)``.
    """
    if name == "bernoulli":
        p = float(params.get("p", 0.5))
        if not 0 < p < 1:
            raise ValueError("need 0 < p < 1")
        q = 1 - p
        return Law(f"b({p:g})", p, p * q, (q - p) / math.sqrt(p * q), p * q * (p * p + q * q),
                   lambda t: q + p * np.exp(1j * np.asarray(t, dtype=float)),
                   lambda rng, size: rng.binomial(1, p, size).astype(float),
                   lambda n: stats.binom(n, p), True)
    if name == "poisson":
        lam = float(params.get("lam", 1.0))
        if lam <= 0:
            raise ValueError("need lam > 0")
        return Law(f"P({lam:g})", lam, lam, 1 / math.sqrt(lam), _abs_third(stats.poisson(lam), lam),
                   lambda t: np.exp(lam * (np.exp(1j * np.asarray(t, dtype=float)) - 1)),
                   lambda rng, size: rng.poisson(lam, size).astype(float),
                   lambda n: stats.poisson(n * lam), True)
    if name in ("exponential", "gamma"):
        rate = float(params.get("rate", 1.0))
        shape = 1.0 if name == "exponential" else float(params.get("shape", 2.0))
        if rate <= 0 or shape <= 0:
            raise ValueError("need rate > 0 and shape > 0")
        label = f"Exp({rate:g})" if name == "exponential" else f"Gamma({shape:g}, {rate:g})"
        mean = shape / rate
        return Law(label, mean, shape / rate ** 2, 2 / math.sqrt(shape),
                   _abs_third(stats.gamma(shape, scale=1 / rate), mean),
                   lambda t: (1 - 1j * np.asarray(t, dtype=float) / rate) ** (-shape),
                   lambda rng, size: rng.gamma(shape, 1 / rate, size),
                   lambda n: stats.gamma(n * shape, scale=1 / rate), False)
    if name == "uniform":
        a, b = float(params.get("a", 0.0)), float(params.get("b", 1.0))
        if not a < b:
            raise ValueError("need a < b")

        def cf(t):
            t = np.asarray(t, dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = (np.exp(1j * t * b) - np.exp(1j * t * a)) / (1j * t * (b - a))
            return np.where(t == 0, 1.0 + 0j, v)
        return Law(f"U({a:g}, {b:g})", (a + b) / 2, (b - a) ** 2 / 12, 0.0, (b - a) ** 3 / 32, cf,
                   lambda rng, size: rng.uniform(a, b, size), None, False)
    raise ValueError(f"unknown law {name!r}; choose one of {LAW_NAMES}")


LAW_NAMES = ("bernoulli", "poisson", "exponential", "gamma", "uniform")


def _law(law) -> Law:
    return law if isinstance(law, Law) else make_law(law)


# ---------------------------------------------------------------------------
# Standardised sums
# ---------------------------------------------------------------------------


def standardise(s, n, mean: float, var: float):
    """``S_n^* = (S_n - n mean) / sqrt(n var)``, vectorised."""
    n = np.asarray(n, dtype=float)
    return (np.asarray(s, dtype=float) - n * mean) / np.sqrt(n * var)


def standardised_cdf(law, n: int, x):
    """Exact ``P(S_n^* <= x)`` for a law whose sum has a closed form."""
    L = _law(law)
    if L.sum_law is None:
        raise ValueError(f"no closed form for the sum of {L.name} variables; simulate instead")
    x = np.asarray(x, dtype=float)
    s = n * L.mean + x * math.sqrt(n * L.var)
    if L.lattice:
        s = np.floor(s + 1e-9 * np.maximum(1.0, np.abs(s)))   # guard atoms against rounding
    return L.sum_law(n).cdf(s)


def kolmogorov_distance(law, n: int, grid: int = 4001) -> float:
    """``D_n = sup_x |F_{S_n^*}(x) - Phi(x)|``, computed exactly.

    For a lattice law ``F_{S_n^*}`` is constant between consecutive atoms ``x_k`` and ``Phi`` is
    increasing, so the supremum is the largest of ``|F(x_k) - Phi(x_k)|`` and
    ``|F(x_k-) - Phi(x_k)|`` over the atoms. For a continuous law the difference is smooth: a grid
    search on ``[-10, 10]`` is refined by a bounded local maximisation.
    """
    L = _law(law)
    if L.sum_law is None:
        raise ValueError(f"no closed form for the sum of {L.name} variables; use empirical_kolmogorov")
    S = L.sum_law(n)
    m, s = n * L.mean, math.sqrt(n * L.var)
    if L.lattice:
        lo, hi = S.support()
        hi = min(hi, int(S.ppf(1 - 1e-15)) + 2)
        k = np.arange(int(lo), int(hi) + 1)
        x = (k - m) / s
        F = S.cdf(k)
        Fm = np.concatenate([[0.0], F[:-1]])
        ph = stats.norm.cdf(x)
        return float(max(np.max(np.abs(F - ph)), np.max(np.abs(Fm - ph)), stats.norm.sf(x[-1])))
    h = lambda x: -abs(S.cdf(m + x * s) - stats.norm.cdf(x))
    xs = np.linspace(-10, 10, grid)
    vals = np.abs(S.cdf(m + xs * s) - stats.norm.cdf(xs))
    i = int(np.argmax(vals))
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, grid - 1)]
    res = optimize.minimize_scalar(h, bounds=(a, b), method="bounded", options={"xatol": 1e-10})
    return float(max(vals[i], -res.fun))


def simulate_standardised(law, n: int, size: int, seed=None) -> np.ndarray:
    """``size`` independent draws of ``S_n^*`` (from the law of ``S_n`` when it is known)."""
    L = _law(law)
    rng = np.random.default_rng(seed)
    if L.sum_law is not None:
        s = L.sum_law(n).rvs(size=size, random_state=rng)
    else:
        if n * size > MAX_DRAWS:
            raise ValueError("too many draws: reduce n * size")
        s = np.zeros(size)
        rows = max(1, 2_000_000 // n)
        for start in range(0, size, rows):
            r = min(rows, size - start)
            s[start:start + r] = L.sample(rng, (r, n)).sum(axis=1)
    return standardise(s, n, L.mean, L.var)


def empirical_kolmogorov(sample) -> float:
    """Kolmogorov distance ``sup_x |F_m(x) - Phi(x)|`` of the empirical CDF of a sample from ``Phi``."""
    x = np.sort(np.asarray(sample, dtype=float))
    m = x.size
    ph = stats.norm.cdf(x)
    i = np.arange(1, m + 1)
    return float(max(np.max(i / m - ph), np.max(ph - (i - 1) / m)))


def standardised_cf(law, t, n: int):
    """``Phi_n(t) = Phi_Y(t / sqrt n)^n``, the characteristic function of ``S_n^*``.

    Here ``Y = (X - mu) / sigma`` and ``Phi_Y(u) = e^{-i u mu / sigma} Phi_X(u / sigma)``.
    """
    L = _law(law)
    u = np.asarray(t, dtype=float) / math.sqrt(n)
    phy = np.exp(-1j * u * L.mean / L.sd) * L.cf(u / L.sd)
    return phy ** n


def berry_esseen_bound(law, n, C: float = 0.4748):
    """The Berry--Esseen bound ``C rho / (sigma^3 sqrt n)`` on the Kolmogorov distance."""
    L = _law(law)
    return C * L.rho / (L.sd ** 3 * np.sqrt(np.asarray(n, dtype=float)))


def edgeworth_constant(skew: float) -> float:
    """``|gamma| / (6 sqrt(2 pi))``: for a non-lattice law with a finite third moment and skewness
    ``gamma``, ``sqrt(n) D_n`` tends to this constant, the supremum of the modulus of the first
    Edgeworth term ``(gamma / 6)(1 - x^2) phi(x)``, attained at ``x = 0``."""
    return abs(skew) / (6 * math.sqrt(2 * math.pi))


# ---------------------------------------------------------------------------
# Normal approximation of the binomial
# ---------------------------------------------------------------------------


def binomial_tail(k, n: int, p: float):
    """Exact ``P(Y > k)`` for ``Y ~ B(n, p)``."""
    return stats.binom.sf(k, n, p)


def normal_tail(k, n: int, p: float, eps: float = 0.5):
    """Normal approximation of ``P(Y > k) = P(Y > k + eps)``, ``Y ~ B(n, p)``, integer ``k``:
    ``1 - Phi((k + eps - n p) / sqrt(n p (1 - p)))``. ``eps = 1/2`` is the continuity correction;
    ``eps = 0`` and ``eps = 1`` are the two naive choices of the poll example."""
    z = (np.asarray(k, dtype=float) + eps - n * p) / math.sqrt(n * p * (1 - p))
    return stats.norm.sf(z)


def poll_numbers(n: int = 100, p: float = 0.4, k: int = 50) -> dict:
    """The election poll: ``P(Y >= k + 1)`` exactly and with ``eps = 1, 0, 1/2``.

    ``z_eps = (k + eps - n p) / sqrt(n p (1 - p))``; the notes read ``Phi`` from a table at ``z``
    rounded to two decimals (``"*_table"``), which gives ``0.0125``, ``0.0207`` and ``0.0162``.
    """
    out = {"exact": float(binomial_tail(k, n, p))}
    s = math.sqrt(n * p * (1 - p))
    for name, eps in (("eps1", 1.0), ("eps0", 0.0), ("eps_half", 0.5)):
        z = (k + eps - n * p) / s
        out[name] = float(stats.norm.sf(z))
        out[name + "_z"] = z
        out[name + "_table"] = float(stats.norm.sf(math.floor(z * 100) / 100))
    return out


def matching_shift(k: int, n: int, p: float) -> float:
    """The ``eps`` for which :func:`normal_tail` equals the exact ``P(Y > k)``."""
    target = float(binomial_tail(k, n, p))
    return float(optimize.brentq(lambda e: float(normal_tail(k, n, p, e)) - target, -2.0, 3.0,
                                 xtol=1e-12))


def max_cdf_error(n: int, p: float, eps: float = 0.5) -> float:
    """``max_k |P(Y <= k) - Phi((k + eps - n p) / sqrt(n p (1 - p)))|`` over ``k = 0, ..., n``."""
    k = np.arange(0, n + 1)
    z = (k + eps - n * p) / math.sqrt(n * p * (1 - p))
    return float(np.max(np.abs(stats.binom.cdf(k, n, p) - stats.norm.cdf(z))))


def normal_rule(n, p, threshold: float = 10.0):
    """The rule of thumb of the notes: ``n p >= threshold`` and ``n (1 - p) >= threshold``."""
    n, p = np.asarray(n, dtype=float), np.asarray(p, dtype=float)
    return (n * p >= threshold) & (n * (1 - p) >= threshold)


def binomial_normal_density(x, n: int, p: float):
    """The density of ``N(n p, n p (1 - p))``, drawn over the histograms of ``B(n, p)``."""
    return stats.norm.pdf(x, n * p, math.sqrt(n * p * (1 - p)))


# ---------------------------------------------------------------------------
# The delta method
# ---------------------------------------------------------------------------


def sample_means(law, n: int, size: int, seed=None) -> np.ndarray:
    """``size`` independent draws of the sample mean ``Xbar_n``."""
    L = _law(law)
    z = simulate_standardised(L, n, size, seed)
    return L.mean + z * L.sd / math.sqrt(n)


def delta_variance(dg_theta, sigma2_theta):
    """The variance ``g'(theta)^2 sigma^2(theta)`` of the delta method."""
    return np.asarray(dg_theta, dtype=float) ** 2 * np.asarray(sigma2_theta, dtype=float)


def delta_sample(law, g: Callable, n: int, size: int, seed=None) -> np.ndarray:
    """Draws of ``sqrt(n) (g(Xbar_n) - g(theta))``, with ``theta`` the mean of the law."""
    L = _law(law)
    xb = sample_means(L, n, size, seed)
    return math.sqrt(n) * (g(xb) - g(L.mean))


def second_order_sample(law, g: Callable, n: int, size: int, seed=None) -> np.ndarray:
    """Draws of ``n (g(Xbar_n) - g(theta))``, for the second-order delta method."""
    L = _law(law)
    xb = sample_means(L, n, size, seed)
    return n * (g(xb) - g(L.mean))


def second_order_cdf(x, sigma2: float, g2: float):
    """Distribution function of ``c W`` with ``W ~ chi^2(1)`` and ``c = sigma^2 g''(theta) / 2``."""
    c = sigma2 * g2 / 2
    x = np.asarray(x, dtype=float)
    if c > 0:
        return stats.chi2.cdf(x / c, 1)
    if c < 0:
        return np.where(x >= 0, 1.0, stats.chi2.sf(x / c, 1))
    return (x >= 0).astype(float)


def _sqrt_pos(x):
    return np.sqrt(np.maximum(np.asarray(x, dtype=float), 0.0))


#: Variance-stabilising transforms: name -> (law family, g, limit variance of sqrt(n)(g(Xbar_n) - g(theta))).
STABILISERS = {
    "poisson": ("poisson", _sqrt_pos, 0.25),
    "bernoulli": ("bernoulli", lambda x: np.arcsin(_sqrt_pos(np.minimum(x, 1.0))), 0.25),
    "exponential": ("exponential", lambda x: np.log(np.asarray(x, dtype=float)), 1.0),
}


def stabilising_transform(sigma: Callable, theta0: float) -> Callable:
    """``g(theta) = int_{theta_0}^{theta} dt / sigma(t)``: then ``g'(theta)^2 sigma^2(theta) = 1``."""
    def g(theta):
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        out = np.array([integrate.quad(lambda t: 1.0 / sigma(t), theta0, v, limit=200)[0] for v in th])
        return out[0] if np.ndim(theta) == 0 else out.reshape(np.shape(theta))
    return g


# ---------------------------------------------------------------------------
# The worked exercises
# ---------------------------------------------------------------------------

#: ``lim sqrt(n) (e^{-n} sum_{k<=n} n^k / k! - 1/2)``: ``1 / (2 sqrt(2 pi))`` from the continuity
#: correction plus ``1 / (6 sqrt(2 pi))`` from the skewness of the Poisson law.
RAMANUJAN_CONSTANT = 2 / (3 * math.sqrt(2 * math.pi))


def poisson_partial_sum(n, x: float = 0.0):
    """``e^{-n} sum_{k <= n + x sqrt(n)} n^k / k! = P(S_n <= n + x sqrt n)``, ``S_n ~ P(n)``."""
    n = np.asarray(n, dtype=float)
    return stats.poisson.cdf(np.floor(n + x * np.sqrt(n) + 1e-9), n)


def variance_counterexample_sample(n: int, size: int, p: float = 0.5, seed=None) -> np.ndarray:
    """Draws of ``Z_n = sqrt(n) Y_n + X_n``: ``Y_n ~ b(1/n)`` independent of the standardised
    binomial ``X_n = (V_n - n p) / sqrt(n p (1 - p))``, ``V_n ~ B(n, p)``."""
    rng = np.random.default_rng(seed)
    v = rng.binomial(n, p, size)
    y = rng.random(size) < 1.0 / n
    return math.sqrt(n) * y + (v - n * p) / math.sqrt(n * p * (1 - p))


def variance_counterexample_var(n):
    """``var(Z_n) = n var(Y_n) + var(X_n) = 2 - 1/n``."""
    n = np.asarray(n, dtype=float)
    return n * (1 / n) * (1 - 1 / n) + 1.0


def exam_cf(t, alpha):
    """``Phi_{X_n}(t) = (e^{-alpha^2 t^2 / 2} + 1) / 2`` for ``X_n = Y Z_n``."""
    t, alpha = np.asarray(t, dtype=float), np.asarray(alpha, dtype=float)
    return 0.5 * (np.exp(-0.5 * alpha ** 2 * t ** 2) + 1.0)


def exam_sample(alpha: float, size: int, seed=None) -> np.ndarray:
    """Draws of ``X_n = Y Z_n``, ``Y ~ N(0, 1)`` independent of ``Z_n``, ``P(Z_n = alpha) = P(Z_n = 0) = 1/2``."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(size)
    z = np.where(rng.random(size) < 0.5, alpha, 0.0)
    return y * z


def gaussian_abs_moment(p):
    """``E|Y|^p = 2^{p/2} Gamma((p + 1) / 2) / sqrt(pi)`` for ``Y ~ N(0, 1)``."""
    p = np.asarray(p, dtype=float)
    return 2 ** (p / 2) * special.gamma((p + 1) / 2) / math.sqrt(math.pi)


def exam_abs_moment(alpha, p):
    """Exact ``E|X_n|^p = |alpha|^p E|Y|^p / 2`` (the notes bound it by ``|alpha|^p E|Y|^p``)."""
    return 0.5 * np.abs(np.asarray(alpha, dtype=float)) ** np.asarray(p, dtype=float) * gaussian_abs_moment(p)
