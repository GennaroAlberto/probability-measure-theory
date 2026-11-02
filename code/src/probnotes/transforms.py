"""Multinomial and Gamma laws; transformations of a random variable (Chapter 8).

Five groups of tools, in the order of the chapter:

* **The multinomial law.** :func:`multinomial_pmf` is the discrete density of
  ``X = (X_1, ..., X_N) ~ Multinomial(n, p_1, ..., p_N)`` in the convention of the notes: ``N`` of
  the ``N + 1`` types are recorded, and the last one has probability ``1 - sum(p)``.
  :func:`multinomial_coefficient` is the exact integer ``n! / (n_1! ... n_{N+1}!)``,
  :func:`multinomial_support` lists ``X(Omega)``, :func:`multinomial_marginal` sums the density
  over all coordinates but one, and :func:`sample_multinomial` draws samples.
* **The Gamma law.** :func:`gamma_pdf`, :func:`gamma_cdf` and :func:`gamma_survival` for
  ``Gamma(alpha, lam)`` with ``lam`` a *rate* (density ``lam^alpha x^(alpha-1) e^(-lam x) / Gamma(alpha)``),
  :func:`gamma_moments` and :func:`gamma_moment`, and :func:`hazard_rate` and
  :func:`conditional_survival` for the ageing remark.
* **The Erlang law and the Poisson process.** :func:`erlang_cdf` is the finite sum
  ``1 - sum_{k<n} (lam t)^k e^(-lam t) / k!`` of the notes, :func:`poisson_pmf` the Poisson density,
  :func:`poisson_arrivals` simulates the arrival times ``T_n = U_1 + ... + U_n`` with i.i.d.
  ``Exp(lam)`` inter-arrival times, :func:`poisson_counts` the counts ``N_t``, and
  :func:`poisson_path` one path on ``[0, horizon]``.
* **The worked exercises.** :func:`poisson_conditionals_table` is the joint density
  ``c nu^n mu^m alpha^(nm) / (n! m!)`` of the first exercise, with its constant ``c``;
  :func:`annulus_marginal_density` and :func:`sample_annulus` are for the uniform law on an annulus.
* **Transformations.** :func:`discrete_image_law` is the law of ``g(X)`` for a discrete ``X``.
  :func:`transformed_density` evaluates the formula of the piecewise-monotone corollary,
  ``f_Y(y) = sum_i f_X(g_i^{-1}(y)) |(g_i^{-1})'(y)| 1_{g(I_i)}(y)``, from a list of
  :class:`Branch` objects (closed-form inverses, as in :func:`square_branches` and
  :func:`affine_branch`, or numerical ones, built by :func:`monotone_branch`).
  :func:`ks_distance` and :func:`histogram_check` compare a sample with a law.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Callable, Sequence

import numpy as np
from scipy import special

# ---------------------------------------------------------------------------
# The multinomial law
# ---------------------------------------------------------------------------


def _probabilities(p) -> np.ndarray:
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if p.ndim != 1:
        raise ValueError("p must be a one-dimensional array of probabilities")
    if (p < 0).any() or p.sum() > 1 + 1e-12:
        raise ValueError("need p_i >= 0 and p_1 + ... + p_N <= 1")
    return p


def multinomial_coefficient(counts: Sequence[int]) -> int:
    """The multinomial coefficient ``n! / (n_1! ... n_k!)`` with ``n = n_1 + ... + n_k``, exactly.

    It is the number of ways of placing ``n_1`` outcomes of type 1, ..., ``n_k`` of type ``k``
    among ``n`` trials, and the product of binomial coefficients
    ``C(n, n_1) C(n - n_1, n_2) ... C(n_k, n_k)`` of the notes.
    """
    counts = [int(c) for c in counts]
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    out, left = 1, sum(counts)
    for c in counts:
        out *= math.comb(left, c)
        left -= c
    return out


def multinomial_pmf(x, n: int, p) -> np.ndarray:
    """Discrete density of ``X ~ Multinomial(n, p_1, ..., p_N)`` at ``x = (n_1, ..., n_N)``.

    ``P(X = x) = n! / (n_1! ... n_N! (n - sum n_i)!) p_1^{n_1} ... p_N^{n_N} (1 - sum p_i)^{n - sum n_i}``,
    and ``0`` outside ``X(Omega) = {x in {0, ..., n}^N : sum x_i <= n}``. The last axis of ``x``
    has length ``N = len(p)``; the other axes are vectorised. The convention ``0^0 = 1`` is used.
    """
    p = _probabilities(p)
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != len(p):
        raise ValueError("the last axis of x must have length len(p)")
    rest = n - x.sum(axis=-1)
    q = max(1.0 - p.sum(), 0.0)
    ok = (x >= 0).all(axis=-1) & (rest >= 0) & (x == np.round(x)).all(axis=-1)
    xs = np.where(ok[..., None], x, 0.0)
    rs = np.where(ok, rest, 0.0)
    logc = special.gammaln(n + 1) - special.gammaln(xs + 1).sum(axis=-1) - special.gammaln(rs + 1)
    logp = special.xlogy(xs, p).sum(axis=-1) + special.xlogy(rs, q)
    return np.where(ok, np.exp(logc + logp), 0.0)


def multinomial_support(n: int, N: int) -> np.ndarray:
    """All points of ``X(Omega)`` for ``X ~ Multinomial(n, p_1, ..., p_N)``, one per row.

    There are ``C(n + N, N)`` of them.
    """
    pts = [c for c in itertools.product(range(n + 1), repeat=N) if sum(c) <= n]
    return np.array(pts, dtype=int).reshape(-1, N)


def multinomial_marginal(n: int, p, i: int) -> np.ndarray:
    """The discrete density of ``X_i`` obtained by summing the multinomial density over the other
    coordinates (the "not easy" computation of the notes). Returns ``f[k] = P(X_i = k)``,
    ``k = 0, ..., n``, which is the binomial density ``B(n, p_i)``.
    """
    p = _probabilities(p)
    pts = multinomial_support(n, len(p))
    w = multinomial_pmf(pts, n, p)
    return np.bincount(pts[:, i], weights=w, minlength=n + 1)


def sample_multinomial(n: int, p, size: int, rng=None) -> np.ndarray:
    """``size`` draws of ``(X_1, ..., X_N)``, as an array of shape ``(size, N)``."""
    p = _probabilities(p)
    rng = np.random.default_rng(rng)
    full = np.append(p, max(1.0 - p.sum(), 0.0))
    return rng.multinomial(n, full / full.sum(), size=size)[:, : len(p)]


# ---------------------------------------------------------------------------
# The Gamma law
# ---------------------------------------------------------------------------


def _check_gamma(alpha: float, lam: float) -> None:
    if not (alpha > 0 and lam > 0):
        raise ValueError("the Gamma law needs alpha > 0 and lam > 0")


def gamma_pdf(x, alpha: float, lam: float) -> np.ndarray:
    """Density of ``Gamma(alpha, lam)``: ``lam^alpha / Gamma(alpha) x^(alpha-1) e^(-lam x)`` for ``x > 0``.

    ``lam`` is a rate: the mean is ``alpha / lam``. The value at ``x <= 0`` is ``0``, as the
    indicator ``1_{(0, inf)}`` of the notes prescribes.
    """
    _check_gamma(alpha, lam)
    x = np.asarray(x, dtype=float)
    xp = np.where(x > 0, x, 1.0)
    logf = alpha * math.log(lam) - special.gammaln(alpha) + (alpha - 1) * np.log(xp) - lam * xp
    return np.where(x > 0, np.exp(logf), 0.0)


def gamma_cdf(x, alpha: float, lam: float) -> np.ndarray:
    """Distribution function of ``Gamma(alpha, lam)``: the regularised incomplete Gamma function
    ``P(alpha, lam x)`` for ``x > 0``, and ``0`` for ``x <= 0``."""
    _check_gamma(alpha, lam)
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, special.gammainc(alpha, lam * np.maximum(x, 0.0)), 0.0)


def gamma_survival(x, alpha: float, lam: float) -> np.ndarray:
    """``P(X > x)`` for ``X ~ Gamma(alpha, lam)``, computed without cancellation."""
    _check_gamma(alpha, lam)
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, special.gammaincc(alpha, lam * np.maximum(x, 0.0)), 1.0)


def gamma_moments(alpha: float, lam: float) -> tuple[float, float]:
    """Mean and variance of ``Gamma(alpha, lam)``: ``alpha / lam`` and ``alpha / lam^2``."""
    _check_gamma(alpha, lam)
    return alpha / lam, alpha / lam**2


def gamma_moment(k: float, alpha: float, lam: float) -> float:
    """``E[X^k] = Gamma(alpha + k) / (Gamma(alpha) lam^k)`` for real ``k > -alpha``; ``inf`` otherwise."""
    _check_gamma(alpha, lam)
    if k <= -alpha:
        return math.inf
    return math.exp(special.gammaln(alpha + k) - special.gammaln(alpha) - k * math.log(lam))


def hazard_rate(x, alpha: float, lam: float) -> np.ndarray:
    """Hazard rate ``h(x) = f(x) / P(X > x)`` of ``Gamma(alpha, lam)`` for ``x > 0``.

    It is constant (``lam``) for ``alpha = 1``, increasing for ``alpha > 1`` and decreasing for
    ``alpha < 1``, with limit ``lam`` at infinity. Accurate while ``P(X > x)`` does not underflow
    (``lam x`` up to several hundred).
    """
    x = np.asarray(x, dtype=float)
    s = gamma_survival(x, alpha, lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = gamma_pdf(x, alpha, lam) / s
    return np.where(x > 0, h, np.nan)


def conditional_survival(s, t, alpha: float, lam: float) -> np.ndarray:
    """``P(X > s + t | X > s) = P(X > s + t) / P(X > s)`` for ``X ~ Gamma(alpha, lam)``."""
    return gamma_survival(np.asarray(s) + np.asarray(t), alpha, lam) / gamma_survival(s, alpha, lam)


# ---------------------------------------------------------------------------
# The Erlang law and the Poisson process
# ---------------------------------------------------------------------------


def poisson_pmf(k, mean) -> np.ndarray:
    """``P(N = k) = e^(-mean) mean^k / k!`` for ``N ~ P(mean)`` (``0`` for ``k`` not in ``N``)."""
    k = np.asarray(k, dtype=float)
    mean = np.asarray(mean, dtype=float)
    ok = (k >= 0) & (k == np.round(k))
    kk = np.where(ok, k, 0.0)
    return np.where(ok, np.exp(special.xlogy(kk, mean) - mean - special.gammaln(kk + 1)), 0.0)


def erlang_cdf(t, n: int, lam: float) -> np.ndarray:
    """Distribution function of the Erlang law ``Gamma(n, lam)``, ``n >= 1`` an integer, by the
    finite sum of the notes: ``F_n(t) = 1 - sum_{k=0}^{n-1} (lam t)^k e^(-lam t) / k!`` for
    ``t > 0``, which is ``P(N_t >= n)`` for ``N_t ~ P(lam t)``.
    """
    if int(n) != n or n < 1:
        raise ValueError("the Erlang law needs an integer n >= 1")
    t = np.asarray(t, dtype=float)
    mean = lam * np.maximum(t, 0.0)
    tail = sum(poisson_pmf(k, mean) for k in range(int(n)))
    return np.where(t > 0, 1.0 - tail, 0.0)


def poisson_arrivals(lam: float, n: int, size: int, rng=None) -> np.ndarray:
    """Arrival times ``T_1 < ... < T_n`` of a Poisson process of rate ``lam``, ``size`` paths.

    ``T_k = U_1 + ... + U_k`` with ``U_i`` i.i.d. ``Exp(lam)``; the result has shape ``(size, n)``.
    """
    rng = np.random.default_rng(rng)
    return np.cumsum(rng.exponential(1.0 / lam, size=(size, n)), axis=1)


def poisson_counts(lam: float, t: float, size: int, rng=None) -> np.ndarray:
    """``size`` independent draws of ``N_t = #{n >= 1 : T_n <= t}``, the number of arrivals of a
    Poisson process of rate ``lam`` in ``[0, t]``. By the Erlang identity ``N_t ~ P(lam t)``."""
    rng = np.random.default_rng(rng)
    m = int(lam * t + 8 * math.sqrt(lam * t) + 16)
    T = poisson_arrivals(lam, m, size, rng)
    counts = (T <= t).sum(axis=1)
    last = T[:, -1]
    for i in np.flatnonzero(last <= t):          # rare: the path needs more arrivals
        s = last[i]
        while s <= t:
            s += rng.exponential(1.0 / lam)
            counts[i] += s <= t
    return counts


def poisson_path(lam: float, horizon: float, rng=None) -> np.ndarray:
    """The arrival times in ``[0, horizon]`` of one path of a Poisson process of rate ``lam``."""
    rng = np.random.default_rng(rng)
    times, s = [], 0.0
    while True:
        s += rng.exponential(1.0 / lam)
        if s > horizon:
            return np.array(times)
        times.append(s)


# ---------------------------------------------------------------------------
# The worked exercises
# ---------------------------------------------------------------------------


def poisson_conditionals_constant(nu: float, mu: float, alpha: float, tol: float = 1e-16) -> float:
    """The constant ``c`` of the joint density ``c nu^n mu^m alpha^(nm) / (n! m!)`` on ``N^2``.

    ``1 / c = sum_n nu^n / n! exp(mu alpha^n)``, the series of the marginal of ``X``; for
    ``alpha = 1`` this is ``e^(nu + mu)``.
    """
    if not (nu > 0 and mu > 0 and 0 < alpha <= 1):
        raise ValueError("need nu, mu > 0 and 0 < alpha <= 1")
    total, n, term = 0.0, 0, 1.0                  # term = nu^n / n!
    while True:
        add = term * math.exp(mu * alpha**n)
        total += add
        if n > nu and add < tol * total:
            return 1.0 / total
        n += 1
        term *= nu / n


def poisson_conditionals_table(nu: float, mu: float, alpha: float, nmax: int, mmax: int):
    """The joint density ``f[n, m] = c nu^n mu^m alpha^(nm) / (n! m!)`` for ``n <= nmax``,
    ``m <= mmax``, with the exact constant ``c``; returns ``(c, f)``. Truncation leaves out the
    mass ``1 - f.sum()``."""
    c = poisson_conditionals_constant(nu, mu, alpha)
    n = np.arange(nmax + 1)[:, None]
    m = np.arange(mmax + 1)[None, :]
    logf = (math.log(c) + special.xlogy(n, nu) + special.xlogy(m, mu) + n * m * math.log(alpha)
            - special.gammaln(n + 1) - special.gammaln(m + 1))
    return c, np.exp(logf)


def annulus_marginal_density(x, r_in: float = 1.0, r_out: float = 3.0) -> np.ndarray:
    """Density of ``X`` when ``(X, Y)`` is uniform on ``{r_in^2 <= x^2 + y^2 <= r_out^2}``:
    the length of the section at ``x`` divided by the area ``pi (r_out^2 - r_in^2)``."""
    x = np.asarray(x, dtype=float)
    length = 2 * np.sqrt(np.maximum(r_out**2 - x * x, 0.0)) - 2 * np.sqrt(np.maximum(r_in**2 - x * x, 0.0))
    return length / (math.pi * (r_out**2 - r_in**2))


def sample_annulus(size: int, r_in: float = 1.0, r_out: float = 3.0, rng=None) -> np.ndarray:
    """``size`` points uniform on the annulus, by rejection from the square ``[-r_out, r_out]^2``
    (the square ``Q`` of the notes). Returns an array of shape ``(size, 2)``."""
    rng = np.random.default_rng(rng)
    out = np.empty((0, 2))
    accept = math.pi * (r_out**2 - r_in**2) / (2 * r_out) ** 2
    while len(out) < size:
        k = int((size - len(out)) / accept * 1.1) + 16
        pts = rng.uniform(-r_out, r_out, size=(k, 2))
        r2 = (pts**2).sum(axis=1)
        out = np.vstack([out, pts[(r2 >= r_in**2) & (r2 <= r_out**2)]])
    return out[:size]


# ---------------------------------------------------------------------------
# Transformations of a random variable
# ---------------------------------------------------------------------------


def discrete_image_law(values, probs, g: Callable):
    """Law of ``Y = g(X)`` for a discrete ``X`` with ``P(X = values[i]) = probs[i]``.

    ``f_Y(y) = sum_{x in g^{-1}({y})} f_X(x)``: the masses of the points with the same image are
    added. Returns ``(ys, fy)`` with ``ys`` sorted. Values of ``g`` are compared exactly.
    """
    values = np.asarray(values)
    probs = np.asarray(probs, dtype=float)
    ys, inv = np.unique(np.asarray(g(values)), return_inverse=True)
    return ys, np.bincount(inv.ravel(), weights=probs.ravel(), minlength=len(ys))


@dataclass(frozen=True)
class Branch:
    """One monotone piece ``g_i = g|_{I_i}`` of a transformation, described by its inverse.

    ``inverse(y) = g_i^{-1}(y)`` and ``derivative(y) = (g_i^{-1})'(y)`` for ``y`` in the open
    interval ``g(I_i) = (lo, hi)``.
    """

    inverse: Callable
    derivative: Callable
    lo: float
    hi: float


def transformed_density(y, fx: Callable, branches: Sequence[Branch]) -> np.ndarray:
    """``f_Y(y) = sum_i f_X(g_i^{-1}(y)) |(g_i^{-1})'(y)| 1_{g(I_i)}(y)`` (the piecewise-monotone
    corollary; one branch gives the theorem for a ``C^1`` function with ``g' != 0``)."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    for b in branches:
        inside = (y > b.lo) & (y < b.hi)
        if inside.any():
            yi = y[inside]
            out[inside] += fx(b.inverse(yi)) * np.abs(b.derivative(yi))
    return out


def square_branches() -> list[Branch]:
    """The two branches of ``g(x) = x^2`` on ``I_1 = (-inf, 0)`` and ``I_2 = (0, inf)``."""
    return [Branch(lambda y: -np.sqrt(y), lambda y: -0.5 / np.sqrt(y), 0.0, math.inf),
            Branch(np.sqrt, lambda y: 0.5 / np.sqrt(y), 0.0, math.inf)]


def square_density(y, fx: Callable) -> np.ndarray:
    """Density of ``X^2``: ``(f_X(sqrt y) + f_X(-sqrt y)) / (2 sqrt y)`` for ``y > 0``, else ``0``."""
    return transformed_density(y, fx, square_branches())


def affine_branch(sigma: float, mu: float) -> Branch:
    """The single branch of ``g(x) = sigma x + mu``, ``sigma != 0``: inverse ``(y - mu) / sigma``."""
    if sigma == 0:
        raise ValueError("sigma must be non-zero")
    return Branch(lambda y: (y - mu) / sigma, lambda y: np.full_like(np.asarray(y, dtype=float), 1.0 / sigma),
                  -math.inf, math.inf)


def affine_density(y, fx: Callable, sigma: float, mu: float) -> np.ndarray:
    """Density of ``sigma X + mu``: ``f_X((y - mu) / sigma) / |sigma|``."""
    return transformed_density(y, fx, [affine_branch(sigma, mu)])


def monotone_branch(g: Callable, a: float, b: float, dg: Callable | None = None, iters: int = 80) -> Branch:
    """A :class:`Branch` for a continuous, strictly monotone ``g`` on a bounded open interval
    ``(a, b)``, inverted numerically by vectorised bisection.

    The derivative of the inverse is ``1 / g'(g^{-1}(y))``, with ``g'`` given by ``dg`` or by a
    central difference.
    """
    ga, gb = float(g(np.float64(a))), float(g(np.float64(b)))
    increasing = gb > ga
    lo, hi = min(ga, gb), max(ga, gb)

    def inverse(y):
        y = np.asarray(y, dtype=float)
        left, right = np.full_like(y, a), np.full_like(y, b)
        for _ in range(iters):
            mid = (left + right) / 2
            gm = g(mid)
            go_right = (gm < y) if increasing else (gm > y)
            left = np.where(go_right, mid, left)
            right = np.where(go_right, right, mid)
        return (left + right) / 2

    def derivative(y):
        x = inverse(y)
        if dg is not None:
            d = dg(x)
        else:
            h = 1e-6 * (b - a)
            d = (g(np.minimum(x + h, b)) - g(np.maximum(x - h, a))) / (np.minimum(x + h, b) - np.maximum(x - h, a))
        return 1.0 / d

    return Branch(inverse, derivative, lo, hi)


# ---------------------------------------------------------------------------
# Comparing a sample with a law
# ---------------------------------------------------------------------------


def ks_distance(sample, cdf: Callable) -> float:
    """``sup_x |F_n(x) - F(x)|`` for the empirical distribution function ``F_n`` of ``sample`` and
    a continuous distribution function ``F``, computed exactly at the sample points."""
    x = np.sort(np.asarray(sample, dtype=float).ravel())
    n = len(x)
    F = np.asarray(cdf(x), dtype=float)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))


def histogram_check(sample, cdf: Callable, edges):
    """Histogram of ``sample`` on the bins ``edges`` against the law with distribution function
    ``cdf``. Returns ``(heights, expected, se)``: the empirical heights ``count / (n width)``, the
    exact ``(F(b) - F(a)) / width``, and the standard error of each height. Heights are compared
    with bin averages of the density, not with its value at the midpoint, which matters where
    the density is unbounded."""
    sample = np.asarray(sample, dtype=float).ravel()
    edges = np.asarray(edges, dtype=float)
    n = len(sample)
    width = np.diff(edges)
    counts = np.histogram(sample, bins=edges)[0]
    prob = np.diff(np.asarray(cdf(edges), dtype=float))
    return counts / (n * width), prob / width, np.sqrt(prob * (1 - prob) / n) / width
