"""L^p norms, image measures, laws, distribution functions and densities (Chapter 6).

Six groups of tools, in the order of the chapter:

* **L^p norms on discrete spaces.** A measure on a finite set (or a truncated countable one)
  is an array of non-negative ``weights``; a function is the array of its ``values`` (which
  may contain ``inf``). :func:`lp_norm` computes ``(sum |f|^p w)^(1/p)``, with the convention
  ``0 * inf = 0``, and :func:`ess_sup` the essential supremum of ``|f|``.
  :func:`ess_sup_by_definition` follows the definition ``inf{M > 0 : mu(|f| > M) = 0}`` of the
  notes literally, for the tests. :func:`power_lp_norm` gives ``||x^-a||_p`` on an interval
  of the half-line with Lebesgue measure, in closed form.
* **Image measures on finite spaces.** :func:`image_measure` pushes the weights of ``mu``
  forward along ``phi``: ``mu_phi({y}) = mu(phi^{-1}({y}))``.
* **Discrete laws.** :class:`DiscreteLaw` stores the support ``x_i`` and the discrete density
  ``p_i = P(X = x_i)``; it computes the distribution function ``F_X(x) = sum_{x_i <= x} p_i``,
  expectations ``E[g(X)] = sum g(x_i) p_i`` and samples.
* **Empirical distribution functions.** :func:`ecdf`, the exact Kolmogorov distance
  :func:`ks_distance` between an empirical and a true distribution function (atoms allowed),
  and sample sizes from the Dvoretzky--Kiefer--Wolfowitz and Chebyshev inequalities.
* **Absolute continuity and Radon--Nikodym derivatives on finite spaces.** On a finite space
  ``nu << mu`` means that ``nu`` charges no point that ``mu`` does not charge, and then
  ``d nu / d mu (x) = nu({x}) / mu({x})`` wherever ``mu({x}) > 0``.
* **Laws on the line and Monte Carlo checks.** :func:`quantile` is the generalised inverse of
  a distribution function (so ``quantile(F, U)`` has distribution function ``F`` when ``U`` is
  uniform); :class:`MixedLaw` is a law with atoms and a density; :func:`mc_expectation` and
  :func:`expectation_density` compare ``E[g(X)]`` estimated from samples with
  ``int g f_X dx``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy import integrate, stats

INF = math.inf

# ---------------------------------------------------------------------------
# L^p norms and the essential supremum on discrete spaces
# ---------------------------------------------------------------------------


def _values_weights(values, weights):
    f = np.abs(np.asarray(values, dtype=float)).ravel()
    w = np.ones_like(f) if weights is None else np.asarray(weights, dtype=float).ravel()
    if f.shape != w.shape:
        raise ValueError("values and weights must have the same length")
    if np.isnan(f).any() or np.isnan(w).any() or (w < 0).any():
        raise ValueError("values must not be NaN and weights must be non-negative")
    return f, w


def ess_sup(values, weights=None) -> float:
    """Essential supremum of |f| for the measure with the given point weights (default: counting).

    Points of weight 0 form a null set and are ignored; if every point has weight 0, then
    f = 0 a.e. and the essential supremum is 0 (every M > 0 qualifies).
    """
    f, w = _values_weights(values, weights)
    charged = f[w > 0]
    return float(charged.max()) if charged.size else 0.0


def mass_above(values, weights, M: float) -> float:
    """mu({|f| > M}) = sum of the weights of the points where |f| > M."""
    f, w = _values_weights(values, weights)
    return float(np.sum(w[f > M]))


def ess_sup_by_definition(values, weights=None) -> float:
    """inf{M > 0 : mu(|f| > M) = 0}, with inf of the empty set = +inf, as in the notes.

    On a finite space M -> mu(|f| > M) is a step function that changes only at the values
    |f(x)|, so it suffices to test those values: the set of admissible M is [c, inf) with c the
    smallest tested value that works (c > 0), or (0, inf) if c = 0, or empty.
    """
    f, w = _values_weights(values, weights)
    for c in np.unique(np.concatenate([[0.0], f[np.isfinite(f)]])):
        if mass_above(f, w, c) == 0:
            return float(c)              # c = 0 means (0, inf) is admissible: the infimum is 0
    return INF


def lp_norm(values, weights=None, p: float = 2.0) -> float:
    """||f||_p = (sum_x |f(x)|^p mu({x}))^(1/p) for 1 <= p < inf, and ess sup |f| for p = inf.

    Uses 0 * inf = 0 (points of weight 0 never contribute) and rescales by the largest value,
    so that large p does not overflow.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    if math.isinf(p):
        return ess_sup(values, weights)
    f, w = _values_weights(values, weights)
    keep = (w > 0) & (f > 0)                 # points with f = 0 contribute 0, even with w = inf
    f, w = f[keep], w[keep]
    if f.size == 0:
        return 0.0
    top = f.max()
    if math.isinf(top) or math.isinf(w.sum()):
        return INF
    return float(top * np.sum(w * (f / top) ** p) ** (1.0 / p))


def conjugate_exponent(p: float) -> float:
    """q with 1/p + 1/q = 1 (q = inf for p = 1 and q = 1 for p = inf)."""
    if p == 1:
        return INF
    if math.isinf(p):
        return 1.0
    return p / (p - 1.0)


def power_integral(b: float, lo: float, hi: float) -> float:
    """int_lo^hi x^(-b) dx for 0 <= lo < hi <= inf (Lebesgue measure), possibly +inf."""
    if not 0 <= lo < hi:
        raise ValueError("need 0 <= lo < hi")
    if b == 1:
        return INF if lo == 0 or math.isinf(hi) else math.log(hi / lo)
    if (lo == 0 and b > 1) or (math.isinf(hi) and b < 1):
        return INF
    top = 0.0 if math.isinf(hi) else hi ** (1 - b)
    bottom = 0.0 if lo == 0 else lo ** (1 - b)
    return (top - bottom) / (1 - b)


def power_lp_norm(a: float, p: float, lo: float, hi: float) -> float:
    """||x^(-a)||_p on (lo, hi) with Lebesgue measure; p = inf gives the essential supremum."""
    if math.isinf(p):                    # the supremum of x^-a on (lo, hi), at an endpoint
        if a > 0:
            return INF if lo == 0 else lo ** -a
        if a < 0:
            return INF if math.isinf(hi) else hi ** -a
        return 1.0
    return power_integral(a * p, lo, hi) ** (1.0 / p)


# ---------------------------------------------------------------------------
# Image measures on finite spaces
# ---------------------------------------------------------------------------


def image_measure(weights, phi_values):
    """The image measure mu_phi of mu = sum_x w(x) delta_x under phi, on the finite set phi(X).

    ``weights[i]`` is mu({x_i}) and ``phi_values[i]`` is phi(x_i). Returns ``(support, masses)``
    with ``support`` the sorted distinct values of phi and ``masses[j] = mu(phi^{-1}({y_j}))``.
    """
    w = np.asarray(weights, dtype=float).ravel()
    y = np.asarray(phi_values).ravel()
    if w.shape != y.shape:
        raise ValueError("weights and phi_values must have the same length")
    support, inverse = np.unique(y, return_inverse=True)
    return support, np.bincount(inverse.ravel(), weights=w, minlength=len(support))


def preimage_mass(weights, phi_values, B) -> float:
    """mu(phi^{-1}(B)) for a set B (any container, or a predicate applied to the values)."""
    w = np.asarray(weights, dtype=float).ravel()
    y = np.asarray(phi_values).ravel()
    inside = np.asarray(B(y), dtype=bool) if callable(B) else np.isin(y, list(B))
    return float(np.sum(w[inside]))


# ---------------------------------------------------------------------------
# Discrete laws
# ---------------------------------------------------------------------------


class DiscreteLaw:
    """The law of a discrete random variable: P(X = x_i) = p_i on a finite set of points.

    Repeated points are merged (their masses add up), so ``DiscreteLaw(values, weights)`` is the
    image measure of ``weights`` under ``x -> value``. Countable laws (Poisson, geometric) are
    truncated where the tail is below ``tol`` (see :func:`poisson_law`).
    """

    def __init__(self, support, masses, tol: float = 1e-9):
        x, p = image_measure(masses, support)
        if (p < 0).any():
            raise ValueError("masses must be non-negative")
        if abs(p.sum() - 1.0) > tol:
            raise ValueError(f"masses must add up to 1, not {p.sum()}")
        self.support = x.astype(float)
        self.masses = p
        self._cum = np.cumsum(p)

    def pmf(self, x) -> np.ndarray:
        """The discrete density f_X(x) = P(X = x) (0 off the support)."""
        x = np.asarray(x, dtype=float)
        i = np.searchsorted(self.support, x)
        hit = (i < len(self.support)) & (self.support[np.minimum(i, len(self.support) - 1)] == x)
        return np.where(hit, self.masses[np.minimum(i, len(self.support) - 1)], 0.0)

    def cdf(self, x) -> np.ndarray:
        """F_X(x) = P(X <= x) = sum of the p_i with x_i <= x (right-continuous)."""
        i = np.searchsorted(self.support, np.asarray(x, dtype=float), side="right")
        return np.where(i > 0, self._cum[np.maximum(i - 1, 0)], 0.0)

    def cdf_left(self, x) -> np.ndarray:
        """F_X(x-) = P(X < x)."""
        i = np.searchsorted(self.support, np.asarray(x, dtype=float), side="left")
        return np.where(i > 0, self._cum[np.maximum(i - 1, 0)], 0.0)

    def prob(self, B) -> float:
        """P(X in B) = sum_{x_i in B} p_i, for a predicate or a container B."""
        return preimage_mass(self.masses, self.support, B)

    def expectation(self, g: Callable = lambda x: x) -> float:
        """E[g(X)] = sum_i g(x_i) p_i (an error if sum |g(x_i)| p_i is infinite)."""
        gx = np.asarray(g(self.support), dtype=float)
        if not np.isfinite(np.sum(np.abs(gx) * self.masses)):
            raise ValueError("g(X) is not integrable")
        return float(np.sum(gx * self.masses))

    def mean(self) -> float:
        return self.expectation()

    def var(self) -> float:
        m = self.mean()
        return self.expectation(lambda x: (x - m) ** 2)

    def quantile(self, u) -> np.ndarray:
        """F_X^{-1}(u) = min{x_i : F_X(x_i) >= u} for u in (0, 1]."""
        i = np.searchsorted(self._cum, np.asarray(u, dtype=float), side="left")
        return self.support[np.minimum(i, len(self.support) - 1)]

    def sample(self, n: int, rng=None) -> np.ndarray:
        """n independent draws F_X^{-1}(U_k), with U_k uniform on (0, 1)."""
        rng = np.random.default_rng(rng)
        return self.quantile(rng.random(n) * self._cum[-1])


def poisson_law(lam: float, tol: float = 1e-15) -> DiscreteLaw:
    """Poisson(lam), truncated at the first k with P(X > k) < tol (mass renormalised)."""
    kmax = int(stats.poisson.isf(tol, lam)) + 1
    k = np.arange(kmax + 1)
    p = stats.poisson.pmf(k, lam)
    return DiscreteLaw(k, p / p.sum(), tol=1e-6)


def binomial_law(n: int, p: float) -> DiscreteLaw:
    """Binomial(n, p) on {0, ..., n}."""
    k = np.arange(n + 1)
    return DiscreteLaw(k, stats.binom.pmf(k, n, p))


# ---------------------------------------------------------------------------
# Empirical distribution functions
# ---------------------------------------------------------------------------


def ecdf(sample, x) -> np.ndarray:
    """The empirical distribution function F_n(x) = #{i : X_i <= x} / n, at the points x."""
    s = np.sort(np.asarray(sample, dtype=float).ravel())
    return np.searchsorted(s, np.asarray(x, dtype=float), side="right") / len(s)


def ks_distance(sample, cdf: Callable, cdf_left: Callable | None = None) -> float:
    """sup_x |F_n(x) - F(x)|, computed exactly.

    Between two consecutive distinct sample points F_n is constant and F is monotone, so the
    supremum is attained, or approached, at a sample point or just to its left. ``cdf_left`` is
    x -> F(x-); omit it when F is continuous.
    """
    s = np.sort(np.asarray(sample, dtype=float).ravel())
    u = np.unique(s)
    n = len(s)
    fn = np.searchsorted(s, u, side="right") / n
    fn_left = np.searchsorted(s, u, side="left") / n
    F = np.asarray(cdf(u), dtype=float)
    F_left = F if cdf_left is None else np.asarray(cdf_left(u), dtype=float)
    return float(max(np.max(np.abs(fn - F)), np.max(np.abs(fn_left - F_left))))


def dkw_epsilon(n: int, alpha: float) -> float:
    """Half-width eps with P(sup_x |F_n - F| > eps) <= alpha (DKW inequality, Massart's constant)."""
    return math.sqrt(math.log(2 / alpha) / (2 * n))


def dkw_sample_size(eps: float, alpha: float) -> int:
    """Smallest n with 2 exp(-2 n eps^2) <= alpha: a uniform guarantee over all x."""
    return math.ceil(math.log(2 / alpha) / (2 * eps**2) - 1e-9)


def chebyshev_sample_size(eps: float, alpha: float, var_bound: float = 0.25) -> int:
    """Smallest n with var_bound / (n eps^2) <= alpha: Chebyshev at one point x.

    F_n(x) has variance F(x)(1 - F(x)) / n <= 1 / (4 n), hence the default var_bound = 1/4.
    """
    return math.ceil(var_bound / (alpha * eps**2) - 1e-9)


def kolmogorov_tail(lam: float, terms: int = 100) -> float:
    """Limit of P(sqrt(n) sup|F_n - F| > lam) for continuous F: 2 sum (-1)^(k-1) e^(-2 k^2 lam^2)."""
    k = np.arange(1, terms + 1)
    return float(2 * np.sum((-1.0) ** (k - 1) * np.exp(-2 * k**2 * lam**2)))


# ---------------------------------------------------------------------------
# Absolute continuity and Radon-Nikodym derivatives on finite spaces
# ---------------------------------------------------------------------------


def _pair(nu, mu):
    nu = np.asarray(nu, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    if nu.shape != mu.shape:
        raise ValueError("the two measures must live on the same points")
    if (nu < 0).any() or (mu < 0).any():
        raise ValueError("weights must be non-negative")
    return nu, mu


def is_absolutely_continuous(nu, mu) -> bool:
    """nu << mu on a finite space: every point of mu-measure 0 has nu-measure 0."""
    nu, mu = _pair(nu, mu)
    return bool(np.all(nu[mu == 0] == 0))


def are_equivalent(nu, mu) -> bool:
    """nu ~ mu: nu << mu and mu << nu (the same null sets)."""
    return is_absolutely_continuous(nu, mu) and is_absolutely_continuous(mu, nu)


def are_singular(nu, mu) -> bool:
    """nu and mu are concentrated on disjoint sets: no point is charged by both."""
    nu, mu = _pair(nu, mu)
    return bool(np.all((nu == 0) | (mu == 0)))


def radon_nikodym(nu, mu) -> np.ndarray:
    """d nu / d mu on a finite space: nu({x}) / mu({x}) where mu({x}) > 0, and 0 elsewhere.

    The value on mu-null points is arbitrary (the derivative is unique only mu-a.e.); raises
    ValueError if nu is not absolutely continuous with respect to mu.
    """
    nu, mu = _pair(nu, mu)
    if not is_absolutely_continuous(nu, mu):
        bad = np.flatnonzero((mu == 0) & (nu > 0))
        raise ValueError(f"nu is not << mu: mu({{x}}) = 0 < nu({{x}}) at indices {bad[:5].tolist()}")
    out = np.zeros_like(nu)
    np.divide(nu, mu, out=out, where=mu > 0)
    return out


def density_measure(f, mu) -> np.ndarray:
    """The measure nu(A) = int_A f d mu, as point weights nu({x}) = f(x) mu({x}) (f >= 0)."""
    f = np.asarray(f, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    if (f < 0).any():
        raise ValueError("a density must be non-negative")
    return np.where(mu > 0, f * mu, 0.0)


def integral(g_values, weights) -> float:
    """int g d mu = sum_x g(x) mu({x}) on a finite space (0 * inf = 0)."""
    g = np.asarray(g_values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    m = (w > 0) & (g != 0)                    # 0 * inf = 0
    return float(np.sum(g[m] * w[m]))


# ---------------------------------------------------------------------------
# Laws on the line, quantiles and Monte Carlo checks
# ---------------------------------------------------------------------------


def quantile(cdf: Callable, u, lo: float, hi: float, tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """The generalised inverse F^{-1}(u) = inf{x : F(x) >= u}, by vectorised bisection on [lo, hi].

    ``F(lo) < u <= F(hi)`` is assumed (points with u <= F(lo) return lo). If U is uniform on
    (0, 1), quantile(F, U) has distribution function F: its law is the image of Lebesgue measure
    on (0, 1) under F^{-1}.
    """
    u = np.asarray(u, dtype=float)
    a = np.full(u.shape, float(lo))
    b = np.full(u.shape, float(hi))
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        right = np.asarray(cdf(mid), dtype=float) >= u
        b = np.where(right, mid, b)
        a = np.where(right, a, mid)
        if np.max(b - a, initial=0.0) <= tol * max(1.0, abs(lo), abs(hi)):
            break
    return np.where(np.asarray(cdf(np.full(u.shape, float(lo)))) >= u, lo, b)


class MixedLaw:
    """A law on R with atoms and a density: P = sum_i p_i delta_{a_i} + f(x) dx.

    ``density`` is a vectorised function, zero outside ``(lo, hi)``, with integral
    ``1 - sum(masses)``. It is neither discrete nor absolutely continuous when both parts are
    present (as in the Ross-style exercises).
    """

    def __init__(self, atoms, masses, density: Callable, lo: float, hi: float, tol: float = 1e-7):
        self.atoms = np.asarray(atoms, dtype=float)
        self.masses = np.asarray(masses, dtype=float)
        self.density = density
        self.lo, self.hi = float(lo), float(hi)
        self.ac_mass = integrate.quad(density, lo, hi, limit=200)[0]
        if abs(self.ac_mass + self.masses.sum() - 1) > tol:
            raise ValueError(f"total mass {self.ac_mass + self.masses.sum()} is not 1")

    def _ac_cdf(self, x: float) -> float:
        if x <= self.lo:
            return 0.0
        return integrate.quad(self.density, self.lo, min(x, self.hi), limit=200)[0]

    def cdf(self, x) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        jumps = (self.atoms[None, :] <= x[:, None]) @ self.masses
        return jumps + np.array([self._ac_cdf(t) for t in x])

    def cdf_left(self, x) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        jumps = (self.atoms[None, :] < x[:, None]) @ self.masses
        return jumps + np.array([self._ac_cdf(t) for t in x])

    def expectation(self, g: Callable = lambda x: x) -> float:
        """E[g(X)] = sum_i g(a_i) p_i + int g(x) f(x) dx."""
        atoms = float(np.sum(np.asarray(g(self.atoms), dtype=float) * self.masses))
        return atoms + integrate.quad(lambda x: g(x) * self.density(x), self.lo, self.hi, limit=200)[0]

    def sample(self, n: int, rng=None) -> np.ndarray:
        """n draws: an atom with probability p_i, else the density part by numerical inversion."""
        rng = np.random.default_rng(rng)
        probs = np.append(self.masses, self.ac_mass)
        which = rng.choice(len(probs), size=n, p=probs / probs.sum())
        out = np.empty(n)
        is_atom = which < len(self.masses)
        out[is_atom] = self.atoms[which[is_atom]]
        m = int((~is_atom).sum())
        if m:
            grid = np.linspace(self.lo, self.hi, 4097)
            F = np.concatenate([[0.0], np.cumsum(np.diff(grid) * 0.5 * (self.density(grid[1:]) + self.density(grid[:-1])))])
            out[~is_atom] = np.interp(rng.random(m) * F[-1], F, grid)
        return out


@dataclass
class MCResult:
    """A Monte Carlo estimate of E[g(X)] with its standard error and a 95% CLT interval."""

    estimate: float
    stderr: float
    n: int

    @property
    def ci(self) -> tuple:
        return (self.estimate - 1.96 * self.stderr, self.estimate + 1.96 * self.stderr)


def mc_expectation(g: Callable, sample) -> MCResult:
    """The sample mean of g(X_1), ..., g(X_n) and its standard error."""
    y = np.asarray(g(np.asarray(sample, dtype=float)), dtype=float)
    return MCResult(float(y.mean()), float(y.std(ddof=1) / math.sqrt(len(y))), len(y))


def expectation_density(g: Callable, density: Callable, lo: float = -INF, hi: float = INF,
                        points=None) -> float:
    """E[g(X)] = int g(x) f_X(x) dx by adaptive quadrature (``points``: kinks inside (lo, hi))."""
    h = lambda x: g(x) * density(x)
    if points is None:
        return integrate.quad(h, lo, hi, limit=200)[0]
    edges = [lo, *sorted(points), hi]
    return float(sum(integrate.quad(h, a, b, limit=200)[0] for a, b in zip(edges[:-1], edges[1:])))
