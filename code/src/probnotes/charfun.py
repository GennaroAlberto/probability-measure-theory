"""Characteristic functions (Chapter 10).

Five groups of tools, in the order of the chapter:

* **Closed forms.** ``cf_normal``, ``cf_uniform``, ``cf_exponential``, ``cf_gamma``,
  ``cf_laplace``, ``cf_cauchy``, ``cf_binomial``, ``cf_poisson``, ``cf_geometric`` and the two
  laws of the worked exercises, :func:`cf_density_2x` (the density ``2x`` on ``(0, 1)``) and
  :func:`cf_product_normals` (``(1 + t^2)^(-n/2)``, the law of a sum of ``n`` independent
  products of two independent standard Gaussians). :data:`LAWS` collects a few laws with their
  characteristic function, a sampler, a density and their first four moments.
* **Empirical characteristic functions.** :func:`ecf` computes
  ``phi_n(t) = (1/n) sum_k exp(i t^T X_k)`` for a sample in ``R`` or ``R^N``, and
  :func:`ecf_standard_error` its standard error ``sqrt((1 - |phi_n(t)|^2) / n)``.
  :func:`cf_independence_defect` measures the distance from the factorisation of the
  independence criterion (property 8) on a grid.
* **Fourier transforms by quadrature.** :func:`cf_from_density` computes
  ``int_a^b exp(itx) f(x) dx`` by Gauss--Legendre quadrature.
* **Inversion.** :func:`invert_density` computes the inversion formula
  ``f(x) = (1/2 pi) int exp(-itx) phi(t) dt`` truncated to ``[-L, L]``; :func:`levy_inversion`
  the truncated inversion formula for an interval, which also works when ``phi`` is not
  integrable; :func:`atom_mass` the formula ``(1/2L) int_{-L}^L exp(-itx) phi(t) dt``, whose limit
  is ``P(X = x)``.
* **Moments from derivatives at 0.** :func:`fd_weights` (Fornberg's algorithm) gives finite
  difference weights, :func:`cf_derivative` the ``k``-th derivative of ``phi`` at a point, and
  :func:`moments_from_cf` the moments ``E[X^k] = phi^(k)(0) / i^k`` (property 6).

Everything is vectorised and works by chunks, so that no intermediate array exceeds a few
million entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import math

import numpy as np
from scipy import special

_CHUNK = 2_000_000          # largest number of complex entries formed at once


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------


def _t(t) -> np.ndarray:
    return np.asarray(t, dtype=float)


def cf_normal(t, mu=0.0, sigma=1.0):
    """Characteristic function of N(mu, sigma^2): ``exp(i t mu - sigma^2 t^2 / 2)``."""
    t = _t(t)
    return np.exp(1j * t * mu - 0.5 * (sigma * t) ** 2)


def cf_uniform(t, a=-1.0, b=1.0):
    """Characteristic function of U(a, b): ``(exp(itb) - exp(ita)) / (it(b - a))``, and 1 at t = 0."""
    t = _t(t)
    c, h = 0.5 * (a + b), 0.5 * (b - a)
    return np.exp(1j * t * c) * np.sinc(t * h / np.pi)        # np.sinc(x) = sin(pi x)/(pi x)


def cf_exponential(t, lam=1.0):
    """Characteristic function of Exp(lam): ``lam / (lam - it)``."""
    t = _t(t)
    return lam / (lam - 1j * t)


def cf_gamma(t, alpha, lam=1.0):
    """Characteristic function of Gamma(alpha, lam) (lam a rate): ``(1 - it/lam)^(-alpha)``.

    The power is the principal one, which is the right branch because ``1 - it/lam`` has
    positive real part (see the exercises).
    """
    t = _t(t)
    return (1.0 - 1j * t / lam) ** (-alpha)


def cf_laplace(t, b=1.0):
    """Characteristic function of the Laplace density ``exp(-|x|/b) / (2b)``: ``1 / (1 + b^2 t^2)``."""
    t = _t(t)
    return (1.0 / (1.0 + (b * t) ** 2)).astype(complex)


def cf_cauchy(t, x0=0.0, gamma=1.0):
    """Characteristic function of the Cauchy law: ``exp(i x0 t - gamma |t|)``."""
    t = _t(t)
    return np.exp(1j * x0 * t - gamma * np.abs(t))


def cf_binomial(t, n, p):
    """Characteristic function of B(n, p): ``(1 - p + p exp(it))^n``."""
    t = _t(t)
    return (1.0 - p + p * np.exp(1j * t)) ** n


def cf_poisson(t, lam):
    """Characteristic function of the Poisson law P(lam): ``exp(lam (exp(it) - 1))``."""
    t = _t(t)
    return np.exp(lam * (np.exp(1j * t) - 1.0))


def cf_geometric(t, p):
    """Characteristic function of the geometric law on {1, 2, ...}, P(N = n) = p (1-p)^(n-1)."""
    t = _t(t)
    e = np.exp(1j * t)
    return p * e / (1.0 - (1.0 - p) * e)


def cf_density_2x(t):
    """Characteristic function of the density ``2x`` on ``(0, 1)`` (worked exercise).

    ``2 (exp(it)(1 - it) - 1) / t^2`` for ``t != 0``, and 1 at ``t = 0``. Near 0 the formula
    loses digits to cancellation, so the Taylor series ``sum_k 2 (it)^k / (k! (k + 2))`` is used
    for ``|t| < 0.1``.
    """
    t = _t(t)
    out = np.empty(t.shape, dtype=complex)
    small = np.abs(t) < 0.1
    ts = t[~small]
    out[~small] = 2.0 * (np.exp(1j * ts) * (1.0 - 1j * ts) - 1.0) / ts ** 2
    tz = t[small]
    acc = np.zeros(tz.shape, dtype=complex)
    term = np.ones(tz.shape, dtype=complex)
    for k in range(0, 14):                      # E[X^k] = 2/(k+2)
        acc += term * 2.0 / (k + 2)
        term = term * 1j * tz / (k + 1)
    out[small] = acc
    return out


def cf_product_normals(t, n=1):
    """``(1 + t^2)^(-n/2)``: the characteristic function of a sum of ``n`` independent copies of
    ``Z_1 Z_2``, with ``Z_1, Z_2`` independent standard Gaussians (worked exercise and exercises).
    ``n = 1`` is the function of the worked exercise, ``n = 2`` the Laplace law, ``n = 3`` the
    law of ``T = X + Y + Z``."""
    t = _t(t)
    return ((1.0 + t * t) ** (-0.5 * n)).astype(complex)


def density_product_normals(x, n=1):
    """Density of a sum of ``n`` independent copies of ``Z_1 Z_2`` (a variance-gamma law):

    ``|x|^((n-1)/2) K_((n-1)/2)(|x|) / (sqrt(pi) 2^((n-1)/2) Gamma(n/2))``, with ``K_nu`` the
    modified Bessel function of the second kind. For ``n = 1`` this is ``K_0(|x|)/pi``, which is
    unbounded near 0; for ``n = 2`` it is ``exp(-|x|)/2``; for ``n = 3`` it is ``|x| K_1(|x|)/pi``,
    whose value at 0 is ``1/pi``. At ``x = 0`` the value for ``n >= 2`` is the limit.
    """
    x = np.abs(np.asarray(x, dtype=float))
    nu = 0.5 * (n - 1)
    c = math.sqrt(math.pi) * 2.0 ** nu * math.gamma(0.5 * n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = x ** nu * special.kv(nu, x) / c
    if n >= 2:      # x^nu K_nu(x) -> 2^(nu-1) Gamma(nu) as x -> 0
        out = np.where(x == 0, 2.0 ** (nu - 1) * math.gamma(nu) / c, out)
    else:
        out = np.where(x == 0, np.inf, out)
    return out


@dataclass
class Law:
    """A law on the real line: its characteristic function, a sampler and what is known about it."""

    name: str
    cf: Callable[[np.ndarray], np.ndarray]
    sample: Callable[[np.random.Generator, int], np.ndarray]
    density: Callable[[np.ndarray], np.ndarray] | None = None
    pmf: Callable[[np.ndarray], np.ndarray] | None = None
    moments: dict = field(default_factory=dict)      # k -> E[X^k], k = 1..4 when finite
    integrable_cf: bool = False
    symmetric: bool = False


def _poisson_moments(lam):
    return {1: lam, 2: lam + lam ** 2, 3: lam ** 3 + 3 * lam ** 2 + lam,
            4: lam ** 4 + 6 * lam ** 3 + 7 * lam ** 2 + lam}


LAWS = {
    "normal": Law("N(0, 1)", cf_normal, lambda rng, n: rng.standard_normal(n),
                  density=lambda x: np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2 * math.pi),
                  moments={1: 0.0, 2: 1.0, 3: 0.0, 4: 3.0}, integrable_cf=True, symmetric=True),
    "uniform": Law("U(-1, 1)", cf_uniform, lambda rng, n: rng.uniform(-1.0, 1.0, n),
                   density=lambda x: 0.5 * (np.abs(np.asarray(x)) < 1),
                   moments={1: 0.0, 2: 1 / 3, 3: 0.0, 4: 1 / 5}, symmetric=True),
    "exponential": Law("Exp(1)", cf_exponential, lambda rng, n: rng.exponential(1.0, n),
                       density=lambda x: np.where(np.asarray(x) > 0, np.exp(-np.abs(x)), 0.0),
                       moments={1: 1.0, 2: 2.0, 3: 6.0, 4: 24.0}),
    "laplace": Law("Laplace", cf_laplace, lambda rng, n: rng.laplace(0.0, 1.0, n),
                   density=lambda x: 0.5 * np.exp(-np.abs(np.asarray(x))),
                   moments={1: 0.0, 2: 2.0, 3: 0.0, 4: 24.0}, integrable_cf=True, symmetric=True),
    "cauchy": Law("Cauchy", cf_cauchy, lambda rng, n: rng.standard_cauchy(n),
                  density=lambda x: 1.0 / (math.pi * (1.0 + np.asarray(x) ** 2)),
                  integrable_cf=True, symmetric=True),
    "product": Law("Z1 Z2", cf_product_normals,
                   lambda rng, n: rng.standard_normal(n) * rng.standard_normal(n),
                   density=density_product_normals,
                   moments={1: 0.0, 2: 1.0, 3: 0.0, 4: 9.0}, symmetric=True),
    "three products": Law("X + Y + Z", lambda t: cf_product_normals(t, 3),
                          lambda rng, n: (rng.standard_normal((3, n)) * rng.standard_normal((3, n))).sum(0),
                          density=lambda x: density_product_normals(x, 3),
                          moments={1: 0.0, 2: 3.0, 3: 0.0, 4: 45.0}, integrable_cf=True, symmetric=True),
    "density 2x": Law("density 2x on (0, 1)", cf_density_2x, lambda rng, n: np.sqrt(rng.random(n)),
                      density=lambda x: np.where((np.asarray(x) > 0) & (np.asarray(x) < 1), 2 * np.asarray(x), 0.0),
                      moments={k: 2 / (k + 2) for k in range(1, 5)}),
    "poisson": Law("P(2)", lambda t: cf_poisson(t, 2.0), lambda rng, n: rng.poisson(2.0, n).astype(float),
                   pmf=lambda k: np.exp(-2.0) * 2.0 ** np.asarray(k) / special.factorial(k),
                   moments=_poisson_moments(2.0)),
}


# ---------------------------------------------------------------------------
# Empirical characteristic functions
# ---------------------------------------------------------------------------


def _as_points(sample) -> np.ndarray:
    """A sample as an (n, N) array: one row per observation."""
    a = np.asarray(sample, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError("a sample is an array of shape (n,) or (n, N)")
    return a


def _as_freqs(t, N) -> tuple[np.ndarray, tuple]:
    """Frequencies as an (m, N) array, and the shape of the output."""
    t = np.asarray(t, dtype=float)
    if N == 1:
        return t.reshape(-1, 1), t.shape
    if t.shape[-1] != N:
        raise ValueError(f"frequencies in R^{N} must have last dimension {N}")
    return t.reshape(-1, N), t.shape[:-1]


def ecf(sample, t) -> np.ndarray:
    """Empirical characteristic function ``phi_n(t) = (1/n) sum_k exp(i t^T X_k)``.

    ``sample`` has shape ``(n,)`` (real observations) or ``(n, N)`` (observations in ``R^N``);
    ``t`` is an array of reals, or of points of ``R^N`` (last axis of length ``N``). It is the
    characteristic function of the empirical law ``(1/n) sum_k delta_{X_k}``.
    """
    X = _as_points(sample)
    n, N = X.shape
    T, shape = _as_freqs(t, N)
    out = np.zeros(len(T), dtype=complex)
    step = max(1, _CHUNK // max(len(T), 1))
    for i in range(0, n, step):
        out += np.exp(1j * (X[i:i + step] @ T.T)).sum(axis=0)
    return (out / n).reshape(shape)


def ecf_standard_error(sample, t) -> np.ndarray:
    """Estimated standard error of the empirical characteristic function at ``t``.

    ``E|exp(itX) - phi(t)|^2 = 1 - |phi(t)|^2``, so the standard error of the mean of ``n``
    copies is ``sqrt((1 - |phi(t)|^2) / n)``; the unknown ``phi`` is replaced by ``phi_n``.
    """
    n = _as_points(sample).shape[0]
    p = ecf(sample, t)
    return np.sqrt(np.clip(1.0 - np.abs(p) ** 2, 0.0, None) / n)


def cf_independence_defect(x, y, s, t) -> float:
    """``max |phi_{(X,Y)}(s_i, t_j) - phi_X(s_i) phi_Y(t_j)|`` over a grid, from a sample of pairs.

    By the independence criterion (property 8) the population version is 0 for every grid if and
    only if ``X`` and ``Y`` are independent. ``x`` and ``y`` are paired samples of shape ``(n,)``.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be paired one-dimensional samples")
    s, t = np.atleast_1d(np.asarray(s, dtype=float)), np.atleast_1d(np.asarray(t, dtype=float))
    n = len(x)
    joint = np.zeros((len(s), len(t)), dtype=complex)
    step = max(1, _CHUNK // (len(s) + len(t)))
    for i in range(0, n, step):
        A = np.exp(1j * np.outer(x[i:i + step], s))
        B = np.exp(1j * np.outer(y[i:i + step], t))
        joint += A.T @ B
    joint /= n
    return float(np.max(np.abs(joint - np.outer(ecf(x, s), ecf(y, t)))))


# ---------------------------------------------------------------------------
# Fourier transforms of densities by quadrature
# ---------------------------------------------------------------------------


def cf_from_density(density, t, a, b, n=200) -> np.ndarray:
    """``int_a^b exp(itx) f(x) dx`` by ``n``-point Gauss--Legendre quadrature on ``[a, b]``.

    Accurate when ``f`` is smooth on ``[a, b]`` (truncate the support if it is unbounded and
    the tails are negligible) and ``|t| (b - a)`` is small compared with ``n``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    w = 0.5 * (b - a) * weights * np.asarray(density(x), dtype=float)
    t = _t(t)
    flat = t.ravel()
    out = np.empty(flat.shape, dtype=complex)
    step = max(1, _CHUNK // n)
    for i in range(0, len(flat), step):
        out[i:i + step] = np.exp(1j * np.outer(flat[i:i + step], x)) @ w
    return out.reshape(t.shape)


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------


def _half_line_integral(g, phi, x, L, n):
    """``int_0^L Re(g(t, x) phi(t)) dt`` by the trapezoidal rule with ``n`` intervals, for each x.

    ``g(t, x)`` returns an array of shape ``(len(x), len(t))``.
    """
    if L <= 0 or n < 1:
        raise ValueError("need L > 0 and n >= 1")
    t = np.linspace(0.0, L, n + 1)
    w = np.full(n + 1, L / n)
    w[0] = w[-1] = 0.5 * L / n
    p = np.asarray(phi(t), dtype=complex) * w
    x = np.atleast_1d(np.asarray(x, dtype=float))
    flat = x.ravel()
    out = np.empty(flat.shape)
    step = max(1, _CHUNK // (n + 1))
    for i in range(0, len(flat), step):
        out[i:i + step] = np.real(g(t, flat[i:i + step]) @ p)
    return out.reshape(x.shape)


def invert_density(phi, x, L=50.0, n=20_000) -> np.ndarray:
    """Truncated inversion formula ``f_L(x) = (1/2pi) int_{-L}^{L} exp(-itx) phi(t) dt``.

    Uses ``phi(-t) = conj(phi(t))`` (property 2) to write ``f_L(x) = (1/pi) int_0^L
    Re(exp(-itx) phi(t)) dt``, computed by the trapezoidal rule with ``n`` intervals. If
    ``phi`` is integrable, ``f_L(x)`` tends to the continuous density as ``L -> infinity``, and
    the truncation error is at most ``(1/pi) int_L^infinity |phi(t)| dt``. The step ``L/n``
    must be small compared with ``1/|x|`` and with ``pi`` divided by the width of the law
    (the trapezoidal rule replaces the law by its periodisation with period ``2 pi n / L``).
    """
    return _half_line_integral(lambda t, xs: np.exp(-1j * np.outer(xs, t)), phi, x, L, n) / np.pi


def levy_inversion(phi, a, b, L=200.0, n=200_000) -> np.ndarray:
    """Truncated inversion formula for an interval:

    ``(1/2pi) int_{-L}^{L} (exp(-ita) - exp(-itb)) / (it) phi(t) dt``, which tends to
    ``P(a < X < b) + (P(X = a) + P(X = b)) / 2`` as ``L -> infinity`` for every law
    (Billingsley, Section 26). ``a`` may be an array; ``b`` is a number.
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))

    def g(t, aa):
        tt = np.where(t == 0, 1.0, t)
        val = (np.exp(-1j * np.outer(aa, tt)) - np.exp(-1j * b * tt)[None, :]) / (1j * tt)
        val[:, t == 0] = (b - aa)[:, None]
        return val

    return _half_line_integral(g, phi, a, L, n) / np.pi


def atom_mass(phi, x, L=200.0, n=200_000) -> np.ndarray:
    """``(1/2L) int_{-L}^{L} exp(-itx) phi(t) dt``, which tends to ``P(X = x)`` as ``L -> infinity``."""
    return _half_line_integral(lambda t, xs: np.exp(-1j * np.outer(xs, t)), phi, x, L, n) / L


# ---------------------------------------------------------------------------
# Derivatives at 0 and moments
# ---------------------------------------------------------------------------


def fd_weights(k, nodes, x0=0.0) -> np.ndarray:
    """Weights ``w_j`` with ``f^(k)(x0) ~ sum_j w_j f(nodes_j)``, exact for polynomials of degree
    ``< len(nodes)`` (Fornberg's algorithm, Math. Comp. 51 (1988), 699--706)."""
    z = np.asarray(nodes, dtype=float)
    m = len(z) - 1
    if k > m:
        raise ValueError("need more nodes than the order of the derivative")
    c = np.zeros((m + 1, k + 1))
    c[0, 0] = 1.0
    c1, c4 = 1.0, z[0] - x0
    for i in range(1, m + 1):
        mn = min(i, k)
        c2, c5, c4 = 1.0, c4, z[i] - x0
        for j in range(i):
            c3 = z[i] - z[j]
            c2 *= c3
            if j == i - 1:
                for s in range(mn, 0, -1):
                    c[i, s] = c1 * (s * c[i - 1, s - 1] - c5 * c[i - 1, s]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for s in range(mn, 0, -1):
                c[j, s] = (c4 * c[j, s] - s * c[j, s - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, k]


def cf_derivative(phi, k, t0=0.0, h=1e-2, p=4):
    """``k``-th derivative of ``phi`` at ``t0`` by the central difference on the ``2p + 1`` nodes
    ``t0 + j h``, ``|j| <= p`` (error of order ``h^(2p + 2 - k)`` for even ``k``, plus a
    rounding error of order ``1e-16 / h^k``)."""
    if 2 * p < k:
        raise ValueError("need 2p >= k")
    offs = np.arange(-p, p + 1, dtype=float)
    w = fd_weights(k, offs)
    vals = np.asarray(phi(t0 + h * offs), dtype=complex)
    return complex(w @ vals) / h ** k


def moments_from_cf(phi, kmax=4, h=1e-2, p=4) -> np.ndarray:
    """``E[X^k] = phi^(k)(0) / i^k`` for ``k = 1..kmax`` (property 6), by :func:`cf_derivative`.

    Returns complex numbers: for a real random variable with ``E|X|^kmax < infinity`` the
    imaginary parts are 0 up to the discretisation error, which is a useful check.
    """
    return np.array([cf_derivative(phi, k, 0.0, h, p) / 1j ** k for k in range(1, kmax + 1)])


def sample_moments(sample, kmax=4) -> np.ndarray:
    """``(1/n) sum_j X_j^k`` for ``k = 1..kmax``: the moments of the empirical law, whose
    characteristic function is :func:`ecf`."""
    x = np.asarray(sample, dtype=float)
    return np.array([np.mean(x ** k) for k in range(1, kmax + 1)])
