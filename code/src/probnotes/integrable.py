"""Integrable functions, dominated convergence, Riemann versus Lebesgue, discrete measures (Chapter 5).

Four groups of tools, in the order of the chapter:

* **Riemann and Lebesgue sums.** :func:`riemann_sum` and :func:`darboux_sums` partition the
  *domain* ``[a, b]``; :func:`lebesgue_sum` partitions the *codomain*: it integrates the dyadic
  simple function ``f_n = min(n, 2^-n floor(2^n f))`` of Chapter 3 through the measures of the
  level sets, ``int f_n dm = 2^-n sum_{i=1}^{n 2^n} m({f >= i 2^-n})``. The measure of the level
  sets is passed in as a function, exact when it is known, or estimated on a grid by
  :func:`level_measure_on_grid`.
* **Dominated convergence.** The two families of the worked exercises of the notes
  (:func:`siri_unit`, :func:`siri_halfline`) with their exact integrals, three families with no
  integrable dominating function (:func:`spike`, :func:`escaping_block`, :func:`bump`), the
  smallest dominating function :func:`envelope` and the check :func:`is_dominated`.
* **Discrete measures.** :class:`DiscreteMeasure` (finitely many atoms; a countable measure is
  handled through its truncations), the measure of the example of the notes with atoms
  ``x_n = (-1)^(n+1) 2^n / n`` of mass ``2^-n`` (:func:`siri_discrete`), and rearrangements of
  the conditionally convergent series ``sum f(x_n) mu({x_n})`` (:func:`rearrange_pq`,
  :func:`rearrange_to_target`).
* **sin x / x.** The improper integral of ``sin x / x`` from 1 to T, which converges, against
  that of ``|sin x| / x``, which grows like ``(2 / pi) log T``; both exact through the sine
  integral ``Si``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import special

# ---------------------------------------------------------------------------
# Riemann and Lebesgue sums
# ---------------------------------------------------------------------------


def parts(values):
    """Positive and negative parts ``(f+, f-)`` of an array of values, with ``f = f+ - f-``."""
    v = np.asarray(values, dtype=float)
    return np.maximum(v, 0.0), np.maximum(-v, 0.0)


def riemann_sum(f, a: float, b: float, n: int, rule: str = "mid") -> float:
    """Riemann sum of ``f`` on the uniform partition of ``[a, b]`` into ``n`` cells.

    ``rule`` picks the point of each cell where ``f`` is evaluated: ``"left"``, ``"right"``
    or ``"mid"``. ``f`` must accept a NumPy array.
    """
    h = (b - a) / n
    offset = {"left": 0.0, "mid": 0.5, "right": 1.0}[rule]
    x = a + h * (np.arange(n) + offset)
    return float(h * np.sum(f(x)))


def darboux_sums(f, a: float, b: float, n: int, samples: int = 33):
    """Lower and upper Darboux sums of ``f`` on the uniform partition of ``[a, b]`` into ``n`` cells.

    The infimum and supremum of ``f`` on each cell are estimated by the minimum and maximum of
    ``f`` at ``samples`` equally spaced points of the cell, both ends included. The estimate is
    exact for monotone ``f`` (the extremes are at the ends) and for ``f`` monotone on each cell;
    in general it is an inner approximation (lower sum too high, upper sum too low).
    Returns ``(lower, upper)``.
    """
    h = (b - a) / n
    s = np.linspace(0.0, 1.0, samples)
    x = a + h * (np.arange(n)[:, None] + s[None, :])
    y = f(x)
    return float(h * np.sum(y.min(axis=1))), float(h * np.sum(y.max(axis=1)))


def lebesgue_sum(level_measure, n: int) -> float:
    """Integral of the dyadic simple function ``f_n`` of a measurable ``f >= 0``, by horizontal slices.

    ``f_n = min(n, 2^-n floor(2^n f))`` is the n-th function of the approximation theorem of
    Chapter 3, and ``f_n = 2^-n sum_{i=1}^{n 2^n} 1_{f >= i 2^-n}``, so

        int f_n dm = 2^-n * sum_{i=1}^{n 2^n} m({f >= i 2^-n}).

    ``level_measure(t)`` must return ``m({f >= t})`` for an array of levels ``t > 0``. By
    monotone convergence the values increase to ``int f dm`` as ``n`` grows.
    """
    levels = np.arange(1, n * 2**n + 1) / 2.0**n
    return float(np.sum(level_measure(levels)) / 2.0**n)


def level_measure_on_grid(f, a: float, b: float, n_grid: int = 2**16):
    """Return ``t -> m({x in [a, b] : f(x) >= t})``, estimated on the midpoint grid of ``n_grid`` cells.

    Each grid point stands for a cell of length ``(b - a) / n_grid``; the returned function is
    vectorised in ``t``.
    """
    h = (b - a) / n_grid
    values = np.sort(f(a + h * (np.arange(n_grid) + 0.5)))

    def level_measure(t):
        t = np.asarray(t, dtype=float)
        return h * (n_grid - np.searchsorted(values, t, side="left"))

    return level_measure


def lebesgue_sum_on_grid(f, a: float, b: float, n: int, n_grid: int = 2**16) -> float:
    """:func:`lebesgue_sum` of ``f >= 0`` on ``[a, b]`` with the level sets measured on a grid."""
    return lebesgue_sum(level_measure_on_grid(f, a, b, n_grid), n)


# ---------------------------------------------------------------------------
# Dominated convergence: the two worked exercises, and three counterexamples
# ---------------------------------------------------------------------------


def siri_unit(x, n):
    """``f_n(x) = n x^(-3/2) log(1 + x/n)`` on ``(0, 1)``: the first worked exercise.

    ``0 <= f_n(x) <= x^(-1/2)``, and ``f_n(x)`` increases to ``x^(-1/2)`` as ``n`` grows.
    """
    x = np.asarray(x, dtype=float)
    return n * x**-1.5 * np.log1p(x / n)


def siri_unit_integral(n):
    """Exact ``int_0^1 f_n dx`` for :func:`siri_unit`: ``4 sqrt(n) arctan(1/sqrt(n)) - 2 n log(1 + 1/n)``.

    It increases to ``int_0^1 x^(-1/2) dx = 2``, and equals ``2 - 1/(3n) + O(n^-2)``.
    """
    n = np.asarray(n, dtype=float)
    return 4 * np.sqrt(n) * np.arctan(1 / np.sqrt(n)) - 2 * n * np.log1p(1 / n)


def siri_halfline(x, n):
    """``f_n(x) = x^(-3/2) log(1 + x/n)`` on ``(0, +inf)``: the second worked exercise. Decreases to 0."""
    x = np.asarray(x, dtype=float)
    return x**-1.5 * np.log1p(x / n)


def siri_halfline_integral(n):
    """Exact ``int_0^inf f_n dx = 2 pi / sqrt(n)`` for :func:`siri_halfline` (substitute ``x = n u``)."""
    return 2 * math.pi / np.sqrt(np.asarray(n, dtype=float))


def spike(x, n):
    """``n 1_(0, 1/n)(x)``: tends to 0 at every point, but every integral over ``(0, 1)`` is 1."""
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    return np.where((x > 0) & (x < 1.0 / n), n, 0.0)


def escaping_block(x, n):
    """``1_[n, n+1)(x)``: tends to 0 at every point, but every integral over the real line is 1."""
    x = np.asarray(x, dtype=float)
    return ((x >= n) & (x < n + 1)).astype(float)


def bump(x, n):
    """``n^2 x e^(-n x)`` on ``[0, 1]``: tends to 0 at every point, while the integrals tend to 1."""
    x = np.asarray(x, dtype=float)
    return n**2 * x * np.exp(-n * x)


def bump_integral(n):
    """Exact ``int_0^1 n^2 x e^(-n x) dx = 1 - (1 + n) e^(-n)``."""
    n = np.asarray(n, dtype=float)
    return 1 - (1 + n) * np.exp(-n)


def envelope(family, x, ns, chunk: int = 256):
    """The smallest dominating function ``sup_{n in ns} |f_n(x)|`` of a family, on the points ``x``.

    ``family(x, n)`` must broadcast over an array of ``x`` and a column of ``n``. The values are
    computed in chunks of ``chunk`` indices, so memory stays at ``chunk * len(x)`` floats.
    """
    x = np.asarray(x, dtype=float)
    ns = np.asarray(ns)
    out = np.zeros_like(x)
    for start in range(0, len(ns), chunk):
        block = ns[start:start + chunk][:, None]
        out = np.maximum(out, np.max(np.abs(family(x[None, :], block)), axis=0))
    return out


def is_dominated(family, g, x, ns, rtol: float = 1e-12) -> bool:
    """Whether ``|f_n(x)| <= g(x)`` (up to a relative tolerance) for every ``n`` in ``ns`` and ``x`` in ``x``."""
    x = np.asarray(x, dtype=float)
    gx = np.asarray(g(x), dtype=float)
    return bool(np.all(envelope(family, x, ns) <= gx * (1 + rtol)))


# ---------------------------------------------------------------------------
# Discrete measures and the series of the notes
# ---------------------------------------------------------------------------


@dataclass
class DiscreteMeasure:
    """A measure ``mu = sum_i alpha_i delta_{x_i}`` with finitely many atoms.

    A measure with countably many atoms is studied through its truncations to the first
    ``N`` atoms; :meth:`integral` is then the N-th partial sum of ``sum_i f(x_i) alpha_i``.
    """

    points: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.points.shape != self.weights.shape or self.points.ndim != 1:
            raise ValueError("points and weights must be 1-d arrays of the same length")
        if np.any(self.weights <= 0):
            raise ValueError("the masses alpha_i must be positive")
        if len(np.unique(self.points)) != len(self.points):
            raise ValueError("the atoms x_i must be distinct")

    def measure(self, indicator) -> float:
        """``mu(A) = sum_{x_i in A} alpha_i``, where ``indicator(points)`` is true on ``A``."""
        return float(np.sum(self.weights[np.asarray(indicator(self.points), dtype=bool)]))

    def terms(self, f) -> np.ndarray:
        """The terms ``f(x_i) alpha_i`` of the series that defines the integral, in the order of the atoms."""
        return np.asarray(f(self.points), dtype=float) * self.weights

    def integral(self, f) -> float:
        """``sum_i f(x_i) alpha_i`` (a partial sum, for a truncated countable measure)."""
        return float(np.sum(self.terms(f)))

    def abs_integral(self, f) -> float:
        """``sum_i |f(x_i)| alpha_i``: ``f`` is integrable if and only if this stays bounded."""
        return float(np.sum(np.abs(self.terms(f))))

    def parts_integrals(self, f):
        """``(int f+ dmu, int f- dmu)`` for the (truncated) measure."""
        t = self.terms(f)
        return float(np.sum(np.maximum(t, 0))), float(np.sum(np.maximum(-t, 0)))


def siri_discrete(N: int) -> DiscreteMeasure:
    """The first ``N`` atoms of the measure of the notes: ``x_n = (-1)^(n+1) 2^n / n``, ``mu({x_n}) = 2^-n``, ``n >= 1``.

    The atoms are distinct, and the total mass is ``sum_{n >= 1} 2^-n = 1``. ``2^n`` overflows a
    double for ``n > 1023``, so ``N <= 1000`` here; use :func:`siri_terms` for longer series.
    """
    if not 1 <= N <= 1000:
        raise ValueError("N must be between 1 and 1000 (2^n overflows beyond that)")
    n = np.arange(1, N + 1)
    x = (-1.0) ** (n + 1) * 2.0**n / n
    return DiscreteMeasure(x, 2.0**-n)


def siri_terms(N: int) -> np.ndarray:
    """The terms ``f(x_n) mu({x_n}) = (-1)^(n+1) / n``, ``n = 1..N``, for ``f(x) = x``, computed exactly."""
    n = np.arange(1, N + 1)
    return (-1.0) ** (n + 1) / n


def alternating_parts(N: int):
    """The positive terms ``1/(2k - 1)`` and the negative terms ``-1/(2k)``, ``k = 1..N``, of :func:`siri_terms`.

    They are the contributions of the atoms where ``f = x`` is positive (n odd) and negative
    (n even); each part has an infinite sum: ``int f+ dmu = int f- dmu = +inf``.
    """
    k = np.arange(1, N + 1)
    return 1.0 / (2 * k - 1), -1.0 / (2 * k)


def rearrange_pq(pos, neg, p: int, q: int) -> np.ndarray:
    """Rearrange a series: ``p`` positive terms, then ``q`` negative terms, and so on.

    ``pos`` and ``neg`` are the positive and the negative terms in their original order; as
    many complete blocks as both arrays allow are used. Every term of the original series
    appears exactly once in the limit, so this is a rearrangement.
    """
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    blocks = min(len(pos) // p, len(neg) // q)
    return np.hstack([pos[:p * blocks].reshape(blocks, p), neg[:q * blocks].reshape(blocks, q)]).ravel()


def pq_limit(p: int, q: int) -> float:
    """Sum of the alternating harmonic series rearranged as ``p`` positive, ``q`` negative terms: ``log 2 + log(p/q) / 2``."""
    return math.log(2) + 0.5 * math.log(p / q)


def rearrange_to_target(pos, neg, target: float, n_terms: int) -> np.ndarray:
    """Riemann's rearrangement: positive terms until the partial sum exceeds ``target``, then negative ones until it falls below, and so on.

    Works when both parts diverge and the terms tend to 0; then the partial sums converge to
    ``target``. Returns the first ``n_terms`` rearranged terms (fewer if ``pos`` or ``neg`` run out).
    """
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    out = np.empty(n_terms)
    i = j = 0
    s = 0.0
    for k in range(n_terms):
        if s <= target:
            if i == len(pos):
                return out[:k]
            out[k] = pos[i]
            i += 1
        else:
            if j == len(neg):
                return out[:k]
            out[k] = neg[j]
            j += 1
        s += out[k]
    return out


# ---------------------------------------------------------------------------
# sin x / x: an improper Riemann integral that is not a Lebesgue integral
# ---------------------------------------------------------------------------

SI_1 = float(special.sici(1.0)[0])
#: The improper integral from 1 to infinity of sin x / x, pi/2 - Si(1).
SINE_LIMIT = math.pi / 2 - SI_1


def sine_integral_from_1(T):
    """``int_1^T sin(x)/x dx = Si(T) - Si(1)`` (vectorised); it tends to ``pi/2 - Si(1) = 0.62471...``."""
    return special.sici(np.asarray(T, dtype=float))[0] - SI_1


def abs_sine_integral_from_1(T):
    """``int_1^T |sin x| / x dx``, exactly, through ``Si`` on the half-periods ``[k pi, (k+1) pi]``.

    On ``[k pi, (k+1) pi]``, ``|sin x| = (-1)^k sin x``, so each piece is a signed difference of
    values of ``Si``. Vectorised in ``T >= 1``; memory grows like ``max(T) / pi`` floats.
    """
    T = np.asarray(T, dtype=float)
    if np.any(T < 1):
        raise ValueError("T must be at least 1")
    K = np.floor(T / np.pi).astype(np.int64)          # multiples of pi in (1, T]
    kmax = int(K.max())
    si_br = special.sici(np.pi * np.arange(1, kmax + 1))[0]
    si_starts = np.concatenate([[SI_1], si_br])        # Si at 1, pi, 2 pi, ..., kmax pi
    pieces = np.diff(si_starts) * (-1.0) ** np.arange(kmax)
    cum = np.concatenate([[0.0], np.cumsum(pieces)])   # cum[k] = integral from 1 to k pi
    tail = (-1.0) ** K * (special.sici(T)[0] - si_starts[K])
    return cum[K] + tail


def abs_sine_constant(n_terms: int = 10**6) -> float:
    """``C = lim_T (int_1^T |sin x|/x dx - (2/pi) log T) = (4/pi) sum_k Ci(2k) / (4k^2 - 1) = 0.16684...``.

    From the Fourier series ``|sin x| = 2/pi - (4/pi) sum_k cos(2kx) / (4k^2 - 1)``; the
    truncation error is below ``1 / (8 n_terms^2)``.
    """
    k = np.arange(1, n_terms + 1, dtype=float)
    return float(4 / math.pi * np.sum(special.sici(2 * k)[1] / (4 * k**2 - 1)))
