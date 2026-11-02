"""The integral of non-negative functions (Chapter 4).

Five groups of tools, in the order of the chapter:

* **Arithmetic in [0, +inf]** with the convention ``0 * inf = 0`` of the notes (:func:`mul`).
* **Integrals of non-negative simple functions** from any representation
  ``sum_i a_i 1_{A_i}``: :func:`simple_integral` from the numbers ``a_i`` and ``mu(A_i)``,
  :func:`simple_integral_on` for a measure given as a function of sets (for instance a
  :class:`probnotes.measures.DiscreteMeasure` on a finite or countable space, with finite or
  cofinite subsets of N), and :func:`integral_finite` on a finite space, where every function
  is simple.
* **Step functions on R with Lebesgue measure.** :class:`StepFunction` stores a non-negative
  function that is constant on the intervals ``[breaks[j], breaks[j+1])`` and zero elsewhere:
  this is the common refinement of any representation ``sum_i a_i 1_{A_i}`` with ``A_i``
  finite unions of intervals (:func:`step_function`). It evaluates, integrates over a set
  ``E``, returns its canonical form, and :func:`combine` takes sums, minima and maxima of
  several step functions on their common refinement.
* **The dyadic approximants** ``f_n = min(n, floor(2^n f) / 2^n)`` of Chapter 3 and their
  integrals. Since ``f_n = 2^-n sum_{k=1}^{n 2^n} 1_{f >= k 2^-n}``, the integral of ``f_n``
  only needs the *tail function* ``t -> mu({f >= t})`` (:func:`dyadic_integral`); on a finite
  space it is computed directly (:func:`dyadic_integral_finite`). :data:`EXAMPLES` lists
  functions with known tails and integrals.
* **Sequences for the convergence theorems**: the sliding bump ``1_[n, n+1)`` of the notes, the
  spike, the spreading plateau and an alternating sequence (:data:`SEQUENCES`), the tail infima
  ``inf_{n <= k <= K} f_k`` that build ``liminf f_n`` (:func:`tail_infimum`), the two sides of
  Fatou's lemma (:func:`fatou_sides`), and Fatou's lemma for periodic sequences on a finite
  space (:func:`periodic_fatou`), where both sides are exact.

Values ``+inf`` are ``math.inf``. All functions are vectorised with NumPy where it matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from probnotes.measurable import dyadic_approximation, lebesgue_measure, normalize_intervals

INF = math.inf

# ---------------------------------------------------------------------------
# Arithmetic in [0, +inf]
# ---------------------------------------------------------------------------


def _nonneg(a, name="values") -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if np.isnan(a).any() or (a < 0).any():
        raise ValueError(f"{name} must lie in [0, +inf]")
    return a


def mul(a, b) -> np.ndarray:
    """Product in [0, +inf] with the convention 0 * inf = 0 (elementwise, broadcasting)."""
    a, b = _nonneg(a, "factors"), _nonneg(b, "factors")
    with np.errstate(invalid="ignore"):
        p = a * b
    return np.where((a == 0) | (b == 0), 0.0, p)


# ---------------------------------------------------------------------------
# Integrals of non-negative simple functions
# ---------------------------------------------------------------------------


def simple_integral(coeffs, measures) -> float:
    """sum_i a_i mu(A_i), with 0 * inf = 0: the integral of sum_i a_i 1_{A_i}.

    ``coeffs`` are the values a_i in [0, +inf] and ``measures`` the numbers mu(A_i) in
    [0, +inf]. For a partition A_1, ..., A_n this is the definition of the notes; by
    additivity and homogeneity the same formula holds for any finite family of measurable
    sets (Exercise 4.1).
    """
    return float(np.sum(mul(coeffs, measures)))


def simple_integral_on(coeffs, sets, mu: Callable, E=None) -> float:
    """Integral over E of sum_i a_i 1_{A_i} with respect to mu: sum_i a_i mu(A_i cap E).

    ``mu`` is any function of sets, for instance a :class:`probnotes.measures.DiscreteMeasure`
    (whose sets may be Python sets or :class:`probnotes.measures.FinCofin` subsets of N);
    ``sets`` and ``E`` must support ``&`` (intersection). ``E = None`` integrates over X.
    """
    sets = list(sets)
    if E is not None:
        sets = [A & E for A in sets]
    return simple_integral(coeffs, [mu(A) for A in sets])


def integral_finite(values, weights, E=None) -> float:
    """Integral of f >= 0 on X = {0, ..., N-1}, where mu({x}) = weights[x]: sum_x f(x) mu({x}).

    On a finite space every function is simple (its level sets partition X), so this is the
    integral of the notes. ``E`` (a boolean mask or a collection of points) restricts the
    integral to E, as in the integral of f 1_E.
    """
    f, w = _nonneg(values), _nonneg(weights, "weights")
    if f.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    terms = mul(f, w)
    if E is not None:
        mask = np.asarray(E)
        if mask.dtype != bool:
            m = np.zeros(f.shape, dtype=bool)
            m[list(E)] = True
            mask = m
        terms = np.where(mask, terms, 0.0)
    return float(np.sum(terms))


def integral_series(f: Callable, weights: Callable, n_terms: int) -> np.ndarray:
    """Partial sums S_N = sum_{n < N} f(n) w(n), N = 1, ..., n_terms, on X = N with mu({n}) = w(n).

    S_N is the integral of f 1_{{0, ..., N-1}}, a simple function; these increase to f, so by
    monotone convergence S_N increases to the integral of f with respect to mu. ``f`` and
    ``weights`` are vectorised functions of an integer array.
    """
    n = np.arange(int(n_terms))
    return np.cumsum(mul(f(n), weights(n)))


# ---------------------------------------------------------------------------
# Step functions on R (Lebesgue measure)
# ---------------------------------------------------------------------------


def _midpoints(breaks: np.ndarray) -> np.ndarray:
    """A point inside each elementary interval [breaks[j], breaks[j+1]), even if unbounded."""
    a, b = breaks[:-1], breaks[1:]
    with np.errstate(invalid="ignore"):
        mid = (a + b) / 2
    mid = np.where(np.isinf(a) & np.isfinite(b), b - 1.0, mid)
    mid = np.where(np.isfinite(a) & np.isinf(b), a + 1.0, mid)
    return np.where(np.isinf(a) & np.isinf(b), 0.0, mid)


@dataclass(frozen=True)
class StepFunction:
    """A non-negative step function on R: equal to ``values[j]`` on [breaks[j], breaks[j+1]).

    ``breaks`` is strictly increasing (endpoints may be -inf or inf) and the function is 0
    outside [breaks[0], breaks[-1]). Values lie in [0, +inf]. Every non-negative simple
    function sum_i a_i 1_{A_i} whose sets A_i are finite unions of intervals can be written so
    (:func:`step_function`); this is the common refinement of the notes, on the line.
    """

    breaks: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        b = np.asarray(self.breaks, dtype=float)
        v = _nonneg(self.values)
        if b.ndim != 1 or v.ndim != 1 or len(b) != len(v) + 1:
            raise ValueError("need len(breaks) == len(values) + 1")
        if len(b) > 1 and not np.all(np.diff(b) > 0):
            raise ValueError("breaks must be strictly increasing")
        object.__setattr__(self, "breaks", b)
        object.__setattr__(self, "values", v)

    @property
    def lengths(self) -> np.ndarray:
        """Lebesgue measure of each elementary interval."""
        return np.diff(self.breaks)

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(self.values) == 0:
            return np.zeros(x.shape)
        j = np.searchsorted(self.breaks, x, side="right") - 1
        inside = (j >= 0) & (j < len(self.values))
        return np.where(inside, self.values[np.clip(j, 0, len(self.values) - 1)], 0.0)

    def integral(self, E=None) -> float:
        """Integral over E (a finite union of intervals; ``None`` = R): sum_j v_j m(I_j cap E)."""
        if E is None:
            return simple_integral(self.values, self.lengths)
        Eb = normalize_intervals(E)
        if len(Eb) == 0 or len(self.values) == 0:
            return 0.0
        lo = np.maximum.outer(self.breaks[:-1], Eb[:, 0])
        hi = np.minimum.outer(self.breaks[1:], Eb[:, 1])
        overlap = np.clip(hi - lo, 0.0, None).sum(axis=1)
        return simple_integral(self.values, overlap)

    def canonical(self):
        """Canonical form: the distinct non-zero values a_1 < ... < a_m and their level sets.

        Returns ``(a, sets)`` where ``sets[i]`` is the normalised union of intervals
        {f = a_i}; the remaining level set {f = 0} carries no integral.
        """
        a = np.unique(self.values[self.values > 0])
        sets = [normalize_intervals(np.column_stack([self.breaks[:-1][self.values == v],
                                                      self.breaks[1:][self.values == v]]))
                for v in a]
        return a, sets

    def scale(self, alpha: float) -> "StepFunction":
        """alpha f for alpha in [0, +inf] (with 0 * inf = 0)."""
        return StepFunction(self.breaks, mul(alpha, self.values))

    def restrict(self, E) -> "StepFunction":
        """f 1_E for a finite union of intervals E."""
        return combine([self, step_function([1.0], [E])], "prod")


def step_function(coeffs, sets) -> StepFunction:
    """The step function sum_i a_i 1_{A_i}, where each A_i is a finite union of intervals.

    The sets need not be disjoint; intervals are half-open [a, b), as in
    :func:`probnotes.measurable.contains`. The result is the common refinement: the breaks are
    all the endpoints, and the value on each elementary interval is the sum of the a_i whose
    set contains it.
    """
    coeffs = _nonneg(coeffs, "coefficients")
    blocks = [normalize_intervals(A) for A in sets]
    if len(coeffs) != len(blocks):
        raise ValueError("one coefficient per set")
    ends = [B.ravel() for B in blocks if len(B)]
    if not ends:
        return StepFunction(np.array([0.0]), np.array([]))
    breaks = np.unique(np.concatenate(ends))
    mid = _midpoints(breaks)
    vals = np.zeros(len(mid))
    for a, B in zip(coeffs, blocks):
        if len(B):
            inside = ((mid[:, None] >= B[:, 0]) & (mid[:, None] < B[:, 1])).any(axis=1)
            vals = vals + mul(a, inside.astype(float))
    return _simplify(breaks, vals)


def _simplify(breaks: np.ndarray, vals: np.ndarray) -> StepFunction:
    """Merge adjacent elementary intervals with equal values and trim zeros at both ends."""
    if len(vals) == 0:
        return StepFunction(np.array([0.0]), np.array([]))
    keep = np.ones(len(breaks), dtype=bool)
    keep[1:-1] = vals[1:] != vals[:-1]          # break j separates intervals j-1 and j
    idx = np.flatnonzero(keep)
    b = breaks[idx]
    v = vals[idx[:-1]]
    nz = np.flatnonzero(v > 0)
    if len(nz) == 0:
        return StepFunction(np.array([0.0]), np.array([]))
    lo, hi = nz[0], nz[-1]
    return StepFunction(b[lo:hi + 2], v[lo:hi + 1])


_OPS = {"sum": lambda M: np.sum(M, axis=0),
        "min": lambda M: np.min(M, axis=0),
        "max": lambda M: np.max(M, axis=0),
        "prod": lambda M: np.prod(np.where(np.any(M == 0, axis=0), 0.0, M), axis=0)}


def combine(fs: Sequence[StepFunction], op: str = "sum") -> StepFunction:
    """Pointwise sum, minimum, maximum or product of step functions, on their common refinement.

    ``op`` is ``"sum"``, ``"min"``, ``"max"`` or ``"prod"`` (with 0 * inf = 0). Outside the
    union of the supports every function is 0, so the result is 0 there as well.
    """
    if op not in _OPS:
        raise ValueError(f"op must be one of {sorted(_OPS)}")
    fs = list(fs)
    if not fs:
        raise ValueError("need at least one step function")
    pts = [f.breaks for f in fs if len(f.values)]
    if not pts:
        return StepFunction(np.array([0.0]), np.array([]))
    breaks = np.unique(np.concatenate(pts))
    if len(breaks) < 2:
        return StepFunction(np.array([0.0]), np.array([]))
    mid = _midpoints(breaks)
    M = np.stack([f(mid) for f in fs])          # f(x) = 0 outside the support of f
    return _simplify(breaks, _OPS[op](M))


def lebesgue_simple_integral(coeffs, sets) -> float:
    """sum_i a_i m(A_i) for finite unions of intervals A_i: the definition, without refining."""
    return simple_integral(coeffs, [lebesgue_measure(A) for A in sets])


# ---------------------------------------------------------------------------
# The dyadic approximants and their integrals
# ---------------------------------------------------------------------------

MAX_DYADIC = 22  # n 2^n tail evaluations: about 9e7 for n = 22, done in chunks


def dyadic_integral(tail: Callable, n: int, chunk: int = 1 << 20) -> float:
    """Integral of the n-th dyadic approximant f_n of f, from the tail t -> mu({f >= t}).

    Since f_n = sum_{k=1}^{n 2^n} 2^-n 1_{{f >= k 2^-n}} (a sum of indicators, one per level
    crossed), its integral is 2^-n sum_k mu({f >= k 2^-n}): a lower Riemann sum of the
    non-increasing tail function. ``tail`` must be vectorised over t > 0; ``inf`` values are
    allowed. Summed in chunks, so memory stays bounded.
    """
    n = int(n)
    if not 1 <= n <= MAX_DYADIC:
        raise ValueError(f"need 1 <= n <= {MAX_DYADIC}")
    h, K = 2.0**-n, n * 2**n
    total = 0.0
    for start in range(1, K + 1, chunk):
        k = np.arange(start, min(start + chunk, K + 1), dtype=float)
        tv = _nonneg(tail(k * h), "tail values")
        if np.isinf(tv).any():
            return INF
        total += math.fsum(tv)
    return h * total


def dyadic_integrals(tail: Callable, n_max: int) -> np.ndarray:
    """The integrals of f_1, ..., f_{n_max} (an increasing sequence, by monotonicity)."""
    return np.array([dyadic_integral(tail, n) for n in range(1, int(n_max) + 1)])


def dyadic_as_indicator_sum(values, n: int) -> np.ndarray:
    """f_n computed as 2^-n sum_{k=1}^{n 2^n} 1_{f >= k 2^-n} (to check the identity)."""
    v = _nonneg(values)
    k = np.arange(1, n * 2**n + 1) / 2.0**n
    return (v[..., None] >= k).sum(axis=-1) / 2.0**n


def dyadic_integral_finite(values, weights, n: int) -> float:
    """Integral of the n-th dyadic approximant on a finite space X = {0, ..., N-1}."""
    return integral_finite(dyadic_approximation(values, n), weights)


def tail_from_values(values, weights) -> Callable:
    """The tail t -> mu({f >= t}) of a function on a finite space, as a vectorised function."""
    f, w = _nonneg(values), _nonneg(weights, "weights")
    order = np.argsort(f)
    fs, ws = f[order], w[order]
    suffix = np.concatenate([np.cumsum(ws[::-1])[::-1], [0.0]])   # mass of {f >= fs[i]}

    def tail(t):
        i = np.searchsorted(fs, np.asarray(t, dtype=float), side="left")
        return suffix[i]

    return tail


@dataclass(frozen=True)
class DyadicExample:
    """A non-negative function on an interval with its tail t -> m({f >= t}) and its integral."""

    f: Callable
    domain: tuple
    tail: Callable
    integral: float
    note: str


def _tail_square(t):
    t = np.asarray(t, dtype=float)
    return np.clip(1 - np.sqrt(np.minimum(t, 1.0)), 0.0, 1.0)


def _tail_cauchy(t):
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(t < 1, 2 * np.sqrt(np.maximum(1 / t - 1, 0.0)), 0.0)


EXAMPLES = {
    "x on [0, 1]": DyadicExample(
        lambda x: x, (0.0, 1.0), lambda t: np.clip(1 - np.asarray(t, dtype=float), 0.0, 1.0), 0.5,
        "bounded, on a set of measure 1: only the rounding error, exactly 2^-(n+1)"),
    "x^2 on [0, 1]": DyadicExample(
        lambda x: x**2, (0.0, 1.0), _tail_square, 1 / 3,
        "bounded: rounding error about 2^-(n+1)"),
    "x^(-1/2) on (0, 1)": DyadicExample(
        lambda x: x**-0.5, (0.0, 1.0),
        lambda t: np.minimum(1.0, np.asarray(t, dtype=float) ** -2.0), 2.0,
        "unbounded: the cut at height n costs 1/n"),
    "-log x on (0, 1)": DyadicExample(
        lambda x: -np.log(x), (0.0, 1.0),
        lambda t: np.minimum(1.0, np.exp(-np.asarray(t, dtype=float))), 1.0,
        "unbounded with a thin tail: the cut costs e^-n"),
    "1/(1+x^2) on R": DyadicExample(
        lambda x: 1 / (1 + x**2), (-INF, INF), _tail_cauchy, math.pi,
        "on a set of infinite measure: the small values discarded cost about 2^(1-n/2)"),
    "1/x on (0, 1)": DyadicExample(
        lambda x: 1 / x, (0.0, 1.0),
        lambda t: np.minimum(1.0, 1 / np.asarray(t, dtype=float)), INF,
        "not integrable: the integrals of f_n grow like 1 + log n"),
}


# ---------------------------------------------------------------------------
# Sequences for the convergence theorems
# ---------------------------------------------------------------------------


def indicator_step(a: float, b: float, height: float = 1.0) -> StepFunction:
    """height * 1_[a, b) as a step function."""
    if not b > a:
        return StepFunction(np.array([0.0]), np.array([]))
    return StepFunction(np.array([a, b], dtype=float), np.array([height], dtype=float))


def sliding_bump(n: int) -> StepFunction:
    """The sliding bump 1_[n, n+1) of the notes: it tends to 0 pointwise, with integral 1."""
    return indicator_step(n, n + 1)


def spike(n: int) -> StepFunction:
    """n 1_(0, 1/n), stored as n 1_[0, 1/n) (the same function a.e.); integral 1, limit 0 a.e."""
    return indicator_step(0.0, 1.0 / n, float(n))


def spreading(n: int) -> StepFunction:
    """(1/n) 1_[0, n): tends to 0 uniformly, with integral 1."""
    return indicator_step(0.0, float(n), 1.0 / n)


def alternating(n: int) -> StepFunction:
    """1_[0, 1) for even n and 1_[1, 2) for odd n: liminf = 0, limsup = 1_[0, 2)."""
    return indicator_step(0.0, 1.0) if n % 2 == 0 else indicator_step(1.0, 2.0)


def bump_plus_fixed(n: int) -> StepFunction:
    """1_[0, 1) + 1_[n, n+1): liminf = 1_[0, 1) has integral 1, while every f_n (n >= 1) has 2."""
    return combine([indicator_step(0.0, 1.0), sliding_bump(n)], "sum")


def increasing_plateau(n: int) -> StepFunction:
    """1_[0, n): an increasing sequence, for which Fatou's lemma is an equality (both sides +inf)."""
    return indicator_step(0.0, float(n))


SEQUENCES = {
    "sliding bump 1_[n,n+1)": (sliding_bump, StepFunction(np.array([0.0]), np.array([]))),
    "spike n 1_(0,1/n)": (spike, StepFunction(np.array([0.0]), np.array([]))),
    "spreading (1/n) 1_[0,n)": (spreading, StepFunction(np.array([0.0]), np.array([]))),
    "alternating 1_[0,1), 1_[1,2)": (alternating, StepFunction(np.array([0.0]), np.array([]))),
    "fixed + sliding 1_[0,1) + 1_[n,n+1)": (bump_plus_fixed, indicator_step(0.0, 1.0)),
}
"""name -> (n -> f_n, liminf_n f_n a.e.) for sequences of step functions indexed by n >= 1."""


def tail_infimum(seq: Callable[[int], StepFunction], n: int, K: int) -> StepFunction:
    """inf_{n <= k <= K} f_k, a step function; it decreases to g_n = inf_{k >= n} f_k as K grows."""
    if K < n:
        raise ValueError("need K >= n")
    return combine([seq(k) for k in range(n, K + 1)], "min")


def fatou_sides(seq: Callable[[int], StepFunction], n_max: int, K: int):
    """For n = 1..n_max: the integrals of f_n and of the truncated tail infima inf_{n<=k<=K} f_k.

    The second sequence is non-decreasing in n; for K large its values approximate, from
    above, the integrals of g_n = inf_{k >= n} f_k, whose limit is the integral of liminf f_n
    (monotone convergence). Returns two arrays of length n_max.
    """
    ints = np.array([seq(n).integral() for n in range(1, n_max + 1)])
    infs = np.array([tail_infimum(seq, n, K).integral() for n in range(1, n_max + 1)])
    return ints, infs


def periodic_fatou(h, weights):
    """Fatou's lemma for the periodic sequence f_n = h[n mod p] on a finite space.

    ``h`` has shape (p, N): the p functions of one period. Then liminf_n f_n = min_j h_j
    pointwise and liminf_n of the integrals is min_j of the integrals of h_j. Returns
    ``(integral of liminf, liminf of the integrals)``; Fatou's lemma says the first is <= the
    second.
    """
    h = _nonneg(h)
    if h.ndim != 2:
        raise ValueError("h must have shape (period, N)")
    w = _nonneg(weights, "weights")
    lhs = integral_finite(h.min(axis=0), w)
    rhs = min(integral_finite(row, w) for row in h)
    return lhs, rhs
