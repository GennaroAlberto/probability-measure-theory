"""Lebesgue measure, the Cantor set, measurable and simple functions (Chapter 3).

Four groups of tools, in the order of the chapter:

* **Lebesgue measure of finite unions of intervals.** A set is an array of shape ``(k, 2)``
  whose rows ``[a, b]`` are intervals (``a = b`` is an empty interval and is dropped).
  Open, closed and half-open intervals have the same measure, so only the endpoints are
  stored; endpoints may be ``-inf`` or ``inf``. :func:`normalize_intervals` merges a union
  into disjoint sorted blocks, and :func:`lebesgue_measure` adds up their lengths.
* **The Cantor set** and its "fat" relatives: the stages ``C_k`` as arrays of intervals,
  their measures, and the ternary-to-binary map that shows the Cantor set is uncountable.
* **Simple functions.** The dyadic approximation ``f_n = min(n, floor(2^n f) / 2^n)`` of a
  non-negative function (the sequence of the approximation theorem), the canonical form of
  a function with finitely many values, and the common refinement of two partitions.
* **Finite measurable spaces** ``X = {0, ..., N-1}``: preimages, generated sigma-algebras,
  sigma(f), and the preimage criterion for measurability. Subsets are frozensets of ints.

A Monte Carlo estimator of the Lebesgue measure of a set closes the module.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, combinations
import math

import numpy as np

# ---------------------------------------------------------------------------
# Lebesgue measure of finite unions of intervals
# ---------------------------------------------------------------------------


def as_intervals(intervals) -> np.ndarray:
    """Return the non-empty intervals of ``intervals`` as a float array of shape (k, 2).

    Accepts a pair ``(a, b)``, a list of pairs or an array of shape (k, 2). Rows with
    ``b <= a`` are empty intervals and are dropped.
    """
    arr = np.asarray(intervals, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2))
    arr = arr.reshape(-1, 2)
    if np.isnan(arr).any():
        raise ValueError("interval endpoints must not be NaN")
    return arr[arr[:, 1] > arr[:, 0]]


def normalize_intervals(intervals) -> np.ndarray:
    """Write a finite union of intervals as a sorted union of disjoint, non-adjacent blocks.

    Overlapping or touching intervals (``[0, 1)`` and ``[1, 2)``) are merged. The result is
    the unique representation of the union with the fewest intervals.
    """
    arr = as_intervals(intervals)
    if len(arr) == 0:
        return arr
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    a, b = arr[:, 0], arr[:, 1]
    reach = np.maximum.accumulate(b)              # right end of the union so far
    new_block = np.empty(len(arr), dtype=bool)
    new_block[0] = True
    new_block[1:] = a[1:] > reach[:-1]            # a gap before interval i starts a block
    starts = np.flatnonzero(new_block)
    return np.column_stack([a[starts], np.maximum.reduceat(b, starts)])


def lebesgue_measure(intervals) -> float:
    """Lebesgue measure of a finite union of intervals (possibly overlapping or unbounded)."""
    blocks = normalize_intervals(intervals)
    return float(np.sum(blocks[:, 1] - blocks[:, 0]))


def union(*sets) -> np.ndarray:
    """Union of finite unions of intervals, normalised."""
    parts = [as_intervals(s) for s in sets]
    return normalize_intervals(np.concatenate(parts) if parts else np.empty((0, 2)))


def intersection(A, B) -> np.ndarray:
    """Intersection of two finite unions of intervals, normalised (vectorised over pairs)."""
    A, B = normalize_intervals(A), normalize_intervals(B)
    lo = np.maximum.outer(A[:, 0], B[:, 0]).ravel()
    hi = np.minimum.outer(A[:, 1], B[:, 1]).ravel()
    return normalize_intervals(np.column_stack([lo, hi]))


def complement(A, within=(0.0, 1.0)) -> np.ndarray:
    """Complement of A inside the interval ``within`` (default ``[0, 1)``), normalised."""
    lo, hi = float(within[0]), float(within[1])
    A = intersection(A, [(lo, hi)])
    edges = np.concatenate([[lo], A.ravel(), [hi]])
    return normalize_intervals(edges.reshape(-1, 2))


def contains(A, x) -> np.ndarray:
    """Boolean array: which points of ``x`` lie in the union A of intervals ``[a, b)``.

    Membership of endpoints follows the half-open convention of the notes, ``[a, b)``.
    """
    A = normalize_intervals(A)
    x = np.asarray(x, dtype=float)
    if len(A) == 0:
        return np.zeros(x.shape, dtype=bool)
    idx = np.searchsorted(A[:, 0], x, side="right") - 1
    ok = idx >= 0
    out = np.zeros(x.shape, dtype=bool)
    out[ok] = x[ok] < A[idx[ok], 1]
    return out


# ---------------------------------------------------------------------------
# The Cantor set and fat Cantor sets
# ---------------------------------------------------------------------------

MAX_STAGE = 24  # 2^24 intervals: 256 MB of float64 endpoints; keep stages below this


def cantor_stage(k: int, r: float = 1 / 3) -> np.ndarray:
    """The 2^k closed intervals of the k-th stage of a Cantor-type set, as rows [a, b].

    Stage 0 is [0, 1]. Stage j is obtained from stage j - 1 by removing, from the middle of
    each of its 2^(j-1) intervals, an open interval of length ``r**j``. For ``r = 1/3`` this is
    the middle-thirds Cantor set; ``0 < r < 1/3`` gives a "fat" Cantor set of positive
    measure (``r = 1/4`` is the Smith-Volterra-Cantor set, of measure 1/2). The removals fit
    inside the intervals exactly when ``r <= 1/3``.
    """
    if not 0 < r <= 1 / 3:
        raise ValueError("need 0 < r <= 1/3")
    if not 0 <= k <= MAX_STAGE:
        raise ValueError(f"need 0 <= k <= {MAX_STAGE}")
    I = np.array([[0.0, 1.0]])
    for j in range(1, k + 1):
        a, b = I[:, 0], I[:, 1]
        mid, half_gap = (a + b) / 2, r**j / 2
        I = np.column_stack([a, mid - half_gap, mid + half_gap, b]).reshape(-1, 2)
    return I


def cantor_stage_measure(k: int, r: float = 1 / 3) -> float:
    """Exact measure of the k-th stage: 1 - sum_{j<=k} 2^(j-1) r^j; equal to (2/3)^k for r = 1/3."""
    if r == 1 / 3:
        return (2 / 3) ** k
    return 1.0 - r * (1.0 - (2 * r) ** k) / (1.0 - 2 * r)


def cantor_limit_measure(r: float = 1 / 3) -> float:
    """Measure of the limit set, the intersection of all stages: (1 - 3r) / (1 - 2r)."""
    if not 0 < r <= 1 / 3:
        raise ValueError("need 0 < r <= 1/3")
    return (1 - 3 * r) / (1 - 2 * r)


def in_cantor_stage(x, k: int, r: float = 1 / 3) -> np.ndarray:
    """Boolean array: which points of ``x`` lie in the (closed) k-th stage.

    The endpoints are computed in floating point, so a tolerance ``tol`` (much smaller than
    the gaps r^k) absorbs rounding: 1/3 and 2/3 are in every stage of the Cantor set.
    """
    I = cantor_stage(k, r)
    x = np.asarray(x, dtype=float)
    tol = min(1e-13, r**max(k, 1) / 100)
    idx = np.searchsorted(I[:, 0], x + tol, side="right") - 1
    ok = idx >= 0
    out = np.zeros(x.shape, dtype=bool)
    out[ok] = x[ok] <= I[idx[ok], 1] + tol
    return out


def cantor_point(bits) -> np.ndarray:
    """The point sum_j 2 b_j 3^-j of the Cantor set coded by the 0/1 digits b_1, ..., b_k.

    ``bits`` has shape (..., k); the result has shape (...). Every point of the Cantor set
    has a unique ternary expansion with digits in {0, 2}, so this is a bijection between
    infinite 0/1 sequences and the Cantor set.
    """
    b = np.asarray(bits, dtype=float)
    k = b.shape[-1]
    return b @ (2.0 / 3.0 ** np.arange(1, k + 1))


def binary_value(bits) -> np.ndarray:
    """The number sum_j b_j 2^-j in [0, 1] coded by the binary digits b_1, ..., b_k."""
    b = np.asarray(bits, dtype=float)
    k = b.shape[-1]
    return b @ (0.5 ** np.arange(1, k + 1))


def cantor_function(x, depth: int = 34) -> np.ndarray:
    """The Cantor (Cantor-Lebesgue) function c: [0, 1] -> [0, 1], from ``depth`` ternary digits.

    On the Cantor set, c replaces the ternary digits 0, 2 by the binary digits 0, 1, so c maps
    the Cantor set onto [0, 1]: this is why the Cantor set is uncountable. Off the Cantor set c
    is constant on each removed interval. ``depth = 34`` exhausts double precision.
    """
    x = np.asarray(x, dtype=float)
    if np.any((x < 0) | (x > 1)):
        raise ValueError("the Cantor function is defined on [0, 1]")
    t = np.where(x == 1.0, 0.0, x)
    y = np.where(x == 1.0, 1.0, 0.0)
    alive = x < 1.0
    scale = 0.5
    for _ in range(depth):
        t = 3 * t
        d = np.floor(t)
        t = t - d
        y = y + scale * alive * (d >= 1)       # digit 2 -> binary 1; digit 1 -> 1 and stop
        alive = alive & (d != 1)
        scale /= 2
    return y


# ---------------------------------------------------------------------------
# Simple functions and the dyadic approximation
# ---------------------------------------------------------------------------


def dyadic_index(values, n: int) -> np.ndarray:
    """The index k(x) = min(floor(2^n f(x)), n 2^n) of the level set of f_n containing x.

    With f_n = k / 2^n: k = i - 1 on {(i-1)/2^n <= f < i/2^n}, i = 1, ..., n 2^n, and
    k = n 2^n on {f >= n}. So f_n has at most n 2^n + 1 values. Returned as floats.
    """
    v = np.asarray(values, dtype=float)
    if n < 1 or int(n) != n:
        raise ValueError("n must be an integer >= 1")
    if np.isnan(v).any() or (v < 0).any():
        raise ValueError("the dyadic approximation needs values in [0, +inf]")
    scale = 2.0**n
    return np.minimum(np.floor(v * scale), n * scale)


def dyadic_approximation(values, n: int) -> np.ndarray:
    """The n-th dyadic simple function f_n = min(n, floor(2^n f) / 2^n) of the notes.

    ``values`` are the values f(x) >= 0 of a non-negative function (``inf`` allowed); the
    function is vectorised over them. For each x the sequence f_n(x) increases to f(x), and
    0 <= f(x) - f_n(x) < 2^-n whenever f(x) < n.
    """
    return dyadic_index(values, n) / 2.0**n


def dyadic_levels(n: int) -> np.ndarray:
    """The n 2^n + 1 possible values 0, 1/2^n, ..., n - 1/2^n, n of f_n."""
    return np.arange(n * 2**n + 1) / 2.0**n


def canonical_form(values):
    """Canonical form of a function with finitely many values on a finite grid.

    Returns ``(a, labels)``: the distinct values a_1 < ... < a_m and, for each point, the index
    of its level set A_i = f^{-1}({a_i}); then f = sum_i a_i 1_{A_i}.
    """
    a, labels = np.unique(np.asarray(values), return_inverse=True)
    return a, labels.reshape(np.shape(values))


def common_refinement(labels_f, labels_g) -> np.ndarray:
    """Labels of the common partition {A_i cap B_j} of two partitions given by labels."""
    pairs = np.stack([np.ravel(labels_f), np.ravel(labels_g)], axis=1)
    _, labels = np.unique(pairs, axis=0, return_inverse=True)
    return labels.reshape(np.shape(labels_f))


# ---------------------------------------------------------------------------
# Finite measurable spaces X = {0, ..., N-1}
# ---------------------------------------------------------------------------


def preimage(f, B) -> frozenset:
    """f^{-1}(B) = {x : f(x) in B}, for f given by its values f[0], ..., f[N-1]."""
    f = np.asarray(f)
    return frozenset(np.flatnonzero(np.isin(f, list(B))).tolist())


def atoms(family, n_points: int) -> list:
    """Atoms of sigma(family) on X = {0, ..., n_points - 1}.

    Two points lie in the same atom exactly when every set of the family contains both or
    neither of them; the atoms are the classes of this relation.
    """
    family = [frozenset(S) for S in family]
    member = np.zeros((n_points, len(family)), dtype=bool)
    for j, S in enumerate(family):
        member[list(S), j] = True
    _, cls = np.unique(member, axis=0, return_inverse=True)
    cls = np.ravel(cls)
    return [frozenset(np.flatnonzero(cls == c).tolist()) for c in range(cls.max() + 1)]


def unions_of(atom_list) -> set:
    """All unions of sub-families of ``atom_list`` (2^m sets for m atoms)."""
    m = len(atom_list)
    if m > 20:
        raise ValueError("too many atoms to enumerate (more than 2^20 sets)")
    out = set()
    for mask in range(2**m):
        out.add(frozenset(chain.from_iterable(atom_list[i] for i in range(m) if mask >> i & 1)))
    return out


def sigma_generated(family, n_points: int) -> set:
    """sigma(family) on X = {0, ..., n_points - 1}: all unions of its atoms."""
    return unions_of(atoms(family, n_points))


def is_sigma_algebra(family, n_points: int) -> bool:
    """Check the axioms on a finite X: X in it, and closed under complements and unions.

    On a finite set closure under finite unions gives closure under countable unions, so an
    algebra is a sigma-algebra.
    """
    fam = {frozenset(S) for S in family}
    X = frozenset(range(n_points))
    if X not in fam:
        return False
    if any(X - S not in fam for S in fam):
        return False
    return all(S | T in fam for S in fam for T in fam)


def is_measurable(f, A, E) -> bool:
    """Is f measurable from (X, A) to (Y, E), i.e. f^{-1}(B) in A for every B in E?

    By the generator criterion of the notes, E may be replaced by any family that
    generates the target sigma-algebra.
    """
    A = {frozenset(S) for S in A}
    return all(preimage(f, B) in A for B in E)


def power_set(Y) -> list:
    """All subsets of a finite set Y, as frozensets."""
    Y = list(Y)
    return [frozenset(c) for c in chain.from_iterable(combinations(Y, r) for r in range(len(Y) + 1))]


def sigma_of_function(f) -> set:
    """sigma(f) for f: X -> (Y, P(Y)) on a finite X, computed as {f^{-1}(B) : B subset of f(X)}.

    Subsets B of Y outside the image f(X) add nothing, since f^{-1}(B) = f^{-1}(B cap f(X)).
    """
    image = np.unique(np.asarray(f)).tolist()
    return {preimage(f, B) for B in power_set(image)}


def indicator(A, n_points: int) -> np.ndarray:
    """The indicator 1_A on X = {0, ..., n_points - 1}, as an integer array."""
    out = np.zeros(n_points, dtype=int)
    out[list(A)] = 1
    return out


# ---------------------------------------------------------------------------
# Monte Carlo estimate of a Lebesgue measure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCEstimate:
    """A Monte Carlo estimate of m(A cap box) with two error bars.

    ``ci`` is the asymptotic (central limit) confidence interval. ``hoeffding_halfwidth``
    gives a non-asymptotic one: by Hoeffding's inequality, the estimate is within this
    distance of the true value with probability at least ``confidence``, for every sample
    size.
    """
    estimate: float
    std_error: float
    ci: tuple
    hoeffding_halfwidth: float
    n_samples: int
    confidence: float


def monte_carlo_measure(indicator_fn, low, high, n_samples: int, seed=None,
                        confidence: float = 0.95, batch: int = 1_000_000) -> MCEstimate:
    """Estimate the Lebesgue measure of A cap [low, high] by uniform sampling of the box.

    ``indicator_fn`` maps an array of points to booleans (1_A): shape (n,) in dimension one,
    shape (n, d) in dimension d. The estimator is vol(box) times the fraction of hits; it is
    unbiased, with variance vol^2 p (1 - p) / n where p = m(A cap box) / vol(box). Samples are
    drawn in batches, so memory stays bounded for any ``n_samples``.
    """
    from scipy.stats import norm

    low = np.atleast_1d(np.asarray(low, dtype=float))
    high = np.atleast_1d(np.asarray(high, dtype=float))
    d = low.size
    vol = float(np.prod(high - low))
    rng = np.random.default_rng(seed)
    hits, left = 0, int(n_samples)
    while left > 0:
        m = min(left, batch)
        u = rng.uniform(low, high, size=(m, d))
        hits += int(np.count_nonzero(indicator_fn(u[:, 0] if d == 1 else u)))
        left -= m
    p = hits / n_samples
    se = vol * math.sqrt(p * (1 - p) / n_samples)
    z = float(norm.ppf(0.5 + confidence / 2))
    est = vol * p
    hoeff = vol * math.sqrt(math.log(2 / (1 - confidence)) / (2 * n_samples))
    return MCEstimate(est, se, (est - z * se, est + z * se), hoeff, int(n_samples), confidence)
