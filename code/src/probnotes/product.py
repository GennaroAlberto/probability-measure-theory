"""Product measures, random vectors and independence (Chapter 7).

Four groups of tools, in the order of the chapter:

* **Product measures on finite spaces.** A finite measure on ``{0, ..., m-1}`` is a vector of
  masses; the product measure ``mu_1 (x) mu_2`` is their outer product (:func:`product_measure`),
  an ``m x n`` table. :func:`section_measures` computes ``mu_2(E_x)`` and ``mu_1(E^y)`` for a set
  ``E`` given as a boolean table, and :func:`product_measure_of_set` the two iterated sums.
* **Discrete random vectors.** A joint discrete density is a table ``f[i, j] = P(X = x_i, Y = y_j)``.
  :func:`joint_table` builds it from a finite probability space, :func:`marginals` sums it,
  :func:`is_independent` tests the factorisation ``f = f_X f_Y`` (equivalently, rank one),
  :func:`joint_cdf` and :func:`cdf_independence_defect` check the factorisation of the joint
  distribution function, and :func:`expectation` computes ``E[g(X, Y)]``.
* **Iterated integrals on a grid.** :func:`inner_integrals` and :func:`iterated_integral` apply
  the midpoint rule first in the inner variable, then in the outer one, with separate
  resolutions; :func:`marginal_density` integrates a joint density in one variable. The
  classical function whose two iterated integrals differ is :func:`fubini_counterexample`.
* **Monte Carlo.** :func:`monte_carlo_expectation` estimates ``E[g(X, Y)]`` from samples, with a
  standard error, and :func:`empirical_cdf_defect` measures how far a sample is from having a
  product distribution function. :func:`sample_pair` draws the pairs used in the chapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# ---------------------------------------------------------------------------
# Product measures on finite spaces
# ---------------------------------------------------------------------------


def _masses(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("a measure on a finite space is a one-dimensional array of masses")
    if np.isnan(p).any() or (p < 0).any():
        raise ValueError("masses must be non-negative numbers")
    return p


def product_measure(*masses) -> np.ndarray:
    """Product of finite measures on finite spaces: the outer product of their mass vectors.

    ``product_measure(p, q)[i, j] = p[i] * q[j]``, the mass of the point ``(i, j)``. With ``k``
    arguments the result is a ``k``-dimensional table. Infinite masses follow the convention
    ``0 * inf = 0`` of the notes.
    """
    if not masses:
        raise ValueError("give at least one measure")
    out = _masses(masses[0])
    for p in masses[1:]:
        p = _masses(p)
        with np.errstate(invalid="ignore"):
            out = np.multiply.outer(out, p)
        out = np.where(np.isnan(out), 0.0, out)          # 0 * inf = 0
    return out


def rectangle_measure(table, *index_sets) -> float:
    """Measure of the rectangle ``A_1 x ... x A_k`` under a measure given as a ``k``-dim table.

    Each ``A_i`` is a boolean mask or an array of indices along axis ``i``.
    """
    table = np.asarray(table, dtype=float)
    if len(index_sets) != table.ndim:
        raise ValueError("give one index set per axis")
    idx = [np.flatnonzero(np.asarray(a)) if np.asarray(a).dtype == bool else np.asarray(a, dtype=int)
           for a in index_sets]
    return float(table[np.ix_(*idx)].sum())


def section_measures(p, q, E):
    """Measures of the sections of ``E`` (a boolean ``m x n`` table) under ``p`` and ``q``.

    Returns ``(mu2_Ex, mu1_Ey)``: ``mu2_Ex[i] = q(E_x)`` with ``x = i`` (the section
    ``{y : (x, y) in E}``) and ``mu1_Ey[j] = p(E^y)`` with ``y = j``.
    """
    p, q = _masses(p), _masses(q)
    E = np.asarray(E, dtype=bool)
    if E.shape != (len(p), len(q)):
        raise ValueError("E must have shape (len(p), len(q))")
    return E.astype(float) @ q, p @ E.astype(float)


def product_measure_of_set(p, q, E):
    """The two iterated sums ``sum_x q(E_x) p(x)`` and ``sum_y p(E^y) q(y)`` for a finite ``E``.

    On a finite space they agree (finite sums commute) and equal the product measure of ``E``.
    """
    p, q = _masses(p), _masses(q)
    mu2_Ex, mu1_Ey = section_measures(p, q, E)
    return float(mu2_Ex @ p), float(mu1_Ey @ q)


# ---------------------------------------------------------------------------
# Discrete random vectors: joint and marginal densities, independence
# ---------------------------------------------------------------------------


def joint_table(x, y, weights=None):
    """Joint discrete density of ``(X, Y)`` on a finite probability space.

    ``x[k]``, ``y[k]`` are the values of ``X`` and ``Y`` at the outcome ``k``, which has
    probability ``weights[k]`` (uniform if omitted). Returns ``(xs, ys, f)`` with ``xs``, ``ys``
    the sorted values taken and ``f[i, j] = P(X = xs[i], Y = ys[j])``.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    w = np.full(x.size, 1.0 / x.size) if weights is None else np.asarray(weights, dtype=float).ravel()
    xs, ix = np.unique(x.ravel(), return_inverse=True)
    ys, iy = np.unique(y.ravel(), return_inverse=True)
    f = np.zeros((len(xs), len(ys)))
    np.add.at(f, (ix, iy), w)
    return xs, ys, f


def marginals(table):
    """Marginal densities of a joint table: the sums over all axes but one, one per axis."""
    table = np.asarray(table, dtype=float)
    axes = range(table.ndim)
    return tuple(table.sum(axis=tuple(a for a in axes if a != k)) for k in axes)


def comonotone_table(p, q) -> np.ndarray:
    """The joint table of ``(F_X^{-1}(U), F_Y^{-1}(U))`` for one uniform ``U``: the most dependent
    pair with marginals ``p`` and ``q`` (the "north-west corner" coupling).

    Its marginals are ``p`` and ``q``, like those of ``product_measure(p, q)``: a joint law is
    not determined by its marginal laws.
    """
    p, q = _masses(p), _masses(q)
    cp, cq = np.cumsum(p), np.cumsum(q)
    total = min(cp[-1], cq[-1])
    cuts = np.unique(np.concatenate([[0.0], cp, cq]))
    cuts = cuts[cuts <= total]
    mids, w = (cuts[:-1] + cuts[1:]) / 2, np.diff(cuts)
    i = np.minimum(np.searchsorted(cp, mids), len(p) - 1)
    j = np.minimum(np.searchsorted(cq, mids), len(q) - 1)
    out = np.zeros((len(p), len(q)))
    np.add.at(out, (i, j), w)
    return out


def product_of_marginals(table) -> np.ndarray:
    """The table ``f_X (x) f_Y`` (or the product of all marginals) built from a joint table."""
    return product_measure(*marginals(table))


def independence_defect(table) -> float:
    """``max |f_Z - f_X f_Y|`` over all points: zero exactly for a product (independent) table."""
    table = np.asarray(table, dtype=float)
    return float(np.max(np.abs(table - product_of_marginals(table))))


def is_independent(table, atol=1e-12) -> bool:
    """Whether a joint probability table factorises into its marginals, ``f = f_X f_Y``."""
    return independence_defect(table) <= atol


def table_rank(table, rtol=1e-10) -> int:
    """Numerical rank of a two-dimensional table; a probability table is a product iff it is 1."""
    table = np.asarray(table, dtype=float)
    s = np.linalg.svd(table, compute_uv=False)
    return int(np.sum(s > rtol * s[0])) if s.size and s[0] > 0 else 0


def joint_cdf(table) -> np.ndarray:
    """Joint distribution function on the grid of values: ``F[i, j] = P(X <= xs[i], Y <= ys[j])``.

    The values ``xs``, ``ys`` indexing the table must be sorted in increasing order.
    """
    return np.cumsum(np.cumsum(np.asarray(table, dtype=float), axis=0), axis=1)


def cdf_independence_defect(table) -> float:
    """``max |F_(X,Y)(x, y) - F_X(x) F_Y(y)|`` over the grid of values of a joint table."""
    F = joint_cdf(table)
    return float(np.max(np.abs(F - np.outer(F[:, -1], F[-1, :]))))


def expectation(g, table, xs, ys) -> float:
    """``E[g(X, Y)] = sum_ij g(xs[i], ys[j]) f[i, j]`` for a joint table ``f`` (``g`` vectorised)."""
    X, Y = np.meshgrid(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), indexing="ij")
    return float(np.sum(g(X, Y) * np.asarray(table, dtype=float)))


def covariance(table, xs, ys) -> float:
    """``cov(X, Y) = E[XY] - E[X] E[Y]`` for a joint table."""
    ex = expectation(lambda x, y: x, table, xs, ys)
    ey = expectation(lambda x, y: y, table, xs, ys)
    return expectation(lambda x, y: x * y, table, xs, ys) - ex * ey


def sample_table(table, n, seed=None):
    """Draw ``n`` independent copies of the indices ``(I, J)`` with ``P(I=i, J=j) = f[i, j]``."""
    table = np.asarray(table, dtype=float)
    rng = np.random.default_rng(seed)
    flat = rng.choice(table.size, size=n, p=table.ravel() / table.sum())
    return np.unravel_index(flat, table.shape)


def empirical_table(i, j, shape) -> np.ndarray:
    """Relative frequencies of the index pairs ``(i[k], j[k])`` as a table of the given shape."""
    counts = np.bincount(np.ravel_multi_index((np.asarray(i), np.asarray(j)), shape),
                         minlength=shape[0] * shape[1])
    return counts.reshape(shape) / len(i)


# ---------------------------------------------------------------------------
# Iterated integrals on a grid
# ---------------------------------------------------------------------------


def midpoints(a, b, n) -> np.ndarray:
    """The ``n`` midpoints of the uniform partition of ``[a, b]``."""
    h = (b - a) / n
    return a + h * (np.arange(n) + 0.5)


def inner_integrals(f, outer, inner_range, n_inner, inner="y", chunk=2_000_000):
    """Midpoint-rule integrals of ``f`` in the inner variable, at each outer point.

    With ``inner="y"`` this returns ``int f(x, y) dy`` over ``inner_range`` for every ``x`` in
    ``outer``; with ``inner="x"`` it returns ``int f(x, y) dx`` for every ``y`` in ``outer``.
    ``f`` must be vectorised. Work is split into chunks of at most ``chunk`` evaluations.
    """
    outer = np.asarray(outer, dtype=float)
    c, d = inner_range
    t = midpoints(c, d, n_inner)
    h = (d - c) / n_inner
    rows = max(1, chunk // n_inner)
    out = np.empty(len(outer))
    for s in range(0, len(outer), rows):
        o = outer[s:s + rows, None]
        vals = f(o, t[None, :]) if inner == "y" else f(t[None, :], o)
        out[s:s + rows] = h * np.sum(vals, axis=1)
    return out


def iterated_integral(f, x_range, y_range, n_outer, n_inner, order="dy dx"):
    """Iterated midpoint rule for ``int int f``, with ``n_inner`` points inside, ``n_outer`` outside.

    ``order="dy dx"`` integrates first in ``y`` (for fixed ``x``), then in ``x``: an
    approximation of ``int_X (int_Y f(x, y) dy) dx``; ``order="dx dy"`` the other way round.
    With ``n_inner = n_outer`` both orders give the same double sum; the iterated integral is the
    limit in which the inner resolution goes to infinity first.
    """
    (a, b), (c, d) = x_range, y_range
    if order == "dy dx":
        xs = midpoints(a, b, n_outer)
        return float((b - a) / n_outer * np.sum(inner_integrals(f, xs, (c, d), n_inner, "y")))
    if order == "dx dy":
        ys = midpoints(c, d, n_outer)
        return float((d - c) / n_outer * np.sum(inner_integrals(f, ys, (a, b), n_inner, "x")))
    raise ValueError('order must be "dy dx" or "dx dy"')


def marginal_density(f, x, y_range, n=4000):
    """``f_X(x) = int f(x, y) dy`` over ``y_range`` by the midpoint rule, for each ``x``."""
    return inner_integrals(f, np.atleast_1d(x), y_range, n, "y")


def fubini_counterexample(x, y):
    """``f(x, y) = (x^2 - y^2) / (x^2 + y^2)^2`` on ``(0, 1]^2``: its iterated integrals differ.

    ``int_0^1 f(x, y) dy = 1 / (1 + x^2)`` and ``int_0^1 f(x, y) dx = -1 / (1 + y^2)``, so the
    two iterated integrals are ``pi/4`` and ``-pi/4``; ``f`` is not integrable on the square.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    r2 = x * x + y * y
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x * x - y * y) / (r2 * r2)


def fubini_inner_exact(t, inner="y"):
    """Exact inner integrals of :func:`fubini_counterexample` over ``[0, 1]``."""
    t = np.asarray(t, dtype=float)
    return 1.0 / (1.0 + t * t) if inner == "y" else -1.0 / (1.0 + t * t)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


@dataclass
class MCResult:
    """A Monte Carlo estimate with its standard error and asymptotic 95% interval."""

    estimate: float
    std_error: float
    n: int

    @property
    def ci(self):
        return (self.estimate - 1.96 * self.std_error, self.estimate + 1.96 * self.std_error)


def monte_carlo_expectation(g, sampler, n, seed=None, chunk=1_000_000) -> MCResult:
    """Estimate ``E[g(X, Y)]`` from ``n`` independent draws of ``(X, Y)``.

    ``sampler(rng, k)`` returns two arrays of length ``k``; ``g`` is vectorised. Draws are made
    in chunks of at most ``chunk`` so that memory stays small. The draws, hence the estimate,
    depend on the chunk size, so keep it fixed when reproducing a run.
    """
    rng = np.random.default_rng(seed)
    total = total2 = 0.0
    done = 0
    while done < n:
        k = min(chunk, n - done)
        x, y = sampler(rng, k)
        v = np.asarray(g(x, y), dtype=float)
        total += v.sum()
        total2 += np.dot(v, v)
        done += k
    mean = total / n
    var = max(total2 / n - mean * mean, 0.0) * n / max(n - 1, 1)
    return MCResult(mean, math.sqrt(var / n), n)


def empirical_cdf_defect(x, y, grid_x=None, grid_y=None) -> float:
    """``max |F_n(s, t) - F_n^X(s) F_n^Y(t)|`` over a grid, for a sample ``(x[k], y[k])``.

    ``F_n`` is the empirical joint distribution function. For an independent pair it tends to
    0 like ``1/sqrt(n)``, so a positive limit proves dependence; on a fixed grid a limit 0 does
    not prove independence (the grid may miss where ``F`` and ``F_X F_Y`` differ). By default
    the grid is made of the sample quantiles of levels ``1/40, ..., 39/40`` of each coordinate.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    qs = np.arange(1, 40) / 40
    gx = np.quantile(x, qs) if grid_x is None else np.asarray(grid_x, dtype=float)
    gy = np.quantile(y, qs) if grid_y is None else np.asarray(grid_y, dtype=float)
    ix = np.searchsorted(gx, x, side="left")     # x <= gx[k]  iff  ix <= k
    iy = np.searchsorted(gy, y, side="left")
    K, L = len(gx) + 1, len(gy) + 1
    counts = np.bincount(ix * L + iy, minlength=K * L).reshape(K, L)
    F = np.cumsum(np.cumsum(counts, axis=0), axis=1)[:-1, :-1] / len(x)
    Fx = np.cumsum(np.bincount(ix, minlength=K))[:-1] / len(x)
    Fy = np.cumsum(np.bincount(iy, minlength=L))[:-1] / len(x)
    return float(np.max(np.abs(F - np.outer(Fx, Fy))))


def sample_pair(kind, n, seed=None):
    """Draw ``n`` copies of one of the pairs ``(X, Y)`` used in the chapter.

    * ``"independent"``: ``X, Y`` independent and uniform on ``(-1, 1)``;
    * ``"square"``: ``X`` uniform on ``(-1, 1)`` and ``Y = X^2`` (uncorrelated, dependent);
    * ``"circle"``: ``(cos T, sin T)`` with ``T`` uniform on ``(0, 2 pi)`` (uncorrelated,
      dependent, and the pair has no joint density);
    * ``"disk"``: uniform on the unit disk (uncorrelated, dependent);
    * ``"gamma-mix"``: the joint density ``x e^{-x(1+y)}`` on ``(0, inf)^2``, drawn as
      ``X ~ Exp(1)`` and then ``Y = E / X`` with ``E ~ Exp(1)`` independent of ``X``.
    """
    rng = np.random.default_rng(seed) if not isinstance(seed, np.random.Generator) else seed
    if kind == "independent":
        return rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    if kind == "square":
        x = rng.uniform(-1, 1, n)
        return x, x * x
    if kind == "circle":
        t = rng.uniform(0, 2 * np.pi, n)
        return np.cos(t), np.sin(t)
    if kind == "disk":
        r, t = np.sqrt(rng.random(n)), rng.uniform(0, 2 * np.pi, n)
        return r * np.cos(t), r * np.sin(t)
    if kind == "gamma-mix":
        x = rng.exponential(1.0, n)
        return x, rng.exponential(1.0, n) / x
    raise ValueError(f"unknown pair {kind!r}")


PAIRS = ("independent", "square", "circle", "disk", "gamma-mix")
