"""Transformations of random vectors (Chapter 9).

Six groups of tools, in the order of the chapter:

* **Discrete transformations and discrete convolution.** :func:`law_of_function` computes the law
  of ``h(X, Y)`` from a joint discrete density (a table), as in the discrete case of the notes;
  :func:`convolve_pmfs` is the discrete convolution ``f_X * f_Y`` of two densities on the integers,
  and :func:`convolve_power` the law of a sum of ``n`` independent copies. The notable laws of the
  chapter are :func:`poisson_pmf`, :func:`binomial_pmf`, :func:`geometric_pmf` (on ``{1, 2, ...}``)
  and :func:`pascal_pmf`.
* **The Jacobian formula.** :func:`transformed_density` returns the density
  ``f_W(w) = f_Z(g^{-1}(w)) |det J_{g^{-1}}(w)| 1_{g(A)}(w)`` of ``W = g(Z)`` for a
  ``C^1``-diffeomorphism ``g``, given ``g^{-1}`` (the Jacobian determinant is computed by central
  differences unless it is supplied); :func:`linear_transformed_density` is the case
  ``W = A Z + b`` and :func:`piecewise_transformed_density` the version with a partition into
  pieces.
* **Densities of sums.** :func:`sum_density` computes ``f_{X+Y}(s) = int f_Z(x, s - x) dx`` from a
  joint density by the midpoint rule, and :func:`convolve_densities_grid` the convolution
  ``f_X * f_Y`` of two densities sampled on uniform grids.
* **Minimum and maximum.** :func:`min_max_cdf` gives the distribution functions of
  ``min(X, Y)`` and ``max(X, Y)`` from the joint (or, for independent variables, the marginal)
  distribution functions.
* **Gamma and Beta.** :func:`gamma_pdf`, :func:`beta_pdf`, :func:`beta_function`,
  :func:`beta_mean_var` and :func:`sample_gamma_beta`, which draws ``S = X + Y`` and
  ``T = X / (X + Y)`` for independent Gamma variables. :func:`joint_cdf_gap` measures how far a
  sample of pairs is from having a product distribution function: correlation is not enough.
* **The worked exercises in closed form**, and samplers for them (:func:`sample_worked`).

Every function is vectorised in its numerical arguments.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special

# ---------------------------------------------------------------------------
# Discrete laws and discrete convolution
# ---------------------------------------------------------------------------


def poisson_pmf(k, lam):
    """``P(X = k) = e^{-lam} lam^k / k!`` for ``X ~ P(lam)``, ``k`` in ``N`` (0 elsewhere)."""
    k = np.asarray(k, dtype=float)
    ok = (k >= 0) & (k == np.floor(k))
    kk = np.where(ok, k, 0.0)
    with np.errstate(divide="ignore"):
        out = np.exp(-lam + kk * np.log(lam) - special.gammaln(kk + 1)) if lam > 0 else (kk == 0) * 1.0
    return np.where(ok, out, 0.0)


def binomial_pmf(k, n, p):
    """``P(X = k) = C(n, k) p^k (1-p)^(n-k)`` for ``X ~ B(n, p)``."""
    k = np.asarray(k, dtype=float)
    ok = (k >= 0) & (k <= n) & (k == np.floor(k))
    kk = np.where(ok, k, 0.0)
    out = special.comb(n, kk) * p ** kk * (1 - p) ** (n - kk)
    return np.where(ok, out, 0.0)


def geometric_pmf(k, p):
    """``P(X = k) = p (1-p)^(k-1)``, ``k = 1, 2, ...``: the law ``G(p)`` of the first success."""
    return pascal_pmf(k, 1, p)


def pascal_pmf(k, n, p):
    """``P(X = k) = C(k-1, n-1) p^n (1-p)^(k-n)``, ``k = n, n+1, ...``: the law ``Pascal(n, p)``.

    ``X`` is the trial of the ``n``-th success in independent Bernoulli trials with success
    probability ``p``; ``Pascal(1, p) = G(p)``.
    """
    k = np.asarray(k, dtype=float)
    ok = (k >= n) & (k == np.floor(k))
    kk = np.where(ok, k, float(n))
    out = special.comb(kk - 1, n - 1) * p ** n * (1 - p) ** (kk - n)
    return np.where(ok, out, 0.0)


def convolve_pmfs(p, q, p0=0, q0=0):
    """Discrete convolution of two densities on the integers.

    ``p[i] = P(X = p0 + i)`` and ``q[j] = P(Y = q0 + j)``. Returns ``(r, r0)`` with
    ``r[k] = sum_i p[i] q[k - i] = P(X + Y = r0 + k)`` for independent ``X`` and ``Y``, and
    ``r0 = p0 + q0``. If ``p`` and ``q`` are truncations of laws with infinite support,
    ``r[k]`` is exact for ``k < min(len(p), len(q))``.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.ndim != 1 or q.ndim != 1:
        raise ValueError("densities must be one-dimensional arrays")
    return np.convolve(p, q), int(p0) + int(q0)


def convolve_power(p, n, p0=0):
    """Law of ``X_1 + ... + X_n`` for independent copies of ``X``, with ``p[i] = P(X = p0 + i)``.

    Returns ``(r, n * p0)``. Uses repeated squaring, so ``n`` may be large.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    result, r0 = None, 0
    base, b0 = np.asarray(p, dtype=float), int(p0)
    while n:
        if n & 1:
            if result is None:
                result, r0 = base.copy(), b0
            else:
                result, r0 = convolve_pmfs(result, base, r0, b0)
        n >>= 1
        if n:
            base, b0 = convolve_pmfs(base, base, b0, b0)
    return result, r0


def law_of_function(table, xs, ys, h, decimals=10):
    """Law of ``T = h(X, Y)`` for a discrete vector with joint density ``table[i, j]``.

    ``table[i, j] = P(X = xs[i], Y = ys[j])`` and ``h`` is vectorised. Returns ``(ts, ft)``, the
    sorted values ``t`` of ``h`` on the support and ``f_T(t) = sum_{z in h^{-1}(t)} f_Z(z)``.
    Values that agree to ``decimals`` decimal places are identified.
    """
    table = np.asarray(table, dtype=float)
    X, Y = np.meshgrid(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), indexing="ij")
    vals = np.round(np.asarray(h(X, Y), dtype=float), decimals).ravel()
    w = table.ravel()
    keep = w > 0
    ts, inv = np.unique(vals[keep], return_inverse=True)
    ft = np.zeros(len(ts))
    np.add.at(ft, inv, w[keep])
    return ts, ft


# ---------------------------------------------------------------------------
# The Jacobian formula
# ---------------------------------------------------------------------------


def numerical_jacobian_det(ginv, u, v, h=1e-6):
    """``det J_{g^{-1}}(u, v)`` by central differences; ``ginv(u, v)`` returns ``(x, y)``."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    hu = h * np.maximum(1.0, np.abs(u))
    hv = h * np.maximum(1.0, np.abs(v))
    xu1, yu1 = ginv(u + hu, v)
    xu0, yu0 = ginv(u - hu, v)
    xv1, yv1 = ginv(u, v + hv)
    xv0, yv0 = ginv(u, v - hv)
    dxdu, dydu = (xu1 - xu0) / (2 * hu), (yu1 - yu0) / (2 * hu)
    dxdv, dydv = (xv1 - xv0) / (2 * hv), (yv1 - yv0) / (2 * hv)
    return dxdu * dydv - dxdv * dydu


def transformed_density(f_Z, ginv, jac_det=None, image=None):
    """Density of ``W = g(Z)`` for a ``C^1``-diffeomorphism ``g`` of an open set ``A`` onto ``g(A)``.

    ``f_Z(x, y)`` is the density of ``Z`` (with ``P(Z in A) = 1``), ``ginv(u, v)`` returns
    ``g^{-1}(u, v)``, ``jac_det(u, v)`` is ``det J_{g^{-1}}(u, v)`` (computed numerically if
    omitted) and ``image(u, v)`` is the indicator of ``g(A)`` (everywhere 1 if omitted). Returns the
    function ``f_W(u, v) = f_Z(g^{-1}(u, v)) |det J_{g^{-1}}(u, v)| 1_{g(A)}(u, v)``.
    """
    def f_W(u, v):
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        u, v = np.broadcast_arrays(u, v)
        inside = np.ones(u.shape, dtype=bool) if image is None else np.asarray(image(u, v), dtype=bool)
        out = np.zeros(u.shape)
        if not inside.any():
            return out
        uu, vv = u[inside], v[inside]
        x, y = ginv(uu, vv)
        J = jac_det(uu, vv) if jac_det is not None else numerical_jacobian_det(ginv, uu, vv)
        out[inside] = f_Z(x, y) * np.abs(J)
        return out
    return f_W


def linear_transformed_density(f_Z, A, b=(0.0, 0.0)):
    """Density of ``W = A Z + b`` for an invertible ``2 x 2`` matrix ``A``:
    ``f_W(w) = f_Z(A^{-1}(w - b)) / |det A|``."""
    A = np.asarray(A, dtype=float)
    det = float(np.linalg.det(A))
    if abs(det) < 1e-14:
        raise ValueError("A must be invertible")
    Ainv = np.linalg.inv(A)
    b = np.asarray(b, dtype=float)

    def f_W(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        du, dv = u - b[0], v - b[1]
        x = Ainv[0, 0] * du + Ainv[0, 1] * dv
        y = Ainv[1, 0] * du + Ainv[1, 1] * dv
        return f_Z(x, y) / abs(det)
    return f_W


def piecewise_transformed_density(f_Z, pieces):
    """Density of ``W = g(Z)`` when ``g`` is a ``C^1``-diffeomorphism on each piece ``A_i`` of an
    almost sure partition: ``f_W(w) = sum_i f_Z(g_i^{-1}(w)) |det J_{g_i^{-1}}(w)| 1_{g(A_i)}(w)``.

    ``pieces`` is a list of tuples ``(ginv_i, jac_det_i, image_i)`` as in
    :func:`transformed_density` (``jac_det_i`` may be ``None``).
    """
    parts = [transformed_density(f_Z, gi, ji, im) for gi, ji, im in pieces]

    def f_W(u, v):
        return sum(p(u, v) for p in parts)
    return f_W


# ---------------------------------------------------------------------------
# Densities of sums
# ---------------------------------------------------------------------------


def midpoints(a, b, n):
    """The ``n`` midpoints of the uniform partition of ``[a, b]``."""
    return a + (b - a) / n * (np.arange(n) + 0.5)


def sum_density(f_Z, s, x_range, n=4000):
    """``f_{X+Y}(s) = int f_Z(x, s - x) dx`` over ``x_range`` by the midpoint rule, for each ``s``."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    a, b = x_range
    x = midpoints(a, b, n)
    out = np.empty(len(s))
    rows = max(1, 2_000_000 // n)
    for k in range(0, len(s), rows):
        ss = s[k:k + rows, None]
        out[k:k + rows] = (b - a) / n * np.sum(f_Z(x[None, :], ss - x[None, :]), axis=1)
    return out


def convolution(f_X, f_Y, s, x_range, n=4000):
    """``(f_X * f_Y)(s) = int f_X(x) f_Y(s - x) dx`` over ``x_range`` by the midpoint rule."""
    return sum_density(lambda x, y: f_X(x) * f_Y(y), s, x_range, n)


def convolve_densities_grid(fx, fy, h, x0=0.0, y0=0.0):
    """Convolution of two densities sampled at midpoints of uniform grids with the same step ``h``.

    ``fx[k] = f_X(x0 + (k + 1/2) h)`` and ``fy[l] = f_Y(y0 + (l + 1/2) h)``. Returns
    ``(s, fs)`` with ``s[j] = x0 + y0 + (j + 1) h`` and
    ``fs[j] = h sum_{k + l = j} fx[k] fy[l]``, a midpoint-rule approximation of
    ``(f_X * f_Y)(s[j])``.
    """
    fx, fy = np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)
    fs = h * np.convolve(fx, fy)
    s = x0 + y0 + h * (np.arange(len(fs)) + 1)
    return s, fs


def uniform_sum_pdf(s, n):
    """Density of ``U_1 + ... + U_n`` for independent ``U(0, 1)`` variables (Irwin--Hall):
    ``(1/(n-1)!) sum_{k <= s} (-1)^k C(n, k) (s - k)^(n-1)`` on ``(0, n)``."""
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    for k in range(n + 1):
        out = out + (-1) ** k * math.comb(n, k) * np.where(s > k, np.abs(s - k) ** (n - 1), 0.0)
    out = out / math.factorial(n - 1)
    return np.where((s > 0) & (s < n), out, 0.0)


def uniform_sum_cdf(s, n):
    """Distribution function of ``U_1 + ... + U_n`` (Irwin--Hall); ``F(1) = 1/n!``."""
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    for k in range(n + 1):
        out = out + (-1) ** k * math.comb(n, k) * np.where(s > k, (s - k), 0.0) ** n
    out = out / math.factorial(n)
    return np.clip(np.where(s >= n, 1.0, np.where(s <= 0, 0.0, out)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Minimum and maximum
# ---------------------------------------------------------------------------


def min_max_cdf(F_X, F_Y, t, F_joint=None):
    """Distribution functions of ``min(X, Y)`` and ``max(X, Y)`` at ``t``.

    ``F_max(t) = F_(X,Y)(t, t)`` and ``F_min(t) = F_X(t) + F_Y(t) - F_(X,Y)(t, t)``. If
    ``F_joint`` is omitted, ``X`` and ``Y`` are taken independent: ``F_(X,Y)(t, t) = F_X(t) F_Y(t)``.
    """
    t = np.asarray(t, dtype=float)
    Fx, Fy = F_X(t), F_Y(t)
    Fxy = Fx * Fy if F_joint is None else F_joint(t, t)
    return Fx + Fy - Fxy, Fxy


# ---------------------------------------------------------------------------
# Gamma and Beta
# ---------------------------------------------------------------------------


def gamma_pdf(x, alpha, lam):
    """Density of ``Gamma(alpha, lam)`` (``lam`` a rate): ``lam^alpha x^(alpha-1) e^{-lam x} / Gamma(alpha)``."""
    x = np.asarray(x, dtype=float)
    pos = x > 0
    xx = np.where(pos, x, 1.0)
    out = np.exp(alpha * math.log(lam) + (alpha - 1) * np.log(xx) - lam * xx - special.gammaln(alpha))
    return np.where(pos, out, 0.0)


def beta_function(a, b):
    """``B(a, b) = int_0^1 t^(a-1) (1-t)^(b-1) dt = Gamma(a) Gamma(b) / Gamma(a + b)``."""
    return np.exp(special.gammaln(a) + special.gammaln(b) - special.gammaln(np.add(a, b)))


def beta_pdf(t, a, b):
    """Density of ``Beta(a, b)``: ``t^(a-1) (1-t)^(b-1) / B(a, b)`` on ``(0, 1)``."""
    t = np.asarray(t, dtype=float)
    inside = (t > 0) & (t < 1)
    tt = np.where(inside, t, 0.5)
    out = np.exp((a - 1) * np.log(tt) + (b - 1) * np.log1p(-tt)
                 - (special.gammaln(a) + special.gammaln(b) - special.gammaln(a + b)))
    return np.where(inside, out, 0.0)


def beta_mean_var(a, b):
    """Mean ``a/(a+b)`` and variance ``a b / ((a+b)^2 (a+b+1))`` of ``Beta(a, b)``."""
    return a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1))


def gamma_beta_density(s, t, alpha, beta, lam):
    """Joint density of ``(S, T) = (X + Y, X / (X + Y))`` for independent ``X ~ Gamma(alpha, lam)``,
    ``Y ~ Gamma(beta, lam)``: the product ``Gamma(alpha + beta, lam)(s) * Beta(alpha, beta)(t)``."""
    return gamma_pdf(s, alpha + beta, lam) * beta_pdf(t, alpha, beta)


def sample_gamma_beta(alpha, beta, lam, n, seed=None, lam2=None):
    """Draw ``n`` copies of ``(S, T) = (X + Y, X / (X + Y))``.

    ``X ~ Gamma(alpha, lam)`` and ``Y ~ Gamma(beta, lam2)`` are independent (``lam2 = lam`` by
    default). For ``lam2 = lam``, ``S`` and ``T`` are independent; otherwise they are not.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    lam2 = lam if lam2 is None else lam2
    x = rng.gamma(alpha, 1.0 / lam, n)
    y = rng.gamma(beta, 1.0 / lam2, n)
    s = x + y
    return s, x / s


# ---------------------------------------------------------------------------
# Empirical distribution functions
# ---------------------------------------------------------------------------


def empirical_cdf(sample, x):
    """``F_n(x) = #{k : sample[k] <= x} / n`` at each point of ``x``."""
    srt = np.sort(np.asarray(sample, dtype=float))
    return np.searchsorted(srt, np.asarray(x, dtype=float), side="right") / len(srt)


def ks_distance(sample, cdf):
    """``sup_x |F_n(x) - F(x)|`` for a continuous distribution function ``F`` (vectorised)."""
    srt = np.sort(np.asarray(sample, dtype=float))
    n = len(srt)
    F = cdf(srt)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))


def dkw_sample_size(eps, delta):
    """Smallest ``n`` with ``2 exp(-2 n eps^2) <= delta``: by the Dvoretzky--Kiefer--Wolfowitz
    inequality (Massart's constant), ``P(sup |F_n - F| > eps) <= delta`` for such ``n``."""
    return int(math.ceil(math.log(2.0 / delta) / (2.0 * eps * eps)))


def joint_cdf_gap(x, y, grid_x=None, grid_y=None, levels=39):
    """``max |F_n(s, t) - F_n^X(s) F_n^Y(t)|`` over a grid, for a sample of pairs ``(x[k], y[k])``.

    ``F_n`` is the empirical joint distribution function and ``F_n^X``, ``F_n^Y`` the empirical
    marginal ones. For independent coordinates it tends to ``0`` like ``n^{-1/2}``; a positive limit
    proves dependence. By default the grid is made of the sample quantiles of levels
    ``1/(levels+1), ..., levels/(levels+1)`` of each coordinate.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    qs = np.arange(1, levels + 1) / (levels + 1)
    gx = np.quantile(x, qs) if grid_x is None else np.asarray(grid_x, dtype=float)
    gy = np.quantile(y, qs) if grid_y is None else np.asarray(grid_y, dtype=float)
    ix = np.searchsorted(gx, x, side="left")      # x <= gx[k]  iff  ix <= k
    iy = np.searchsorted(gy, y, side="left")
    K, L = len(gx) + 1, len(gy) + 1
    counts = np.bincount(ix * L + iy, minlength=K * L).reshape(K, L)
    F = np.cumsum(np.cumsum(counts, axis=0), axis=1)[:-1, :-1] / len(x)
    Fx = np.cumsum(np.bincount(ix, minlength=K))[:-1] / len(x)
    Fy = np.cumsum(np.bincount(iy, minlength=L))[:-1] / len(x)
    return float(np.max(np.abs(F - np.outer(Fx, Fy))))


def sample_correlation(x, y):
    """Sample correlation coefficient of two arrays."""
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


# ---------------------------------------------------------------------------
# The worked exercises in closed form
# ---------------------------------------------------------------------------


def integer_part_pmf(k, lam):
    """``P([X] = k) = (1 - e^{-lam}) e^{-lam k}``, ``k in N``, for ``X ~ Exp(lam)``; ``[X] + 1 ~ G(1 - e^{-lam})``."""
    return geometric_pmf(np.asarray(k, dtype=float) + 1, 1.0 - math.exp(-lam))


def half_integer_part_pmf(k, lam):
    """``P(Z = k/2) = (1 - e^{-lam/2}) e^{-lam k / 2}``, ``k in N``, for ``Z = [2X]/2``, ``X ~ Exp(lam)``."""
    return integer_part_pmf(k, lam / 2.0)


def fractional_part_pdf(t, lam):
    """Density ``lam e^{-lam t} / (1 - e^{-lam})`` on ``[0, 1)`` of ``X - [X]``, ``X ~ Exp(lam)``."""
    t = np.asarray(t, dtype=float)
    return np.where((t >= 0) & (t < 1), lam * np.exp(-lam * t) / (-math.expm1(-lam)), 0.0)


def fractional_part_cdf(t, lam):
    """Distribution function ``(1 - e^{-lam t}) / (1 - e^{-lam})`` on ``[0, 1)`` of ``X - [X]``."""
    t = np.asarray(t, dtype=float)
    mid = -np.expm1(-lam * np.clip(t, 0, 1)) / (-math.expm1(-lam))
    return np.where(t < 0, 0.0, np.where(t >= 1, 1.0, mid))


def geometric_pair_pmf(i, j, a):
    """``f_(X,Y)(i, j) = (1 - a)^2 a^(i + j)``, ``i, j in N``: independent, each ``(1-a) a^i``."""
    i, j = np.asarray(i, dtype=float), np.asarray(j, dtype=float)
    ok = (i >= 0) & (j >= 0)
    return np.where(ok, (1 - a) ** 2 * a ** (np.where(ok, i + j, 0)), 0.0)


def min_diff_pmf(k, r, a):
    """``f_(U,V)(k, r) = (1 - a)^2 a^(2k + |r|)`` for ``U = min(X, Y)``, ``V = X - Y``."""
    k, r = np.asarray(k, dtype=float), np.asarray(r, dtype=float)
    ok = k >= 0
    return np.where(ok, (1 - a) ** 2 * a ** (2 * np.where(ok, k, 0) + np.abs(r)), 0.0)


def poisson_bernoulli_pmf(k, lam, p):
    """Law of ``U = X Y`` with ``X ~ P(lam)``, ``Y ~ b(p)`` independent:
    ``P(U = 0) = 1 - p + p e^{-lam}``, ``P(U = k) = p e^{-lam} lam^k / k!`` for ``k >= 1``."""
    k = np.asarray(k, dtype=float)
    return np.where(k == 0, 1 - p + p * math.exp(-lam), p * poisson_pmf(k, lam))


def poisson_bernoulli_moments(lam, p):
    """``(E[U], Var(U), cov(X, U), rho(X, U))`` for ``U = X Y`` as in :func:`poisson_bernoulli_pmf`."""
    mean = lam * p
    var = lam * p + lam ** 2 * p - lam ** 2 * p ** 2
    cov = lam * p
    rho = cov / math.sqrt(lam * var)
    return mean, var, cov, rho


def laplace_pdf(x):
    """``f(x) = e^{-|x|} / 2`` on ``R``."""
    return 0.5 * np.exp(-np.abs(np.asarray(x, dtype=float)))


def sqrt_abs_laplace_pdf(u):
    """Density ``2 u e^{-u^2}`` on ``(0, inf)`` of ``U = sqrt(|X|)``, ``X`` with density ``e^{-|x|}/2``."""
    u = np.asarray(u, dtype=float)
    return np.where(u > 0, 2 * u * np.exp(-u * u), 0.0)


def laplace_uniform_sum_pdf(v):
    """Density of ``V = U + Y``, ``U = sqrt(|X|)``, ``Y ~ U(0, 1)`` independent:
    ``1 - e^{-v^2}`` on ``(0, 1]`` and ``e^{-(v-1)^2} - e^{-v^2}`` on ``(1, inf)``."""
    v = np.asarray(v, dtype=float)
    return np.where(v <= 0, 0.0, np.where(v <= 1, -np.expm1(-v * v), np.exp(-(v - 1) ** 2) - np.exp(-v * v)))


def sum_difference_pdf(s, t):
    """Density ``e^{-|s+t|/2} / 4`` on ``D = {t < s < t + 2}`` of ``(X + Y, X - Y)`` for the Laplace
    ``X`` and the uniform ``Y`` of the worked exercise."""
    s, t = np.asarray(s, dtype=float), np.asarray(t, dtype=float)
    return np.where((t < s) & (s < t + 2), 0.25 * np.exp(-np.abs(s + t) / 2), 0.0)


def uniform_product_pdf(u, n):
    """Density ``(-log u)^(n-1) / (n-1)!`` on ``(0, 1)`` of a product of ``n`` independent ``U(0, 1)``."""
    u = np.asarray(u, dtype=float)
    inside = (u > 0) & (u < 1)
    uu = np.where(inside, u, 0.5)
    return np.where(inside, (-np.log(uu)) ** (n - 1) / math.factorial(n - 1), 0.0)


def uniform_product_cdf(u, n):
    """Distribution function ``u sum_{k<n} (-log u)^k / k!`` on ``(0, 1]`` of a product of ``n``
    independent ``U(0, 1)``: the Erlang tail ``P(T_n >= -log u)``, ``T_n ~ Gamma(n, 1)``."""
    u = np.asarray(u, dtype=float)
    inside = (u > 0) & (u < 1)
    L = -np.log(np.where(inside, u, 0.5))
    tot = sum(L ** k / math.factorial(k) for k in range(n))
    return np.where(u <= 0, 0.0, np.where(u >= 1, 1.0, np.where(inside, u, 0.5) * tot))


def sample_worked(name, n, seed=None, **params):
    """Samples for the worked exercises of the chapter.

    * ``"integer-part"`` (``lam``): returns ``([X], X - [X])`` for ``X ~ Exp(lam)``;
    * ``"geometric-pair"`` (``a``): returns ``(X, Y)`` with ``P(X = i) = (1-a) a^i`` independent;
    * ``"poisson-bernoulli"`` (``lam``, ``p``): returns ``(X, U)`` with ``U = X Y``;
    * ``"laplace-uniform"``: returns ``(X, Y)``, ``X`` with density ``e^{-|x|}/2``, ``Y ~ U(0, 1)``;
    * ``"uniform-product"`` (``m``): returns the product of ``m`` independent ``U(0, 1)``.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    if name == "integer-part":
        x = rng.exponential(1.0 / params.get("lam", 1.0), n)
        return np.floor(x), x - np.floor(x)
    if name == "geometric-pair":
        a = params.get("a", 0.5)
        return rng.geometric(1 - a, n) - 1, rng.geometric(1 - a, n) - 1
    if name == "poisson-bernoulli":
        lam, p = params.get("lam", 2.0), params.get("p", 0.3)
        x = rng.poisson(lam, n)
        return x, x * (rng.random(n) < p)
    if name == "laplace-uniform":
        return rng.laplace(0.0, 1.0, n), rng.random(n)
    if name == "uniform-product":
        m = params.get("m", 2)
        out = np.ones(n)
        for _ in range(m):
            out *= rng.random(n)
        return out
    raise ValueError(f"unknown worked exercise {name!r}")
