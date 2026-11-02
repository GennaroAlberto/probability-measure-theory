"""Checks of the Chapter 7 code against the notes and the exercises."""
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import integrate, special

from probnotes.product import (
    MCResult,
    comonotone_table,
    covariance,
    cdf_independence_defect,
    empirical_cdf_defect,
    empirical_table,
    expectation,
    fubini_counterexample,
    fubini_inner_exact,
    independence_defect,
    inner_integrals,
    is_independent,
    iterated_integral,
    joint_cdf,
    joint_table,
    marginal_density,
    marginals,
    midpoints,
    monte_carlo_expectation,
    product_measure,
    product_measure_of_set,
    product_of_marginals,
    rectangle_measure,
    sample_pair,
    sample_table,
    section_measures,
    table_rank,
)

# ------------------------------------------------------- product measures on finite spaces


def test_product_measure_is_the_outer_product():
    p, q = [0.2, 0.3, 0.5], [0.1, 0.9]
    P = product_measure(p, q)
    assert P.shape == (3, 2)
    assert np.allclose(P, np.outer(p, q))
    assert P.sum() == pytest.approx(1.0)
    assert product_measure(p, q, [0.5, 0.5]).shape == (3, 2, 2)
    # convention 0 * inf = 0 for infinite masses
    assert np.array_equal(product_measure([0.0, 1.0], [np.inf, 2.0]), [[0.0, 0.0], [np.inf, 2.0]])
    with pytest.raises(ValueError):
        product_measure([-1.0, 1.0], [1.0])


def test_rectangles_have_product_measure():
    # mu1 (x) mu2 (A x B) = mu1(A) mu2(B), the defining property of the product measure
    rng = np.random.default_rng(0)
    p, q = rng.random(5), 3 * rng.random(7)              # finite measures, not probabilities
    P = product_measure(p, q)
    for _ in range(50):
        A, B = rng.random(5) < 0.5, rng.random(7) < 0.5
        assert rectangle_measure(P, A, B) == pytest.approx(p[A].sum() * q[B].sum())


def test_verify_on_rectangles_as_in_the_notes():
    # E = A x B: mu2(E_x) = mu2(B) 1_A(x), mu1(E^y) = mu1(A) 1_B(y), both integrals mu1(A) mu2(B)
    p, q = np.array([1.0, 2.0, 0.5, 4.0]), np.array([0.3, 0.7, 2.0])
    A, B = np.array([True, False, True, True]), np.array([False, True, True])
    E = np.outer(A, B)
    mu2_Ex, mu1_Ey = section_measures(p, q, E)
    assert np.allclose(mu2_Ex, q[B].sum() * A)
    assert np.allclose(mu1_Ey, p[A].sum() * B)
    s1, s2 = product_measure_of_set(p, q, E)
    assert s1 == pytest.approx(p[A].sum() * q[B].sum()) and s2 == pytest.approx(s1)


def test_iterated_sums_agree_on_any_set():
    rng = np.random.default_rng(1)
    p, q = rng.random(6), rng.random(9)
    for _ in range(30):
        E = rng.random((6, 9)) < 0.4
        s1, s2 = product_measure_of_set(p, q, E)
        assert s1 == pytest.approx(s2)
        assert s1 == pytest.approx(product_measure(p, q)[E].sum())


# ------------------------------------------------------------ discrete random vectors

DICE = np.meshgrid(np.arange(1, 7), np.arange(1, 7), indexing="ij")


def test_max_min_of_two_dice():
    d1, d2 = DICE
    xs, ys, f = joint_table(np.maximum(d1, d2), np.minimum(d1, d2))
    expected = np.tril(np.full((6, 6), 2)) - np.eye(6)          # 2 below the diagonal, 1 on it
    assert np.allclose(36 * f, expected)
    fx, fy = marginals(f)
    k = np.arange(1, 7)
    assert np.allclose(36 * fx, 2 * k - 1) and np.allclose(36 * fy, 13 - 2 * k)
    assert not is_independent(f)
    assert table_rank(f) == 6
    assert independence_defect(f) == pytest.approx(49 / 1296)        # at (max, min) = (6, 1)
    assert cdf_independence_defect(f) == pytest.approx(112 / 1296)   # = 7/81, at (4, 2)
    assert covariance(f, xs, ys) == pytest.approx(1225 / 1296)       # = (35/36)^2
    assert Fraction(1225, 1296) == Fraction(35, 36) ** 2


def test_sum_and_difference_are_uncorrelated_but_dependent():
    d1, d2 = DICE
    xs, ys, g = joint_table(d1 + d2, d1 - d2)
    assert covariance(g, xs, ys) == pytest.approx(0.0, abs=1e-14)
    assert independence_defect(g) == pytest.approx(1 / 36)
    assert table_rank(g) == 6


def test_parity_of_sum_is_independent_of_first_die():
    d1, d2 = DICE
    _, _, h = joint_table((d1 + d2) % 2, d1)
    assert is_independent(h) and table_rank(h) == 1
    assert cdf_independence_defect(h) == pytest.approx(0.0, abs=1e-15)


def test_independence_iff_rank_one():
    rng = np.random.default_rng(2)
    for _ in range(40):
        p, q = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(5))
        P = product_measure(p, q)
        assert is_independent(P) and table_rank(P) == 1
        assert np.allclose(product_of_marginals(P), P)
        f = rng.dirichlet(np.ones(20)).reshape(4, 5)             # a generic joint table
        assert not is_independent(f) and table_rank(f) > 1


def test_comonotone_coupling_has_the_same_marginals():
    rng = np.random.default_rng(5)
    for _ in range(20):
        p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(7))
        C = comonotone_table(p, q)
        fx, fy = marginals(C)
        assert np.allclose(fx, p) and np.allclose(fy, q)
        assert not is_independent(C)
        # a mixture with the product keeps the marginals; only theta = 0 is independent
        for theta in (0.0, 0.3, 1.0):
            M = (1 - theta) * product_measure(p, q) + theta * C
            assert np.allclose(marginals(M)[0], p) and is_independent(M) == (theta == 0.0)
    assert np.allclose(comonotone_table([0.5, 0.5], [0.5, 0.5]), [[0.5, 0], [0, 0.5]])


def test_cdf_determines_the_table():
    # f(i,j) = F(i,j) - F(i-1,j) - F(i,j-1) + F(i-1,j-1): the discrete (iii) => (ii)
    rng = np.random.default_rng(3)
    f = rng.dirichlet(np.ones(12)).reshape(3, 4)
    F = np.pad(joint_cdf(f), ((1, 0), (1, 0)))
    assert np.allclose(F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1], f)
    assert F[-1, -1] == pytest.approx(1.0)
    assert cdf_independence_defect(product_of_marginals(f)) == pytest.approx(0.0, abs=1e-15)


def test_expectation_from_a_table():
    xs, ys = np.array([-1.0, 0.0, 1.0]), np.array([0.0, 1.0])
    f = np.array([[0.0, 1 / 3], [1 / 3, 0.0], [0.0, 1 / 3]])      # Y = X^2, X uniform on {-1,0,1}
    assert expectation(lambda x, y: x * y, f, xs, ys) == pytest.approx(0.0)
    assert covariance(f, xs, ys) == pytest.approx(0.0)
    assert not is_independent(f)
    assert expectation(lambda x, y: y, f, xs, ys) == pytest.approx(2 / 3)


def test_sampling_a_table():
    f = np.array([[0.1, 0.2], [0.3, 0.4]])
    i, j = sample_table(f, 200_000, seed=4)
    assert np.allclose(empirical_table(i, j, f.shape), f, atol=5e-3)


def test_empirical_defect_of_dice_pairs():
    # the numbers quoted in the solution of the dice exercise (seed 2026)
    P = product_measure(np.full(6, 1 / 6), np.full(6, 1 / 6))
    d = np.arange(1, 7)
    for n, indep, dep in ((100, 0.0231, 0.0357), (10_000, 0.00373, 0.0285), (1_000_000, 0.000404, 0.0280)):
        i, j = sample_table(P, n, seed=2026)
        d1, d2 = d[i], d[j]
        assert independence_defect(joint_table((d1 + d2) % 2, d1)[2]) == pytest.approx(indep, rel=0.02)
        assert independence_defect(joint_table(d1 + d2, d1 - d2)[2]) == pytest.approx(dep, rel=0.02)


# ------------------------------------------------------------- iterated integrals


def test_midpoint_rule_on_a_product():
    f = lambda x, y: x * y**2
    for order in ("dy dx", "dx dy"):
        assert iterated_integral(f, (0, 1), (0, 1), 200, 200, order) == pytest.approx(1 / 6, rel=1e-4)
    assert np.allclose(midpoints(0, 1, 4), [0.125, 0.375, 0.625, 0.875])
    with pytest.raises(ValueError):
        iterated_integral(f, (0, 1), (0, 1), 10, 10, "dz")


def test_sections_of_a_disk():
    disk = lambda x, y: (x * x + y * y <= 1).astype(float)
    x = np.array([0.0, 0.5, 0.9])
    assert np.allclose(inner_integrals(disk, x, (-1, 1), 200_000), 2 * np.sqrt(1 - x**2), atol=1e-4)
    area = iterated_integral(disk, (-1, 1), (-1, 1), 2000, 20_000)
    assert area == pytest.approx(math.pi, abs=1e-4)


def test_fubini_counterexample_inner_integrals():
    f = fubini_counterexample
    x = np.array([0.05, 0.3, 1.0])
    assert np.allclose(inner_integrals(f, x, (0, 1), 400_000, "y"), fubini_inner_exact(x, "y"), atol=1e-6)
    assert np.allclose(inner_integrals(f, x, (0, 1), 400_000, "x"), fubini_inner_exact(x, "x"), atol=1e-6)
    # an antiderivative in y: d/dy [y / (x^2 + y^2)] = f(x, y)
    xx, yy, h = 0.7, 0.4, 1e-6
    F = lambda y: y / (xx**2 + y**2)
    assert (F(yy + h) - F(yy - h)) / (2 * h) == pytest.approx(f(xx, yy), rel=1e-8)


def test_fubini_counterexample_iterated_integrals_differ():
    f = fubini_counterexample
    # same grid in both variables: the double sum is 0 by antisymmetry
    for n in (10, 100, 1000):
        assert abs(iterated_integral(f, (0, 1), (0, 1), n, n)) < 1e-12
    # inner resolution much finer than the outer one: the two iterated integrals, +-pi/4
    a = iterated_integral(f, (0, 1), (0, 1), 200, 200 * 64, "dy dx")
    b = iterated_integral(f, (0, 1), (0, 1), 200, 200 * 64, "dx dy")
    assert a == pytest.approx(math.pi / 4, abs=1e-5) and b == pytest.approx(-math.pi / 4, abs=1e-5)
    # the table of the solution (n = 200 outer points, 200 k inner points)
    for k, value, tol in ((4, 0.7851233, 5e-8), (16, 0.7853986822, 5e-11), (64, 0.7853986841, 5e-11),
                          (256, 0.7853986842, 5e-11)):
        assert iterated_integral(f, (0, 1), (0, 1), 200, 200 * k, "dy dx") == pytest.approx(value, abs=tol)
        assert iterated_integral(f, (0, 1), (0, 1), 200, 200 * k, "dx dy") == pytest.approx(-value, abs=tol)
    # nested adaptive quadrature agrees
    inner = lambda x: integrate.quad(lambda y: f(x, y), 0, 1, points=[x], limit=200)[0]
    assert integrate.quad(inner, 0, 1, limit=200)[0] == pytest.approx(math.pi / 4, abs=1e-9)


def test_monte_carlo_of_a_non_integrable_function():
    # E|f(U,V)| = +inf: the averages quoted in the solution (10^6 samples, seeds 0..4) do not settle
    quoted = ((-0.554, 0.92), (-1.125, 0.65), (-0.030, 1.00), (3.270, 3.56), (2.026, 3.21))
    for seed, (avg, se) in enumerate(quoted):
        r = monte_carlo_expectation(fubini_counterexample, lambda rng, k: (rng.random(k), rng.random(k)),
                                    10**6, seed=seed)
        assert r.estimate == pytest.approx(avg, abs=5e-4) and r.std_error == pytest.approx(se, abs=5e-3)


def test_fubini_counterexample_is_not_integrable():
    # int int_{[eps,1]^2} |f| = log(1/eps) - pi/2 + 2 arctan(eps), which diverges
    absf = lambda x, y: np.abs(fubini_counterexample(x, y))
    for eps, exact in ((0.1, 0.9311260711814734), (0.01, 3.054373192566526)):
        assert math.log(1 / eps) - math.pi / 2 + 2 * math.atan(eps) == pytest.approx(exact)
        assert iterated_integral(absf, (eps, 1), (eps, 1), 2000, 20_000) == pytest.approx(exact, rel=1e-4)
    # on [eps,1]^2 the function is bounded and both iterated integrals vanish
    eps = 0.05
    assert iterated_integral(fubini_counterexample, (eps, 1), (eps, 1), 1000, 20_000) == pytest.approx(0, abs=5e-5)


def test_marginal_density_of_the_joint_density_exercise():
    f = lambda x, y: x * np.exp(-x * (1 + y))
    x = np.array([0.5, 1.0, 2.0])
    assert np.allclose(marginal_density(f, x, (0, 200), 400_000), np.exp(-x), rtol=1e-6)
    total = integrate.dblquad(lambda y, x: f(x, y), 0, np.inf, 0, np.inf)[0]
    assert total == pytest.approx(1.0, abs=1e-9)
    p = integrate.dblquad(lambda y, x: f(x, y), 0, np.inf, 0, lambda x: x)[0]
    closed = 1 - math.exp(0.25) * math.sqrt(math.pi) / 2 * special.erfc(0.5)
    assert p == pytest.approx(closed, abs=1e-9) and closed == pytest.approx(0.454359, abs=1e-6)


# ------------------------------------------------------------------------ Monte Carlo


def test_monte_carlo_expectation():
    r = monte_carlo_expectation(lambda x, y: x * y, lambda rng, k: sample_pair("gamma-mix", k, rng),
                                10**6, seed=2026)
    assert isinstance(r, MCResult) and r.n == 10**6
    assert abs(r.estimate - 1.0) < 4 * r.std_error                # E[XY] = 1
    assert r.ci[0] < r.estimate < r.ci[1]
    assert r.estimate == pytest.approx(1.00056, abs=1e-5)


def test_sample_sizes_for_the_covariance_exercise():
    # Var(XY) = 1/7, 1/8, 1/24; N = ceil((1.96/0.005)^2 Var) gives a half-width of about 0.005
    for kind, v, N, est in (("square", 1 / 7, 21952, 0.0023863), ("circle", 1 / 8, 19208, -0.0000544),
                            ("disk", 1 / 24, 6403, -0.0007436)):
        assert math.ceil((1.96 / 0.005) ** 2 * v) == N
        r = monte_carlo_expectation(lambda x, y: x * y, lambda rng, k: sample_pair(kind, k, rng), N, seed=2026)
        assert 1.96 * r.std_error == pytest.approx(0.005, rel=0.01)
        assert r.estimate == pytest.approx(est, abs=1e-6)
        x, y = sample_pair(kind, 10**6, seed=1)
        assert np.var(x * y) == pytest.approx(v, rel=0.01)


def test_empirical_cdf_defect_table():
    # the table of the solution of the exercise on uncorrelated pairs (seed 2026)
    table = {"independent": (0.0166, 0.0064, 0.0020, 0.0005), "square": (0.1375, 0.1250, 0.1250, 0.1250),
             "circle": (0.0668, 0.0625, 0.0625, 0.0625), "disk": (0.0228, 0.0153, 0.0141, 0.0136)}
    for kind, row in table.items():
        for n, value in zip((10**3, 10**4, 10**5, 10**6), row):
            assert empirical_cdf_defect(*sample_pair(kind, n, 2026)) == pytest.approx(value, abs=5e-5)


def test_cdf_gap_on_a_fixed_grid_can_vanish_for_dependent_pairs():
    # X = (I + U)/40, Y = (J + U)/40 with I, J uniform on {0..39} and one U: dependent, yet
    # F = F_X F_Y at every point of the grid {1/40, ..., 39/40}^2
    I, J = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
    g = np.arange(1, 40) / 40
    Fx = np.minimum(g, 1)                                     # X is uniform on (0, 1)
    F = np.array([[np.mean((I <= 40 * s - 1) & (J <= 40 * t - 1)) for t in g] for s in g])
    assert np.allclose(F, np.outer(Fx, Fx))
    # ... but off the grid: F(1/80, 1/80) = P(I = J = 0, U <= 1/2) = 1/3200, F_X F_Y = 1/6400
    assert Fraction(1, 1600) * Fraction(1, 2) - Fraction(1, 80) ** 2 == Fraction(1, 6400)
    # and X, Y are dependent: 40X and 40Y have the same fractional part U
    rng = np.random.default_rng(0)
    u = rng.random(10**5)
    x = (rng.integers(0, 40, 10**5) + u) / 40
    y = (rng.integers(0, 40, 10**5) + u) / 40
    assert empirical_cdf_defect(x, y, g, g) < 0.01             # sampling noise only, on the grid
    assert np.allclose((40 * x) % 1, (40 * y) % 1)


def test_empirical_cdf_defect():
    x, y = sample_pair("independent", 10**5, seed=2026)
    assert empirical_cdf_defect(x, y) < 0.005
    x, y = sample_pair("square", 10**5, seed=2026)
    assert empirical_cdf_defect(x, y) == pytest.approx(0.125, abs=0.003)   # sup |F - F_X F_Y| = 1/8
    # exact value at (s, t) = (-1/2, 1/4): P(X <= -1/2, X^2 <= 1/4) = 0, F_X F_Y = 1/4 * 1/2
    assert empirical_cdf_defect(x, y, [-0.5], [0.25]) == pytest.approx(0.125, abs=0.003)


def test_joint_density_exercise_by_simulation():
    x, y = sample_pair("gamma-mix", 10**6, seed=2026)
    assert np.mean(x > y) == pytest.approx(0.454359, abs=4 * 0.0005)
    assert np.mean((x <= 1) & (x * y <= 1)) == pytest.approx((1 - math.exp(-1)) ** 2, abs=0.002)
    assert empirical_cdf_defect(x, x * y) < 0.003            # X and XY are independent
    assert empirical_cdf_defect(x, y) > 0.1                  # X and Y are not


def test_broken_stick():
    rng = np.random.default_rng(2026)
    N = 10**6
    u, v = rng.random(N), rng.random(N)
    a, b = np.minimum(u, v), np.maximum(u, v)
    assert np.mean((a < 0.5) & (b - a < 0.5) & (b > 0.5)) == pytest.approx(0.25, abs=0.002)
    u, w = rng.random(N), rng.random(N)
    L = np.maximum(u, 1 - u)
    tri = (w * L < 0.5) & ((1 - w) * L < 0.5)
    assert np.mean(tri) == pytest.approx(2 * math.log(2) - 1, abs=0.002)
    c = rng.random(N) < 0.5                                  # break a piece chosen at random
    L3 = np.where(c, u, 1 - u)
    tri3 = (w * L3 < 0.5) & ((1 - w) * L3 < 0.5) & (1 - L3 < 0.5)
    assert np.mean(tri3) == pytest.approx(math.log(2) - 0.5, abs=0.002)
    assert np.mean(tri3) == pytest.approx(0.193709, abs=1e-6)          # the value quoted (seed 2026)
    # the same probability as an iterated integral of an indicator over the unit square
    ind = lambda u, w: ((w * np.maximum(u, 1 - u) < 0.5) & ((1 - w) * np.maximum(u, 1 - u) < 0.5)).astype(float)
    assert iterated_integral(ind, (0, 1), (0, 1), 2001, 2001) == pytest.approx(2 * math.log(2) - 1, abs=1e-3)
    # the midpoint values quoted in the solution (odd n: no ties at the boundaries)
    two = lambda u, v: ((np.minimum(u, v) < 0.5) & (np.abs(u - v) < 0.5) & (np.maximum(u, v) > 0.5)).astype(float)
    assert midpoints(0, 1, 2001)[1000] == 0.5
    assert iterated_integral(two, (0, 1), (0, 1), 2001, 2001) == pytest.approx(0.249500, abs=1e-6)
    assert iterated_integral(ind, (0, 1), (0, 1), 2001, 2001) == pytest.approx(0.386292, abs=1e-6)


def test_sample_pair_kinds():
    for kind in ("independent", "square", "circle", "disk", "gamma-mix"):
        x, y = sample_pair(kind, 1000, seed=0)
        assert x.shape == y.shape == (1000,)
    x, y = sample_pair("circle", 1000, seed=0)
    assert np.allclose(x**2 + y**2, 1)
    with pytest.raises(ValueError):
        sample_pair("nope", 10)
