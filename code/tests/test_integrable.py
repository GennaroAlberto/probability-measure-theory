"""Checks of the Chapter 5 code against the notes and the exercises."""
import math

import numpy as np
import pytest
from scipy import integrate

from probnotes.integrable import (
    SINE_LIMIT,
    DiscreteMeasure,
    abs_sine_constant,
    abs_sine_integral_from_1,
    alternating_parts,
    bump,
    bump_integral,
    darboux_sums,
    envelope,
    escaping_block,
    is_dominated,
    lebesgue_sum,
    lebesgue_sum_on_grid,
    level_measure_on_grid,
    parts,
    pq_limit,
    rearrange_pq,
    rearrange_to_target,
    riemann_sum,
    sine_integral_from_1,
    siri_discrete,
    siri_halfline,
    siri_halfline_integral,
    siri_terms,
    siri_unit,
    siri_unit_integral,
    spike,
)
from probnotes.measurable import dyadic_approximation

# ---------------------------------------------------------------- f = f+ - f-


def test_parts():
    v = np.array([-2.0, 0.0, 3.5, -np.inf, np.inf])
    p, m = parts(v)
    assert np.array_equal(p, [0, 0, 3.5, 0, np.inf])
    assert np.array_equal(m, [2, 0, 0, np.inf, 0])
    finite = np.isfinite(v)
    assert np.array_equal(p[finite] - m[finite], v[finite])
    assert np.array_equal(p + m, np.abs(v))                  # |f| = f+ + f-


# ---------------------------------------------------------------- Riemann and Lebesgue sums


def test_riemann_sums_of_a_parabola():
    f = lambda x: x**2
    for n in (1, 7, 100):
        assert riemann_sum(f, 0, 1, n, "mid") == pytest.approx(1 / 3 - 1 / (12 * n**2), rel=1e-12)
        assert riemann_sum(f, 0, 1, n, "left") == pytest.approx((n - 1) * (2 * n - 1) / (6 * n**2), rel=1e-12)
        assert riemann_sum(f, 0, 1, n, "right") == pytest.approx((n + 1) * (2 * n + 1) / (6 * n**2), rel=1e-12)


def test_darboux_sums_are_exact_for_monotone_functions():
    lo, up = darboux_sums(lambda x: x**2, 0, 1, 50)
    assert lo == pytest.approx(riemann_sum(lambda x: x**2, 0, 1, 50, "left"), rel=1e-12)
    assert up == pytest.approx(riemann_sum(lambda x: x**2, 0, 1, 50, "right"), rel=1e-12)
    # sin on [0, pi]: lower <= integral = 2 <= upper, and the gap shrinks like 1/n
    gaps = []
    for n in (10, 100, 1000):
        lo, up = darboux_sums(np.sin, 0, math.pi, n)
        assert lo <= 2 <= up
        gaps.append(up - lo)
    assert gaps[1] < gaps[0] / 9 and gaps[2] < gaps[1] / 9


def test_lebesgue_sum_of_the_identity():
    # f(x) = x on [0, 1]: m(f >= t) = 1 - t, and int f_n = 1/2 - 2^(-n-1) exactly.
    meas = lambda t: np.clip(1 - t, 0, 1)
    for n in range(1, 12):
        assert lebesgue_sum(meas, n) == pytest.approx(0.5 - 2.0 ** (-n - 1), abs=1e-14)


def test_lebesgue_sum_agrees_with_integrating_the_dyadic_function():
    # The horizontal sum equals the integral of f_n = min(n, floor(2^n f)/2^n) (Chapter 3),
    # here computed on a fine grid by the midpoint rule.
    f = lambda x: 3 * np.sin(math.pi * x) ** 2
    x = (np.arange(2**16) + 0.5) / 2**16
    for n in (1, 2, 3, 5):
        direct = np.mean(dyadic_approximation(f(x), n))
        assert lebesgue_sum_on_grid(f, 0, 1, n, 2**16) == pytest.approx(direct, abs=1e-12)


def test_lebesgue_sums_increase_to_the_integral_of_an_unbounded_function():
    # f(x) = x^(-1/2) on (0, 1]: m(f >= t) = min(1, t^-2); int f = 2 (monotone convergence).
    meas = lambda t: np.minimum(1.0, t**-2.0)
    ns = np.arange(1, 15)
    vals = np.array([lebesgue_sum(meas, n) for n in ns])
    assert np.all(np.diff(vals) > 0) and np.all(vals < 2.0)
    # the cut at height n costs int_n^inf m(f >= t) dt = 1/n; the rounding costs about 2^(-n-1)
    assert np.all(np.abs(vals - (2 - 1 / ns)) <= 2.0**-ns)
    # the grid estimate of the level sets agrees with the exact one
    lm = level_measure_on_grid(lambda x: x**-0.5, 0, 1, 2**18)
    t = np.array([0.5, 1.0, 2.0, 5.0])
    assert np.allclose(lm(t), meas(t), atol=1e-5)


# ---------------------------------------------------------------- the worked exercises


@pytest.mark.parametrize("n", [1, 2, 5, 37, 1000])
def test_first_worked_exercise_integrals(n):
    q = integrate.quad(siri_unit, 0, 1, args=(n,), limit=200)[0]
    assert siri_unit_integral(n) == pytest.approx(q, rel=1e-9)


def test_first_worked_exercise_limit_is_two():
    n = np.array([1, 10, 100, 10**4, 10**6])
    vals = siri_unit_integral(n)
    assert np.all(np.diff(vals) > 0)
    assert np.allclose(vals[1:], 2 - 1 / (3 * n[1:]), atol=2 / n[1:] ** 2)
    assert vals[-1] == pytest.approx(2.0, abs=1e-6)
    assert siri_unit_integral(1) == pytest.approx(4 * math.pi / 4 - 2 * math.log(2), rel=1e-14)


def test_first_worked_exercise_domination_and_monotonicity():
    x = np.geomspace(1e-8, 1 - 1e-9, 2001)
    g = lambda x: x**-0.5
    ns = np.arange(1, 400)
    assert is_dominated(siri_unit, g, x, ns)
    vals = siri_unit(x[None, :], ns[:, None])
    assert np.all(np.diff(vals, axis=0) >= 0)                 # f_n increases with n ...
    assert np.allclose(siri_unit(x, 10**8), g(x), rtol=1e-7)  # ... to g = x^(-1/2)


@pytest.mark.parametrize("n", [1, 4, 9, 100])
def test_second_worked_exercise_integrals(n):
    q = (integrate.quad(siri_halfline, 0, 1, args=(n,))[0]
         + integrate.quad(siri_halfline, 1, np.inf, args=(n,), limit=500)[0])
    assert siri_halfline_integral(n) == pytest.approx(q, rel=1e-8)
    assert siri_halfline_integral(1) == pytest.approx(2 * math.pi)


def test_second_worked_exercise_decreases_to_zero():
    x = np.geomspace(1e-6, 1e6, 801)
    vals = siri_halfline(x[None, :], np.arange(1, 50)[:, None])
    assert np.all(np.diff(vals, axis=0) <= 0)
    assert np.max(siri_halfline(x, 1e12) / siri_halfline(x, 1)) < 1e-7    # ~ (x/n) / log(1+x) at x = 1e6
    # f_1 dominates the whole family, so dominated convergence also gives the limit 0
    assert is_dominated(siri_halfline, lambda x: siri_halfline(x, 1), x, np.arange(1, 50))


# ---------------------------------------------------------------- failures of domination


def test_counterexamples_converge_to_zero_but_integrals_do_not():
    x = np.linspace(0.001, 0.999, 999)
    assert np.all(spike(x, 10**4) == 0) and np.all(bump(x, 10**6) < 1e-300)
    assert np.all(escaping_block(np.linspace(-5, 50, 1001), 60) == 0)
    for n in (1, 3, 10, 100):
        assert integrate.quad(spike, 0, 1, args=(n,), points=[1 / n])[0] == pytest.approx(1.0)
        assert integrate.quad(escaping_block, n - 1, n + 2, args=(n,), points=[n, n + 1])[0] == pytest.approx(1.0)
        assert bump_integral(n) == pytest.approx(integrate.quad(bump, 0, 1, args=(n,))[0], rel=1e-10)
    assert bump_integral(50) == pytest.approx(1.0, abs=1e-19)


def test_the_envelopes_are_not_integrable():
    # sup_n n 1_(0,1/n)(x) = floor-like ~ 1/x, and sup_n n^2 x e^(-nx) = 4/(e^2 x) at x = 2/k.
    k = np.arange(2, 400)
    x = 2.0 / k
    env = envelope(bump, x, np.arange(1, 1000))
    assert np.allclose(env * x, 4 / math.e**2, rtol=1e-12)
    xs = np.geomspace(1e-4, 1, 400)
    env_spike = envelope(spike, xs, np.arange(1, 20001))
    assert np.all(env_spike <= 1 / xs + 1e-9) and np.all(env_spike >= 1 / xs - 1)
    assert not is_dominated(bump, lambda x: 1 / np.sqrt(x), xs, np.arange(1, 500))


# ---------------------------------------------------------------- discrete measures


def test_discrete_measure_of_the_notes():
    mu = siri_discrete(1000)
    assert mu.measure(lambda x: np.ones_like(x, dtype=bool)) == pytest.approx(1.0)   # a probability
    assert mu.measure(lambda x: x > 0) == pytest.approx(2 / 3)                        # odd n
    assert mu.points[:4].tolist() == [2.0, -2.0, 8 / 3, -4.0]
    # the terms f(x_n) alpha_n are (-1)^(n+1)/n
    assert np.allclose(mu.terms(lambda x: x), siri_terms(1000), rtol=1e-12, atol=0)
    # sum |f(x_n)| alpha_n is the harmonic series: f(x) = x is not integrable
    assert mu.abs_integral(lambda x: x) == pytest.approx(np.sum(1 / np.arange(1, 1001)))
    pos, neg = mu.parts_integrals(lambda x: x)
    assert pos == pytest.approx(np.sum(1 / np.arange(1, 1001, 2)))
    assert neg == pytest.approx(np.sum(1 / np.arange(2, 1001, 2)))
    # |x|^(1/2) is integrable: sum 2^(-n/2) n^(-1/2)
    assert mu.integral(lambda x: np.abs(x) ** 0.5) == pytest.approx(1.6208739036039665, rel=1e-12)


def test_integral_is_the_series_on_a_finite_measure():
    mu = DiscreteMeasure([0.0, 1.0, 2.5], [0.2, 0.3, 0.5])
    f = lambda x: x**2 - 1
    assert mu.integral(f) == pytest.approx(-0.2 + 0 + 0.5 * 5.25)
    p, m = mu.parts_integrals(f)
    assert p - m == pytest.approx(mu.integral(f))
    with pytest.raises(ValueError):
        DiscreteMeasure([0.0, 0.0], [0.5, 0.5])
    with pytest.raises(ValueError):
        DiscreteMeasure([0.0, 1.0], [0.5, 0.0])


def test_partial_sums_converge_but_the_parts_diverge():
    s = np.cumsum(siri_terms(10**6))
    assert s[-1] == pytest.approx(math.log(2), abs=1e-6)
    pos, neg = alternating_parts(10**6)
    # each part grows like (1/2) log N: no finite integral of f+ or f-
    assert np.sum(pos) - 0.5 * math.log(10**6) == pytest.approx(math.log(2) + 0.5 * 0.5772156649, abs=1e-6)
    assert -np.sum(neg) - 0.5 * math.log(10**6) == pytest.approx(0.5 * 0.5772156649, abs=1e-6)


@pytest.mark.parametrize("p,q", [(1, 1), (2, 1), (1, 2), (3, 1), (1, 4)])
def test_rearrangements_change_the_sum(p, q):
    pos, neg = alternating_parts(4 * 10**5)
    r = rearrange_pq(pos, neg, p, q)
    assert np.sum(r) == pytest.approx(pq_limit(p, q), abs=5e-5)
    if (p, q) == (1, 1):
        assert np.sum(r) == pytest.approx(math.log(2), abs=5e-6)


def test_rearrangement_uses_every_term_once():
    pos, neg = alternating_parts(30)
    r = rearrange_pq(pos, neg, 2, 1)
    assert len(r) == 45 and len(set(r.tolist())) == 45
    assert set(r.tolist()) == set(pos.tolist()) | set(neg[:15].tolist())


@pytest.mark.parametrize("target", [-1.0, 0.0, math.pi, 1.5])
def test_riemann_rearrangement_reaches_any_target(target):
    pos, neg = alternating_parts(10**6)
    r = rearrange_to_target(pos, neg, target, 2 * 10**5)
    s = np.cumsum(r)
    assert abs(s[-1] - target) < 1e-3
    assert np.max(np.abs(s[-1000:] - target)) < 2e-3


def test_integrable_function_is_insensitive_to_rearrangement():
    # f(x) = sign(x) |x|^(1/2) is integrable for the measure of the notes:
    # every rearrangement of its series has the same sum.
    mu = siri_discrete(1000)
    t = mu.terms(lambda x: np.sign(x) * np.abs(x) ** 0.5)
    pos, neg = t[t > 0], t[t < 0]
    exact = float(np.sum(t))
    for p, q in [(1, 1), (2, 1), (5, 2)]:
        r = rearrange_pq(pos, neg, p, q)
        assert np.sum(r) == pytest.approx(exact, abs=1e-12)


# ---------------------------------------------------------------- sin x / x


def test_sine_integral_converges():
    assert SINE_LIMIT == pytest.approx(0.6247132564277136, rel=1e-13)
    T = np.array([2.0, 10.0, 37.5])
    q = [integrate.quad(lambda x: math.sin(x) / x, 1, t, limit=200)[0] for t in T]
    assert np.allclose(sine_integral_from_1(T), q, rtol=1e-12)
    assert abs(sine_integral_from_1(1e6) - SINE_LIMIT) < 1e-6


def test_abs_sine_integral_exact_and_logarithmic():
    for t in (1.5, 3.0, 4.0, 10.0, 50.0):
        q = integrate.quad(lambda x: abs(math.sin(x)) / x, 1, t, limit=500, points=np.pi * np.arange(1, int(t / np.pi) + 1))[0]
        assert abs_sine_integral_from_1(t) == pytest.approx(q, rel=1e-10)
    C = abs_sine_constant()
    assert C == pytest.approx(0.166840740, abs=2e-9)
    T = np.array([1e3, 1e4, 1e5, 1e6])
    gap = abs_sine_integral_from_1(T) - 2 / math.pi * np.log(T)
    assert np.all(np.abs(gap - C) < 1.0 / T * 1.1)
    # lower bound used in the notes' example: int_{k pi}^{(k+1) pi} |sin x|/x >= 2/((k+1) pi)
    k = np.arange(1, 50)
    assert np.all(abs_sine_integral_from_1((k + 1) * np.pi) - abs_sine_integral_from_1(k * np.pi)
                  >= 2 / ((k + 1) * np.pi))


# ---------------------------------------------------------------- numbers quoted by the exercises

ZETA_HALF = -1.4603545088095868          # zeta(1/2), the Riemann zeta function at 1/2


def test_alternating_series_error_bound():
    # Exercise 5.7(a): |S_N - log 2| <= 1/(N+1), and S_N - log 2 ~ -1/(2N) for even N.
    s = np.cumsum(siri_terms(10**6))
    for k in range(1, 7):
        N = 10**k
        err = s[N - 1] - math.log(2)
        assert abs(err) <= 1 / (N + 1)
        assert err == pytest.approx(-1 / (2 * N), rel=2 / N)


def test_sine_integral_tail_bound():
    # Exercise 5.8(b): |I(T) - (pi/2 - Si(1))| <= 2/T; the largest value of I is I(pi) = 0.905854.
    T = np.concatenate([10.0 ** np.arange(1, 7), np.linspace(1, 200, 4001)])
    assert np.all(np.abs(sine_integral_from_1(T) - SINE_LIMIT) <= 2 / T)
    assert float(sine_integral_from_1(math.pi)) == pytest.approx(0.905854, abs=1e-6)
    assert np.max(sine_integral_from_1(np.linspace(1, 300, 300001))) <= float(sine_integral_from_1(math.pi)) + 1e-12
    # Exercise 5.8(d): J(T) = 10 near T = 5.106e6
    assert float(abs_sine_integral_from_1(5.10e6)) < 10 < float(abs_sine_integral_from_1(5.11e6))


def test_riemann_sum_constants_for_the_unbounded_function():
    # Exercise 5.10(c): 2 - R_N ~ |zeta(1/2)| / sqrt(N), 2 - M_N ~ (sqrt 2 - 1)|zeta(1/2)| / sqrt(N).
    f = lambda x: x**-0.5
    k = np.arange(1, 10**6 + 1)
    assert np.sum(k**-0.5) - 2 * 1000.0 == pytest.approx(ZETA_HALF + 0.5e-3, abs=1e-8)   # Euler-Maclaurin
    for N in (10**4, 10**6):
        right = (2 - riemann_sum(f, 0, 1, N, "right")) * math.sqrt(N)
        mid = (2 - riemann_sum(f, 0, 1, N, "mid")) * math.sqrt(N)
        assert right == pytest.approx(-ZETA_HALF - 0.5 / math.sqrt(N), abs=1e-6)
        assert mid == pytest.approx(-(math.sqrt(2) - 1) * ZETA_HALF, abs=1e-6)


def test_uniform_level_sums_without_a_cut():
    # Exercise 5.10(b): L(h) = h sum_{i>=1} m(f >= ih) = 1 + M psi'(M + 1), M = 1/h, and 2 - L(h) ~ h/2.
    from scipy.special import polygamma
    for k in range(2, 13, 2):
        M = 2**k
        h = 1.0 / M
        i = np.arange(1, 400 * M + 1)
        direct = h * np.sum(np.minimum(1.0, (i * h) ** -2.0)) + M * float(polygamma(1, 400 * M + 1))
        closed = 1 + M * float(polygamma(1, M + 1))
        assert direct == pytest.approx(closed, rel=1e-12)
        assert abs((2 - closed) - h / 2) <= h**2 / 5


def test_cantor_upper_darboux_sums():
    # Exercise 5.10(d): (5/2) 2^k - 2 of the 3^k cells of mesh 3^-k meet the Cantor set.
    from probnotes.measurable import cantor_stage
    for k in range(1, 13):
        left = np.rint(cantor_stage(k, 1 / 3)[:, 0] * 3**k).astype(np.int64)
        cells = np.unique(np.concatenate([left - 1, left, left + 1]))
        cells = cells[(cells >= 0) & (cells < 3**k)]
        assert len(cells) == 5 * 2 ** (k - 1) - 2
    assert (5 * 2**11 - 2) / 3**12 == pytest.approx(0.019265, abs=1e-6)
