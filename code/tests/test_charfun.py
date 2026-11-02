"""Checks of the Chapter 10 code against the notes and the exercises."""
import math

import numpy as np
import pytest
from scipy import integrate, special, stats

from probnotes.charfun import (
    LAWS,
    atom_mass,
    cf_binomial,
    cf_cauchy,
    cf_density_2x,
    cf_derivative,
    cf_exponential,
    cf_from_density,
    cf_gamma,
    cf_geometric,
    cf_independence_defect,
    cf_laplace,
    cf_normal,
    cf_poisson,
    cf_product_normals,
    cf_uniform,
    density_product_normals,
    ecf,
    ecf_standard_error,
    fd_weights,
    invert_density,
    levy_inversion,
    moments_from_cf,
    sample_moments,
)

T_GRID = np.linspace(-12.0, 12.0, 241)

# ----------------------------------------------------------------------------- closed forms


@pytest.mark.parametrize("name", sorted(LAWS))
def test_property_1_and_2_for_every_law(name):
    # Phi(0) = 1, |Phi| <= 1, Phi(-t) = conj Phi(t); real exactly for the symmetric laws
    phi = LAWS[name].cf
    assert phi(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-14)
    v = phi(T_GRID)
    assert np.all(np.abs(v) <= 1 + 1e-12)
    assert np.allclose(phi(-T_GRID), np.conj(v), atol=1e-13)
    if LAWS[name].symmetric:
        assert np.max(np.abs(v.imag)) < 1e-13
    else:
        assert np.max(np.abs(v.imag)) > 1e-2


@pytest.mark.parametrize("name", ["normal", "uniform", "exponential", "laplace", "cauchy", "density 2x",
                                  "three products"])
def test_closed_forms_match_quadrature_of_the_density(name):
    law = LAWS[name]
    lo, hi = {"normal": (-12, 12), "uniform": (-1, 1), "exponential": (0, 60), "laplace": (0, 60),
              "cauchy": (0, 1), "density 2x": (0, 1), "three products": (0, 60)}[name]
    t = np.linspace(-6, 6, 25)
    if name in ("laplace", "three products"):      # symmetric: integrate on (0, 60) and double the real part
        q = 2 * cf_from_density(law.density, t, lo, hi, 600).real
    elif name == "cauchy":                          # heavy tails: Fourier integral on [0, inf) (QAWF)
        q = np.array([2 * integrate.quad(law.density, 0, np.inf, weight="cos", wvar=abs(s))[0]
                      if s != 0 else 1.0 for s in t])
    else:
        q = cf_from_density(law.density, t, lo, hi, 600)
    assert np.allclose(q, law.cf(t), atol=1e-9)


def test_discrete_closed_forms():
    t = np.linspace(-4, 4, 33)
    k = np.arange(0, 80)
    assert np.allclose(cf_binomial(t, 7, 0.3), np.exp(1j * np.outer(t, np.arange(8))) @ stats.binom.pmf(np.arange(8), 7, 0.3))
    assert np.allclose(cf_poisson(t, 2.5), np.exp(1j * np.outer(t, k)) @ stats.poisson.pmf(k, 2.5))
    assert np.allclose(cf_geometric(t, 0.4), np.exp(1j * np.outer(t, k[1:])) @ (0.4 * 0.6 ** (k[1:] - 1)))


def test_gamma_and_its_special_cases():
    t = np.linspace(-5, 5, 41)
    assert np.allclose(cf_gamma(t, 1.0, 2.0), cf_exponential(t, 2.0))
    # Exp + Exp = Gamma(2, lam); Gamma(a) Gamma(b) = Gamma(a + b): property 3 on closed forms
    assert np.allclose(cf_exponential(t, 2.0) ** 2, cf_gamma(t, 2.0, 2.0))
    assert np.allclose(cf_gamma(t, 0.7, 3.0) * cf_gamma(t, 1.6, 3.0), cf_gamma(t, 2.3, 3.0))
    # principal branch is the right one: compare with quadrature of the Gamma(1/2, 1) density
    dens = lambda x: stats.gamma.pdf(x, 0.5)
    q = np.array([integrate.quad(lambda x: math.cos(s * x) * dens(x), 0, np.inf, limit=400)[0]
                  + 1j * integrate.quad(lambda x: math.sin(s * x) * dens(x), 0, np.inf, limit=400)[0]
                  for s in (-3.0, 0.5, 2.0)])
    assert np.allclose(q, cf_gamma(np.array([-3.0, 0.5, 2.0]), 0.5), atol=1e-6)


# ------------------------------------------------------------- empirical characteristic functions


def test_ecf_is_the_cf_of_the_empirical_law_and_converges():
    rng = np.random.default_rng(2026)
    x = rng.exponential(1.0, 20_000)
    t = np.linspace(-8, 8, 81)
    e = ecf(x, t)
    assert ecf(x, np.zeros(3)) == pytest.approx(np.ones(3))
    assert np.allclose(e[::-1], np.conj(e))                         # property 2 holds for phi_n too
    se = ecf_standard_error(x, t)
    assert np.all(np.abs(e - cf_exponential(t)) <= 5 * se + 1e-12)
    # standard error formula: sqrt((1 - |phi|^2) / n), which vanishes at t = 0
    assert se[40] == 0.0
    assert se[0] == pytest.approx(math.sqrt((1 - abs(e[0]) ** 2) / len(x)))


def test_ecf_in_several_dimensions_and_marginals():
    # property 5: Phi_{X_j}(t) = Phi_X(t e_j), exactly for empirical laws
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5000, 3)) @ np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 2.0]])
    t = np.linspace(-3, 3, 7)
    for j in range(3):
        pts = np.zeros((len(t), 3))
        pts[:, j] = t
        assert np.allclose(ecf(X, pts), ecf(X[:, j], t))
    assert ecf(X, np.zeros((2, 2, 3))).shape == (2, 2)
    with pytest.raises(ValueError):
        ecf(X, np.zeros((4, 2)))


def test_affine_maps_property_4():
    # Phi_{AX+b}(t) = e^{i t^T b} Phi_X(A^T t), exactly for empirical laws
    rng = np.random.default_rng(3)
    X = rng.exponential(1.0, (4000, 3))
    A = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, 1.0]])
    b = np.array([0.7, -1.2])
    Y = X @ A.T + b
    pts = rng.normal(size=(20, 2))
    assert np.allclose(ecf(Y, pts), np.exp(1j * pts @ b) * ecf(X, pts @ A))


def test_independence_property_3_and_criterion_8():
    rng = np.random.default_rng(2026)
    n = 100_000
    x, y = rng.uniform(-1, 1, n), rng.exponential(1.0, n)
    t = np.linspace(-5, 5, 41)
    e = ecf(x + y, t)
    assert np.all(np.abs(e - cf_uniform(t) * cf_exponential(t)) < 5 * ecf_standard_error(x + y, t) + 1e-3)
    s = np.linspace(-3, 3, 13)
    d_indep = cf_independence_defect(x, y, s, s)
    d_dep = cf_independence_defect(x, x ** 2, s, s)
    assert d_indep < 0.02
    assert d_dep > 0.1
    # the population defect of (X, X^2) at (s, t) = (2, 3): E[e^{i(2X+3X^2)}] - phi_X(2) phi_{X^2}(3)
    j = integrate.quad(lambda u: 0.5 * math.cos(2 * u + 3 * u * u), -1, 1)[0] \
        + 1j * integrate.quad(lambda u: 0.5 * math.sin(2 * u + 3 * u * u), -1, 1)[0]
    p2 = integrate.quad(lambda u: 0.5 * math.cos(3 * u * u), -1, 1)[0] \
        + 1j * integrate.quad(lambda u: 0.5 * math.sin(3 * u * u), -1, 1)[0]
    assert abs(j - cf_uniform(2.0) * p2) > 0.1
    with pytest.raises(ValueError):
        cf_independence_defect(x, y[:-1], s, s)


# ------------------------------------------------------------------- the worked exercises


def test_gaussian_ode_worked_exercise():
    # Phi_N' = -t Phi_N, with Phi_N computed by quadrature from the density, not from the formula
    phi = lambda t: cf_from_density(LAWS["normal"].density, t, -12, 12, 400)
    for t0 in (-2.0, -0.3, 0.0, 1.1, 2.5):
        assert cf_derivative(phi, 1, t0, h=1e-2) == pytest.approx(-t0 * phi(np.array(t0)), abs=1e-9)
    # b) and c): affine map and sums of independent Gaussians
    t = np.linspace(-3, 3, 31)
    assert np.allclose(np.exp(1j * t * 1.5) * cf_normal(2.0 * t), cf_normal(t, 1.5, 2.0))
    assert np.allclose(cf_normal(t, 1, 2) * cf_normal(t, -3, 0.5), cf_normal(t, -2, math.sqrt(4.25)))


def test_one_over_sqrt_worked_exercise():
    phi = cf_product_normals
    m = moments_from_cf(phi).real
    assert m[0] == pytest.approx(0, abs=1e-8) and m[1] == pytest.approx(1, abs=1e-8)        # E X, var X
    assert m[3] == pytest.approx(9, abs=1e-5)                                                 # E X^4
    m2 = moments_from_cf(lambda t: phi(t) ** 2).real
    assert m2[3] == pytest.approx(24, abs=1e-5)                                               # E (X+Y)^4
    # direct computation (Exercise): E(X+Y)^4 = 2 E X^4 + 6 (E X^2)^2
    assert 2 * m[3] + 6 * m[1] ** 2 == pytest.approx(24, abs=1e-5)
    # Maclaurin coefficients of 1/(1+t^2): Phi^{(2k)}(0) = (2k)! (-1)^k
    for k in (1, 2, 3):
        assert cf_derivative(lambda t: phi(t) ** 2, 2 * k, h=0.05, p=6).real == pytest.approx(
            math.factorial(2 * k) * (-1) ** k, rel=1e-4)
    # c) T = X + Y + Z: integrable cf, density |x| K_1(|x|)/pi, symmetric, P(T >= 0) = 1/2, E T^4 = 45
    phiT = lambda t: phi(t) ** 3
    assert integrate.quad(lambda t: (1 + t * t) ** -1.5, -np.inf, np.inf)[0] == pytest.approx(2.0)
    x = np.linspace(-6, 6, 25)
    assert np.allclose(invert_density(phiT, x, L=400, n=100_000), density_product_normals(x, 3), atol=5e-6)
    assert invert_density(phiT, 0.0, L=400, n=100_000)[0] == pytest.approx(1 / math.pi, abs=5e-6)
    pos = integrate.quad(lambda u: density_product_normals(u, 3), 0, np.inf)[0]
    assert pos == pytest.approx(0.5, abs=1e-10)
    assert moments_from_cf(phiT).real[3] == pytest.approx(45, abs=1e-4)


def test_density_2x_worked_exercise_two_methods():
    t = np.concatenate([np.linspace(-20, -0.2, 50), np.linspace(0.2, 20, 50)])
    closed = 2 * (np.exp(1j * t) * (1 - 1j * t) - 1) / t ** 2
    assert np.allclose(cf_density_2x(t), closed, atol=1e-13)
    # first method: integration by parts = quadrature of 2x e^{itx}
    assert np.allclose(cf_from_density(lambda x: 2 * x, t, 0, 1, 80), closed, atol=1e-12)
    # second method: -2i Phi_Y'(t) with Y ~ U(0, 1)
    phiY = lambda s: cf_uniform(s, 0.0, 1.0)
    second = np.array([-2j * cf_derivative(phiY, 1, s, h=1e-3, p=4) for s in t])
    assert np.allclose(second, closed, atol=1e-9)
    # continuity at 0 and the moments 2/(k+2)
    assert cf_density_2x(np.array([0.0]))[0] == 1
    assert np.allclose(cf_density_2x(np.array([0.099999, 0.1])), cf_from_density(lambda x: 2 * x, np.array([0.099999, 0.1]), 0, 1, 40))
    assert np.allclose(moments_from_cf(cf_density_2x).real, [2 / 3, 1 / 2, 2 / 5, 1 / 3], atol=1e-6)


def test_product_of_gaussians_has_cf_one_over_sqrt():
    # E[e^{itZ1Z2}] = E[e^{-t^2 Z2^2/2}] = (1+t^2)^{-1/2}: Gauss-Hermite in z2
    t = np.linspace(-6, 6, 13)
    inner = [integrate.quad(lambda z: math.exp(-0.5 * (s * z) ** 2 - 0.5 * z * z), -np.inf, np.inf)[0]
             / math.sqrt(2 * math.pi) for s in t]
    assert np.allclose(inner, cf_product_normals(t), atol=1e-9)
    # density K_0(|x|)/pi integrates to 1, is unbounded at 0; the Laplace case n = 2
    assert 2 * integrate.quad(lambda u: special.k0(u) / math.pi, 0, np.inf)[0] == pytest.approx(1.0)
    assert np.isinf(density_product_normals(0.0))
    assert np.allclose(density_product_normals(np.array([0.0, 1.3]), 2), 0.5 * np.exp(-np.array([0.0, 1.3])))
    rng = np.random.default_rng(2026)
    s = rng.standard_normal(200_000) * rng.standard_normal(200_000)
    assert np.mean(s ** 4) == pytest.approx(9, rel=0.05)
    # P(|Z1 Z2| <= a) from the density against the sample, within 5 standard errors
    for a in (0.1, 0.5, 1.0, 2.0):
        exact = 2 * integrate.quad(lambda v: special.k0(v) / math.pi, 0, a)[0]
        assert abs(np.mean(np.abs(s) <= a) - exact) < 5 * math.sqrt(exact * (1 - exact) / len(s))


# --------------------------------------------------------------------------------- inversion


@pytest.mark.parametrize("name,L,n,tol", [("normal", 40, 20_000, 1e-12), ("cauchy", 60, 200_000, 1e-8),
                                          ("laplace", 2000, 400_000, 2e-4)])
def test_inversion_of_integrable_cfs(name, L, n, tol):
    law = LAWS[name]
    x = np.linspace(-5, 5, 21)
    f = invert_density(law.cf, x, L=L, n=n)
    assert np.allclose(f, law.density(x), atol=tol)
    # the truncation error is at most (1/pi) int_L^infinity |phi|; the rest is the aliasing of the
    # trapezoidal rule, which replaces f by its periodisation with period 2 pi n / L
    tail = integrate.quad(lambda t: abs(law.cf(np.array(t))), L, np.inf)[0] / math.pi
    assert np.max(np.abs(f - law.density(x))) <= tail + 1e-8


def test_sums_of_uniforms_by_inversion():
    # (sin t / t)^n is integrable for n >= 2: the Irwin-Hall density, centred
    x = np.linspace(-2.5, 2.5, 11)
    f3 = invert_density(lambda t: cf_uniform(t) ** 3, x, L=400, n=200_000)
    exact = np.where(np.abs(x) <= 1, (3 - x ** 2) / 8, np.where(np.abs(x) <= 3, (3 - np.abs(x)) ** 2 / 16, 0))
    assert np.allclose(f3, exact, atol=1e-5)


def test_non_integrable_cfs():
    # Exp(1): the truncated inversion integral tends to 1/2 at the jump, and to e^{-x}, 0 elsewhere
    x = np.array([-1.0, 0.0, 1.0])
    f = invert_density(cf_exponential, x, L=2000, n=400_000)
    assert np.allclose(f, [0, 0.5, math.exp(-1)], atol=2e-3)
    # |phi| = (1+t^2)^{-1/2} is not integrable: int_0^T grows like log T
    assert integrate.quad(lambda t: abs(cf_exponential(t)), 0, 1e4, limit=200)[0] > math.log(1e4)
    # interval formula works for any law; atoms are recovered by the averaged formula
    assert levy_inversion(cf_uniform, np.array([-0.5]), 0.5, L=2000, n=400_000)[0] == pytest.approx(0.5, abs=1e-3)
    assert levy_inversion(cf_uniform, np.array([-1.0]), 0.5, L=2000, n=400_000)[0] == pytest.approx(0.75, abs=1e-3)
    p = atom_mass(lambda t: cf_poisson(t, 2.0), np.arange(5), L=1000, n=200_000)
    assert np.allclose(p, stats.poisson.pmf(np.arange(5), 2.0), atol=2e-3)
    b = atom_mass(lambda t: cf_binomial(t, 1, 0.3), np.array([0.0, 0.5, 1.0]), L=1000, n=200_000)
    assert np.allclose(b, [0.7, 0.0, 0.3], atol=2e-3)


# ----------------------------------------------------------------- derivatives and moments


def test_fd_weights_are_exact_on_polynomials():
    nodes = np.arange(-3, 4, dtype=float)
    for k in range(0, 5):
        w = fd_weights(k, nodes)
        for deg in range(0, 7):
            exact = math.factorial(deg) if deg == k else 0.0          # d^k/dx^k x^deg at 0
            assert w @ nodes ** deg == pytest.approx(exact, abs=1e-9)
    assert np.allclose(fd_weights(2, [-1.0, 0.0, 1.0]), [1, -2, 1])
    with pytest.raises(ValueError):
        fd_weights(3, [0.0, 1.0])


@pytest.mark.parametrize("name", ["normal", "uniform", "exponential", "laplace", "product",
                                  "three products", "density 2x", "poisson"])
def test_moments_from_derivatives_at_0(name):
    law = LAWS[name]
    m = moments_from_cf(law.cf, 4)
    exact = np.array([law.moments[k] for k in range(1, 5)])
    assert np.allclose(m.real, exact, atol=1e-5 * max(1, abs(exact).max()))
    assert np.max(np.abs(m.imag)) < 1e-6


def test_moments_of_the_empirical_law():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 3000)
    # the error is rounding, of order 1e-16 / h^4 times the size of the weights
    assert np.allclose(moments_from_cf(lambda t: ecf(x, t), 4, h=1e-2).real, sample_moments(x), atol=1e-5)


def test_cauchy_cf_is_not_differentiable_at_0():
    # one-sided difference quotients of e^{-|t|} at 0 are -1 and +1
    h = 1e-6
    assert (cf_cauchy(h) - 1) / h == pytest.approx(-1, abs=1e-5)
    assert (cf_cauchy(-h) - 1) / (-h) == pytest.approx(1, abs=1e-5)


# ----------------------------------------------------------------------------- exercises


def test_ross_style_exercises():
    t = np.linspace(-3, 3, 61)
    # chi^2(n) via Z^2: (1 - 2it)^{-1/2} = Gamma(1/2, 1/2); by Gauss-Hermite quadrature
    s3 = np.array([-3.0, -0.5, 1.0, 2.5])
    phi = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    q = np.array([integrate.quad(lambda z: math.cos(s * z * z) * phi(z), -np.inf, np.inf, limit=400)[0]
                  + 1j * integrate.quad(lambda z: math.sin(s * z * z) * phi(z), -np.inf, np.inf, limit=400)[0]
                  for s in s3])
    assert np.allclose(q, (1 - 2j * s3) ** -0.5, atol=1e-7)
    assert np.allclose(((1 - 2j * t) ** -0.5) ** 5, cf_gamma(t, 2.5, 0.5))
    # sums of independent Poisson and binomial laws
    assert np.allclose(cf_poisson(t, 1.3) * cf_poisson(t, 2.2), cf_poisson(t, 3.5))
    assert np.allclose(cf_binomial(t, 4, 0.3) * cf_binomial(t, 6, 0.3), cf_binomial(t, 10, 0.3))
    # cos(t)^n: sum of n Rademacher signs, E S^2 = n, E S^4 = 3n^2 - 2n
    for n in (1, 3, 10):
        m = moments_from_cf(lambda s: np.cos(s).astype(complex) ** n, 4, h=1e-2).real
        assert m[1] == pytest.approx(n, abs=1e-6) and m[3] == pytest.approx(3 * n * n - 2 * n, rel=1e-6)
    # geometric sum of Exp(lam): E[e^{itS}] = G_N(phi_X(t)) = Exp(p lam)
    p, lam = 0.25, 2.0
    G = lambda s: p * s / (1 - (1 - p) * s)
    assert np.allclose(G(cf_exponential(t, lam)), cf_exponential(t, p * lam))
    # moments of random sums: E S = E N E X, var S = E N var X + var N (E X)^2
    phiS = lambda s: G(cf_poisson(s, 1.5))          # N geometric, X Poisson(1.5)
    m = moments_from_cf(phiS, 2).real
    EN, VN, EX, VX = 1 / p, (1 - p) / p ** 2, 1.5, 1.5
    assert m[0] == pytest.approx(EN * EX, rel=1e-7)
    assert m[1] - m[0] ** 2 == pytest.approx(EN * VX + VN * EX ** 2, rel=1e-6)


def test_cauchy_via_laplace_inversion_and_stability():
    # inversion formula applied to the Laplace cf gives the Cauchy cf (Exercise)
    t = np.array([0.0, 0.5, 2.0])
    vals = [2 * integrate.quad(lambda x: 1 / (math.pi * (1 + x * x)), 0, np.inf, weight="cos", wvar=s)[0]
            if s else 1.0 for s in t]
    assert np.allclose(vals, np.exp(-np.abs(t)), atol=1e-8)
    # and the Laplace density is recovered from 1/(1+t^2) by the inversion formula
    x = np.array([0.3, 1.7])
    f = [integrate.quad(lambda s: 1 / (1 + s * s), 0, np.inf, weight="cos", wvar=u)[0] / math.pi for u in x]
    assert np.allclose(f, 0.5 * np.exp(-x), atol=1e-8)
    # the mean of n i.i.d. Cauchy variables is Cauchy: phi(t/n)^n = phi(t)
    assert np.allclose(cf_cauchy(t / 7) ** 7, cf_cauchy(t))
    assert np.allclose(cf_laplace(t), 1 / (1 + t * t))
