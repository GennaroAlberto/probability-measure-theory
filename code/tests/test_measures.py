"""Checks of the Chapter 2 code against the notes (examples, Properties 1-7, Borel--Cantelli)
and against the numbers quoted in the exercises."""
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from probnotes.measures import (
    INF,
    DiscreteMeasure,
    FinCofin,
    bonferroni_sums,
    check_additivity,
    counting_measure,
    counts_at,
    derangements,
    dirac,
    expected_count,
    harmonic_probs,
    inclusion_exclusion,
    intersection_sums,
    liminf_limsup_periodic,
    matching_distribution,
    mu_two_power,
    mu_zero_one,
    no_match_probability,
    notes_example_sequence,
    occurrence_times,
    occurrence_times_exact,
    prob_no_event,
    prob_occurrence_between,
    simulate_events,
    square_probs,
    tail_inf,
    tail_intersections,
    tail_sup,
    tail_unions,
)


def random_fincofin(rng, size=8):
    pts = set(rng.choice(size, size=rng.integers(0, size), replace=False).tolist())
    return FinCofin(pts, cofinite=bool(rng.integers(2)))


# ---------- the worked exercise: additive but not sigma-additive -------------------------
def test_two_power_values_from_the_notes():
    for n in range(10):
        assert mu_two_power({n}) == 2.0 ** (-n)
    assert mu_two_power(FinCofin.naturals()) == INF
    # partial sums of sum_{n in N} 2^{-n}: mu({0,...,N-1}) = 2 - 2^{1-N} -> 2, while mu(N) = +inf
    for N in (1, 5, 30):
        assert mu_two_power(range(N)) == pytest.approx(2 - 2.0 ** (1 - N))
    assert mu_two_power(range(60)) == pytest.approx(2.0)


@pytest.mark.parametrize("mu", [mu_two_power, mu_zero_one])
def test_additive_on_the_cofinite_algebra(mu):
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(3000):
        A, B = random_fincofin(rng), random_fincofin(rng)
        if A.isdisjoint(B):
            checked += 1
            assert mu(A | B) == mu(A) + mu(B)
    assert checked > 300
    assert mu(FinCofin()) == 0


def test_zero_one_not_sigma_additive_on_countable_N():
    # N = U {n}: each singleton has measure 0, N has measure 1 (the hint: take X countable)
    assert mu_zero_one(FinCofin.naturals()) == 1
    assert all(mu_zero_one({n}) == 0 for n in range(100))


def test_fincofin_algebra_operations():
    A, B = FinCofin({1, 2}), FinCofin.tail(3)
    assert (A | B) == FinCofin({0}, cofinite=True)
    assert (FinCofin.tail(2) & FinCofin({0, 1, 2, 5})) == FinCofin({2, 5})
    assert (B - A) == B and A.complement().complement() == A
    assert 7 in B and 2 not in B and 2 in A


# ---------- examples of measures (counting, Dirac, discrete; restriction) ----------------
def test_examples_are_additive_on_a_finite_power_set():
    X = range(6)
    w = {x: 0.1 * (x + 1) for x in X}
    for mu in (counting_measure(X), dirac(3), DiscreteMeasure(w)):
        assert check_additivity(mu, X) is None
    assert check_additivity(lambda A: len(A) ** 2, range(3)) is not None


def test_dirac_and_counting():
    d = dirac(4)
    assert d.is_probability and d(range(10)) == 1 and d(set(range(10)) - {4}) == 0
    assert d(FinCofin.tail(3)) == 1 and d(FinCofin.tail(5)) == 0
    c = counting_measure()
    assert c({1, 2, 3}) == 3 and c(FinCofin.tail(5)) == INF and not c.is_finite


def test_discrete_measure_and_restriction():
    f = {0: 0.2, 1: 0.3, 2: 0.5}
    mu = DiscreteMeasure(f)
    assert mu.is_probability and all(mu({i}) == f[i] for i in f)
    A = {1, 2}
    r = mu.restrict(A)
    assert r({0}) == 0 and r({0, 1, 2}) == pytest.approx(0.8)  # concentrated on A
    geo = DiscreteMeasure(lambda n: 0.5 ** (n + 1), total=1.0)
    assert geo(FinCofin.tail(3)) == pytest.approx(0.125)
    assert geo.restrict(FinCofin.tail(2))({0, 1, 5}) == pytest.approx(0.5**6)
    with pytest.raises(ValueError):
        DiscreteMeasure({0: -1.0})


# ---------- Properties 1-5 on a finite space with random weights -------------------------
@pytest.fixture
def finite_space():
    rng = np.random.default_rng(1)
    X = frozenset(range(40))
    mu = DiscreteMeasure({x: float(rng.exponential()) for x in X})
    sets = [frozenset(x for x in X if rng.random() < 0.35) for _ in range(7)]
    return X, mu, sets


def test_properties_1_to_4(finite_space):
    X, mu, sets = finite_space
    A, B = sets[0], sets[1]
    assert mu(X - A) == pytest.approx(mu(X) - mu(A))                       # Property 1
    assert mu(A - (A & B)) == pytest.approx(mu(A) - mu(A & B))             # Property 2
    assert mu(A & B) <= mu(A) + 1e-12
    assert mu(A | B) == pytest.approx(mu(A) + mu(B) - mu(A & B))           # Property 3
    assert mu(frozenset.union(*sets)) <= sum(mu(S) for S in sets) + 1e-12  # Property 4


def test_inclusion_exclusion(finite_space):
    X, mu, sets = finite_space
    for k in range(1, len(sets) + 1):
        assert inclusion_exclusion(mu, sets[:k]) == pytest.approx(mu(frozenset.union(*sets[:k])))
    A1, A2, A3 = sets[:3]  # the case n = 3 written out in the notes
    rhs = (mu(A1) + mu(A2) + mu(A3) - mu(A1 & A2) - mu(A1 & A3) - mu(A2 & A3) + mu(A1 & A2 & A3))
    assert inclusion_exclusion(mu, [A1, A2, A3]) == pytest.approx(rhs)


def test_bonferroni_alternate(finite_space):
    _, mu, sets = finite_space
    P, U = bonferroni_sums(mu, sets), mu(frozenset.union(*sets))
    assert np.all(P[0::2] >= U - 1e-9) and np.all(P[1::2] <= U + 1e-9)


def test_inclusion_exclusion_on_permutations_gives_derangements():
    n = 5
    perms = list(itertools.permutations(range(n)))
    count = DiscreteMeasure({p: 1.0 for p in perms})
    fixes = [frozenset(p for p in perms if p[i] == i) for i in range(n)]
    S = intersection_sums(count, fixes)
    assert list(S) == [math.comb(n, r) * math.factorial(n - r) for r in range(1, n + 1)]
    assert inclusion_exclusion(count, fixes) == math.factorial(n) - derangements(n)


def test_derangements_and_matching():
    assert [derangements(n) for n in range(9)] == [1, 0, 1, 2, 9, 44, 265, 1854, 14833]
    for n in range(1, 8):
        brute = sum(all(p[i] != i for i in range(n)) for p in itertools.permutations(range(n)))
        assert brute == derangements(n)
        dist = matching_distribution(n, exact=True)
        assert sum(dist) == 1 and dist[n - 1] == 0
    assert no_match_probability(5, exact=True) == Fraction(11, 30)
    assert abs(no_match_probability(20) - math.exp(-1)) < 1e-15


# ---------- continuity (Properties 6, 7) and the counterexample --------------------------
def test_continuity_from_below_and_above():
    geo = DiscreteMeasure(lambda n: 0.5 ** (n + 1), total=1.0)
    up = [geo(range(n + 1)) for n in range(40)]  # A_n = {0..n} increases to N
    assert np.all(np.diff(up) > 0) and up[-1] == pytest.approx(geo(FinCofin.naturals()))
    down = [geo(FinCofin.tail(n)) for n in range(40)]  # decreases to the empty set
    assert down == pytest.approx([0.5**n for n in range(40)])
    c = counting_measure()
    assert all(c(FinCofin.tail(n)) == INF for n in range(40))  # yet the intersection is empty


# ---------- lim inf / lim sup ------------------------------------------------------------
def test_real_examples_of_the_aside():
    _, a = notes_example_sequence(1, 50)
    assert np.all(tail_inf(a)[:-1] == -1) and np.all(tail_sup(a)[:-1] == 1)
    _, a = notes_example_sequence(2, 50)
    lo, hi = tail_inf(a), tail_sup(a)
    assert lo[:3] == pytest.approx([-1, -1 / 3, -1 / 3])   # inf over k>=1, k>=2, k>=3
    assert hi[:3] == pytest.approx([1 / 2, 1 / 2, 1 / 4])  # sup over k>=1, k>=2, k>=3
    assert np.all(np.diff(lo[:-1]) >= 0) and np.all(np.diff(hi[:-1]) <= 0)
    assert abs(lo[-3]) < 0.05 and abs(hi[-3]) < 0.05       # both tend to 0


def test_sets_liminf_subset_limsup_and_proposition():
    rng = np.random.default_rng(2)
    X = range(12)
    mu = DiscreteMeasure({x: float(rng.random()) for x in X})
    for _ in range(200):
        N = int(rng.integers(3, 9))
        p = int(rng.integers(1, N + 1))
        seq = [frozenset(x for x in X if rng.random() < 0.6) for _ in range(N)]
        lo, hi = liminf_limsup_periodic(seq, p)
        assert lo <= hi
        assert lo == frozenset.intersection(*seq[-p:]) and hi == frozenset.union(*seq[-p:])
        vals = [mu(S) for S in seq[-p:]]  # mu(A_n) is eventually periodic
        assert mu(lo) <= min(vals) + 1e-12 and max(vals) <= mu(hi) + 1e-12


def test_tail_set_operations():
    ind = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=bool)
    assert tail_intersections(ind).tolist() == [[False, False, False], [False, True, False], [False, True, True]]
    assert tail_unions(ind).tolist() == [[True, True, True], [True, True, True], [False, True, True]]


# ---------- Borel--Cantelli -------------------------------------------------------------
def test_one_minus_x_below_exp():
    x = np.linspace(0, 1, 1001)
    assert np.all(1 - x <= np.exp(-x))


def test_tail_probabilities_exact():
    for n, m in ((1, 10), (10, 100), (7, 1000)):
        assert prob_occurrence_between("harmonic", n, m) == pytest.approx(1 - prob_no_event(1 / np.arange(n + 1, m + 1)))
        assert prob_occurrence_between("square", n, m) == pytest.approx(1 - prob_no_event(1 / np.arange(n + 1, m + 1) ** 2))
    assert prob_occurrence_between("square", 99, math.inf) == pytest.approx(0.01)
    assert prob_occurrence_between("harmonic", 10, 100) == pytest.approx(0.9)


def test_simulated_counts_match_expectations():
    N = 2000
    for kind, probs in (("harmonic", harmonic_probs(N)), ("square", square_probs(N))):
        brute = simulate_events(probs, n_paths=4000, seed=3).sum(axis=1)
        exact = [len(t) for t in occurrence_times_exact(kind, N, n_paths=4000, seed=4)]
        chunked = [len(t) for t in occurrence_times(probs, n_paths=4000, seed=5, chunk=100_000)]
        target = expected_count(kind, N)
        for sample in (brute, exact, chunked):
            sd = np.std(sample) / np.sqrt(len(sample))
            assert abs(np.mean(sample) - target) < 4 * sd
    assert expected_count("square", 10**6) == pytest.approx(math.pi**2 / 6, abs=1e-6)


def test_exact_sampler_reaches_long_horizons():
    T = occurrence_times_exact("harmonic", 1e12, n_paths=5000, seed=6)
    counts = np.array([len(t) for t in T])
    assert abs(counts.mean() - (math.log(1e12) + 0.5772156649)) < 0.3
    assert all(t[0] == 1 and np.all(np.diff(t) > 0) for t in T[:50])
    # p_n = 1/n^2: no occurrence after n = 1 with probability 1/2 (exactly)
    S = occurrence_times_exact("square", 1e12, n_paths=20000, seed=7)
    assert np.mean([len(t) == 1 for t in S]) == pytest.approx(0.5, abs=0.015)
    assert counts_at(S[:3], [1, 10**12]).shape == (3, 2)


# ---------- numbers quoted in the exercises and their solutions -------------------------
def test_numbers_quoted_in_the_exercises():
    # matching problem: smallest n with |P(no match) - 1/e| < 1e-6 is 9
    err = [abs(no_match_probability(n) - math.exp(-1)) for n in range(1, 13)]
    assert err[7] == pytest.approx(2.503e-6, rel=1e-3) and err[8] == pytest.approx(2.525e-7, rel=1e-3)
    assert min(n for n in range(1, 13) if err[n - 1] < 1e-6) == 9
    assert np.max(np.abs(matching_distribution(52) - [math.exp(-1) / math.factorial(k) for k in range(53)])) < 1e-16
    # birthdays
    p = lambda n: 1 - math.prod(1 - j / 365 for j in range(n))
    assert p(22) == pytest.approx(0.475695, abs=1e-6) and p(23) == pytest.approx(0.507297, abs=1e-6)
    lower = lambda n: 1 - math.exp(-n * (n - 1) / 730)
    assert min(n for n in range(2, 60) if lower(n) > 0.5) == 23 and lower(23) == pytest.approx(0.500002, abs=1e-6)
    assert min(n for n in range(2, 60) if math.comb(n, 2) / 365 > 0.5) == 20
    S1, S2 = 253 / 365, math.comb(253, 2) / 365**2
    assert (S1, S2, S1 - S2) == pytest.approx((0.693151, 0.239279, 0.453871), abs=1e-6)
    # Borel--Cantelli: expected counts and exact late-occurrence probabilities at N = 1e5
    assert expected_count("harmonic", 10**5) == pytest.approx(12.090, abs=1e-3)
    assert expected_count("square", 10**5) == pytest.approx(1.645, abs=1e-3)
    k = np.arange(10**4 + 1, 10**5 + 1, dtype=float)
    assert 1 - prob_no_event(k**-1.0) == pytest.approx(0.9)
    assert 1 - prob_no_event(k**-1.5) == pytest.approx(0.0136, abs=1e-4)
    assert 1 - prob_no_event(k**-2.0) == pytest.approx(9.0e-5, rel=1e-2)
    assert np.sum(np.log1p(-k**-0.5)) / math.log(10) == pytest.approx(-188.3, abs=0.05)
    kk = np.arange(2, 10**5 + 1, dtype=float)
    borderline = np.sum(1 / (kk * np.log(kk)))
    assert borderline == pytest.approx(3.238, abs=1e-3)
    assert borderline - math.log(math.log(1e5)) == pytest.approx(0.795, abs=1e-3)
    assert 1 - prob_no_event(1 / (k * np.log(k))) == pytest.approx(0.200, abs=1e-3)


def test_more_numbers_quoted_in_the_solutions():
    from scipy.special import zeta
    # matching: exact distance to Poisson(1), with e^{-1} as an exact rational to 1e-190
    e1 = sum(Fraction((-1) ** r, math.factorial(r)) for r in range(120))
    def gap(n):
        d = matching_distribution(n, exact=True)
        return max(abs(d[k] - e1 / math.factorial(k)) for k in range(n + 1))
    assert float(gap(10)) == pytest.approx(1.0e-5, rel=0.02)
    assert float(gap(52)) == pytest.approx(2.2e-55, rel=0.01)
    # birthday table: exact, union bound, S1 - S2, exponential lower bound
    p = lambda n: 1 - math.prod(1 - j / 365 for j in range(n))
    S1 = lambda n: math.comb(n, 2) / 365
    B2 = lambda n: S1(n) - math.comb(math.comb(n, 2), 2) / 365**2
    E = lambda n: 1 - math.exp(-n * (n - 1) / 730)
    rows = {10: (0.1169, 0.1233, 0.1159, 0.1160), 20: (0.4114, 0.5205, 0.3858, 0.4058),
            23: (0.5073, 0.6932, 0.4539, 0.5000), 30: (0.7063, 1.1918, 0.4832, 0.6963),
            40: (0.8912, 2.1370, -0.1434, 0.8820), 60: (0.9941, 4.8493, -6.902, 0.9922)}
    for n, row in rows.items():
        assert (p(n), S1(n), B2(n), E(n)) == pytest.approx(row, abs=6e-4)
    assert min(n for n in range(2, 61) if S1(n) > 1) == 28
    assert max(range(2, 61), key=B2) == 28 and B2(28) == pytest.approx(0.5008, abs=1e-4)
    assert [n for n in range(2, 61) if B2(n) > 0.5] == [27, 28]
    assert min(n for n in range(2, 61) if B2(n) < 0) == 39
    gaps = {n: p(n) - E(n) for n in range(1, 61)}
    assert max(gaps, key=gaps.get) == 33 and gaps[33] == pytest.approx(0.0103, abs=1e-4)
    assert abs(S1(10) - p(10)) == pytest.approx(0.0063, abs=1e-4) and abs(p(15) - B2(15)) == pytest.approx(0.0062, abs=1e-4)
    # borderline p_n = 1/(n log n): E[N_n] = log log n + 0.795 + o(1)
    kk = np.arange(2, 10**5 + 1, dtype=float)
    c = np.sum(1 / (kk * np.log(kk))) - math.log(math.log(1e5))
    assert c + math.log(math.log(1e12)) == pytest.approx(4.11, abs=0.005)
    assert c + math.log(math.log(1e100)) == pytest.approx(6.23, abs=0.005)
    assert zeta(1.1) == pytest.approx(10.58, abs=0.005)
