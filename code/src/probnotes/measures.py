"""Measures on finite and countable sets, lim inf / lim sup, and Borel--Cantelli (Chapter 2).

Conventions
-----------
* Points of a finite or countable space are hashable Python objects; on N = {0, 1, 2, ...}
  they are integers. A subset is a Python ``set``/``frozenset`` (finite) or, for subsets of
  N that are finite or cofinite, a :class:`FinCofin`.
* Measures take values in [0, +inf]; +inf is ``math.inf``.
* Sequences of events are indexed from n = 1, as in the notes: column ``j`` of an array
  of indicators is the event A_{j+1}.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable, Mapping

import numpy as np

INF = math.inf


# --------------------------------------------------------------------------------------
# Finite and cofinite subsets of N
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FinCofin:
    """A subset of N = {0, 1, 2, ...} that is finite or has finite complement.

    ``FinCofin(F)`` is the finite set F; ``FinCofin(F, cofinite=True)`` is N minus F.
    These sets form the algebra of Chapter 1 (finite or cofinite sets), which is closed
    under complements and finite unions but not under countable unions.
    """

    points: frozenset = frozenset()
    cofinite: bool = False

    def __init__(self, points: Iterable[int] = (), cofinite: bool = False):
        object.__setattr__(self, "points", frozenset(int(p) for p in points))
        object.__setattr__(self, "cofinite", bool(cofinite))
        if any(p < 0 for p in self.points):
            raise ValueError("points of N are non-negative integers")

    @classmethod
    def naturals(cls) -> "FinCofin":
        """N itself (its complement is empty)."""
        return cls((), cofinite=True)

    @classmethod
    def tail(cls, n: int) -> "FinCofin":
        """{n, n+1, n+2, ...}."""
        return cls(range(n), cofinite=True)

    @property
    def is_finite(self) -> bool:
        return not self.cofinite

    def __contains__(self, x: int) -> bool:
        return (x in self.points) != self.cofinite

    def complement(self) -> "FinCofin":
        return FinCofin(self.points, cofinite=not self.cofinite)

    def __or__(self, other: "FinCofin") -> "FinCofin":
        a, b = self, _as_fincofin(other)
        if not a.cofinite and not b.cofinite:
            return FinCofin(a.points | b.points)
        if a.cofinite and b.cofinite:
            return FinCofin(a.points & b.points, cofinite=True)
        fin, cof = (a, b) if b.cofinite else (b, a)
        return FinCofin(cof.points - fin.points, cofinite=True)

    def __and__(self, other: "FinCofin") -> "FinCofin":
        return (self.complement() | _as_fincofin(other).complement()).complement()

    def __sub__(self, other: "FinCofin") -> "FinCofin":
        return self & _as_fincofin(other).complement()

    def isdisjoint(self, other: "FinCofin") -> bool:
        inter = self & other
        return inter.is_finite and not inter.points

    def __repr__(self) -> str:
        pts = sorted(self.points)
        return f"N \\ {set(pts) or '{}'}" if self.cofinite else f"{set(pts) or '{}'}"


def _as_fincofin(A) -> FinCofin:
    return A if isinstance(A, FinCofin) else FinCofin(A)


# --------------------------------------------------------------------------------------
# Measures on finite and countable sets (Examples of measures, items 1-3)
# --------------------------------------------------------------------------------------
class DiscreteMeasure:
    """The measure mu(A) = sum_{x in A} f(x) on a finite or countable set, with f >= 0.

    ``weights`` is either a mapping point -> f(point) (finite support), or a function
    f on N = {0, 1, 2, ...}; in the second case ``total`` = sum_n f(n) (possibly ``inf``)
    must be given, so that cofinite sets can be measured as total - mu(complement).
    Every such mu is a measure on the power set (Example 3 of the notes); it is finite iff
    the total mass is finite, and a probability measure iff the total mass is 1.
    """

    def __init__(self, weights: Mapping | Callable[[int], float], total: float | None = None):
        if callable(weights):
            if total is None:
                raise ValueError("for weights given by a function, pass total = sum of all weights")
            self._dict = None
            self._f = weights
            self.total = float(total)
        else:
            d = {x: float(w) for x, w in dict(weights).items()}
            bad = [x for x, w in d.items() if not (0.0 <= w < INF)]
            if bad:
                raise ValueError(f"weights must be finite and non-negative; bad points: {bad[:5]}")
            self._dict = d
            self._f = lambda x: d.get(x, 0.0)
            self.total = math.fsum(d.values()) if total is None else float(total)

    @property
    def support(self) -> frozenset:
        """Points of positive mass (finite-support measures only)."""
        if self._dict is None:
            raise ValueError("the support of a measure given by a function is not stored")
        return frozenset(x for x, w in self._dict.items() if w > 0)

    def mass(self, x) -> float:
        """mu({x}) = f(x)."""
        return float(self._f(x))

    def __call__(self, A) -> float:
        if isinstance(A, FinCofin):
            if A.cofinite:  # total - (mass of the finite complement); inf - finite = inf
                return self.total - math.fsum(self._f(x) for x in A.points)
            A = A.points
        return math.fsum(self._f(x) for x in A)

    @property
    def is_finite(self) -> bool:
        return self.total < INF

    @property
    def is_probability(self) -> bool:
        return math.isclose(self.total, 1.0, rel_tol=0, abs_tol=1e-12)

    def restrict(self, A) -> "DiscreteMeasure":
        """The restriction mu|_A(B) = mu(B cap A); it is concentrated on A."""
        if isinstance(A, FinCofin) and A.cofinite:
            f, out = self._f, frozenset(A.points)
            return DiscreteMeasure(lambda x: 0.0 if x in out else f(x), total=self(A))
        pts = A.points if isinstance(A, FinCofin) else A
        return DiscreteMeasure({x: self._f(x) for x in pts})


def dirac(x0) -> DiscreteMeasure:
    """The Dirac measure at x0: delta(A) = 1 if x0 in A, 0 otherwise."""
    return DiscreteMeasure({x0: 1.0})


def counting_measure(points: Iterable | None = None) -> DiscreteMeasure:
    """card(A) for finite A and +inf for infinite A.

    With ``points=None`` the space is N (sets may be FinCofin); otherwise the given finite set.
    """
    if points is None:
        return DiscreteMeasure(lambda n: 1.0, total=INF)
    return DiscreteMeasure({x: 1.0 for x in points})


def mu_two_power(A) -> float:
    """The set function of the notes' worked exercise on the cofinite algebra of N:
    sum_{n in A} 2^{-n} if A is finite, +inf if A is infinite. Additive, not sigma-additive."""
    A = _as_fincofin(A)
    if A.cofinite:
        return INF
    return math.fsum(2.0 ** (-n) for n in A.points)


def mu_zero_one(A) -> float:
    """0 on finite sets and 1 on cofinite sets of N (the notes' 'further point')."""
    return 1.0 if _as_fincofin(A).cofinite else 0.0


def _equal(x: float, y: float) -> bool:
    return x == y or math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)


def check_additivity(mu: Callable, ground: Iterable):
    """Check mu(empty)=0 and mu(A u B) = mu(A) + mu(B) for all disjoint A, B of a finite set.

    On a finite set this is the same as sigma-additivity on the power set (a sequence of
    disjoint subsets has only finitely many non-empty terms). Each point goes to A, to B or
    to neither: 3^|ground| pairs. Returns None, or the first counterexample (A, B).
    """
    ground = list(ground)
    if len(ground) > 12:
        raise ValueError("at most 12 points (3^12 pairs)")
    if not _equal(mu(frozenset()), 0.0):
        return (frozenset(), frozenset())
    for labels in itertools.product((0, 1, 2), repeat=len(ground)):
        A = frozenset(x for x, l in zip(ground, labels) if l == 1)
        B = frozenset(x for x, l in zip(ground, labels) if l == 2)
        if not _equal(mu(A | B), mu(A) + mu(B)):
            return (A, B)
    return None


# --------------------------------------------------------------------------------------
# Inclusion-exclusion (Property 5) and the matching problem
# --------------------------------------------------------------------------------------
def intersection_sums(mu: Callable, sets: list) -> np.ndarray:
    """S_r = sum over 1 <= i_1 < ... < i_r <= n of mu(A_{i_1} cap ... cap A_{i_r}), r = 1..n."""
    sets = [frozenset(s) for s in sets]
    n = len(sets)
    if n > 20:
        raise ValueError("2^n intersections: keep n <= 20")
    S = np.zeros(n)
    for r in range(1, n + 1):
        S[r - 1] = math.fsum(mu(frozenset.intersection(*c)) for c in itertools.combinations(sets, r))
    return S


def bonferroni_sums(mu: Callable, sets: list) -> np.ndarray:
    """Partial sums P_k = sum_{r<=k} (-1)^{r+1} S_r, k = 1..n. P_n is the inclusion-exclusion
    formula; for a finite measure the P_k alternate around mu(union) (Bonferroni)."""
    S = intersection_sums(mu, sets)
    signs = (-1.0) ** np.arange(len(S))
    return np.cumsum(signs * S)


def inclusion_exclusion(mu: Callable, sets: list) -> float:
    """Right side of Poincare's formula; equals mu(A_1 u ... u A_n) when every mu(A_i) is finite."""
    return float(bonferroni_sums(mu, sets)[-1]) if sets else 0.0


def derangements(n: int) -> int:
    """D_n, the number of permutations of n objects with no fixed point (D_0 = 1, D_1 = 0)."""
    a, b = 1, 0  # D_0, D_1
    if n == 0:
        return 1
    for k in range(2, n + 1):
        a, b = b, (k - 1) * (a + b)
    return b


def no_match_probability(n: int, exact: bool = False):
    """P(no fixed point) for a uniform random permutation of n objects = sum_{r=0}^n (-1)^r / r!."""
    p = sum(Fraction((-1) ** r, math.factorial(r)) for r in range(n + 1))
    return p if exact else float(p)


def matching_distribution(n: int, exact: bool = False):
    """P(exactly k fixed points), k = 0..n: C(n,k) D_{n-k} / n!."""
    probs = [Fraction(math.comb(n, k) * derangements(n - k), math.factorial(n)) for k in range(n + 1)]
    return probs if exact else np.array([float(p) for p in probs])


# --------------------------------------------------------------------------------------
# lim inf and lim sup: real sequences and sequences of sets
# --------------------------------------------------------------------------------------
def tail_inf(a) -> np.ndarray:
    """inf_{k >= n} a_k over the given finite stretch a_1, ..., a_N (entry n-1 for index n).

    For an infinite sequence this is exact at index n only if the infimum of the tail is
    attained before N; for the notes' two examples it is, for every n < N.
    """
    a = np.asarray(a, dtype=float)
    return np.minimum.accumulate(a[::-1])[::-1]


def tail_sup(a) -> np.ndarray:
    """sup_{k >= n} a_k over the given finite stretch (same caveat as :func:`tail_inf`)."""
    a = np.asarray(a, dtype=float)
    return np.maximum.accumulate(a[::-1])[::-1]


def notes_example_sequence(which: int, n_max: int):
    """The two sequences of the aside on lim inf / lim sup, for n = 1..n_max:
    1: a_n = +1 (n even), -1 (n odd);   2: a_n = +1/n (n even), -1/n (n odd)."""
    n = np.arange(1, n_max + 1)
    sign = np.where(n % 2 == 0, 1.0, -1.0)
    if which == 1:
        return n, sign
    if which == 2:
        return n, sign / n
    raise ValueError("which is 1 or 2")


def tail_intersections(indicators) -> np.ndarray:
    """Row n-1 is the indicator of A_n cap A_{n+1} cap ... cap A_N; input shape (N, |X|)."""
    ind = np.asarray(indicators, dtype=bool)
    return np.logical_and.accumulate(ind[::-1], axis=0)[::-1]


def tail_unions(indicators) -> np.ndarray:
    """Row n-1 is the indicator of A_n cup A_{n+1} cup ... cup A_N; input shape (N, |X|)."""
    ind = np.asarray(indicators, dtype=bool)
    return np.logical_or.accumulate(ind[::-1], axis=0)[::-1]


def liminf_limsup_periodic(sets: list, period: int):
    """Exact lim inf and lim sup of the eventually periodic sequence of sets that starts with
    ``sets`` = [A_1, ..., A_N] and then repeats its last ``period`` sets forever.

    Computed from the definitions: lim inf = U_n (cap_{k>=n} A_k), lim sup = cap_n (U_{k>=n} A_k).
    For n <= N the sets A_k, k >= n, are A_n, ..., A_N and the periodic block, so each tail is
    a finite intersection or union; for n > N the tails repeat. Returns two frozensets.
    """
    sets = [frozenset(s) for s in sets]
    if not 1 <= period <= len(sets):
        raise ValueError("need 1 <= period <= len(sets)")
    ext = sets + sets[-period:]  # one more period covers every set that appears after index n
    tails_cap = [frozenset.intersection(*ext[n:]) for n in range(len(sets))]
    tails_cup = [frozenset.union(*ext[n:]) for n in range(len(sets))]
    return frozenset.union(*tails_cap), frozenset.intersection(*tails_cup)


# --------------------------------------------------------------------------------------
# Borel--Cantelli: independent events A_n with P(A_n) = p_n
# --------------------------------------------------------------------------------------
def harmonic_probs(n_max: int) -> np.ndarray:
    """p_n = 1/n, n = 1..n_max (sum diverges: by part 2, infinitely many A_n occur a.s.)."""
    return 1.0 / np.arange(1, n_max + 1)


def square_probs(n_max: int) -> np.ndarray:
    """p_n = 1/n^2, n = 1..n_max (sum = pi^2/6: by part 1, finitely many A_n occur a.s.)."""
    return 1.0 / np.arange(1, n_max + 1) ** 2


def simulate_events(probs, n_paths: int = 1, seed=None) -> np.ndarray:
    """Indicators of independent events: A_n occurs on path i iff U_{i,n} < p_n, U uniform.

    Returns a boolean array of shape (n_paths, N), column n-1 for A_n. Memory n_paths * N bytes.
    """
    p = np.asarray(probs, dtype=float)
    if n_paths * p.size > 200_000_000:
        raise ValueError("more than 2e8 indicators: use occurrence_times instead")
    rng = np.random.default_rng(seed)
    return rng.random((n_paths, p.size)) < p


def occurrence_times(probs, n_paths: int = 1, seed=None, chunk: int = 2_000_000) -> list:
    """Times n at which A_n occurs, one array per path, by drawing U_{i,n} < p_n in chunks.

    Same law as :func:`simulate_events` but in bounded memory, for long horizons N = len(probs).
    """
    p = np.asarray(probs, dtype=float)
    rng = np.random.default_rng(seed)
    step = max(1, chunk // max(1, n_paths))
    found = [[] for _ in range(n_paths)]
    for a in range(0, p.size, step):
        b = min(p.size, a + step)
        rows, cols = np.nonzero(rng.random((n_paths, b - a)) < p[a:b])
        for i in np.unique(rows):
            found[i].append(cols[rows == i] + a + 1)
    return [np.concatenate(t) if t else np.zeros(0, dtype=np.int64) for t in found]


def occurrence_times_exact(kind: str, horizon: float, n_paths: int = 1, seed=None) -> list:
    """Exact occurrence times n <= horizon for p_n = 1/n ('harmonic') or 1/n^2 ('square').

    No event is drawn one by one: given an occurrence at m, the next one is sampled directly.
      harmonic: P(no A_k for m < k <= j) = prod (1 - 1/k) = m/j, so next = floor(m/U) + 1;
      square:   P(no A_k for m < k <= j) = m(j+1) / ((m+1) j), which tends to m/(m+1):
                with probability m/(m+1) there is no further occurrence at all; otherwise
                next = floor(m / (U(m+1) - m)) + 1.
    A_1 always occurs (p_1 = 1). Horizons up to 1e15 (exact integers in float64).
    """
    if kind not in ("harmonic", "square"):
        raise ValueError("kind is 'harmonic' or 'square'")
    if horizon > 1e15:
        raise ValueError("horizon <= 1e15")
    rng = np.random.default_rng(seed)
    m = np.ones(n_paths)  # last occurrence time
    times = [[1] for _ in range(n_paths)]
    active = np.arange(n_paths)
    while active.size:
        u = rng.random(active.size)
        mm = m[active]
        if kind == "harmonic":
            nxt = np.floor(mm / np.where(u > 0, u, np.nextafter(0, 1))) + 1
        else:
            gap = u * (mm + 1) - mm
            with np.errstate(divide="ignore"):
                nxt = np.where(gap > 0, np.floor(mm / np.where(gap > 0, gap, 1.0)) + 1, np.inf)
        keep = nxt <= horizon
        for i, t in zip(active[keep], nxt[keep]):
            times[i].append(int(t))
        m[active[keep]] = nxt[keep]
        active = active[keep]
    return [np.array(t, dtype=np.int64) for t in times]


def counts_at(times: list, checkpoints) -> np.ndarray:
    """N_n = number of occurrences among A_1..A_n, for each path and each checkpoint n."""
    cp = np.asarray(checkpoints)
    return np.array([np.searchsorted(t, cp, side="right") for t in times])


def expected_count(kind: str, n: int) -> float:
    """E[N_n] = sum_{k<=n} p_k: the harmonic number H_n, or sum_{k<=n} 1/k^2."""
    k = np.arange(1, int(n) + 1, dtype=float)
    if kind == "harmonic":
        return float(math.fsum(1.0 / k))
    if kind == "square":
        return float(math.fsum(1.0 / k**2))
    raise ValueError("kind is 'harmonic' or 'square'")


def prob_no_event(probs) -> float:
    """P(none of the independent events occurs) = prod (1 - p_k), computed with log1p."""
    p = np.asarray(probs, dtype=float)
    if np.any(p >= 1):
        return 0.0
    return float(np.exp(np.sum(np.log1p(-p))))


def prob_occurrence_between(kind: str, n: int, m: float) -> float:
    """P(some A_k occurs with n < k <= m), exactly, for p_k = 1/k or 1/k^2 (n >= 1).

    harmonic: 1 - n/m (tends to 1 as m -> inf); square: 1 - n(m+1)/(m(n+1)) (tends to 1/(n+1)).
    """
    if kind == "harmonic":
        return 1.0 - n / m
    if kind == "square":
        return 1.0 - n / (n + 1) if math.isinf(m) else 1.0 - n * (m + 1) / (m * (n + 1))
    raise ValueError("kind is 'harmonic' or 'square'")
