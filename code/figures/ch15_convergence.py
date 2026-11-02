"""Figures for Chapter 15 (convergence of random variables and the laws of large numbers)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probnotes.convergence import (bernoulli_mean_tail, chebyshev_bound, deviation_probability,
                                   sample_means, tail_sup_deviation, typewriter, typewriter_interval)
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def counterexamples(out, omega=0.3, n_type=10, n_spike=8):
    """The typewriter rows A_1, ..., A_10 and the supports (0, 1/n) of the spike, as in the notes."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.4))
    for ax, n_max, title in ((a1, n_type, "Counterexample 1: typewriter"),
                             (a2, n_spike, r"Counterexample 2: $2^n\mathbf{1}_{(0,1/n)}$")):
        ax.set(xlim=(-0.02, 1.02), ylim=(n_max + 0.7, 0.3), xlabel=r"$\omega$", title=title)
        ax.set_yticks(range(1, n_max + 1), [f"$X_{{{n}}}$" for n in range(1, n_max + 1)])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.hlines(range(1, n_max + 1), 0, 1, color=GREY, lw=0.8)
        ax.axvline(omega, color=GOLD, lw=1.2, label=rf"$\omega={omega:g}$")
    n = np.arange(1, n_type + 1)
    a, b = typewriter_interval(n)
    a1.hlines(n, a, b, color=COBALT, lw=3.2, label=r"$A_n$, where $X_n=1$")
    hit = typewriter(n, omega) == 1
    a1.plot(np.full(hit.sum(), omega), n[hit], "o", color=GOLD, ms=4.5, label=r"$X_n(\omega)=1$")
    kk = 1
    while kk * (kk + 1) // 2 < n_type:                    # dotted lines between the blocks
        a1.axhline(kk * (kk + 1) // 2 + 0.5, color=GREY, lw=0.5, ls=":")
        kk += 1
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7, handlelength=1.5,
              columnspacing=1.0)
    m = np.arange(1, n_spike + 1)
    a2.hlines(m, 0, 1 / m, color=COBALT, lw=3.2, label=r"$(0,\frac{1}{n})$, where $X_n=2^n$")
    for mm in m:
        a2.text(1.04, mm, f"$2^{{{mm}}}$", va="center", fontsize=8, color=INK)
    hit2 = m * omega < 1
    a2.plot(np.full(hit2.sum(), omega), m[hit2], "o", color=GOLD, ms=4.5, label=r"$X_n(\omega)\neq0$")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7, handlelength=1.5,
              columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out / "ch15-counterexamples.pdf")
    plt.close(fig)


def lln(out, eps=0.05, n_paths=20, n_show=2000, n_big=10_000, paths_big=2000, chunk=250, seed=2026):
    """Sample means of b(1/2) variables: paths, and the probabilities of the weak and strong laws."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.3))
    m = sample_means("bernoulli", n_show, n_paths, seed=seed)
    n = np.arange(1, n_show + 1)
    for r in range(n_paths):
        a1.plot(n, m[r], color=COBALT, lw=0.6, alpha=0.55)
    a1.fill_between(n, 0.5 - eps, 0.5 + eps, color=GOLD, alpha=0.18, lw=0, label=r"$\frac{1}{2}\pm\varepsilon$")
    a1.set(xscale="log", xlim=(1, n_show), ylim=(0, 1), xlabel="$n$", title=r"paths $n\mapsto\bar X_n$")
    a1.legend(loc="upper right", fontsize=7)
    # weak law and strong law probabilities, from paths_big paths in chunks
    dev = np.zeros(n_big)
    sup = np.zeros(n_big)
    rng = np.random.default_rng(seed + 1)
    for c in range(paths_big // chunk):
        mm = sample_means("bernoulli", n_big, chunk, seed=rng.integers(2 ** 32))
        dev += chunk * deviation_probability(mm, 0.5, eps)        # |Xbar_n - 1/2| > eps, with a tolerance
        sup += (tail_sup_deviation(mm, 0.5) > eps + 1e-9).sum(axis=0)
    dev /= paths_big
    sup /= paths_big
    pts = np.unique(np.round(np.logspace(1, 4, 16)).astype(int))
    nn = np.union1d(np.round(np.logspace(1, 4, 120)).astype(int), pts)   # the exact curve passes the dots' n
    a2.plot(nn, bernoulli_mean_tail(nn, 0.5, eps), color=COBALT, lw=1.4,
            label=r"$\mathbb{P}(|\bar X_n-\frac{1}{2}|>\varepsilon)$, exact")
    a2.plot(pts, dev[pts - 1], "o", color=COBALT, ms=3, label="same, simulated")
    a2.plot(nn, sup[nn - 1], color=GOLD, lw=1.4,
            label=r"$\mathbb{P}(\max_{n\leq m\leq N}|\bar X_m-\frac{1}{2}|>\varepsilon)$")
    a2.plot(nn, chebyshev_bound(0.25, nn, eps), color=GREEN, lw=1.2, ls="--", label="Chebyshev bound")
    a2.set(xscale="log", xlim=(10, n_big), ylim=(-0.02, 1.05), xlabel="$n$",
           title=rf"$\varepsilon={eps:g}$, {paths_big} paths, $N=10^4$")
    a2.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, fontsize=6.5, handlelength=1.8,
              columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out / "ch15-lln.pdf")
    plt.close(fig)


def main(out):
    use_style()
    counterexamples(out)
    lln(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
