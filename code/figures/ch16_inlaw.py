"""Figures for Chapter 16 (convergence in law)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from probnotes.inlaw import (binomial_poisson_distances, kolmogorov_distance_sample, le_cam_bound,
                             sample_scaled_max, scaled_max_cdf, scaled_max_kolmogorov)
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def _step(ax, jump, closed_right, color, label):
    """Draw 0 before ``jump`` and 1 after it, with the value at ``jump`` marked by a filled dot."""
    ax.hlines(0, -1, jump, color=color, lw=1.8, label=label)
    ax.hlines(1, jump, 1, color=color, lw=1.8)
    if closed_right:                 # F(jump) = 1: filled dot above, open dot below
        ax.plot([jump], [1], "o", color=color, ms=5)
        ax.plot([jump], [0], "o", mfc="white", mec=color, ms=5)
    else:                            # G(jump) = 0
        ax.plot([jump], [0], "o", color=color, ms=5)
        ax.plot([jump], [1], "o", mfc="white", mec=color, ms=5)


def one_over_n(out, n=4):
    """The three distribution functions of the example X_n = 1/n, as drawn in the notes."""
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1), sharey=True)
    _step(axes[0], 1 / n, True, COBALT, rf"$F_n$, $n={n}$")
    axes[0].vlines(1 / n, 0, 1, color=GREY, lw=0.6, ls=":")
    axes[0].set_xticks([-1, 0, 1 / n, 1], ["$-1$", "$0$", rf"$\frac{{1}}{{{n}}}$", "$1$"])
    _step(axes[1], 0.0, True, GOLD, r"$F$, law of $0$")
    _step(axes[2], 0.0, False, GREEN, r"$G=\lim_n F_n$")
    for ax in axes:
        ax.set(xlim=(-1, 1), ylim=(-0.15, 1.25))
        ax.set_yticks([0, 1])
        ax.legend(loc="upper left", fontsize=7, handlelength=1.2)
    for ax in axes[1:]:
        ax.set_xticks([-1, 0, 1])
    fig.tight_layout()
    fig.savefig(out / "ch16-one-over-n.pdf")
    plt.close(fig)


def binomial_poisson(out, lam=2.0, ns=(3, 6, 20)):
    """Discrete densities of B(n, lam/n) against P(lam), and the two distances as functions of n."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.9))
    k = np.arange(0, 9)
    a1.bar(k, stats.poisson.pmf(k, lam), width=0.75, color=GREY, alpha=0.35,
           label=rf"$\mathcal{{P}}({lam:g})$")
    for (n, c, mk), dx in zip(zip(ns, (COBALT, GOLD, GREEN), ("o", "s", "^")), (-0.22, 0.0, 0.22)):
        kk = k[k <= n]
        a1.plot(kk + dx, stats.binom.pmf(kk, n, lam / n), mk, color=c, ms=4,
                label=rf"$B({n},\frac{{{lam:g}}}{{{n}}})$")
    a1.set(xlabel="$k$", title="discrete densities")
    a1.legend(fontsize=7)
    n = np.unique(np.round(np.logspace(np.log10(2), 4, 60)).astype(int))
    tv, ks = binomial_poisson_distances(n, lam)
    a2.loglog(n, tv, color=COBALT, label="total variation")
    a2.loglog(n, ks, color=GOLD, label="Kolmogorov")
    a2.loglog(n, le_cam_bound(n, lam), color=GREEN, ls="--", lw=1.2, label=r"$\lambda^2/n$")
    a2.set(xlabel="$n$", title=rf"distance from $\mathcal{{P}}({lam:g})$")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch16-binomial-poisson.pdf")
    plt.close(fig)


def maximum(out, size=10_000, seed=2026):
    """Distribution functions of Z_n = n(1 - max U_i) and their distance to Exp(1)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.9))
    x = np.linspace(0, 5, 801)
    for n, c in zip((1, 2, 5), (COBALT, GOLD, GREEN)):
        a1.plot(x, scaled_max_cdf(n, x), color=c, label=f"$F_{{{n}}}$")
    a1.plot(x, 1 - np.exp(-x), color=INK, ls="--", lw=1.2, label=r"$1-e^{-x}$")
    a1.set(xlabel="$x$", ylim=(0, 1.03), title=r"$Z_n=n(1-\max_{i\leq n}X_i)$")
    a1.legend(fontsize=7, loc="lower right")
    n = np.unique(np.round(np.logspace(0, 3, 50)).astype(int))
    a2.loglog(n, scaled_max_kolmogorov(n), color=COBALT, label="exact")
    a2.loglog(n, 2 * np.exp(-2) / n, color=GREY, ls=":", lw=1.2, label=r"$2e^{-2}/n$")
    rng = np.random.default_rng(seed)
    expcdf = lambda t: np.where(t < 0, 0.0, -np.expm1(-np.maximum(t, 0.0)))
    pts = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    emp = [kolmogorov_distance_sample(sample_scaled_max(m, size, seed=rng), expcdf) for m in pts]
    a2.loglog(pts, emp, "o", color=GOLD, ms=4, label=f"empirical, {size} draws")
    a2.set(xlabel="$n$", title=r"$\sup_x|F_n(x)-(1-e^{-x})|$")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch16-maximum.pdf")
    plt.close(fig)


def main(out):
    use_style()
    one_over_n(out)
    binomial_poisson(out)
    maximum(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
