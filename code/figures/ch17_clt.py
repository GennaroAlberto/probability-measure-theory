"""Figures for Chapter 17 (the central limit theorem and the delta method)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from probnotes.clt import (binomial_normal_density, binomial_tail, kolmogorov_distance, make_law,
                           normal_tail, second_order_cdf, standardised_cdf)
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def binomial(out):
    """The six histograms of B(n, p) of the notes, with the density of N(np, np(1-p))."""
    panels = [(20, 0.1, (0, 20)), (50, 0.1, (0, 20)), (100, 0.1, (0, 20)),
              (20, 0.5, (0, 20)), (50, 0.5, (15, 35)), (100, 0.5, (30, 70))]
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.6))
    for ax, (n, p, (lo, hi)) in zip(axes.ravel(), panels):
        k = np.arange(lo, hi + 1)
        bars = ax.bar(k, stats.binom.pmf(k, n, p), width=1.0, color=COBALT, alpha=0.28, edgecolor=COBALT,
                      linewidth=0.5, label=r"$\mathbb{P}(X=k)$")
        x = np.linspace(lo - 0.5, hi + 0.5, 600)
        curve, = ax.plot(x, binomial_normal_density(x, n, p), color=GOLD, lw=1.4,
                         label=r"$\mathcal{N}(np,np(1-p))$")
        ok = n * p >= 10 and n * (1 - p) >= 10
        ax.set_title(rf"$B({n},{p:g})$", fontsize=9)
        ax.text(0.97, 0.72 if (n, p) == (20, 0.5) else 0.95, f"$np={n * p:g}$\n$n(1-p)={n * (1 - p):g}$",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=INK if ok else "#E34948")
        ax.set_xlim(lo - 1, hi + 1)
        ax.tick_params(labelsize=7)
        ax.grid(False)
        if (n, p) == (20, 0.5):
            xs = np.linspace(11.5, hi + 0.5, 200)
            ax.fill_between(xs, binomial_normal_density(xs, n, p), color=GOLD, alpha=0.35, lw=0)
            lines = []
            for c, col, ls in ((11.0, COBALT, "--"), (11.5, GOLD, "-"), (12.0, GREEN, "--")):
                lines.append(ax.axvline(c, ymax=0.62, color=col, ls=ls, lw=1.0,
                                        label=rf"area beyond ${c:g}$: {float(normal_tail(11, n, p, c - 11)):.3f}"))
            ax.set_ylim(0, 0.29)
            ax.legend(handles=lines, loc="upper left", fontsize=6.2, handlelength=1.4, borderaxespad=0.2,
                      title=rf"$\mathbb{{P}}(X\geq12)={float(binomial_tail(11, n, p)):.3f}$",
                      title_fontsize=6.2)
    axes[0, 0].legend(handles=[bars, curve], loc="center right", fontsize=6.2, handlelength=1.2,
                      borderaxespad=0.2, bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    fig.savefig(out / "ch17-binomial.pdf")
    plt.close(fig)


def clt(out):
    """Left: the CDF of S_n^* for exponential summands against Phi. Right: Kolmogorov distances."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.0))
    x = np.linspace(-3, 3, 600)
    a1.plot(x, stats.norm.cdf(x), color=INK, lw=1.0, ls="--", label=r"$\Phi$")
    for n, col in ((1, GOLD), (4, GREEN), (32, COBALT)):
        a1.plot(x, standardised_cdf("exponential", n, x), color=col, lw=1.4, label=f"$n={n}$")
    a1.set(xlabel="$x$", title=r"$F_{S_n^*}$, summands $\mathrm{Exp}(1)$")
    a1.legend(loc="upper left", fontsize=7.5)
    ns = np.unique(np.round(np.logspace(0, 4, 25)).astype(int))
    for law, col, lab in ((make_law("exponential"), COBALT, r"$\mathrm{Exp}(1)$"),
                          (make_law("bernoulli", p=0.5), GOLD, r"$b(\frac{1}{2})$"),
                          (make_law("bernoulli", p=0.1), GREEN, r"$b(0.1)$")):
        d = np.array([kolmogorov_distance(law, int(n)) for n in ns])
        a2.loglog(ns, d, "o-", color=col, ms=2.5, lw=1.0, label=lab)
    a2.loglog(ns, 1 / np.sqrt(ns), color=GREY, lw=0.9, ls=":", label=r"$1/\sqrt{n}$")
    a2.set(xlabel="$n$", title=r"$\sup_x|F_{S_n^*}(x)-\Phi(x)|$")
    a2.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch17-clt.pdf")
    plt.close(fig)


def _block_density(edges_k, k, pmf, x_of_k):
    """Exact density of x(S) for an integer S, on bins that hold whole blocks of atoms."""
    mass = np.histogram(k, bins=edges_k, weights=pmf)[0]
    ex = x_of_k(edges_k)
    return ex, mass / np.diff(ex)


def delta(out, n=2000):
    """Variance stabilisation for Poisson means, and the second-order delta method (exact laws)."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.9, 2.5))
    for lam, col in ((0.5, COBALT), (4.0, GOLD)):
        S = stats.poisson(n * lam)
        k = np.arange(int(S.ppf(1e-7)), int(S.ppf(1 - 1e-7)) + 1)
        pmf = S.pmf(k)
        centre = n * lam
        for ax, x_of_k, spacing in ((a1, lambda v: (v - centre) / np.sqrt(n), 1 / np.sqrt(n)),
                                    (a2, lambda v: np.sqrt(n) * (np.sqrt(v / n) - np.sqrt(lam)),
                                     1 / (2 * np.sqrt(n * lam)))):
            c = 2 * int(round(0.08 / spacing / 2)) + 1                  # odd number of atoms per bin
            j = np.arange((k[0] - centre) // c - 1, (k[-1] - centre) // c + 2)
            ex, dens = _block_density(centre + j * c - c / 2, k, pmf, x_of_k)
            ax.stairs(dens, ex, color=col, lw=1.0, label=rf"$\lambda={lam:g}$")
    xx = np.linspace(-1.6, 1.6, 300)
    a2.plot(xx, stats.norm.pdf(xx, 0, 0.5), color=GREEN, lw=1.2, ls="--", label=r"$\mathcal{N}(0,\frac{1}{4})$")
    a1.set(title=r"$\sqrt{n}(\bar X_n-\lambda)$", xlabel="$x$", xlim=(-3.2, 3.2))
    a2.set(title=r"$\sqrt{n}(\sqrt{\bar X_n}-\sqrt{\lambda})$", xlabel="$x$", xlim=(-1.6, 1.6))
    a1.legend(fontsize=6.5, loc="upper right")
    a2.legend(fontsize=6.5, loc="upper right")
    w = np.linspace(-1.5, 0.05, 600)
    for nn, col in ((10, GOLD), (100, COBALT)):
        kk = np.arange(nn + 1)
        val = -(kk - nn / 2) ** 2 / nn                       # n(Xbar(1 - Xbar) - 1/4) = -(S_n - n/2)^2 / n
        cdf = np.array([stats.binom.pmf(kk, nn, 0.5)[val <= v].sum() for v in w])
        a3.plot(w, cdf, color=col, lw=1.0, drawstyle="steps-post", label=f"$n={nn}$")
    a3.plot(w, second_order_cdf(w, 0.25, -2.0), color=GREEN, lw=1.2, ls="--",
            label=r"$-\frac{1}{4}\chi^2(1)$")
    a3.set(title=r"$n(\bar X_n(1-\bar X_n)-\frac{1}{4})$, CDF", xlabel="$x$", ylim=(0, 1.03))
    a3.legend(fontsize=6.5, loc="upper left")
    for ax in (a1, a2, a3):
        ax.tick_params(labelsize=7)
        ax.title.set_fontsize(8.5)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(out / "ch17-delta.pdf")
    plt.close(fig)


def main(out):
    use_style()
    binomial(out)
    clt(out)
    delta(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
