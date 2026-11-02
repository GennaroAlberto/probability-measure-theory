"""Figures for Chapter 6 (L^p spaces, laws and densities)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probnotes.laws import MixedLaw, poisson_law, power_lp_norm
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def ess_sup(out):
    """The example of the notes (p. 80) and p -> ||f||_p for three functions on [0, 1]."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1, 1.15]})
    ax.axhspan(-1, 1, color=COBALT, alpha=0.12, lw=0, label=r"$|f|\leq\|f\|_\infty=1$")
    ax.plot([0, 1], [1, 1], color=COBALT, lw=2.0, label=r"$f=1$ off $\{1/n\}$")
    n = np.arange(2, 7)
    ax.vlines(1 / n, 1, n, color=GOLD, lw=0.8, ls=":")
    ax.plot(1 / n, n, "o", color=GOLD, ms=3.5, label=r"$f(1/n)=n$")
    ax.plot(1 / n, np.ones_like(n), "o", mfc="white", mec=COBALT, ms=3.5)
    ax.set_xticks([0, 1 / 3, 1 / 2, 1], ["0", "1/3", "1/2", "1"])
    ax.set(xlim=(-0.03, 1.05), ylim=(-1.1, 6.6), xlabel="$x$", title="essentially bounded, unbounded")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(False)

    p = np.linspace(1, 40, 800)
    ax2.plot(p, 2 / (p + 1) ** (1 / p), color=COBALT, label=r"$f(x)=2x$: $\|f\|_p\uparrow\|f\|_\infty=2$")
    pp = p[p < 3]
    ax2.plot(pp, [power_lp_norm(1 / 3, r, 0, 1) for r in pp], color=GOLD,
             label=r"$f(x)=x^{-1/3}$: $\|f\|_p=\infty$ for $p\geq3$")
    ax2.plot(p, np.ones_like(p), color=GREEN, ls="--", label="the example on the left")
    ax2.axvline(3, color=GREY, lw=0.8, ls=":")
    ax2.axhline(2, color=COBALT, lw=0.6, ls=":")
    ax2.set(xscale="log", ylim=(0, 3.6), xlabel="$p$", title=r"$\|f\|_p$ on $([0,1],\mathcal{B},m)$")
    ax2.set_xticks([1, 3, 10, 40], ["1", "3", "10", "40"])
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch06-ess-sup.pdf")
    plt.close(fig)


def steps(ax, x, F, color, lw=1.8):
    """Draw a right-continuous step function without vertical segments, with jump markers."""
    for a, b, v in zip(x[:-1], x[1:], F[:-1]):
        ax.plot([a, b], [v, v], color=color, lw=lw)


def laws(out):
    """A discrete, an absolutely continuous and a mixed law: their masses or densities, and F_X."""
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.9), sharex="col",
                             gridspec_kw={"height_ratios": [1, 1.25]})
    # discrete: Poisson(3)
    P = poisson_law(3.0)
    k, pk = P.support[:11], P.masses[:11]
    axes[0, 0].vlines(k, 0, pk, color=COBALT, lw=2)
    axes[0, 0].plot(k, pk, "o", color=COBALT, ms=3)
    axes[0, 0].set_title(r"discrete: $\mathcal{P}(3)$, masses $p_i$")
    xs = np.concatenate([[-1.0], k, [11.0]])
    steps(axes[1, 0], xs, np.concatenate([[0.0], P.cdf(k), [1.0]]), COBALT)
    axes[1, 0].plot(k, P.cdf(k), "o", color=COBALT, ms=3)
    axes[1, 0].plot(k, P.cdf_left(k), "o", mfc="white", mec=COBALT, ms=3)
    axes[1, 0].set(xlim=(-1, 10.5), xlabel="$x$", ylabel="$F_X(x)$")
    # absolutely continuous: Exp(1); F is not differentiable at 0
    x = np.linspace(-1, 5, 1201)
    f = np.where(x > 0, np.exp(-np.clip(x, 0, None)), 0.0)
    axes[0, 1].plot(x[x <= 0], f[x <= 0], color=GOLD)
    axes[0, 1].plot(x[x > 0], f[x > 0], color=GOLD)
    axes[0, 1].set_title(r"abs. continuous: density $f_X$")
    axes[1, 1].plot(x, np.where(x > 0, 1 - np.exp(-np.clip(x, 0, None)), 0.0), color=GOLD)
    axes[1, 1].annotate(r"$F_X'(0)$ does not exist", xy=(0, 0), xytext=(0.9, 0.15), fontsize=7,
                        color=GREY, arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6))
    axes[1, 1].set(xlim=(-1, 5), xlabel="$x$")
    # mixed: the insurance payout of the exercises, in thousands
    lam, d, u = 1.0, 0.5, 2.5
    p0, pu = 1 - math.exp(-lam * d), math.exp(-lam * (d + u))
    dens = lambda y: np.where((y > 0) & (y < u), lam * np.exp(-lam * (y + d)), 0.0)
    Y = MixedLaw([0.0, u], [p0, pu], dens, 0.0, u)
    y = np.linspace(0.0005, u - 0.0005, 400)
    axes[0, 2].plot(y, dens(y), color=GREEN, label="density part")
    axes[0, 2].vlines([0, u], 0, [p0, pu], color=INK, lw=2)
    axes[0, 2].plot([0, u], [p0, pu], "o", color=INK, ms=3, label="atoms")
    axes[0, 2].set_title("mixed: atoms and a density")
    axes[0, 2].legend(fontsize=6.5, loc="upper right")
    yy = np.linspace(0, u, 300)
    Fy = Y.cdf(yy)
    axes[1, 2].plot([-0.5, 0], [0, 0], color=GREEN)
    axes[1, 2].plot(yy[:-1], Fy[:-1], color=GREEN)
    axes[1, 2].plot([u, 3.2], [1, 1], color=GREEN)
    axes[1, 2].plot([0, u], [p0, 1], "o", color=GREEN, ms=3)
    axes[1, 2].plot([0, u], [0, 1 - pu], "o", mfc="white", mec=GREEN, ms=3)
    axes[1, 2].set(xlim=(-0.5, 3.2), xlabel="$y$")
    for ax in axes[1]:
        ax.set_ylim(-0.04, 1.06)
    for ax in axes.ravel():
        ax.grid(False)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(out / "ch06-laws.pdf")
    plt.close(fig)


def main(out):
    use_style()
    ess_sup(out)
    laws(out)


if __name__ == "__main__":
    import pathlib

    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
