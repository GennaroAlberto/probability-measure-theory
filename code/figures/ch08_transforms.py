"""Figures for Chapter 8 (multinomial and Gamma laws, transformations of a random variable)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy import stats

from probnotes.style import COBALT, GOLD, GREY, INK, PALETTE, use_style
from probnotes.transforms import (annulus_marginal_density, erlang_cdf, gamma_cdf, gamma_pdf, hazard_rate,
                                  poisson_arrivals, poisson_path, sample_annulus, square_branches,
                                  transformed_density)

GREEN = PALETTE[2]
PURPLE = PALETTE[3]


def gamma_laws(out):
    """Gamma densities for the four regimes of the notes, and their hazard rates."""
    lam = 1.0
    cases = [(0.5, COBALT, r"$\alpha=\frac{1}{2}$"), (1.0, GOLD, r"$\alpha=1$"),
             (1.5, GREEN, r"$\alpha=\frac{3}{2}$"), (3.0, PURPLE, r"$\alpha=3$")]
    x = np.linspace(1e-4, 8, 1600)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for alpha, col, lab in cases:
        a1.plot(x, gamma_pdf(x, alpha, lam), color=col, label=lab)
        a2.plot(x, hazard_rate(x, alpha, lam), color=col, label=lab)
    a1.plot([0], [lam], "o", color=GOLD, ms=4)
    a1.set(xlim=(0, 8), ylim=(0, 1.3), xlabel="$x$", title=r"densities of $\Gamma(\alpha,1)$")
    a1.legend(fontsize=8)
    a2.axhline(lam, color=GREY, lw=0.8, ls=":")
    a2.set(xlim=(0, 8), ylim=(0, 2.4), xlabel="$x$", title=r"hazard rates $f_X(x)\,/\,P(X>x)$")
    a2.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch08-gamma.pdf")
    plt.close(fig)


def poisson_process(out):
    """One path of a Poisson process with its arrival times, and the law of the third arrival."""
    lam, horizon, n = 2.0, 5.0, 3
    times = poisson_path(lam, horizon, rng=2026)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    steps_t = np.concatenate([[0.0], times, [horizon]])
    steps_n = np.concatenate([[0], np.arange(1, len(times) + 1), [len(times)]])
    a1.step(steps_t, steps_n, where="post", color=COBALT, label=r"$N_t$")
    a1.plot(times, np.zeros_like(times), "|", color=GOLD, ms=8, mew=1.4, label=r"arrival times $T_k$")
    a1.axvline(times[n - 1], color=GOLD, lw=0.8, ls=":")
    a1.text(times[n - 1] + 0.06, 0.45 * len(times), f"$T_{n}$", color=GOLD, fontsize=9)
    a1.set(xlim=(0, horizon), ylim=(-0.6, len(times) + 0.6), xlabel="$t$",
           title=rf"a Poisson process, $\lambda={lam:g}$")
    a1.legend(fontsize=7.5, loc="upper left")
    T = poisson_arrivals(lam, n, 100_000, rng=2026)[:, n - 1]
    edges = np.linspace(0, 5, 51)
    a2.hist(T, bins=edges, density=True, color=COBALT, alpha=0.35, label=r"$10^5$ simulated $T_3$")
    x = np.linspace(0, 5, 400)
    a2.plot(x, gamma_pdf(x, n, lam), color=GOLD, label=r"$\Gamma(3,2)$ density")
    a2.set(xlim=(0, 5), xlabel="$t$", title=r"$T_3=U_1+U_2+U_3$, $U_i\sim\mathrm{Exp}(2)$")
    a2.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch08-poisson.pdf")
    plt.close(fig)


def annulus(out):
    """The annulus D in the square Q, and the marginal density of X against a histogram."""
    pts = sample_annulus(100_000, rng=2026)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.75), gridspec_kw={"width_ratios": [1, 1.35]})
    a1.add_patch(Rectangle((-3, -3), 6, 6, facecolor=GOLD, alpha=0.18, edgecolor=GOLD, lw=0.8))
    a1.add_patch(Circle((0, 0), 3, facecolor="white", edgecolor=INK, lw=0.8))
    a1.add_patch(Circle((0, 0), 3, facecolor=COBALT, alpha=0.18, edgecolor=INK, lw=0.8))
    a1.add_patch(Circle((0, 0), 1, facecolor="white", edgecolor=INK, lw=0.8))
    a1.add_patch(Circle((0, 0), 1, facecolor=GOLD, alpha=0.18, edgecolor=INK, lw=0.8))
    a1.scatter(pts[:400, 0], pts[:400, 1], s=1.5, color=COBALT, lw=0)
    a1.text(1.7, 1.7, "$D$", color=COBALT, fontsize=10)
    a1.text(0, 0, r"$Q\setminus D$", color=GOLD, fontsize=8, ha="center", va="center")
    a1.set(xlim=(-3.3, 3.3), ylim=(-3.3, 3.3), xticks=[-3, -1, 1, 3], yticks=[-3, -1, 1, 3],
           title=r"$D$ inside the square $Q$")
    a1.set_aspect("equal")
    a1.grid(False)
    edges = np.linspace(-3, 3, 61)
    a2.hist(pts[:, 0], bins=edges, density=True, color=COBALT, alpha=0.35, label=r"$10^5$ samples of $X$")
    x = np.linspace(-3.3, 3.3, 1201)
    a2.plot(x, annulus_marginal_density(x), color=GOLD, label=r"$f_X$")
    a2.set(xlim=(-3.3, 3.3), ylim=(0, 0.3), xlabel="$x$", title="the marginal density of $X$ (and of $Y$)")
    a2.legend(fontsize=7.5, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "ch08-annulus.pdf")
    plt.close(fig)


def transformations(out):
    """X^2 for a standard Gaussian X, and the two branches of x^2 for X uniform on (-1, 2)."""
    rng = np.random.default_rng(2026)
    Z = rng.standard_normal(200_000)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    edges = np.linspace(0, 4, 41)
    a1.hist(Z**2, bins=edges, density=True, color=COBALT, alpha=0.35, label=r"$2\cdot10^5$ samples of $X^2$")
    y = np.linspace(1e-3, 4, 800)
    a1.plot(y, gamma_pdf(y, 0.5, 0.5), color=GOLD, label=r"$\Gamma(\frac{1}{2},\frac{1}{2})$ density")
    a1.set(xlim=(0, 4), ylim=(0, 2.2), xlabel="$y$", title=r"$X\sim\mathcal{N}(0,1)$, $Y=X^2$")
    a1.legend(fontsize=7.5)
    fx = lambda x: np.where((x > -1) & (x < 2), 1 / 3, 0.0)
    b1, b2 = square_branches()
    y = np.linspace(1e-3, 4.5, 900)
    left = transformed_density(y, fx, [b1])
    right = transformed_density(y, fx, [b2])
    X = rng.uniform(-1, 2, 200_000)
    a2.hist(X**2, bins=np.linspace(0, 4.5, 46), density=True, color=GREY, alpha=0.25, label=r"samples of $X^2$")
    a2.plot(y, right, color=COBALT, label=r"branch $x>0$")
    a2.plot(y, left, color=GOLD, label=r"branch $x<0$")
    a2.plot(y, left + right, color=GREEN, lw=1.2, ls="--", label="sum $= f_Y$")
    a2.set(xlim=(0, 4.5), ylim=(0, 1.6), xlabel="$y$", title=r"$X\sim U(-1,2)$, $Y=X^2$")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch08-transform.pdf")
    plt.close(fig)


def branches(out):
    """The two branches of g(x) = x^2 and the two preimages of a level y (the sketch of the notes)."""
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    x1 = np.linspace(-2.1, 0, 200)
    x2 = np.linspace(0, 2.1, 200)
    ax.plot(x1, x1**2, color=GOLD, label=r"$g_1=g|_{I_1}$")
    ax.plot(x2, x2**2, color=COBALT, label=r"$g_2=g|_{I_2}$")
    y0 = 2.0
    r = math.sqrt(y0)
    ax.axhline(y0, color=GREY, lw=0.8, ls=":")
    ax.plot([-r, -r], [0, y0], color=GOLD, lw=0.8, ls="--")
    ax.plot([r, r], [0, y0], color=COBALT, lw=0.8, ls="--")
    ax.plot([-r, r], [0, 0], "o", color=INK, ms=3)
    ax.plot([0], [0], "o", mfc="white", mec=INK, ms=4)
    ax.text(-r, -0.45, r"$g_1^{-1}(y)=-\sqrt{y}$", ha="center", fontsize=8, color=GOLD)
    ax.text(r, -0.45, r"$g_2^{-1}(y)=\sqrt{y}$", ha="center", fontsize=8, color=COBALT)
    ax.text(-2.05, y0 + 0.12, "$y$", fontsize=9, color=INK)
    ax.text(-1.0, 0.15, "$I_1$", fontsize=9, color=GOLD, ha="center")
    ax.text(1.0, 0.15, "$I_2$", fontsize=9, color=COBALT, ha="center")
    ax.set(xlim=(-2.2, 2.2), ylim=(-0.7, 4.5), xticks=[], yticks=[])
    ax.axhline(0, color=GREY, lw=0.6)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(out / "ch08-branches.pdf")
    plt.close(fig)


def main(out):
    use_style()
    gamma_laws(out)
    poisson_process(out)
    annulus(out)
    transformations(out)
    branches(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
