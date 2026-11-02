"""Figures for Chapter 13 (Gaussian vectors)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from probnotes.gaussian import gaussian_density, sample_affine, sample_gaussian, sample_random_sign_pair
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def densities(out):
    """The three examples of the notes: two invertible covariance matrices (density contours and a
    sample) and the degenerate vector Y = (X1 + X2, 2X1 + 2X2), which lives on a line."""
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.75), sharex=True, sharey=True)
    lim = 5.2
    g = np.linspace(-lim, lim, 241)
    G1, G2 = np.meshgrid(g, g)
    pts = np.stack([G1, G2], axis=-1)
    for ax, C, title in ((axes[0], [[2, 1], [1, 1]], r"invertible $C$, $\det C=1$"),
                         (axes[1], [[2, -1], [-1, 2]], r"invertible $C$, $\det C=3$")):
        Y = sample_gaussian([0, 0], C, 400, seed=2026)
        ax.scatter(Y[:, 0], Y[:, 1], s=2.5, color=GREY, alpha=0.6, lw=0)
        f = gaussian_density(pts, [0, 0], C)
        # contours of the density through the ellipses holding 50%, 90%, 99% of the mass
        levels = sorted(np.exp(-0.5 * stats.chi2.ppf(p, 2)) / (2 * math.pi * math.sqrt(np.linalg.det(C)))
                        for p in (0.5, 0.9, 0.99))
        ax.contour(G1, G2, f, levels=levels, colors=COBALT, linewidths=1.0)
        ax.set_title(title)
    Y = sample_affine([[1, 1], [2, 2]], [0, 0], 400, seed=2026)
    ax = axes[2]
    ax.plot(g, 2 * g, color=GOLD, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.scatter(Y[:, 0], Y[:, 1], s=2.5, color=GREY, alpha=0.6, lw=0, zorder=2)
    ax.set_title(r"singular $C$: degenerate")
    for ax in axes:
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="$y_1$")
        ax.set_aspect("equal")
    axes[0].set_ylabel("$y_2$")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=GREY, alpha=0.8),
               plt.Line2D([], [], color=COBALT, lw=1.0),
               plt.Line2D([], [], color=GOLD, lw=0.8, ls=(0, (4, 2)))]
    fig.legend(handles, ["sample of 400 points", "ellipses holding 50%, 90%, 99% of the law",
                         r"the line $y_2=2y_1$"], loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "ch13-densities.pdf")
    plt.close(fig)


def not_jointly_gaussian(out):
    """Left: the pair (X, ZX) lives on the two diagonals. Right: the law of X + ZX, an atom of mass
    1/2 at 0 plus half of the N(0, 4) density, against the N(0, 2) law it would have if (X, ZX) were
    a Gaussian vector."""
    X, Y = sample_random_sign_pair(100_000, seed=2026)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7))
    a1.scatter(X[:600], Y[:600], s=3, color=COBALT, alpha=0.6, lw=0)
    a1.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), xlabel="$x$", ylabel="$y$",
           title="600 draws of $(X, ZX)$")
    a1.set_aspect("equal")
    S = X + Y
    cont = S[S != 0]
    bins = np.linspace(-8, 8, 65)
    w = np.full(cont.size, 1.0 / S.size / (bins[1] - bins[0]))      # density of the continuous part
    a2.hist(cont, bins=bins, weights=w, color=GREY, alpha=0.35, label="simulated, continuous part")
    x = np.linspace(-8, 8, 801)
    a2.plot(x, 0.5 * stats.norm.pdf(x, scale=2), color=COBALT, label=r"$\frac{1}{2}\,\mathcal{N}(0,4)$ density")
    a2.plot(x, stats.norm.pdf(x, scale=math.sqrt(2)), color=GOLD, ls=(0, (4, 2)),
            label=r"$\mathcal{N}(0,2)$ density")
    p0 = np.mean(S == 0)
    a2.vlines(0, 0, 0.5, color=GREEN, lw=2.2, label=f"atom at 0: mass {p0:.3f}")
    a2.plot([0], [0.5], "o", color=GREEN, ms=4)
    a2.set(xlabel="$x$", ylim=(0, 0.56), title="the law of $X+ZX$")
    a2.legend(fontsize=6.5, loc="upper left", handlelength=1.4)
    fig.tight_layout()
    fig.savefig(out / "ch13-not-joint.pdf")
    plt.close(fig)


def main(out):
    use_style()
    densities(out)
    not_jointly_gaussian(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
