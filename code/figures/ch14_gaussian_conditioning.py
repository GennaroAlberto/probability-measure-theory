"""Figures for Chapter 14 (Gaussian conditioning)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from probnotes.gaussian_conditioning import (bivariate_conditional, partial_sums_cov,
                                             partial_sums_precision, sample_gaussian, slab_mask)
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
SIGNAL = np.array([[1.0, 1.0], [1.0, 2.0]])      # (X, Y) of the signal example


def bivariate(out, n=1_000_000, h=0.05, seed=2026):
    """The signal example: E[X | Y = y] = y / 2, and slab histograms against N(y/2, 1/2)."""
    s = sample_gaussian(None, SIGNAL, n, seed=seed)
    x, y = s[:, 0], s[:, 1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    a1.scatter(y[:2500], x[:2500], s=1.5, color=GREY, alpha=0.45, lw=0, rasterized=True)
    t = np.linspace(-4.5, 4.5, 50)
    m, s2 = bivariate_conditional(t, 0, 0, 1, 2, 1)
    a1.plot(t, m, color=INK, lw=1.4, label=r"$m(y)=\mathbb{E}[X\mid Y=y]=y/2$")
    a1.plot(t, m + 2 * np.sqrt(s2), color=INK, lw=0.8, ls="--", label=r"$m(y)\pm2\sigma$")
    a1.plot(t, m - 2 * np.sqrt(s2), color=INK, lw=0.8, ls="--")
    cols = (COBALT, GOLD, GREEN)
    y0s = (-2.0, 0.0, 2.0)
    for y0, c in zip(y0s, cols):
        a1.axvspan(y0 - h, y0 + h, color=c, alpha=0.6, lw=0)
    a1.set(xlim=(-4.5, 4.5), ylim=(-3.5, 3.5), xlabel="$y$", ylabel="$x$",
           title=r"$(Y,X)$, signal example")
    a1.legend(fontsize=7, loc="upper left")
    xs = np.linspace(-3.5, 3.5, 300)
    a2.plot(xs, stats.norm.pdf(xs), color=GREY, lw=1.0, ls=":", label=r"law of $X$: $\mathcal{N}(0,1)$")
    for y0, c in zip(y0s, cols):
        sel = x[slab_mask(s, [1], [y0], h)]
        a2.hist(sel, bins=np.linspace(-3.5, 3.5, 36), density=True, color=c, alpha=0.3)
        a2.plot(xs, stats.norm.pdf(xs, y0 / 2, np.sqrt(0.5)), color=c, lw=1.5,
                label=f"$y={y0:g}$: {sel.size} draws in the slab")
    a2.set(xlim=(-3.5, 3.5), ylim=(0, 0.98), xlabel="$x$",
           title=r"$X$ in thin slabs, and $\mathcal{N}(y/2,\,1/2)$")
    a2.legend(fontsize=6.3, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(out / "ch14-bivariate.pdf")
    plt.close(fig)


def _matrix(ax, M, title, cmap):
    v = np.abs(M).max()
    ax.imshow(M, cmap=cmap, vmin=-v, vmax=v)
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:g}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[i, j]) > 0.6 * v else INK)
    ax.set_xticks(range(n), [str(k + 1) for k in range(n)], fontsize=7)
    ax.set_yticks(range(n), [str(k + 1) for k in range(n)], fontsize=7)
    ax.grid(False)
    ax.set_title(title)


def precision(out, N=6, n=1_000_000, h=0.05, seed=2026):
    """Partial sums: C = min(k, l) is full, C^{-1} is tridiagonal; Y_1 and Y_3 given Y_2."""
    cmap = LinearSegmentedColormap.from_list("gc", [GOLD, "white", COBALT])
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.35), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    _matrix(axes[0], partial_sums_cov(N), r"$C=(\min(k,l))_{k,l}$", cmap)
    _matrix(axes[1], partial_sums_precision(N), r"$C^{-1}$", cmap)
    s = sample_gaussian(None, partial_sums_cov(3), n, seed=seed)
    sel = s[slab_mask(s, [1], [1.0], h)]
    a = axes[2]
    a.scatter(s[:1500, 0], s[:1500, 2], s=1.5, color=GREY, alpha=0.5, lw=0, rasterized=True,
              label=f"all draws: corr {np.corrcoef(s[:, 0], s[:, 2])[0, 1]:.3f}")
    a.scatter(sel[:1500, 0], sel[:1500, 2], s=1.5, color=COBALT, alpha=0.6, lw=0, rasterized=True,
              label=f"$|Y_2-1|<{h:g}$: corr {np.corrcoef(sel[:, 0], sel[:, 2])[0, 1]:.3f}")
    a.set(xlim=(-4, 4), ylim=(-6, 6), xlabel="$Y_1$", ylabel="$Y_3$", title=r"$(Y_1,Y_3)$, given $Y_2$")
    a.legend(fontsize=6, loc="upper left", markerscale=4)
    fig.tight_layout()
    fig.savefig(out / "ch14-precision.pdf")
    plt.close(fig)


def main(out):
    use_style()
    bivariate(out)
    precision(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
