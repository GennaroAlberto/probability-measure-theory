"""Figures for Chapter 7 (product measures, random vectors, independence)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import numpy as np

from probnotes.product import (empirical_cdf_defect, fubini_counterexample, fubini_inner_exact,
                               inner_integrals, iterated_integral, sample_pair)
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def blob(x, y):
    """Indicator of the union of two discs: the set E of the sections figure."""
    return ((x * x + y * y <= 1.0) | ((x - 1.3) ** 2 + (y - 0.5) ** 2 <= 0.36)).astype(float)


def sections(out):
    """A set E, one vertical and one horizontal section, and the two section-measure functions."""
    xr, yr = (-1.2, 2.1), (-1.2, 1.3)
    xs = np.linspace(*xr, 661)
    ys = np.linspace(*yr, 501)
    len_x = inner_integrals(blob, xs, yr, 4000, "y")          # m(E_x), x in X_1
    len_y = inner_integrals(blob, ys, xr, 4000, "x")          # m(E^y), y in X_2
    area_x = iterated_integral(blob, xr, yr, 2000, 4000, "dy dx")
    area_y = iterated_integral(blob, xr, yr, 2000, 4000, "dx dy")
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.35), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ax.contourf(X, Y, blob(X, Y), levels=[0.5, 1.5], colors=[GREY], alpha=0.18)
    ax.contour(X, Y, blob(X, Y), levels=[0.5], colors=[INK], linewidths=0.9)
    x0, y0 = 0.35, 0.62
    col = ys[blob(np.full_like(ys, x0), ys) > 0]
    row = xs[blob(xs, np.full_like(xs, y0)) > 0]
    ax.plot([x0, x0], [col.min(), col.max()], color=COBALT, lw=2.6, label=r"$E^2_x$")
    ax.plot([row.min(), row.max()], [y0, y0], color=GOLD, lw=2.6, label=r"$E^1_y$")
    ax.axvline(x0, color=COBALT, lw=0.6, ls=":")
    ax.axhline(y0, color=GOLD, lw=0.6, ls=":")
    ax.text(-0.55, -0.35, "$E$", fontsize=11, color=INK)
    ax.set(xlim=xr, ylim=yr, xticks=[x0], xticklabels=["$x$"], yticks=[y0], yticklabels=["$y$"],
           title="a set and two sections")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(loc="lower right", fontsize=7, handlelength=1.2)
    axes[1].fill_between(xs, len_x, color=COBALT, alpha=0.25)
    axes[1].plot(xs, len_x, color=COBALT)
    axes[1].set(xlabel="$x$", title=r"$x\mapsto m(E^2_x)$")
    axes[1].text(0.03, 0.9, f"area {area_x:.4f}", transform=axes[1].transAxes, fontsize=8, color=INK)
    axes[2].fill_between(ys, len_y, color=GOLD, alpha=0.25)
    axes[2].plot(ys, len_y, color=GOLD)
    axes[2].set(xlabel="$y$", title=r"$y\mapsto m(E^1_y)$")
    axes[2].text(0.03, 0.9, f"area {area_y:.4f}", transform=axes[2].transAxes, fontsize=8, color=INK)
    for a in axes[1:]:
        a.set_ylim(0, 3.3)
    fig.tight_layout()
    fig.savefig(out / "ch07-sections.pdf")
    plt.close(fig)


def fubini(out):
    """f = (x^2 - y^2)/(x^2 + y^2)^2: the function, its inner integrals, and grid sums."""
    cmap = LinearSegmentedColormap.from_list("cg", [GOLD, "#FFFFFF", COBALT])
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3), gridspec_kw={"width_ratios": [1, 1.05, 1.1]})
    g = np.linspace(0.0025, 1, 400)
    X, Y = np.meshgrid(g, g, indexing="ij")
    im = axes[0].pcolormesh(X, Y, fubini_counterexample(X, Y), cmap=cmap,
                            norm=SymLogNorm(linthresh=1, vmin=-1e4, vmax=1e4), shading="auto",
                            rasterized=True)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.05, pad=0.03, ticks=[-1e4, -10, 0, 10, 1e4])
    cb.ax.tick_params(labelsize=6)
    axes[0].set(xlabel="$x$", ylabel="$y$", title=r"$f(x,y)$", xticks=[0, 1], yticks=[0, 1])
    axes[0].set_aspect("equal")
    axes[0].grid(False)
    t = np.linspace(0, 1, 400)
    axes[1].fill_between(t, fubini_inner_exact(t, "y"), color=COBALT, alpha=0.2)
    axes[1].fill_between(t, fubini_inner_exact(t, "x"), color=GOLD, alpha=0.2)
    axes[1].plot(t, fubini_inner_exact(t, "y"), color=COBALT, label=r"$\int_0^1 f(x,y)\,dy$")
    axes[1].plot(t, fubini_inner_exact(t, "x"), color=GOLD, label=r"$\int_0^1 f(x,y)\,dx$")
    axes[1].axhline(0, color=GREY, lw=0.6)
    axes[1].text(0.5, 1.3, r"$\int_0^1 f(x,y)\,dy=\frac{1}{1+x^2}$", fontsize=7.5, color=COBALT,
                 ha="center")
    axes[1].text(0.5, -1.58, r"$\int_0^1 f(x,y)\,dx=-\frac{1}{1+y^2}$", fontsize=7.5, color=GOLD,
                 ha="center")
    axes[1].text(0.5, 0.3, r"area $\pi/4$", fontsize=8, color=INK, ha="center")
    axes[1].text(0.5, -0.38, r"area $-\pi/4$", fontsize=8, color=INK, ha="center")
    axes[1].set(xlabel="$x$ (blue), $y$ (gold)", ylim=(-1.75, 1.75), yticks=[-1, 0, 1], title="inner integrals")
    ks = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    dydx = [iterated_integral(fubini_counterexample, (0, 1), (0, 1), 100, 100 * k, "dy dx") for k in ks]
    dxdy = [iterated_integral(fubini_counterexample, (0, 1), (0, 1), 100, 100 * k, "dx dy") for k in ks]
    axes[2].plot(ks, dydx, color=COBALT, marker="o", ms=3.5, label="inner $dy$, outer $dx$")
    axes[2].plot(ks, dxdy, color=GOLD, marker="s", ms=3.5, label="inner $dx$, outer $dy$")
    axes[2].axhline(math.pi / 4, color=GREEN, ls=(0, (4, 2)), lw=1.1, label=r"$\pm\pi/4$")
    axes[2].axhline(-math.pi / 4, color=GREEN, ls=(0, (4, 2)), lw=1.1)
    axes[2].set(xscale="log", xlabel="inner points / outer points", ylim=(-1.05, 1.05),
                title="midpoint sums, 100 outer points")
    axes[2].set_xticks(ks[::2], [str(k) for k in ks[::2]])
    axes[2].legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "ch07-fubini.pdf")
    plt.close(fig)


def dependence(out):
    """Four pairs with zero covariance; only the first is independent."""
    titles = {"independent": "independent uniforms", "square": r"$(X, X^2)$",
              "circle": r"$(\cos T, \sin T)$", "disk": "uniform on the disk"}
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.05))
    for ax, kind in zip(axes, titles):
        x, y = sample_pair(kind, 1500, seed=7)
        xb, yb = sample_pair(kind, 100_000, seed=2026)
        cov = np.mean(xb * yb) - np.mean(xb) * np.mean(yb)
        d = empirical_cdf_defect(xb, yb)
        ax.scatter(x, y, s=1.2, color=COBALT, alpha=0.5, lw=0, rasterized=True)
        ax.set_title(titles[kind], fontsize=8.5)
        ax.text(0.5, -0.2, f"cov {cov:+.3f}\nCDF gap {d:.3f}", transform=ax.transAxes, ha="center",
                va="top", fontsize=7.5, color=INK)
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out / "ch07-dependence.pdf")
    plt.close(fig)


def main(out):
    use_style()
    sections(out)
    fubini(out)
    dependence(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
