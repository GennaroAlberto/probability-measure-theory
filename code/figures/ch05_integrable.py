"""Figures for Chapter 5 (integrable functions, dominated convergence, Riemann vs Lebesgue)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probnotes.integrable import (
    SINE_LIMIT,
    abs_sine_constant,
    abs_sine_integral_from_1,
    alternating_parts,
    bump_integral,
    pq_limit,
    rearrange_pq,
    sine_integral_from_1,
    siri_halfline_integral,
    siri_unit,
    siri_unit_integral,
)
from probnotes.measurable import dyadic_approximation
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def wave(x):
    """A function shaped like the one drawn in the notes (p. 70): up, a dip, up again, down."""
    x = np.asarray(x, dtype=float)
    return 0.15 + 0.9 * np.sin(1.6 * (x - 0.15)) ** 2 + 0.95 * np.exp(-((x - 2.05) / 0.45) ** 2)


def broken(x, y):
    """Insert NaNs at the jumps of a step function so that no vertical segments are drawn."""
    jumps = np.flatnonzero(np.diff(y) != 0) + 1
    return np.insert(x, jumps, np.nan), np.insert(y.astype(float), jumps, np.nan)


def riemann_lebesgue(out):
    """Partition the domain (Riemann) or the codomain (Lebesgue): the picture of the notes, p. 70."""
    a, b = 0.15, 2.65
    x = np.linspace(a, b, 4001)
    f = wave(x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    edges = np.linspace(a, b, 7)
    for left, right in zip(edges[:-1], edges[1:]):
        seg = wave(np.linspace(left, right, 400))
        ax1.bar(left, seg.max(), width=right - left, align="edge", color=GOLD, alpha=0.22,
                edgecolor=GOLD, linewidth=0.8)
        ax1.bar(left, seg.min(), width=right - left, align="edge", color=GOLD, alpha=0.55,
                edgecolor=GOLD, linewidth=0.8)
    ax1.plot(x, f, color=INK, lw=1.1, label="$f$")
    ax1.bar([0], [0], color=GOLD, alpha=0.55, label="lower sum")
    ax1.bar([0], [0], color=GOLD, alpha=0.22, label="upper sum")
    ax1.set_title("Riemann: partition the domain")
    ax1.legend(loc="upper left", fontsize=7.5)
    n = 2
    fn = dyadic_approximation(f, n)
    for level in np.arange(1, 7) / 2**n:
        ax2.axhline(level, color=GREY, lw=0.5, ls=(0, (4, 3)), alpha=0.6, zorder=0)
    ax2.fill_between(x, 0, fn, step=None, color=COBALT, alpha=0.3, lw=0)
    xs, ys = broken(x, fn)
    ax2.plot(xs, ys, color=COBALT, lw=1.4, label=r"simple $f_2\leq f$")
    ax2.plot(x, f, color=INK, lw=1.1, label="$f$")
    ax2.set_title("Lebesgue: partition the codomain")
    ax2.legend(loc="upper left", fontsize=7.5)
    for ax in (ax1, ax2):
        ax.set_xlim(0, 2.85)
        ax.set_ylim(0, 1.62)
        ax.set_xticks([a, b], ["$a$", "$b$"])
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(out / "ch05-riemann-lebesgue.pdf")
    plt.close(fig)


def dominated(out):
    """Left: the family of the first worked exercise under g. Right: limits of integrals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    x = np.geomspace(1e-3, 1, 600)
    ns = (1, 2, 5, 50)
    for k, n in enumerate(ns):
        ax1.plot(x, siri_unit(x, n), color=COBALT, lw=1.3, alpha=0.35 + 0.65 * k / (len(ns) - 1),
                 label=f"$f_{{{n}}}$" if n in (1, 50) else None)
    ax1.plot(x, x**-0.5, color=INK, lw=1.1, ls=(0, (4, 2)), label=r"$g(x)=x^{-1/2}$")
    ax1.set_ylim(0, 6)
    ax1.set_xlabel("$x$")
    ax1.set_title(r"$f_n(x)=\frac{n}{x^{3/2}}\log(1+\frac{x}{n})$, $n=1,2,5,50$")
    ax1.legend(fontsize=7.5, loc="upper right")
    n = np.unique(np.geomspace(1, 1e4, 40).astype(int))
    ax2.plot(n, siri_unit_integral(n), color=COBALT, marker="o", ms=3, lw=1.4,
             label="first worked exercise")
    ax2.plot(n, siri_halfline_integral(n), color=GOLD, marker="s", ms=3, lw=1.4,
             label="second worked exercise")
    ax2.plot(n, bump_integral(n), color=GREEN, marker="^", ms=3, lw=1.4, ls=(0, (5, 2)),
             label=r"$n^2xe^{-nx}$ on $(0,1)$")
    for y in (0, 2):
        ax2.axhline(y, color=GREY, lw=0.8, ls=":")
    ax2.text(1.2, 2.08, r"$\int\lim f_n=2$", color=GREY, fontsize=7.5)
    ax2.text(1.2, 0.08, r"$\int\lim f_n=0$", color=GREY, fontsize=7.5)
    ax2.set_xscale("log")
    ax2.set_ylim(-0.15, 3.3)
    ax2.set_xlabel("$n$")
    ax2.set_title(r"$\int f_n$ against $\int\lim_n f_n$")
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch05-dominated.pdf")
    plt.close(fig)


def sine(out):
    """int_1^T sin x / x converges; int_1^T |sin x| / x grows like (2/pi) log T."""
    T = np.geomspace(1, 1e4, 3000)
    C = abs_sine_constant()
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(T, abs_sine_integral_from_1(T), color=GOLD, lw=1.6, label=r"$\int_1^T|\sin x|/x\,\mathrm{d}x$")
    ax.plot(T, 2 / math.pi * np.log(T) + C, color=GREEN, lw=1.2, ls=(0, (4, 2)),
            label=rf"$\frac{{2}}{{\pi}}\log T+{C:.4f}$")
    ax.plot(T, sine_integral_from_1(T), color=COBALT, lw=1.6, label=r"$\int_1^T\sin x/x\,\mathrm{d}x$")
    ax.axhline(SINE_LIMIT, color=GREY, lw=0.8, ls=":")
    ax.text(300, SINE_LIMIT - 0.45, r"$\frac{\pi}{2}-\mathrm{Si}(1)=0.6247$", color=GREY, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("$T$")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch05-sine.pdf")
    plt.close(fig)


def rearrangement(out):
    """Partial sums of sum f(x_n) mu({x_n}) = sum (-1)^(n+1)/n, in three orders."""
    pos, neg = alternating_parts(400_000)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    styles = [((1, 1), COBALT, "-", "original order"), ((2, 1), GOLD, (0, (5, 2)), "2 positive, 1 negative"),
              ((1, 4), GREEN, (0, (2, 1.5)), "1 positive, 4 negative")]
    for (p, q), color, ls, label in styles:
        s = np.cumsum(rearrange_pq(pos, neg, p, q))
        k = np.unique(np.geomspace(1, len(s), 1500).astype(int)) - 1
        ax.plot(k + 1, s[k], color=color, lw=1.3, ls=ls, label=f"{label}: limit {pq_limit(p, q):.4f}")
        ax.axhline(pq_limit(p, q), color=GREY, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("number of terms")
    ax.set_ylabel("partial sum")
    ax.set_ylim(-0.25, 1.45)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch05-rearrangement.pdf")
    plt.close(fig)


def main(out):
    use_style()
    riemann_lebesgue(out)
    dominated(out)
    sine(out)
    rearrangement(out)


if __name__ == "__main__":
    import pathlib

    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
