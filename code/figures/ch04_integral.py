"""Figures for Chapter 4 (the integral of non-negative functions)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probnotes.integral import EXAMPLES, dyadic_integrals, spike, spreading, sliding_bump, step_function
from probnotes.measurable import dyadic_approximation
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"

# The simple function of the notes' sketch (p. 51): a partition A_1, ..., A_5 of an interval.
CUTS = np.array([0.0, 1.0, 2.2, 3.0, 3.8, 5.0])
LEVELS = np.array([0.0, 1.2, 0.5, 2.1, 0.0])
E = (1.6, 3.5)


def bump(x):
    """The non-negative function of the Chapter 3 figure (shaped like the notes' sketch, p. 50)."""
    x = np.asarray(x, dtype=float)
    y = (0.42 + 0.72 * np.exp(-((x - 0.3) / 0.16) ** 2) + 1.78 * np.exp(-((x - 1.5) / 0.42) ** 2)
         - 0.05 * x) * (1 - np.exp(-14 * np.clip(x, 0, None)))
    return np.where(x > 0, y, 0.0)


def area(y, x):
    """Trapezoidal area under y(x) on a fine grid."""
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def _steps(ax, cuts, levels, **kw):
    for a, b, v in zip(cuts[:-1], cuts[1:], levels):
        ax.plot([a, b], [v, v], **kw)
        kw.pop("label", None)


def simple(out):
    """The integral of a simple function as an area, over X and over a set E (notes, pp. 51-52)."""
    f = step_function(LEVELS, [[(a, b)] for a, b in zip(CUTS[:-1], CUTS[1:])])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, restrict in zip(axes, (False, True)):
        for a, b, v in zip(CUTS[:-1], CUTS[1:], LEVELS):
            ax.fill_between([a, b], 0, v, color=COBALT, alpha=0.10 if restrict else 0.22, lw=0)
            if restrict:
                lo, hi = max(a, E[0]), min(b, E[1])
                if hi > lo:
                    ax.fill_between([lo, hi], 0, v, color=GOLD, alpha=0.35, lw=0)
        _steps(ax, CUTS, LEVELS, color=COBALT, lw=1.8, label="$f$")
        ax.axhline(0, color=GREY, lw=0.8)
        for c in CUTS[1:-1]:
            ax.axvline(c, color=GREY, lw=0.5, ls=":", ymax=0.9)
        for i, (a, b) in enumerate(zip(CUTS[:-1], CUTS[1:])):
            ax.text((a + b) / 2, -0.32, f"$A_{i + 1}$", ha="center", va="center", fontsize=9, color=INK)
        ax.set_xlim(-0.2, 5.2)
        ax.set_ylim(-0.55, 2.55)
        ax.set_xticks([])
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
    axes[0].set_title(r"$\int_X f\,d\mu=\sum_i a_i\,\mu(A_i)=%.2f$" % f.integral())
    axes[1].set_title(r"$\int_E f\,d\mu=\sum_i a_i\,\mu(A_i\cap E)=%.2f$" % f.integral([E]))
    for x in E:
        axes[1].axvline(x, color=GOLD, lw=1.2)
    axes[1].annotate("", xy=(E[1], 2.35), xytext=(E[0], 2.35),
                     arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.0))
    axes[1].text(sum(E) / 2, 2.43, "$E$", color=GOLD, ha="center", va="bottom", fontsize=9)
    axes[0].set_yticks([0, 0.5, 1, 1.5, 2])
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ch04-simple.pdf")
    plt.close(fig)


def monotone(out):
    """Monotone convergence made visible: the areas under f_n increase to the area under f."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    x = np.linspace(0.0, 3.0, 6001)
    fx = bump(x)
    for n, alpha in ((1, 0.45), (2, 0.25), (4, 0.12)):
        fn = dyadic_approximation(fx, n)
        ax.fill_between(x, 0, fn, color=COBALT, alpha=alpha, lw=0,
                        label=f"area under $f_{n}$: {area(fn, x):.3f}")
    ax.plot(x, fx, color=INK, lw=1.1, label=f"area under $f$: {area(fx, x):.3f}")
    ax.set_xlabel("$x$")
    ax.set_title(r"areas over $[0,3]$: $\int f_n\,dm\uparrow\int f\,dm$")
    ax.set_ylim(0, 3.3)
    ax.set_yticks([0, 1, 2])
    ax.grid(False)
    ax.legend(fontsize=7, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.2)
    ax = axes[1]
    n = np.arange(1, 17)
    styles = (("x^2 on [0, 1]", COBALT, "o", r"$x^2$ on $[0,1]$"),
              ("x^(-1/2) on (0, 1)", GOLD, "s", r"$x^{-1/2}$ on $(0,1)$"),
              ("1/(1+x^2) on R", GREEN, "^", r"$1/(1+x^2)$ on $\mathbb{R}$"))
    for key, color, marker, label in styles:
        ex = EXAMPLES[key]
        gap = ex.integral - dyadic_integrals(ex.tail, n[-1])
        ax.plot(n, gap, color=color, marker=marker, ms=3.5, lw=1.4, label=label)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("$n$")
    ax.set_title(r"gap $\int f\,dm-\int f_n\,dm$")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch04-monotone.pdf")
    plt.close(fig)


def escape(out):
    """Three sequences with integral 1 that tend to 0: the mass escapes, and Fatou is strict."""
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    colors = (COBALT, GOLD, GREEN, "#6250D6")
    panels = ((axes[0], sliding_bump, (1, 2, 3, 4), (-0.3, 5.6), 1.75,
               r"sliding: $\mathbf{1}_{[n,n+1)}$"),
              (axes[1], spike, (1, 2, 4), (-0.05, 1.1), 5.6, r"spike: $n\,\mathbf{1}_{(0,1/n)}$"),
              (axes[2], spreading, (1, 2, 4), (-0.3, 4.6), 1.75, r"spreading: $\frac{1}{n}\mathbf{1}_{[0,n)}$"))
    for ax, seq, ns, xlim, top, title in panels:
        ax.axhline(0, color=GREY, lw=1.0, zorder=1)
        for n, c in zip(ns, colors):
            f = seq(n)
            a, b, v = f.breaks[0], f.breaks[-1], f.values[0]
            ax.fill_between([a, b], 0, v, color=c, alpha=0.12, lw=0)
            ax.plot([a, a], [0, v], color=c, lw=0.7, ls=":")
            ax.plot([b, b], [0, v], color=c, lw=0.7, ls=":")
            ax.plot([a, b], [v, v], color=c, lw=1.8, label=f"$n={n}$")
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.04 * top, top)
        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.grid(False)
        ax.legend(fontsize=6.5, loc="upper right", ncol=2, columnspacing=0.6, handlelength=1.0)
    fig.tight_layout()
    fig.savefig(out / "ch04-escape.pdf")
    plt.close(fig)


def main(out):
    use_style()
    simple(out)
    monotone(out)
    escape(out)


if __name__ == "__main__":
    import pathlib

    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
