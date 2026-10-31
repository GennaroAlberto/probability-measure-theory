"""Figures for Chapter 3 (Lebesgue measure; measurable and simple functions)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probnotes.measurable import cantor_stage, cantor_stage_measure, dyadic_approximation
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def bump(x):
    """A non-negative function shaped like the one drawn in the notes (p. 50)."""
    x = np.asarray(x, dtype=float)
    y = (0.42 + 0.72 * np.exp(-((x - 0.3) / 0.16) ** 2) + 1.78 * np.exp(-((x - 1.5) / 0.42) ** 2)
         - 0.05 * x) * (1 - np.exp(-14 * np.clip(x, 0, None)))
    return np.where(x > 0, y, 0.0)


def broken(x, y):
    """Insert NaNs at the jumps of a step function so that no vertical segments are drawn."""
    jumps = np.flatnonzero(np.diff(y) != 0) + 1
    return np.insert(x, jumps, np.nan), np.insert(y.astype(float), jumps, np.nan)


def dyadic(out):
    """f and its dyadic approximations f_1, f_2, f_4 (the picture of the notes, p. 50)."""
    x = np.linspace(-0.3, 3.0, 6001)
    f = bump(x)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), sharey=True)
    for ax, n in zip(axes, (1, 2, 4)):
        for level in np.arange(1, int(np.ceil(f.max() * 2**n)) + 1) / 2**n:
            ax.axhline(level, color=GREY, lw=0.35, alpha=0.35, zorder=0)
        ax.plot(x, f, color=INK, lw=1.0, label="$f$")
        xs, ys = broken(x, dyadic_approximation(f, n))
        ax.plot(xs, ys, color=COBALT, lw=1.8, label=f"$f_n$")
        ax.set_title(f"$f_{n}$: steps of $2^{{-{n}}}$, cap ${n}$")
        ax.set_xlabel("$x$")
        ax.set_xlim(-0.3, 3.0)
        ax.grid(False)
    axes[0].set_ylim(-0.08, 2.5)
    axes[0].set_yticks([0, 0.5, 1, 1.5, 2])
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ch03-dyadic.pdf")
    plt.close(fig)


def l1_error_log(n):
    """Exact integral over (0, 1) of f - f_n for f(x) = -log x."""
    h = 2.0**-n
    return math.exp(-n) + (1 - math.exp(-n)) * (1 - math.exp(-h) * (1 + h)) / (1 - math.exp(-h))


def dyadic_errors(out):
    """How fast f_n approaches f: uniformly once n >= sup f, and in mean for an unbounded f."""
    n = np.arange(1, 15)
    x = np.linspace(0, 1, 200_001)
    f = 12 * x * (1 - x)                                          # bounded, sup f = 3
    sup_err = np.array([np.max(f - dyadic_approximation(f, k)) for k in n])
    l1 = np.array([l1_error_log(k) for k in n])
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.plot(n, sup_err, color=COBALT, lw=2.6, marker="o", ms=4.5, alpha=0.75, zorder=2,
            label="largest error, $f(x)=12x(1-x)$")
    ax.plot(n, l1, color=GOLD, lw=2.6, marker="s", ms=4.5, alpha=0.75, zorder=2,
            label=r"mean error on $(0,1)$, $f(x)=-\log x$")
    nn = np.linspace(0.6, 14.4, 200)
    ax.plot(nn, 2.0**-nn, color=GREEN, lw=1.3, ls=(0, (4, 2)), zorder=5, label="reference $2^{-n}$")
    ax.axvline(3, color=GREY, lw=0.8, ls=":")
    ax.text(3.15, 1.2, r"$n=\sup f$", color=GREY, fontsize=8)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("$n$")
    ax.set_ylabel("error")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch03-dyadic-errors.pdf")
    plt.close(fig)


def cantor(out):
    """Stages of the Cantor set and of a fat Cantor set, and their measures."""
    K = 6
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    for ax, r, color, title in ((axes[0], 1 / 3, COBALT, r"$C_k$: remove middle thirds"),
                                (axes[1], 1 / 4, GOLD, r"fat Cantor: remove $4^{-k}$")):
        for k in range(K + 1):
            I = cantor_stage(k, r)
            ax.hlines(np.full(len(I), -k), I[:, 0], I[:, 1], color=color, lw=4.5)
        ax.set_yticks(-np.arange(K + 1), [f"$k={k}$" for k in range(K + 1)])
        ax.set_xticks([0, 1 / 3, 2 / 3, 1], ["0", "1/3", "2/3", "1"])
        ax.set_ylim(-K - 0.6, 0.6)
        ax.set_title(title)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ks = np.arange(0, 21)
    axes[2].plot(ks, [cantor_stage_measure(k) for k in ks], color=COBALT, marker="o", ms=2.5,
                 label=r"$m(C_k)=(2/3)^k$")
    axes[2].plot(ks, [cantor_stage_measure(k, 1 / 4) for k in ks], color=GOLD, marker="s", ms=2.5,
                 label="fat Cantor stage")
    axes[2].axhline(0.5, color=GOLD, lw=0.8, ls="--")
    axes[2].set_xlabel("$k$")
    axes[2].set_title("measure of stage $k$")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch03-cantor.pdf")
    plt.close(fig)


def main(out):
    use_style()
    dyadic(out)
    dyadic_errors(out)
    cantor(out)


if __name__ == "__main__":
    import pathlib

    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
