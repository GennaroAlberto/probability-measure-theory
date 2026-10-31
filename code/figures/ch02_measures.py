"""Figures for Chapter 2 (Measures): the lim inf / lim sup examples and Borel--Cantelli."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probnotes.measures import (expected_count, harmonic_probs, notes_example_sequence,
                                occurrence_times, square_probs, tail_inf, tail_sup)
from probnotes.style import COBALT, GOLD, GREY, INK, use_style


def limits(out):
    """The two real sequences of the aside, with their tail infima and suprema."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    n_show = 16
    for ax, which in zip(axes, (1, 2)):
        n, a = notes_example_sequence(which, 200)  # long stretch, so the tails are exact
        lo, hi = tail_inf(a)[:n_show], tail_sup(a)[:n_show]
        n, a = n[:n_show], a[:n_show]
        ax.step(n, hi, where="mid", color=GOLD, lw=1.6, label=r"$\mathrm{sup}_{k\geq n}\,a_k$")
        ax.step(n, lo, where="mid", color=COBALT, lw=1.6, label=r"$\mathrm{inf}_{k\geq n}\,a_k$")
        ax.plot(n, a, "o", color=INK, ms=3.2, label="$a_n$", zorder=3)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_xlabel("$n$")
        ax.set_title(f"Example {which}")
        ax.set_xticks([1, 4, 8, 12, 16])
    axes[0].set_ylim(-1.2, 1.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "ch02-limits.pdf")
    plt.close(fig)


def borel_cantelli(out):
    """Counting paths N_n for independent events with P(A_n) = 1/n and 1/n^2."""
    horizon, n_paths = 10**6, 25
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, kind, probs, title, seed in (
            (axes[0], "harmonic", harmonic_probs(horizon), r"$P(A_n)=1/n$", 21),
            (axes[1], "square", square_probs(horizon), r"$P(A_n)=1/n^2$", 22)):
        times = occurrence_times(probs, n_paths=n_paths, seed=seed)
        for k, t in enumerate(times):
            x = np.concatenate([t, [horizon]])
            y = np.concatenate([np.arange(1, len(t) + 1), [len(t)]])
            ax.step(x, y, where="post", color=COBALT, lw=0.9, alpha=0.45,
                    label="simulated paths" if k == 0 else None)
        grid = np.unique(np.logspace(0, 6, 200).astype(int))
        ax.plot(grid, np.cumsum(probs)[grid - 1], color=GOLD, lw=2.0,
                label=r"$E[N_n]=\sum_{k\leq n}P(A_k)$")
        ax.set_xscale("log")
        ax.set_xlim(1, horizon)
        ax.set_xlabel("$n$")
        ax.set_title(title)
        ax.text(0.03, 0.96, f"$E[N_n]$ at $n=10^6$: {expected_count(kind, horizon):.2f}",
                transform=ax.transAxes, va="top", fontsize=8, color=GREY)
    axes[0].set_ylabel("$N_n$ = occurrences up to $n$")
    axes[1].set_ylim(0, 6)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "ch02-borel-cantelli.pdf")
    plt.close(fig)


def main(out):
    use_style()
    limits(out)
    borel_cantelli(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
