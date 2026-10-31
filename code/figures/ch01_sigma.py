"""Figures for Chapter 1 (sigma-algebras)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse, Patch, Rectangle

from probnotes.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]


def _tint(color, strength):
    """The colour mixed with white: strength 1 is the colour itself, 0 is white."""
    c = np.array(to_rgb(color))
    return tuple(1 - strength * (1 - c))


def _paint(ax, labels, colors, extent):
    """Fill a label image (integers 0..k-1) with the given colours."""
    rgb = np.array([colors[k] for k in range(len(colors))])[labels]
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest", zorder=0)


def _in_ellipse(x, y, cx, cy, w, h):
    return ((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2 <= 1


def disjointification(out):
    """The sets A_1 = B_1, A_2 = B_2 minus A_1, A_3 = B_3 minus (A_1 u A_2) of the proof of
    Proposition sa-algebra-is-sigma (the figure on p. 3 of the handwritten notes)."""
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    extent = (0, 6, 0, 4.6)
    x, y = np.meshgrid(np.linspace(*extent[:2], 900), np.linspace(*extent[2:], 690))
    discs = [(2.2, 2.85, 2.6, 2.0), (3.8, 2.85, 2.6, 2.0), (3.0, 1.55, 2.6, 2.0)]
    B = [_in_ellipse(x, y, *d) for d in discs]
    lab = np.zeros(x.shape, dtype=int)
    lab[B[2] & ~B[0] & ~B[1]] = 3
    lab[B[1] & ~B[0]] = 2
    lab[B[0]] = 1
    _paint(ax, lab, [(1, 1, 1), _tint(COBALT, 0.35), _tint(GOLD, 0.40), _tint(GREEN, 0.40)], extent)
    for (cx, cy, w, h), name, (tx, ty) in zip(discs, ("$B_1$", "$B_2$", "$B_3$"),
                                              ((0.65, 3.85), (5.35, 3.85), (3.0, 0.25))):
        ax.add_patch(Ellipse((cx, cy), w, h, fill=False, color=INK, lw=1.0))
        ax.text(tx, ty, name, ha="center", va="center", color=INK)
    ax.legend(handles=[Patch(color=_tint(COBALT, 0.35), label="$A_1=B_1$"),
                       Patch(color=_tint(GOLD, 0.40), label=r"$A_2=B_2\setminus A_1$"),
                       Patch(color=_tint(GREEN, 0.40), label=r"$A_3=B_3\setminus(A_1\cup A_2)$")],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
    ax.axis("off")
    fig.savefig(out / "ch01-disjoint.pdf")
    plt.close(fig)


def atoms_two_sets(out):
    """The atoms of sigma({A, B}): four in general, three when A and B are disjoint."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    extent = (0, 6, 0, 4)
    x, y = np.meshgrid(np.linspace(*extent[:2], 900), np.linspace(*extent[2:], 600))
    colors = [_tint(GREY, 0.10), _tint(COBALT, 0.40), _tint(GOLD, 0.45), _tint(GREEN, 0.45)]
    cases = [((2.3, 2.0, 3.0, 2.3), (3.7, 2.0, 3.0, 2.3)), ((1.6, 2.0, 2.4, 2.1), (4.4, 2.0, 2.4, 2.1))]
    titles = ["four atoms: $2^4=16$ sets", r"$A\cap B=\emptyset$: three atoms, $2^3=8$ sets"]
    for ax, (ea, eb), title in zip(axes, cases, titles):
        A, B = _in_ellipse(x, y, *ea), _in_ellipse(x, y, *eb)
        lab = np.zeros(x.shape, dtype=int)
        lab[A & ~B] = 1
        lab[B & ~A] = 2
        lab[A & B] = 3
        _paint(ax, lab, colors, extent)
        ax.add_patch(Rectangle((0.02, 0.02), 5.96, 3.96, fill=False, color=GREY, lw=0.8))
        for (cx, cy, w, h), name, dx in ((ea, "$A$", -0.75), (eb, "$B$", 0.75)):
            ax.add_patch(Ellipse((cx, cy), w, h, fill=False, color=INK, lw=1.0))
            ax.text(cx + dx * w / 3, cy + 0.62 * h, name, ha="center", color=INK)
        ax.text(5.75, 3.7, "$X$", ha="right", va="top", color=GREY)
        ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal", title=title)
        ax.axis("off")
    handles = [Patch(color=colors[1], label=r"$A\setminus B$"),
               Patch(color=colors[3], label=r"$A\cap B$"),
               Patch(color=colors[2], label=r"$B\setminus A$"),
               Patch(facecolor=colors[0], edgecolor=GREY, lw=0.5, label=r"$(A\cup B)^c$")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "ch01-atoms.pdf")
    plt.close(fig)


def _interval(ax, a, b, h, left_closed, right_closed, color, lw=2.0, ms=4.2):
    ax.plot([a, b], [h, h], color=color, lw=lw, solid_capstyle="butt")
    for x, closed in ((a, left_closed), (b, right_closed)):
        ax.plot([x], [h], marker="o", ms=ms, color=color, mfc=color if closed else "white", mew=1.2,
                zorder=3)


def monotone_intervals(out):
    """The monotone sequences of intervals of Example sa-intervals and of the proof of
    Proposition sa-borel-generators (the figures on pp. 5 and 11 of the handwritten notes)."""
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True)
    ns = range(2, 8)
    panels = [
        (r"$[\frac{1}{n},1)\uparrow(0,1)$", lambda n: (1 / n, 1, True, False), (0, 1, False, False)),
        (r"$(a,b-\frac{1}{n}]\uparrow(a,b)$", lambda n: (0, 1 - 1 / n, False, True), (0, 1, False, False)),
        (r"$(a,b+\frac{1}{n})\downarrow(a,b]$", lambda n: (0, 1 + 1 / n, False, False), (0, 1, False, True)),
    ]
    for ax, (title, set_n, limit) in zip(axes, panels):
        for k, n in enumerate(ns):
            a, b, lc, rc = set_n(n)
            _interval(ax, a, b, len(ns) - k, lc, rc, COBALT, lw=1.6, ms=3.6)
        a, b, lc, rc = limit
        _interval(ax, a, b, 0, lc, rc, GOLD, lw=2.4)
        ax.set_title(title)
        ax.set_yticks([0] + [len(ns) - k for k in range(len(ns))])
        ax.set_yticklabels(["limit"] + [f"$n={n}$" for n in ns], fontsize=8)
        ax.set_ylim(-0.8, len(ns) + 0.8)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xticks([0, 0.5, 1])
    axes[0].set_xticklabels(["$0$", r"$\frac{1}{2}$", "$1$"])
    for ax in axes[1:]:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["$a$", "$b$"])
    axes[0].set_xlim(-0.1, 1.2)
    axes[1].set_xlim(-0.1, 1.2)
    axes[2].set_xlim(-0.1, 1.6)
    handles = [plt.Line2D([], [], color=COBALT, lw=1.6, label="the sets of the sequence"),
               plt.Line2D([], [], color=GOLD, lw=2.4, label="their union or intersection"),
               plt.Line2D([], [], color=GREY, marker="o", ls="", mfc=GREY, label="endpoint included"),
               plt.Line2D([], [], color=GREY, marker="o", ls="", mfc="white", label="endpoint excluded")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "ch01-intervals.pdf")
    plt.close(fig)


def main(out):
    use_style()
    disjointification(out)
    atoms_two_sets(out)
    monotone_intervals(out)


if __name__ == "__main__":
    import pathlib

    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
