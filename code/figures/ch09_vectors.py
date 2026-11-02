"""Figures for Chapter 9 (transformations of random vectors)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probnotes.style import COBALT, GOLD, GREY, INK, PALETTE, use_style
from probnotes.vectors import (beta_pdf, fractional_part_pdf, gamma_pdf, joint_cdf_gap,
                               laplace_uniform_sum_pdf, sample_gamma_beta, sample_worked,
                               sqrt_abs_laplace_pdf, uniform_product_pdf)

GREEN = PALETTE[2]
PURPLE = PALETTE[3]


def _plane(ax, xlim, ylim, xlabel, ylabel):
    ax.axhline(0, color=GREY, lw=0.7)
    ax.axvline(0, color=GREY, lw=0.7)
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect("equal")
    ax.grid(False)


def regions(out):
    """The three regions drawn in the notes: A_t, the strip of f_V, and the strip D."""
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5))
    # A_t = {x + y <= t}
    ax, t = axes[0], 1.0
    xs = np.linspace(-2, 2.5, 2)
    ax.fill_between(xs, -2.0, t - xs, color=GOLD, alpha=0.22, lw=0)
    ax.plot(xs, t - xs, color=GOLD, lw=1.4, label="$x+y=t$")
    x0 = 0.35
    ax.plot([x0, x0], [-2, t - x0], color=COBALT, lw=2.2, label=r"$y\leq t-x$")
    ax.plot([x0], [t - x0], "o", color=COBALT, ms=3)
    ax.text(-1.6, -1.4, "$A_t$", fontsize=11, color=INK)
    _plane(ax, (-2, 2.5), (-2, 2.5), "$x$", "$y$")
    ax.set_xticks([x0, t], ["$x$", "$t$"])
    ax.set_yticks([t], ["$t$"])
    ax.set_title(r"$A_t=\{x+y\leq t\}$")
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.2)
    # the strip of Exercise 5 b): u > 0, u < v < u + 1
    ax = axes[1]
    us = np.linspace(0, 2.6, 2)
    ax.fill_between(us, us, us + 1, color=GOLD, alpha=0.22, lw=0)
    ax.plot(us, us, color=GOLD, lw=1.2)
    ax.plot(us, us + 1, color=GOLD, lw=1.2)
    for v in (0.6, 1.8):
        ax.plot([max(0.0, v - 1), v], [v, v], color=COBALT, lw=2.2)
        ax.axhline(v, color=COBALT, lw=0.5, ls=":")
    ax.text(1.75, 1.2, "$v=u$", fontsize=7, color=INK)
    ax.text(0.55, 2.15, "$v=u+1$", fontsize=7, color=INK)
    _plane(ax, (-0.5, 2.6), (-0.5, 2.6), "$u$", "$v$")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_title(r"$\{u>0,\ u<v<u+1\}$")
    # the strip D = {t < s < t + 2}
    ax = axes[2]
    ts = np.linspace(-2.5, 2.5, 2)
    ax.fill_betweenx(ts, ts, ts + 2, color=GOLD, alpha=0.22, lw=0)
    ax.plot(ts, ts, color=GOLD, lw=1.2)
    ax.plot(ts + 2, ts, color=GOLD, lw=1.2)
    ax.text(0.55, 0.05, "$D$", fontsize=11, color=INK)
    ax.text(-1.95, -1.55, "$s=t$", fontsize=7, color=INK)
    ax.text(1.3, -1.6, "$s=t+2$", fontsize=7, color=INK)
    _plane(ax, (-2.5, 2.5), (-2.5, 2.5), "$s$", "$t$")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_title(r"$D=\{t<s<t+2\}$")
    fig.tight_layout()
    fig.savefig(out / "ch09-regions.pdf")
    plt.close(fig)


def beta(out):
    """Beta densities in the four cases sketched in the notes, and the uniform law."""
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    t = np.linspace(0.0005, 0.9995, 2000)
    cases = [((0.5, 0.5), COBALT, r"$\alpha=\beta=1/2$"),
             ((1.0, 3.0), GOLD, r"$\alpha=1,\ \beta=3$"),
             ((1.8, 1.5), GREEN, r"$\alpha=1.8,\ \beta=1.5$"),
             ((2.0, 3.0), PURPLE, r"$\alpha=2,\ \beta=3$")]
    for (a, b), c, lab in cases:
        ax.plot(t, beta_pdf(t, a, b), color=c, label=lab)
    ax.plot([0, 1], [1, 1], color=GREY, ls="--", lw=1.1, label=r"$\alpha=\beta=1$")
    ax.set(xlim=(0, 1), ylim=(0, 3.2), xlabel="$t$", title=r"densities of $\mathrm{Beta}(\alpha,\beta)$")
    ax.legend(fontsize=7.5, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "ch09-beta.pdf")
    plt.close(fig)


def _pairs(kind, n, seed):
    rng = np.random.default_rng(seed)
    if kind == "gamma":
        return sample_gamma_beta(2.0, 3.0, 1.0, n, rng)
    if kind == "rates":
        return sample_gamma_beta(2.0, 3.0, 1.0, n, rng, lam2=3.0)
    x, y = rng.random(n), rng.random(n)
    return x + y, x / (x + y)


def gamma_beta(out):
    """(S, T) for Gamma and for uniform summands, and the distribution-function gap."""
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.45), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, kind, title in ((axes[0], "gamma", r"$\Gamma(2,1),\ \Gamma(3,1)$"),
                            (axes[1], "uniform", "uniforms on $(0,1)$")):
        s, t = _pairs(kind, 3000, 7)
        ax.scatter(s, t, s=1.2, color=COBALT, alpha=0.5, lw=0, rasterized=True)
        sb, tb = _pairs(kind, 100_000, 2026)
        r = np.corrcoef(sb, tb)[0, 1]
        ax.set(xlabel="$S=X+Y$", ylabel=r"$T=X/(X+Y)$", ylim=(0, 1), title=title)
        ax.text(0.97, 0.03, f"corr {r:+.3f}", transform=ax.transAxes, ha="right", fontsize=7.5, color=INK)
    axes[0].set_xlim(0, 15)
    axes[1].set_xlim(0, 2)
    ns = np.unique(np.geomspace(300, 1_000_000, 9).astype(int))
    ax = axes[2]
    for kind, c, lab in (("gamma", COBALT, "Gamma, same rate"), ("uniform", GOLD, "uniforms"),
                         ("rates", GREEN, "Gamma, rates 1 and 3")):
        gaps = [joint_cdf_gap(*_pairs(kind, n, 2026)) for n in ns]
        ax.plot(ns, gaps, color=c, marker="o", ms=3, label=lab)
    ax.plot(ns, 0.5 / np.sqrt(ns), color=GREY, ls=":", label=r"$0.5/\sqrt{n}$")
    ax.set(xscale="log", yscale="log", ylim=(2e-5, 0.3), xlabel="samples $n$",
           title=r"largest $|F_{(S,T)}-F_SF_T|$")
    ax.legend(fontsize=6.3, loc="lower left", handlelength=1.4)
    fig.tight_layout()
    fig.savefig(out / "ch09-gamma-beta.pdf")
    plt.close(fig)


def worked(out):
    """Three laws of the worked exercises against histograms."""
    n = 100_000
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
    ax = axes[0]
    grid = np.linspace(0, 1, 400)
    for lam, c in ((0.5, COBALT), (2.0, GOLD), (5.0, GREEN)):
        _, frac = sample_worked("integer-part", n, seed=2026, lam=lam)
        ax.hist(frac, bins=40, range=(0, 1), density=True, color=c, alpha=0.25)
        ax.plot(grid, fractional_part_pdf(grid, lam), color=c, label=rf"$\lambda={lam:g}$")
    ax.set(xlabel="$t$", title=r"$X-[X]$, $X\sim\mathrm{Exp}(\lambda)$")
    ax.legend(fontsize=7)
    ax = axes[1]
    x, y = sample_worked("laplace-uniform", n, seed=2026)
    u = np.sqrt(np.abs(x))
    grid = np.linspace(0, 3.5, 500)
    ax.hist(u, bins=60, range=(0, 3.5), density=True, color=COBALT, alpha=0.25)
    ax.hist(u + y, bins=60, range=(0, 3.5), density=True, color=GOLD, alpha=0.25)
    ax.plot(grid, sqrt_abs_laplace_pdf(grid), color=COBALT, label=r"$U=\sqrt{|X|}$")
    ax.plot(grid, laplace_uniform_sum_pdf(grid), color=GOLD, label="$V=U+Y$")
    ax.set(xlabel="$u$, $v$", title="Laplace and uniform")
    ax.legend(fontsize=7)
    ax = axes[2]
    grid = np.linspace(0.002, 1, 500)
    for m, c in ((2, COBALT), (3, GOLD), (4, GREEN)):
        prod = sample_worked("uniform-product", n, seed=2026, m=m)
        ax.hist(prod, bins=50, range=(0, 1), density=True, color=c, alpha=0.25)
        ax.plot(grid, uniform_product_pdf(grid, m), color=c, label=f"$n={m}$")
    ax.set(xlabel="$u$", ylim=(0, 6), title=r"$U_n=X_1\cdots X_n$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch09-worked.pdf")
    plt.close(fig)


def main(out):
    use_style()
    regions(out)
    beta(out)
    gamma_beta(out)
    worked(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
