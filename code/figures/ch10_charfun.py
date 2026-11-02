"""Figures for Chapter 10 (characteristic functions)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probnotes.charfun import (LAWS, cf_exponential, density_product_normals, ecf,
                               ecf_standard_error, invert_density)
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
VIOLET = "#6250D6"


def cfs(out):
    """Left: four symmetric laws have real characteristic functions. Right: Exp(1), complex,
    with the empirical characteristic function of 200 draws."""
    t = np.linspace(-12, 12, 1201)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for key, col, lab in (("normal", COBALT, r"$\mathcal{N}(0,1)$: $e^{-t^2/2}$"),
                          ("laplace", GOLD, r"Laplace: $(1+t^2)^{-1}$"),
                          ("product", GREEN, r"$(1+t^2)^{-1/2}$"),
                          ("uniform", VIOLET, r"$U(-1,1)$: $\sin t/t$")):
        a1.plot(t, LAWS[key].cf(t).real, color=col, label=lab)
    a1.axhline(0, color=GREY, lw=0.6)
    a1.set(xlabel="$t$", ylim=(-0.3, 1.05), title="symmetric laws: real $\\Phi_X$")
    a1.legend(fontsize=6.5, loc="upper right", handlelength=1.4)
    rng = np.random.default_rng(2026)
    x = rng.exponential(1.0, 200)
    ts = np.linspace(-12, 12, 49)
    e, se = ecf(x, ts), ecf_standard_error(x, ts)
    p = cf_exponential(t)
    a2.plot(t, p.real, color=COBALT, label=r"Re $\Phi_X$")
    a2.plot(t, p.imag, color=GOLD, label=r"Im $\Phi_X$")
    a2.plot(t, np.abs(p), color=GREY, ls=(0, (3, 2)), lw=1.0, label=r"$|\Phi_X|=(1+t^2)^{-1/2}$")
    a2.errorbar(ts, e.real, yerr=2 * se, fmt="o", ms=2.2, color=COBALT, elinewidth=0.6, capsize=0,
                label="empirical, $n=200$")
    a2.errorbar(ts, e.imag, yerr=2 * se, fmt="s", ms=2.2, color=GOLD, elinewidth=0.6, capsize=0)
    a2.axhline(0, color=GREY, lw=0.6)
    a2.set(xlabel="$t$", ylim=(-0.65, 1.05), title=r"$X\sim\mathrm{Exp}(1)$: $\Phi_X(t)=1/(1-it)$")
    a2.legend(fontsize=6.5, loc="upper right", handlelength=1.4)
    fig.tight_layout()
    fig.savefig(out / "ch10-cfs.pdf")
    plt.close(fig)


def inversion(out):
    """Left: the density of T = X + Y + Z of the worked exercise, by the inversion formula, against
    a histogram of 10^5 simulated values. Right: the truncated inversion integral of the
    non-integrable Exp(1) characteristic function."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    x = np.linspace(-6, 6, 481)
    rng = np.random.default_rng(2026)
    sim = LAWS["three products"].sample(rng, 100_000)
    a1.hist(sim, bins=np.linspace(-6, 6, 61), density=True, color=GREY, alpha=0.3, label="simulated, $n=10^5$")
    a1.plot(x, density_product_normals(x, 3), color=INK, lw=1.0, label=r"$|x|K_1(|x|)/\pi$")
    for L, col, ls in ((1.0, GOLD, "-"), (4.0, GREEN, "-"), (100.0, COBALT, (0, (4, 2)))):
        a1.plot(x, invert_density(LAWS["three products"].cf, x, L=L, n=4000), color=col, ls=ls,
                lw=1.3, label=f"inversion, $L={L:g}$")
    a1.set(xlabel="$x$", ylim=(-0.02, 0.42), title=r"$\Phi_T(t)=(1+t^2)^{-3/2}$, integrable")
    a1.legend(fontsize=6.3, loc="upper right", handlelength=1.4)
    x2 = np.linspace(-1.5, 3, 721)
    exact = np.where(x2 > 0, np.exp(-x2), 0.0)
    a2.plot(x2, exact, color=INK, lw=1.0, label=r"$e^{-x}\mathbf{1}_{(0,\infty)}$")
    for L, col in ((5.0, GOLD), (20.0, GREEN), (100.0, COBALT)):
        a2.plot(x2, invert_density(cf_exponential, x2, L=L, n=int(40 * L)), color=col, lw=1.1,
                label=f"$L={L:g}$")
    a2.plot([0], [0.5], "o", color=INK, ms=3.5)
    a2.annotate(r"$\frac{1}{2}$", (0, 0.5), xytext=(-0.75, 0.62), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    a2.set(xlabel="$x$", ylim=(-0.2, 1.2), title=r"$\Phi_X(t)=1/(1-it)$, not integrable")
    a2.legend(fontsize=6.5, loc="upper right", handlelength=1.4)
    fig.tight_layout()
    fig.savefig(out / "ch10-inversion.pdf")
    plt.close(fig)


def main(out):
    use_style()
    cfs(out)
    inversion(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
