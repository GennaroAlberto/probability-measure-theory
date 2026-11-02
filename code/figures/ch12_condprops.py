"""Figures for Chapter 12 (properties of conditional expectation)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from probnotes.condprops import (affine_predictor, bin_index, bin_regression, gamma_posterior,
                                 mse, negbin_pmf, sample_poisson_gamma, sample_signal, signal_mse,
                                 signal_posterior_mean)
from probnotes.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def quantile_edges(y, k):
    """Edges of ``k`` bins with the same number of samples of ``y``; outer edges at -inf, +inf."""
    e = np.quantile(y, np.linspace(0, 1, k + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def bin_regression_mse(x_fit, y_fit, x_test, y_test, k):
    """Out-of-sample mean squared error of the regression on the indicators of k bins of Y."""
    edges = quantile_edges(y_fit, k)
    coef = bin_regression(x_fit, y_fit, edges)
    return mse(x_test, coef[bin_index(y_test, edges)])


def projection(out, sigma=1.0, seed=2026):
    """E[X | Y], the best affine predictor and bin regressions for X ~ Exp(1), Y = X + N(0, 1)."""
    x, y = sample_signal(100_000, sigma, seed=seed)
    xt, yt = sample_signal(1_000_000, sigma, seed=seed + 1)
    a, b = affine_predictor(x, y)
    edges = quantile_edges(y, 12)
    coef = bin_regression(x, y, edges)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    a1.scatter(y[:1500], x[:1500], s=1.6, color=GREY, alpha=0.45, lw=0, rasterized=True)
    t = np.linspace(-3, 6, 600)
    a1.plot(t, signal_posterior_mean(t, sigma), color=COBALT, lw=1.7,
            label=r"$h(y)=\mathbb{E}[X\mid Y=y]$")
    a1.plot(t, a + b * t, color=GOLD, lw=1.4, ls="--", label=r"best affine $a+by$")
    step = coef[bin_index(t, edges)]
    a1.plot(t, step, color=GREEN, lw=1.2, drawstyle="steps-post", label="regression on 12 bins")
    a1.set(xlim=(-3, 6), ylim=(-0.2, 6), xlabel="$y$", ylabel="$x$", title="predicting $X$ from $Y$")
    a1.legend(fontsize=6.8, loc="upper left")
    ks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    for n_fit, colour, marker in ((2000, GOLD, "s"), (100_000, COBALT, "o")):
        e = [bin_regression_mse(x[:n_fit], y[:n_fit], xt, yt, k) for k in ks]
        a2.plot(ks, e, marker=marker, ms=3.5, color=colour, lw=1.2, label=f"bins fitted on {n_fit:,} pairs")
    a2.axhline(signal_mse(sigma), color=INK, lw=0.9, label=r"$\mathbb{E}[(X-h(Y))^2]$")
    a2.axhline(sigma ** 2 / (1 + sigma ** 2), color=GREY, lw=0.9, ls="--", label="best affine")
    a2.set(xscale="log", xlabel="number of bins $K$", ylabel="mean squared error",
           title="error on new data", ylim=(0.38, 1.02))
    a2.set_xticks(ks, [str(k) for k in ks])
    a2.minorticks_off()
    a2.legend(fontsize=6.8)
    fig.tight_layout()
    fig.savefig(out / "ch12-projection.pdf")
    plt.close(fig)


def poisson_gamma(out, alpha=2.0, lam=0.5, n=100_000, seed=2026):
    """The negative binomial law of Y and the Gamma laws of X given Y = k, against simulation."""
    x, y = sample_poisson_gamma(n, alpha, lam, seed=seed)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    k = np.arange(0, 21)
    a1.bar(k, negbin_pmf(k, alpha, lam), color=COBALT, alpha=0.35, width=0.75,
           label=r"$BN(2,\frac{1}{3})$")
    a1.plot(k, np.bincount(y, minlength=k.size)[: k.size] / n, "D", color=GREEN, ms=3,
            label=f"simulation ($10^5$ pairs)")
    a1.plot(k, stats.poisson.pmf(k, alpha / lam), "o", color=GOLD, ms=3.5,
            label=r"$\mathcal{P}(4)$, same mean")
    a1.set(xlabel="$k$", title=r"$\mathbb{P}(Y=k)$", xticks=np.arange(0, 21, 4))
    a1.legend(fontsize=6.8)
    t = np.linspace(0, 16, 500)
    a2.plot(t, stats.gamma.pdf(t, alpha, scale=1 / lam), color=GREY, lw=1.3,
            label=r"$f_X$: $\Gamma(2,\frac{1}{2})$")
    for kk, colour in zip((0, 4, 10), (COBALT, GOLD, GREEN)):
        shape, rate = gamma_posterior(kk, alpha, lam)
        sel = x[y == kk]
        a2.hist(sel, bins=np.linspace(0, 16, 49), density=True, color=colour, alpha=0.2)
        a2.plot(t, stats.gamma.pdf(t, shape, scale=1 / rate), color=colour, lw=1.5,
                label=f"given $Y={kk}$ ({sel.size} pairs)")
        a2.axvline(shape / rate, color=colour, lw=0.9, ls="--")
    a2.set(xlabel="$x$", title=r"$f_{X\mid Y=k}$: $\Gamma(2+k,\frac{3}{2})$", xlim=(0, 16))
    a2.legend(fontsize=6.8)
    fig.tight_layout()
    fig.savefig(out / "ch12-poisson-gamma.pdf")
    plt.close(fig)


def main(out):
    use_style()
    projection(out)
    poisson_gamma(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
