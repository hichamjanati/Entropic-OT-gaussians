#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy.stats import norm, multivariate_normal

from sinkhorn import sinkhorn_log
from closed_forms import closed_form


def get_theoretical_weights(sigma_a, sigma_b, mean_a, mean_b, mass_a, mass_b,
                            epsilon, gamma, grid):
    cov_a = np.array([[sigma_a ** 2]])
    cov_b = np.array([[sigma_b ** 2]])
    mean_a = np.array([mean_a])
    mean_b = np.array([mean_b])

    sigma = (epsilon * 0.5) ** 0.5
    _, plan_cov, plan_mean, plan_mass = closed_form(cov_a, cov_b, sigma,
                                                    mean_a, mean_b,
                                                    return_params=True,
                                                    gamma=gamma)
    pi = multivariate_normal(plan_mean, plan_cov)
    xp, yp = np.meshgrid(grid, grid)
    pos = np.empty(xp.shape + (2,))
    pos[:, :, 0] = xp
    pos[:, :, 1] = yp
    weights = pi.pdf(pos) * plan_mass
    return weights.T


def simulate_1d_gaussians(sigma_a, sigma_b, mean_a, mean_b, mass_a, mass_b,
                          n_samples, bins, seed=42):
    rng = np.random.RandomState(seed)
    samples_a = rng.randn(n_samples) * sigma_a + mean_a
    samples_b = rng.randn(n_samples) * sigma_b + mean_b
    range_min = min(samples_a.min(), samples_b.min()) - 0.1
    range_max = max(samples_a.max(), samples_b.max()) + 0.1
    subsampled_grid = np.linspace(range_min, range_max, bins)
    rv_a = norm(loc=mean_a, scale=sigma_a)
    rv_b = norm(loc=mean_b, scale=sigma_b)
    pdf_a = rv_a.pdf(subsampled_grid) * mass_a
    pdf_b = rv_b.pdf(subsampled_grid) * mass_b
    return ((samples_a, samples_b), (pdf_a, subsampled_grid),
            (subsampled_grid, pdf_b))


def get_empirical_weights(samples_a, samples_b, mass_a, mass_b, epsilon, gamma,
                          grid_range, bins):
    C = ((samples_a[:, None, None] - samples_b[None, :, None]) ** 2).sum(-1)
    C = torch.tensor(C, dtype=torch.float32)
    loss, weights = sinkhorn_log(C, epsilon, gamma, mass_a, mass_b,
                                 return_plan=True)
    samples_plan_x, samples_plan_y = np.meshgrid(samples_a, samples_b)
    samples_plan_x = samples_plan_x.flatten()
    samples_plan_y = samples_plan_y.flatten()
    range_mat = np.array([grid_range, grid_range])
    weights = weights.T
    H, xedges, yedges = np.histogram2d(samples_plan_x, samples_plan_y,
                                       bins=bins, density=True,
                                       range=range_mat,
                                       weights=weights.flatten())
    return H, xedges, yedges


if __name__ == "__main__":
    seed = 42
    n_samples = 2000
    bins = 100
    mean_a, mean_b = 0., 0.5
    sigma_a, sigma_b = 0.2, 0.3
    mass_a = 1.
    epsilons = [0.02, 0.1, 0.25, 1]
    gammas = [0.001, 0.01, 0.25, 1.]
    params = [(e, None, 1.) for e in epsilons]
    params += [(0.1, g, 2.) for g in gammas]
    max_mass = 2.5
    fig_names = list("abcdefghijklmnop")
    titles = len(epsilons) * ["Balanced | "] + len(gammas) * ["Unbalanced | "]

    fontsize = 15
    plt_params = {'axes.labelsize': fontsize + 1,
                  'font.size': fontsize,
                  'legend.fontsize': fontsize + 2,
                  'xtick.labelsize': fontsize + 1,
                  'ytick.labelsize': fontsize + 2,
                  'figure.titlesize': fontsize + 2,
                  'pdf.fonttype': 42}
    plt.rcParams.update(plt_params)

    for ii, (param, fig_name, title) in enumerate(zip(params, fig_names,
                                                      titles)):
        epsilon, gamma, mass_b = param
        sigma = (epsilon / 2) ** 0.5
        # Get 1d Gaussian samples and their evaluated pdf ready to plot
        samples, pdf_data_a, pdf_data_b = \
            simulate_1d_gaussians(sigma_a, sigma_b, mean_a, mean_b, mass_a,
                                  mass_b, n_samples, bins, seed=seed)
        samples_a, samples_b = samples

        # subsampled grid corresponds to a uniform common grid used for all
        # histograms with the specified bins argument
        subsampled_grid = pdf_data_b[0]
        grid_range = [subsampled_grid.min(), subsampled_grid.max()]

        # get empirical weights by computing a 2d histogram
        plan_weights, xedges, yedges = get_empirical_weights(samples_a,
                                                             samples_b,
                                                             mass_a, mass_b,
                                                             epsilon, gamma,
                                                             grid_range, bins)
        # evaluate the theoretical pdf
        expected_weights = get_theoretical_weights(sigma_a, sigma_b, mean_a,
                                                   mean_b, mass_a, mass_b,
                                                   epsilon, gamma,
                                                   subsampled_grid)

        # plt.clf()
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5, wspace=0.0, hspace=0.0)

        ax1 = plt.subplot(gs[0, 1:])
        ax1.plot(*pdf_data_b, 'black', label='Target distribution', lw=3)
        hist_a = np.histogram(samples_a, bins=bins, range=grid_range)
        hist_b = np.histogram(samples_b, bins=bins, range=grid_range)
        integral_a = hist_a[0].dot(np.diff(hist_a[1]))
        integral_b = hist_b[0].dot(np.diff(hist_b[1]))
        weights_a = mass_a * np.ones_like(samples_a) / integral_a
        weights_b = mass_b * np.ones_like(samples_b) / integral_b

        ax1.hist(samples_b, bins=bins,
                 range=grid_range, color="indianred", weights=weights_b)
        ax1.axis('off')
        ylim = np.array(ax1.get_ylim())
        if gamma is None:
            ylim *= max_mass
            ax1.set_ylim(ylim)
        ax2 = plt.subplot(gs[1:, :1])
        ax2.hist(samples_a, bins=bins, orientation="horizontal",
                 range=grid_range, color="indianred", weights=weights_a)

        ax2.invert_xaxis()
        ax2.invert_yaxis()
        ax2.plot(*pdf_data_a, 'black', lw=3, label='Source distribution')
        ax2.axis('off')

        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)

        ax3 = plt.subplot(gs[1:, 1: 5], sharex=ax1, sharey=ax2)
        alphas = plan_weights / plan_weights.max()
        ax3.imshow(alphas, interpolation='nearest', origin='low',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap="Reds", vmin=0., alpha=alphas, zorder=10)
        ax3.grid(True)
        colors = np.linspace(0., 0.5, 4)[::-1]
        colors = colors[:, None] * np.ones(3)[None, :]
        ax3.contour(subsampled_grid, subsampled_grid,
                    expected_weights, colors=colors,
                    levels=4, vmin=0., alpha=0.8)
        ax3.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)
        fig.patch.set_alpha(0)
        if gamma:
            title += r"$\gamma = %s$ | $\varepsilon = %s$" % (gamma, epsilon)
        else:
            title += r"$\varepsilon = %s$" % epsilon
        if gamma is None:
            plt.title(title, y=1.12)
        else:
            plt.title(title, y=1.25)
        plt.savefig('fig/fig2%s.pdf' % fig_name, bbox_inches='tight')
    plt.close("all")
