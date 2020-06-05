#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy.stats import norm, multivariate_normal
from scipy.linalg import sqrtm, inv

from scipy.special import logsumexp


def sinkhorn_plan(x, y, sigma, gamma=None, maxiter=30000, tol=1e-6):
    n_samples_x, dim = x.shape
    n_samples_y, dim_y = y.shape
    assert dim == dim_y

    C = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)

    epsilon = 2 * sigma ** 2
    if gamma:
        tau = gamma / (gamma + epsilon)
    else:
        tau = 1
    v = np.zeros(n_samples_x)
    # weights defined in log-domain
    w_x = - np.log(n_samples_x) * np.ones(n_samples_x)
    w_y = - np.log(n_samples_y) * np.ones(n_samples_y)
    # dual potentials are considered divided by eps
    for ii in range(maxiter):
        vold = v.copy()
        u = - tau * epsilon * logsumexp((- C + v[None, :]) / epsilon
                                        + w_y[None, :], axis=1)
        v = - tau * epsilon * logsumexp((- C + u[:, None]) / epsilon
                                        + w_x[:, None], axis=0)
        err = abs(v - vold).max()
        err /= max(1., abs(v).max())
        if err < tol and ii > 1:
            # print("Converged after %s iterations." % ii)
            break
    if ii == maxiter - 1:
        print("Sinkhorn did not converge. Last err: %s" % err)

    # #### compute Unbalanced loss exactly (without assuming optimality)
    # to be sure
    log_plan = (- C + v[None, :] + u[:, None]) / epsilon
    log_plan += w_x[:, None] + w_y[None, :]
    plan = np.exp(log_plan)

    return plan


def get_balanced_cov(cov_a, cov_b, sigma=0.1):
    n = len(cov_a)
    Id = np.eye(n)

    P = np.zeros((2 * n, 2 * n))
    P[:n, :n] = cov_a
    P[n:, n:] = cov_b

    sA = sqrtm(cov_a)
    D = sqrtm(4 * sA.dot(cov_b).dot(sA) + sigma ** 2 * Id)
    C = (sA.dot(D).dot(np.linalg.inv(sA)) - sigma * Id) / 2.

    P[:n, n:] = C
    P[n:, :n] = C.T
    return P


def get_unbalanced_cov(A, B, gamma=1., sigma=0.1):
    n = len(A)
    Id = np.eye(n)
    s = sigma ** 2
    lb = 2 / (2 * s + gamma)
    X = A + B + lb * Id
    P = np.zeros((2 * n, 2 * n))
    P[:n, :n] = A
    P[n:, n:] = B

    sA = sqrtm(A)
    D = sqrtm(4 * sA.dot(B).dot(sA) + sigma ** 2 * Id)
    C = (sA.dot(D).dot(np.linalg.inv(sA)) - sigma * Id) / 2.

    P[:n, n:] = C
    P[n:, :n] = C.T
    return P


def get_density(cov_a, cov_b, mean_a, mean_b, sigma, grid,
                threshold=1e-2):
    plan_cov = get_balanced_cov(cov_a, cov_b, sigma)
    plan_mean = np.concatenate([mean_a, mean_b]).flatten()
    pi = multivariate_normal(plan_mean, plan_cov)
    xp, yp = grid
    pos = np.empty(xp.shape + (2,))
    pos[:, :, 0] = xp
    pos[:, :, 1] = yp
    pi_vals = pi.pdf(pos)
    pi_vals[pi_vals < threshold] = 0.
    return pi_vals


if __name__ == "__main__":
    eps = [5.]
    seed = 42
    n_samples = 200
    mu_a, mu_b = 0., 0.
    sigma_a, sigma_b = 0.5, 1.
    cov_a = np.eye(1) * sigma_a
    cov_b = np.eye(1) * sigma_b
    mean_a = np.ones(1) * mu_a
    mean_b = np.ones(1) * mu_b
    rv_a = norm(loc=mu_a, scale=sigma_a)
    rv_b = norm(loc=mu_b, scale=sigma_b)

    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    rng = np.randon.RandomState(seed)

    # Plot grid figure -- the figsize and gridspec generation parts
    # are not super clean ##

    plt.clf()
    fig = plt.figure(figsize=(3 * len(eps) + 1, 4))
    gs = gridspec.GridSpec(4, 3 * len(eps) + 1, wspace=0.0, hspace=0.0)

    x = np.linspace(-3, 3, 1000)

    ax2 = plt.subplot(gs[1:, :1])
    ax2.plot(rv_a.pdf(x), x, 'b', label='Source distribution')
    ax2.fill_between(rv_a.pdf(x)[:], x[:], color='b', interpolate=True)
    ax2.set_xlim(xmin=0)
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax2.axis('off')

    levels = np.linspace(0, 1, 40)[1:]
    xp, yp = np.mgrid[-3:3:.005, -3:3:.005]

    for i, e in enumerate(eps):
        sigma = (e / 2) ** 0.5
        pi_vals = get_density(cov_a, cov_b, mean_a, mean_b, sigma, (xp, yp),
                              gamma=None)
        ax1 = plt.subplot(gs[0, 1 + 3 * i: 1 + 3 * (i + 1)])
        ax1.plot(x, rv_b.pdf(x), 'r', label='Target distribution')
        ax1.fill_between(x, rv_b.pdf(x), color='r')
        ax1.set_ylim(ymin=0)
        ax1.set_xlim(xmin=-3, xmax=3)
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)
        ax1.axis('off')

        ax3 = plt.subplot(gs[1:, 1 + 3 * i: 1 + 3 * (i + 1)], sharex=ax1,
                          sharey=ax2)

        ax3.contourf(xp, yp, pi_vals, cmap=plt.get_cmap('Reds'),
                     vmin=levels[0],
                     vmax=pi_vals.max(), levels=levels)
        ax3.contour(xp, yp, pi_vals, cmap=plt.get_cmap('Reds'),
                    vmin=levels[0],
                    vmax=pi_vals.max(), levels=levels)
        ax3.set_ylim(ymin=3)
        ax3.set_xlim(xmin=-3, xmax=3)
        ax3.set_title(r"$\sigma^2 =$ {}".format(e), y=-0.15)
        ax3.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)

    fig.patch.set_alpha(0)
    plt.savefig('fig/sigma_grid.pdf', bbox_inches='tight')
    plt.show()
    plt.close("all")
