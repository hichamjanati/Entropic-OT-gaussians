import numpy as np
import torch
from sinkhorn import sinkhorn_exp, sinkhorn_log
from closed_forms import closed_form


def generate_cost(n_samples, mean_a, mean_b, cov_a, cov_b, seed=42):
    rng = np.random.RandomState(seed)
    A = rng.multivariate_normal(mean_a, cov_a, size=n_samples)
    B = rng.multivariate_normal(mean_b, cov_b, size=n_samples)
    C = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return C


def run_simulation(dim, n_samples, epsilon, gamma, mass_b, seed=None):
    print(dim, epsilon, gamma)
    device = "cpu"
    # here we keep the seed fixed once and for all for the matrices
    # randomness is kept for different sampling trials controlled by
    # the seed argument which is different for each call of this function
    params_rng = np.random.RandomState(42)
    A = params_rng.randn(dim, dim) * 0.2
    cov_a = A @ A.T
    B = params_rng.randn(dim, dim) * 0.2
    cov_b = B @ B.T
    means = params_rng.rand(2, dim) * 2 - 1
    mean_a, mean_b = means
    C = generate_cost(n_samples[-1], mean_a, mean_b, cov_a, cov_b, seed=seed)
    C = torch.tensor(C, dtype=torch.float32, device=device)
    sigma = (epsilon / 2) ** 0.5
    theoretical = closed_form(cov_a, cov_b, sigma, mean_a, mean_b, gamma=gamma,
                              mass_a=1, mass_b=mass_b)
    empirical = np.zeros(len(n_samples))
    for ii, n in enumerate(n_samples):
        C_ = C[:n][:, :n]
        current_emp = sinkhorn_exp(C_, epsilon, gamma, mass_b=mass_b,
                                   device=device)
        if np.isnan(current_emp):
            current_emp = sinkhorn_log(C_, epsilon, gamma, mass_b=mass_b,
                                       device=device)
        empirical[ii] = current_emp
    return theoretical, empirical, means


if __name__ == "__main__":
    from joblib import Parallel, delayed
    import pickle

    dimensions = [5, 10]
    n_samples = np.logspace(1, 3.5, 20).astype(int)
    seed = 42
    rng = np.random.RandomState(seed)
    n_trials = 20
    seeds = rng.randint(1000, size=n_trials)

    epsilons = np.array([0.5, 1., 5.])
    gammas_masses = [(None, 1.), (1., 2.)]
    sigmas = (epsilons / 2) ** 0.5
    params = [(dim, eps, gamma, mass_b, seed) for dim in dimensions
              for eps in epsilons for gamma, mass_b in gammas_masses
              for seed in seeds]
    pll = Parallel(n_jobs=60)
    out = pll((delayed(run_simulation)(dim, n_samples, epsilon, gamma, mass_b,
                                       seed)
              for dim, epsilon, gamma, mass_b, seed in params))

    theoretical, empirical, means = list(zip(*out))
    shape_theo = (len(dimensions), len(epsilons), len(gammas_masses),
                  n_trials)
    shape_empirical = (len(dimensions), len(epsilons), len(gammas_masses),
                       len(n_samples), n_trials)
    theoretical = np.array(theoretical).reshape(shape_theo)
    empirical = np.stack(empirical).reshape(shape_empirical)
    data = dict(exp=empirical, theory=theoretical, seed=seed,
                dimensions=dimensions, epsilons=epsilons, n_trials=n_trials,
                n_samples=n_samples, gammas_masses=gammas_masses, means=means)
    with open("data/gaussiansot.pkl", "wb") as ff:
        pickle.dump(data, ff)
