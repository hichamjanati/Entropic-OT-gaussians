import numpy as np

from scipy.linalg import sqrtm
from run_sample_convergence import closed_form


def bures(A, B):
    As = sqrtm(A)
    D = sqrtm(As.dot(B).dot(As))
    loss = np.trace(A + B - 2 * D)
    return loss


def run_simulation(dim, epsilons, seed=42, device_id=0):
    device = "cpu"
    rng = np.random.RandomState(None)
    A = rng.randn(dim, dim) * 0.2
    cov_a = A @ A.T
    B = rng.randn(dim, dim) * 0.2
    cov_b = B @ B.T
    dists = np.zeros(len(epsilons))
    unreg = bures(cov_a, cov_b)
    for ii, epsilon in enumerate(epsilons):
        sigma = (epsilon / 2) ** 0.5
        current_dist = closed_form(cov_a, cov_b, sigma) - \
                       (closed_form(cov_a, cov_a, sigma) + closed_form(cov_b, cov_b, sigma)) / 2.
        dists[ii] = current_dist
    return dists, unreg


if __name__ == "__main__":
    from joblib import Parallel, delayed
    import pickle

    dimensions = [2, 5, 10]
    epsilons = np.logspace(-2, 2, 20)
    seed = 42
    rng = np.random.RandomState(seed)
    n_trials = 20

    sigmas = (epsilons / 2) ** 0.5
    params = [(dim, kk) for dim in dimensions for kk in range(n_trials)]
    pll = Parallel(n_jobs=60)
    out = pll((delayed(run_simulation)(dim, epsilons, seed, device_id)
              for device_id, (dim, _) in enumerate(params)))

    dists, unreg = list(zip(*out))
    dists = np.array(dists).reshape(len(dimensions), len(epsilons), n_trials)
    unreg = np.array(unreg).reshape(len(dimensions), n_trials)

    data = dict(unreg=unreg, dist=dists, seed=seed, dimensions=dimensions, epsilons=epsilons, n_trials=n_trials)
    with open("data/gaussiansot_sigmas.pkl", "wb") as ff:
        pickle.dump(data, ff)
