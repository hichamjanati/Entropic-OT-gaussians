import numpy as np
import torch
from matplotlib import pyplot as plt
import pickle


def newton_schultz(A, sol, maxiter=100, tol=1e-4, eps=0.1):
    """Matrix square root and inverse square root for SPD matrices"""
    d = len(A)
    normA = torch.norm(A)
    Y = A / ((1 + eps) * normA)
    Id = torch.eye(d, dtype=A.dtype, device=A.device)
    normId = torch.norm(Id)
    ratio = (normA / normId) ** 0.5

    Z = Id / ((1 + eps) * normId)
    for ii in range(maxiter):
        T = 0.5 * (3 * Id - Z @ Y)
        Y = Y @ T
        Z = T @ Z
        if abs(sol - Y * ratio).mean() < tol:
            print("Converged after %d iters" % ii)
            break
    if ii == maxiter - 1:
        print("Did not converge !")
    Y = ratio * Y
    Z = Z / ratio
    return Y, Z


def evd(A):
    """Matrix square root and inverse square root for SPD matrices"""
    ei, ev = torch.symeig(A, eigenvectors=True)
    ei = torch.diag(torch.sqrt(ei))
    ei_inv = torch.diag(1. / torch.sqrt(ei))
    Asqrt = ev @ ei @ ev.t()
    Asqrt_inv = ev @ ei_inv @ ev.t()
    return Asqrt, Asqrt_inv


if __name__ == "__main__":
    from time import time
    seed = 42
    devices = ["cpu", "cuda:0"]
    dimensions = torch.linspace(2, 2000, 20).type(torch.int)
    n_trials = 50
    torch.manual_seed(seed)

    times_ns, times_evd = np.empty((2, 2, len(dimensions), n_trials))

    for dev_id, device in enumerate(devices):
        print(device)
        for dim_id, dim in enumerate(dimensions):
            print("dim = ", dim)
            for ii in range(n_trials):
                print("trial = ", ii)
                A = torch.randn(dim, dim, dtype=torch.float32, device=device)
                A = A @ A.t() + 0.1 * torch.eye(dim, dtype=torch.float32,
                                                device=device)

                t2 = time()
                Asq2, Asqinv2 = evd(A)
                t2 = time() - t2
                times_evd[dev_id, dim_id, ii] = t2

                t = time()
                Asq, Asqinv = newton_schultz(A, Asq2)
                t = time() - t
                times_ns[dev_id, dim_id, ii] = t
colors = ["cornflowerblue", "indianred"]
all_times = [times_ns, times_evd]
names = ["Newton-Schultz", "EVD"]
device_names = ["CPU", "GPU"]
data = dict(all_times=all_times, names=names, device_names=device_names,
            dimensions=dimensions, seed=seed, n_trials=n_trials)
with open("data/ns-data.pkl", "wb") as ff:
    pickle.dump(data, ff)
