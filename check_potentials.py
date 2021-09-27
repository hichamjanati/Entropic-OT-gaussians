import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import cm


def sinkhorn_log(a, b, C, epsilon, gamma=1., mass_a=1., mass_b=1.,
                 maxiter=5000, tol=1e-7, device="cpu", return_plan=False,
                 w_a=None, w_b=None):
    n_samples_a, n_samples_b = C.shape
    C = C / epsilon
    v = torch.zeros(n_samples_b, dtype=torch.float32,
                    device=device)
    # dual potentials are considered divided by eps
    loss = torch.ones(1) * 1e10
    w_a = torch.log(a + 1e-60)
    w_b = torch.log(b + 1e-60)
    mass_a = a.sum()
    mass_b = b.sum()
    tau = gamma / (gamma + epsilon)
    for ii in range(maxiter):
        vold = v.clone()
        u = - tau * torch.logsumexp(- C + v[None, :] + w_b[None, :], dim=1)
        v = - tau * torch.logsumexp(- C + u[:, None] + w_a[:, None], dim=0)
        # weights are uniform so take means of potentials
        err = abs(v / vold - 1).max()
        if err < tol and ii > 1:
            # print("Converged after %s iterations." % ii)
            break
    if ii == maxiter - 1:
        print("Sinkhorn log did not converge. Last err: %s" % err)
    print(ii, err)
    log_plan = - C + v[None, :] + u[:, None]
    log_plan += w_a[:, None] + w_b[None, :]
    plan = torch.exp(log_plan)

    plan_mass = plan.sum()
    loss = gamma * (mass_a + mass_b - 2 * plan_mass)
    loss += epsilon * (mass_a * mass_b - plan_mass)

    return u * epsilon, v * epsilon


def gaussian_density(mean, var, grid, mass):
    pdf = np.exp(- ((grid - mean.squeeze()) ** 2 / (2 * var.squeeze())))
    pdf /= pdf.sum()
    pdf *= mass
    pdf = torch.tensor(pdf, device="cpu")
    return pdf


def get_potentials(grid, gamma, A, a, B, b, mass_a, mass_b):
    At = 2 / gamma * A + 1
    Bt = 2 / gamma * B + 1
    C = ((A * B) / (At * Bt)) ** 0.5
    F = B / (C * Bt)
    u_mat = F - 1
    u_mean = C / (1 + 2 / gamma * C) * (a / A - b / (C * Bt))
    v_mat = 1 / F - 1
    v_mean = C / (1 + 2 / gamma * C) * (b / B - a / (C * At))
    cst = np.log(Bt * mass_a / (At * mass_b)) + (a + v_mean) * a / A - (b + u_mean) * b / B
    u_logmass = gamma / 8 * (cst + 2/gamma * u_mean * v_mean)
    v_logmass = gamma / 8 * (-cst + 2/gamma * u_mean * v_mean)
    f = - u_mat * grid ** 2 + 2 * u_mean * grid + 2 * u_logmass
    g = - v_mat * grid ** 2 + 2 * v_mean * grid + 2 * v_logmass
    return f.squeeze(), g.squeeze()


if __name__ == "__main__":
    grid = np.linspace(-5, 5, 300)
    C = (grid[:, None] - grid[None, :]) ** 2
    C = torch.tensor(C, device="cpu")
    mean_a, mean_b = np.array([-0.5]), np.array([0.5])
    mass_a, mass_b = 1., 2.
    cov_a, cov_b = np.eye(1) * 0.4, np.eye(1) * 0.3
    a = gaussian_density(mean_a, cov_a, grid, mass_a)
    b = gaussian_density(mean_b, cov_b, grid, mass_b)

    gamma = 1.

    theo_f, theo_g = get_potentials(grid, gamma, cov_a.flatten(), mean_a.flatten(), cov_b.flatten(), mean_b.flatten(), mass_a, mass_b)
    epsilons = [1., 0.5, 0.1, 0.05, 0.01, 0.005]
    colors = cm.Reds(np.linspace(0.1, 1., len(epsilons)))
    fs = []
    gs = []
    losses = []
    for eps in epsilons:
        f, g = sinkhorn_log(a, b, C, eps, gamma)
        fs.append(f)
        gs.append(g)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, theo, label in zip(axes, [theo_f, theo_g], ["f(x)", "g(x)"]):
        ax.plot(grid, theo, lw=3, color="k", label=r"theoretical ($\varepsilon=0$)")
        ax.set_ylabel(label)
        ax.grid()
    
    for ax, curves in zip(axes, [fs, gs]):
        for curve, color, eps in zip(curves, colors, epsilons):
            ax.plot(grid, curve, color=color, lw=2, label=r"$\varepsilon = %.4f$" % eps)
    plt.legend()
    plt.show()
    