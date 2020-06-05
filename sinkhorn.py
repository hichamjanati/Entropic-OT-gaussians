import numpy as np
import torch


def sinkhorn_log(C, epsilon, gamma=None, mass_a=1., mass_b=1.,
                 maxiter=5000, tol=1e-4, device="cpu", return_plan=False,
                 w_a=None, w_b=None):
    n_samples_a, n_samples_b = C.shape
    C = C / epsilon
    v = torch.zeros(n_samples_b, dtype=torch.float32,
                    device=device)
    # weights args inside logsumexp
    if w_a is None:
        w_a = torch.ones(n_samples_a, dtype=torch.float32, device=device) \
            * np.log(mass_a / n_samples_a)
    if w_b is None:
        w_b = torch.ones(n_samples_b, dtype=torch.float32, device=device) \
            * np.log(mass_b / n_samples_b)

    # dual potentials are considered divided by eps
    loss = torch.ones(1) * 1e10
    if gamma is None:
        tau = 1.
        assert mass_a == mass_b

    else:
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

    if return_plan or gamma:
        log_plan = - C + v[None, :] + u[:, None]
        log_plan += w_a[:, None] + w_b[None, :]
        plan = torch.exp(log_plan)

    if gamma:
        plan_mass = plan.sum()
        loss = gamma * (mass_a + mass_b - 2 * plan_mass)
        loss += epsilon * (mass_a * mass_b - plan_mass)
    else:
        loss = (u.mean() + v.mean()) * epsilon
    if return_plan:
        return loss.item(), plan.numpy()
    return loss.item(), u, v


def sinkhorn_exp(C, epsilon, gamma=None, mass_a=1, mass_b=1, maxiter=5000,
                 tol=1e-4, device="cpu", return_plan=False,
                 w_a=None, w_b=None):
    dim, _ = C.shape
    K = torch.exp(- C / epsilon)
    if gamma is None:
        tau = 1.
        assert mass_a == mass_b
    else:
        tau = gamma / (gamma + epsilon)
    v = torch.ones(dim, device=device)
    # weights args inside logsumexp
    if w_a is None:
        w_a = torch.ones(dim, device=device) * mass_a / dim
    if w_b is None:
        w_b = torch.ones(dim, device=device) * mass_b / dim

    Kv = K.mv(v * w_a)
    # dual potentials are considered divided by eps
    for ii in range(maxiter):
        vold = v.clone()
        u = (1. / Kv) ** tau
        Ku = K.t().mv(u * w_a)
        v = (1. / Ku) ** tau
        Kv = K.mv(v * w_b)
        err = abs(v - vold).max()
        err /= max(1., abs(v).max())
        if err < tol and ii > 1:
            # print("Converged after %s iterations." % ii)
            break
        if np.isnan(err.item()):
            if return_plan:
                return err.item(), np.zeros(2)
            return err.item()
    if ii == maxiter - 1:
        print("Sinkhorn exp did not converge. Last err: %s" % err)

    if gamma or return_plan:
        plan = (w_a[:, None] * u[:, None] * K * v[None, :] * w_b[None, :])
    if gamma is None:
        # weights are uniform so take means of potentials
        loss = torch.log(u).mean() + torch.log(v).mean()
        # remultiply by eps
        loss *= epsilon
    else:
        plan_mass = (w_a * u * Kv).sum()
        loss = gamma * (mass_a + mass_b - 2 * plan_mass)
        loss += epsilon * (mass_a * mass_b - plan_mass)
    if return_plan:
        return loss.item(), plan.numpy()
    return loss.item(), u, v
