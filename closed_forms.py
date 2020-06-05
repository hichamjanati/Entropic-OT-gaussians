import numpy as np
from scipy.linalg import sqrtm, det, inv


def bures(A, B):
    sqA = sqrtm(A)
    C = sqA.dot(B).dot(sqA)
    sqC = sqrtm(C)
    obj = np.diag(A + B - 2 * sqC).sum()
    return obj


def closed_form_balanced(cov_a, cov_b, sigma, mean_a=None, mean_b=None,
                         return_params=False):
    n = len(cov_a)
    if mean_a is None:
        mean_a = np.zeros(n)
    if mean_b is None:
        mean_b = np.zeros(n)
    if sigma == 0:
        loss = bures(cov_a, cov_b)
        loss += np.linalg.norm(mean_a - mean_b) ** 2
        return loss

    Id = np.eye(n)
    cov_as = sqrtm(cov_a)
    D = sqrtm(4 * cov_as.dot(cov_b).dot(cov_as) + sigma ** 4 * Id)
    loss = np.trace(cov_a + cov_b - D) + sigma ** 2 * n
    loss += - n * sigma ** 2 * np.log(2 * sigma ** 2)
    loss += sigma ** 2 * np.log(np.linalg.det(D + sigma ** 2 * Id))
    loss += np.linalg.norm(mean_a - mean_b) ** 2
    if return_params:
        scov_a = sqrtm(cov_a)
        C = (scov_a.dot(D).dot(np.linalg.inv(scov_a)) - sigma ** 2 * Id) / 2.
        cov_plan = np.block([[cov_a, C], [C.T, cov_b]])
        cov_means = np.concatenate((mean_a, mean_b)).flatten()
        return loss, cov_plan, cov_means, 1.
    return loss


def _get_matrices_unbalanced(cov_a, cov_b, s, gamma):
    Id = np.eye(len(cov_a))
    lb = s + gamma / 2
    tau = gamma / (2 * s + gamma)
    cov_at = 0.5 * gamma * (Id - lb * inv(cov_a + lb * Id))
    cov_bt = 0.5 * gamma * (Id - lb * inv(cov_b + lb * Id))
    C = sqrtm(cov_at.dot(cov_bt) / tau + s ** 2 / 4 * Id) - s / 2 * Id
    Cinv = inv(C)
    F = cov_bt.dot(Cinv)
    G = Cinv.dot(cov_at)

    return C, F, G, cov_at, cov_bt


def closed_form_unbalanced(cov_a, cov_b, sigma, mean_a=None, mean_b=None,
                           return_params=False,
                           gamma=None, mass_a=1, mass_b=1):
    if sigma <= 0:
        raise ValueError("sigma must be positive to compute UOT.")
    d = len(cov_a)
    if mean_a is None:
        mean_a = np.zeros(d)
    if mean_b is None:
        mean_b = np.zeros(d)
    Id = np.eye(d)
    s = sigma ** 2
    tau = gamma / (gamma + 2 * s)
    C, F, G, cov_at, cov_bt = _get_matrices_unbalanced(cov_a, cov_b, s, gamma)
    lb = s + gamma / 2
    X = cov_a + cov_b + lb * Id
    Xinv = inv(X)
    det_ab = det(cov_a.dot(cov_b))
    det_atbt = det(cov_at.dot(cov_bt))

    diff = mean_a - mean_b
    exp_mass = np.exp(- 0.5 * diff.dot(Xinv.dot(diff)) / (tau + 1))
    s_power = d * s / (s + gamma) * np.log(sigma)
    num = mass_a * mass_b * det(C) * (det_atbt ** tau / det_ab) ** 0.5
    num **= 1 / (tau + 1)
    num *= np.exp(s_power)
    num *= exp_mass
    den = det(C - 2 / gamma * cov_at.dot(cov_bt)) ** 0.5
    plan_mass = num / den

    # UOT at optimality depends on the mass of the plan and those of the inputs
    loss = gamma * (mass_a + mass_b - 2 * plan_mass)
    loss += 2 * s * (mass_a * mass_b - plan_mass)

    if return_params:
        H1 = (Id + C / lb).dot(cov_a - cov_a.dot(Xinv).dot(cov_a))
        H4 = (Id + C.T / lb).dot(cov_b - cov_b.dot(Xinv).dot(cov_b))
        H2 = C + (Id + C / lb).dot(cov_a.dot(Xinv).dot(cov_b))
        H3 = C.T + (Id + C.T / lb).dot(cov_b).dot(Xinv).dot(cov_a)
        plan_cov = np.block([[H1, H2], [H3, H4]])
        plan_mean_a = mean_a + cov_a.dot(Xinv).dot(mean_b - mean_a)
        plan_mean_b = mean_b + cov_b.dot(Xinv).dot(mean_a - mean_b)
        plan_mean = np.concatenate([plan_mean_a, plan_mean_b]).flatten()

        return loss, plan_cov, plan_mean, plan_mass
    return loss


def closed_form(cov_a, cov_b, sigma, mean_a=None, mean_b=None,
                return_params=False,
                gamma=None, **params):
    if gamma:
        return closed_form_unbalanced(cov_a, cov_b, sigma, mean_a, mean_b,
                                      return_params=return_params, gamma=gamma,
                                      **params)
    else:
        return closed_form_balanced(cov_a, cov_b, sigma, mean_a, mean_b,
                                    return_params)
