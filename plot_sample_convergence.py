import numpy as np
from matplotlib import pyplot as plt
import pickle


fontsize = 12
params = {'axes.labelsize': fontsize + 1,
          'font.size': fontsize,
          'legend.fontsize': fontsize + 2,
          'xtick.labelsize': fontsize + 1,
          'ytick.labelsize': fontsize + 2,
          'pdf.fonttype': 42,
          'figure.titlesize': 36}
plt.rcParams.update(params)


with open("data/gaussiansot.pkl", "rb") as ff:
    data = pickle.load(ff)

# theoretical values are equal across trials check it first
# then take the first
theory = data["theory"]
assert abs(theory - theory[:, :, :, :1]).max() == 0
theory = theory[:, :, :, 0]

exp = data["exp"]

n_samples = data["n_samples"]
dimensions = data["dimensions"]
epsilons = data["epsilons"]
n_dims, n_eps, n_gammas, n_trials, _ = exp.shape
gammas_masses = data["gammas_masses"]
means = data["means"]
colors = ["cornflowerblue", "indianred", "gold"]

# reshape balanced and unbalanced in one row
exp = np.transpose(exp, (2, 0, 1, 3, 4))
exp = exp.reshape(4, *exp.shape[2:])
theo = np.transpose(theory, (2, 0, 1))
theo = theo.reshape(4, -1)
ot_types = ["Balanced", "Balanced", "Unbalanced", "Unbalanced"]

plt.clf()

f, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True)
for ii, (ax, emp_dim, theo_dim, ot_type) in enumerate(zip(axes, exp, theo,
                                                      ot_types)):
    dim = dimensions[ii % 2]
    for color, eps, theo, emp in zip(colors, epsilons, theo_dim, emp_dim):
        sigma = (eps / 2) ** 0.5
        upper = emp.mean(0) + emp.std(0)
        lower = emp.mean(0) - emp.std(0)
        ax.plot(n_samples, emp.mean(0),
                color=color, lw=2,
                label=r"$\varepsilon = %s$" % np.round(eps, 2),
                alpha=0.7)
        hline = ax.hlines(theo, xmin=n_samples[0], xmax=n_samples[-1],
                          color="black", ls="dashed", lw=1)
        ax.fill_between(n_samples, lower, upper,
                        color=color, alpha=.3)
        ax.grid(True)
        ax.set_xscale("log")
        ax.set_xlabel("# of samples", fontsize=16, labelpad=0)
        # if ot_type == "Unbalanced":
        #     ax.set_title(ot_type + fr" | $d = {dim:d}$, $\gamma = {gammas_masses[1][0]}$", y=1.03, fontsize=19)
        # else:
        ax.set_title(ot_type + r" | $d = %d$" % dim, y=1.03, fontsize=19)
plt.legend()
plt.subplots_adjust(wspace=0.25)
plt.savefig("fig/fig1.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
plt.close("all")
