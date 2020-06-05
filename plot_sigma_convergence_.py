import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pickle


fontsize = 9
params = {'axes.labelsize': fontsize + 1,
          'font.size': fontsize,
          'legend.fontsize': fontsize + 2,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize + 2,
          'pdf.fonttype': 42}
plt.rcParams.update(params)


with open("data/gaussiansot_sigmas.pkl", "rb") as ff:
    data = pickle.load(ff)

dists = data["dist"]
unreg = data['unreg']
dimensions = data["dimensions"]
epsilons = data["epsilons"]
dim, n_eps, n_trials = dists.shape
colors = ["cornflowerblue", "indianred", "gold"]


plt.clf()
f, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)

for i in range(dim):
    ax = axes[i]
    upper = dists[i].mean(0) + dists[i].std(0)
    lower = dists[i].mean(0) - dists[i].std(0)
    ax.plot(epsilons, dists[i].mean(0), c=colors[i], label="Bures-Sinkhorn")
    ax.fill_between(epsilons, lower, upper,
                    color=colors[i], alpha=.3)
    ax.hlines(unreg[i].mean(0), xmin=epsilons[0], xmax=epsilons[-1], ls='--', color=colors[i],
              label='Bures')
    ax.set_ylim(ymin=0)
    ax.set_xscale("log")
    ax.grid(True)
    ax.set_title(r"$d = %d$" % dimensions[i], y=.99)
ax.set_xlabel(r"$\sigma$", y=0.1)

linestyles = ['--', '-']
lines = [Line2D([0], [0], color='black', linestyle=ls) for ls in linestyles]
labels = ['Bures', 'Bures-Sinkhorn']
plt.legend(lines, labels, loc=2, ncol=2, bbox_to_anchor=[.2, -.15])
plt.savefig("fig/fig_sigmas.pdf", tight_layout=True)
plt.close("all")
