import pickle
from matplotlib import pyplot as plt
import numpy as np


fontsize = 9
params = {'axes.labelsize': fontsize + 1,
          'font.size': fontsize,
          'legend.fontsize': fontsize + 2,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize + 2,
          'pdf.fonttype': 42}
plt.rcParams.update(params)

with open("data/ns-data.pkl", "rb") as ff:
    data = pickle.load(ff)
device_names = data["device_names"]
all_times = data["all_times"]
names = data["names"]
colors = ["cornflowerblue", "indianred"]
dimensions = data["dimensions"]

f, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for ii, (ax, device_name) in enumerate(zip(axes, device_names)):
    for color, datum, name in zip(colors, all_times, names):
        upper = datum[ii].mean(1) + datum[ii].std(1)
        lower = datum[ii].mean(1) - datum[ii].std(1)
        ax.plot(dimensions, datum[ii].mean(1),
                color=color, lw=1,
                label=name)
        ax.fill_between(dimensions, lower, upper,
                        color=color, alpha=.3)
    ax.grid(True)
    # ax.set_xscale("log")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("dimension d", y=0.1)
    ax.set_title(device_name, y=0.98)
plt.legend(loc=2, ncol=2, bbox_to_anchor=[-0.6, 1.14])
plt.savefig("fig/fig2.pdf", tight_layout=True)
plt.close("all")
