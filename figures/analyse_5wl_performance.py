import glob
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
from paths import TEST_DATA_PATH

test_data_path = TEST_DATA_PATH + "/baseline/"

ground_truth = np.squeeze(np.load(test_data_path + "baseline.npz")["oxygenations"])

wls = []
results = []
est_oxy = []
names = []

for filename in glob.glob(test_data_path + "*_masking_BASE*"):
    print(filename)
    num_wl = int(filename.split("_")[-1].split(".")[0])
    names.append(filename.split("_")[-2])
    data = np.squeeze(np.load(filename)["estimate"])
    wls.append(num_wl)
    results.append(np.abs(data-ground_truth) * 100)
    est_oxy.append(data)

wls = np.asarray(wls)
sort_idx = np.argsort(wls)
wls = wls[sort_idx]
names = np.asarray(names)[sort_idx]
results = np.asarray(results)[sort_idx]
est_oxy = np.asarray(est_oxy)[sort_idx]
median_results = np.median(results, axis=1)
iqr_results = iqr(results, axis=1)
random_points = np.random.choice(len(ground_truth), 10000, replace=False)


def plot_scatter(ax, idx):
    ax.set_title(f"{names[idx]}")
    ax.set_xlabel("Ground Truth sO$_2$ [%]", fontweight="bold")
    ax.set_ylabel("Estimated sO$_2$ [%]", fontweight="bold")
    ax.plot([0, 100], [0, 100], color="green")
    ax.scatter(ground_truth[random_points] * 100, est_oxy[idx][random_points] * 100, alpha=0.01, c="black")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xticks([0, 25, 50, 75, 100], [0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100], [0, 25, 50, 75, 100])


fig = plt.figure(layout="constrained", figsize=(12, 6))
subfigs = fig.subfigures(1, 2, wspace=0.07)
ax0 = subfigs[0].subplots(1, 1)

ax0.plot(np.arange(len(median_results)), median_results)
ax0.plot(np.arange(len(median_results)), median_results, "o", c="black", zorder=5)
ax0.spines.right.set_visible(False)
ax0.spines.top.set_visible(False)
ax0.set_ylabel("Median absolute sO$_2$ estimation error [%]", fontweight="bold")
ax0.set_xlabel("Method", fontweight="bold")
ax0.set_xticks(np.arange(len(median_results)), names)

((ax1, ax2), (ax3, ax4)) = subfigs[1].subplots(2, 2)

plot_scatter(ax1, 0)
plot_scatter(ax2, 1)
plot_scatter(ax3, 2)
plot_scatter(ax4, 3)

plt.savefig("performance_with_different_data_strategies.png", dpi=300)
