import glob
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
from paths import TEST_DATA_PATH

test_data_path = TEST_DATA_PATH + "/baseline/"

ground_truth = np.squeeze(np.load(test_data_path + "baseline.npz")["oxygenations"])

wls = []
results = []
results_corr = []
est_oxy = []
est_oxy_corr = []

for filename in glob.glob(test_data_path + "*_dist_BASE*"):
    print(filename)
    num_wl = int(filename.split("_")[-1].split(".")[0])
    data = np.squeeze(np.load(filename)["estimate"])
    wls.append(num_wl)
    results.append(np.abs(data-ground_truth) * 100)
    est_oxy.append(data)

wls = np.asarray(wls)
sort_idx = np.argsort(wls)
wls = wls[sort_idx]
results = np.asarray(results)[sort_idx]
est_oxy = np.asarray(est_oxy)[sort_idx]
median_results = np.median(results, axis=1)
iqr_results = iqr(results, axis=1)

random_points = np.random.choice(len(ground_truth), 10000, replace=False)


def plot_scatter(ax, idx, corr=False):
    ax.set_title(f"{wls[idx]} wl ({'corrected' if corr else 'uncorrected'})")
    ax.set_xlabel("Ground Truth sO$_2$ [%]", fontweight="bold")
    ax.set_ylabel("Estimated sO$_2$ [%]", fontweight="bold")
    ax.plot([0, 100], [0, 100], color="green")
    if corr:
        ax.scatter(ground_truth[random_points] * 100, est_oxy_corr[idx][random_points] * 100, alpha=0.01, c="black")
    else:
        ax.scatter(ground_truth[random_points] * 100, est_oxy[idx][random_points] * 100, alpha=0.01, c="black")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xticks([0, 25, 50, 75, 100], [0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100], [0, 25, 50, 75, 100])


fig = plt.figure(layout="constrained", figsize=(12, 8))
subfigs = fig.subfigures(1, 2, wspace=0.07)
ax0 = subfigs[0].subplots(1, 1)

ax0.plot(wls, median_results, c="blue", label="uncorrected")#, yerr=(iqr_results/2), ecolor="red")
ax0.plot(wls, median_results, "o", c="blue", zorder=5)
ax0.spines.right.set_visible(False)
ax0.spines.top.set_visible(False)
ax0.vlines(20, 5, 30, color="green")
ax0.fill_betweenx([median_results[6], median_results[6]+0.75], 2, 41, color="green", alpha=0.3)
ax0.set_ylabel("Absolute sO$_2$ estimation error [%]", fontweight="bold")
ax0.set_xlabel("Number of wavelengths", fontweight="bold")
ax0.set_ylim(5, 20)
ax0.set_xlim(10, 30)
ax0.set_xticks([10, 15, 20, 25, 30], [10, 15, 20, 25, 30])
ax0.legend(loc="upper right")

((ax1, ax2), (ax3, ax4), (ax5, ax6)) = subfigs[1].subplots(3, 2)

plot_scatter(ax1, 3, corr=False)
plot_scatter(ax2, 3, corr=True)
plot_scatter(ax3, 7, corr=False)
plot_scatter(ax4, 7, corr=True)
plot_scatter(ax5, 9, corr=False)
plot_scatter(ax6, 9, corr=True)

plt.savefig("performance_with_distance_from_n_wl.png", dpi=300)