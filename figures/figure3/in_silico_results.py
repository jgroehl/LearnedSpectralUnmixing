from mpl_toolkits.axes_grid1 import make_axes_locatable
from paths import TEST_DATA_PATH
import matplotlib.pyplot as plt
from scipy.stats import iqr
import numpy as np
import glob
import json

ALL_MODELS = ['ALL', 'BASE',
              'BG_0-100', 'BG_60-80', 'BG_H2O', 'HET_0-100', 'HET_60-80',
              'RES_0.15', 'RES_0.15_SMALL',
              'RES_0.6',  'RES_1.2',
              'WATER_2cm', 'WATER_4cm',
              'SKIN',
              'ILLUM_5mm', 'ILLUM_POINT',
               'SMALL',
              'ACOUS',
              'MSOT', 'MSOT_SKIN','MSOT_ACOUS','MSOT_ACOUS_SKIN',
              'INVIS','INVIS_SKIN', 'INVIS_ACOUS', 'INVIS_SKIN_ACOUS',

              ]

ALL_LABELS= ['ALL', 'BASE',
              '0-100', '60-80', 'H2O', 'H/0-100', 'H/60-80',
              '0.15mm', '0.15/S',
              '0.6mm',  '1.2mm',
              'W.2cm', 'W.4cm',
              'SKIN',
              'DISK', 'POINT',
              'SMALL',
              'ACOUS',
              r'$\bf{MSOT}$', 'SKIN', 'ACOUS', 'AC/SK',
              r'$\bf{INVIS}$', 'SKIN', 'ACOUS', 'AC/SK'
              ]


with open(f"{TEST_DATA_PATH}/result_matrix.json", "r+") as json_file:
    result_matrix_dict = json.load(json_file)

indices = np.arange(len(ALL_MODELS)).astype(int)

all_results_heatmap = np.zeros((len(ALL_MODELS), len(ALL_MODELS)))
all_results_heatmap_norm_train = np.zeros((len(ALL_MODELS), len(ALL_MODELS)))
all_results_heatmap_norm_test = np.zeros((len(ALL_MODELS), len(ALL_MODELS)))

node_sizes = []

for testing_idx, model in enumerate(ALL_MODELS):
    for training_idx, dataset in enumerate(ALL_MODELS):
        all_results_heatmap[training_idx, testing_idx] = result_matrix_dict[model][dataset]

all_results_heatmap = all_results_heatmap.T
all_results_heatmap = all_results_heatmap[indices, :][:, indices]
all_results_heatmap = all_results_heatmap.reshape((len(indices), len(indices)))

test_data_path = TEST_DATA_PATH + "/baseline/"

ground_truth = np.squeeze(np.load(test_data_path + "baseline.npz")["oxygenations"])

wls = []
results = []
est_oxy = []

for filename in glob.glob(test_data_path + "*_est_BASE*"):
    print(filename)
    num_wl = int(filename.split("_")[-1].split(".")[0])
    data = np.squeeze(np.load(filename)["estimate"])
    wls.append(num_wl)
    results.append(np.abs(data-ground_truth) * 100)
    est_oxy.append(data)

wls = np.asarray(wls)
sort_idx = np.argsort(wls)
wls = np.delete(wls[sort_idx], [4])
results = np.asarray(results)[sort_idx]
est_oxy = np.asarray(est_oxy)[sort_idx]
median_results = np.delete(np.median(results, axis=1), [4])
perc025 = np.delete(np.percentile(results, 25, axis=1), [4])
perc075 = np.delete(np.percentile(results, 75, axis=1), [4])

dist_ground_truth = np.squeeze(np.load(test_data_path + "baseline.npz")["oxygenations"])

dist_wls = []
dist_results = []
dist_results_corr = []
dist_est_oxy = []
dist_est_oxy_corr = []

for filename in glob.glob(test_data_path + "*_dist_BASE*"):
    print(filename)
    num_wl = int(filename.split("_")[-1].split(".")[0])
    dist_data = np.squeeze(np.load(filename)["estimate"])
    dist_wls.append(num_wl)
    dist_results.append(np.abs(dist_data-dist_ground_truth) * 100)
    dist_est_oxy.append(dist_data)

dist_wls = np.asarray(dist_wls)
dist_sort_idx = np.argsort(dist_wls)
dist_wls = dist_wls[dist_sort_idx]
dist_results = np.asarray(dist_results)[dist_sort_idx]
dist_est_oxy = np.asarray(dist_est_oxy)[dist_sort_idx]
dist_median_results = np.median(dist_results, axis=1)
dist_iqr_results = iqr(dist_results, axis=1)

dist_random_points = np.random.choice(len(dist_ground_truth), 10000, replace=False)

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12, 7), layout="constrained", dpi=200)
# plt.suptitle('UMAP projection of all datasets with semantic encoding', fontsize=14, fontweight="bold")

GridSpec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
subfigure_1 = fig.add_subfigure(GridSpec[0:2,0:2])
subfigure_1.text(0, 0.95, "A", size=30, weight='bold')

ax0 = subfigure_1.subplots(1, 1)
im = ax0.imshow(all_results_heatmap * 100)
ax0.set_yticks(np.arange(len(indices)), np.asarray(ALL_LABELS)[indices], fontsize=12)
ax0.set_xticks(np.arange(len(indices)), np.asarray(ALL_LABELS)[indices], rotation=90, fontsize=12)
ax0.set_xlabel("Testing Data Set", fontweight="bold", fontsize=14)
ax0.set_ylabel("Training Data Set", fontweight="bold", fontsize=14)
divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label(r"$\epsilon {sO_2}$ [%]", fontsize=14)

subfigure_4 = fig.add_subfigure(GridSpec[0, 2])
subfigure_4.text(0, 0.90, "B", size=30, weight='bold')
ax0 = subfigure_4.subplots(1, 1)
ax0.errorbar(wls, median_results)
ax0.plot(wls, median_results, "o", c="black", zorder=5)
ax0.spines.right.set_visible(False)
ax0.spines.top.set_visible(False)
ax0.fill_between(wls, perc025, perc075, color="blue", alpha=0.1)
ax0.set_ylabel(r"$\epsilon {sO_2}$ [%]", fontsize=14)
ax0.set_xlabel("Number of wavelengths", fontweight="bold", fontsize=14)


subfigure_5 = fig.add_subfigure(GridSpec[1, 2])
subfigure_5.text(0, 0.90, "C", size=30, weight='bold')
ax0 = subfigure_5.subplots(1, 1)

ax0.plot(dist_wls, dist_median_results, c="blue")
ax0.plot(dist_wls, dist_median_results, "o", c="blue", zorder=5)
ax0.spines.right.set_visible(False)
ax0.spines.top.set_visible(False)
ax0.vlines(20, 5, 30, color="green")
ax0.fill_betweenx([dist_median_results[6], dist_median_results[6]+0.75], 2, 41, color="green", alpha=0.3)
ax0.set_ylabel(r"$\epsilon {sO_2}$ [%]", fontsize=14)
ax0.set_xlabel("Number of wavelengths", fontweight="bold", fontsize=14)
ax0.set_ylim(7, 15)
ax0.set_xlim(14.5, 25.5)

plt.savefig(f"figure3.png", dpi=500)
plt.close()
