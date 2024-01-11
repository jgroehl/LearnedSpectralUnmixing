from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from paths import TEST_DATA_PATH, EXAMPLE_PATH, TRAINING_DATA_PATH
from matplotlib_scalebar.scalebar import ScaleBar
import umap
from matplotlib.colors import LinearSegmentedColormap
from utils.io import preprocess_data
import matplotlib.pyplot as plt
from scipy.stats import iqr
from umap.parametric_umap import load_ParametricUMAP
import numpy as np
import glob
import json
import pickle
import os

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
              'INVIS','INVIS_SKIN', 'INVIS_ACOUS', 'INVIS_SKIN_ACOUS'
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
              'MSOT', 'SKIN', 'ACOUS', 'AC/SK',
              'INVIS', 'SKIN', 'ACOUS', 'AC/SK'
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

fig = plt.figure(figsize=(12, 7), layout="constrained", dpi=200)

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
cbar.set_label(r"$\epsilon {sO_2}$ [%]", fontsize=14, labelpad=-37, color="white")

subfigure_4 = fig.add_subfigure(GridSpec[0, 2])
subfigure_4.text(0, 0.90, "B", size=30, weight='bold')
((ax0, ax1), (ax2, ax3)) = subfigure_4.subplots(2, 2)

subfigure_5 = fig.add_subfigure(GridSpec[1, 2])
subfigure_5.text(0, 0.90, "C", size=30, weight='bold')
subplots_2 = subfigure_5.subplots(1, 1)

# old figure 2 code:

COLOURS = ["#201923", "#fcff5d", "#7dfc00", "#0ec434", "#228c68",
           "#8ad8e8", "#235b54", "#29bdab", "#3998f5", "#37294f", "#277da7",
           "#3750db", "#f22020", "#991919", "#ffcba5", "#e68f66", "#c56133",
           "#96341c", "#632819", "#ffc413", "#f47a22", "#2f2aa0", "#b732cc",
           "#772b9d", "#f07cab", "#d30b94", "#edeff3", "#c3a5b4", "#946aa2", "#5d4c86"]

PATHS = glob.glob(TRAINING_DATA_PATH + "/*")
PATHS = [PATHS[0]] + PATHS[2:]
np.random.seed(1337)
rnd_state = np.random.RandomState(1337)

NUM_RANDOM_DATAPOINTS = 200000

print("Loading data...")
all_data = []
all_oxy = []
all_ds_idx = []
datasets = []

for ds_idx, path in enumerate(PATHS):
    base_filename = path.split("/")[-1].split("\\")[-1]
    if base_filename == "ALL":
        continue
    datasets.append(base_filename)
    spectra, oxy = preprocess_data(f"{path}/{base_filename}_train.npz", 41)
    all_data.append(spectra)
    all_oxy.append(oxy)
    all_ds_idx.append(np.ones_like(oxy) * ds_idx)

all_data = np.hstack(all_data)
all_oxy = np.hstack(all_oxy)
all_ds_idx = np.hstack(all_ds_idx)
print("Loading data...[Done]")

random_idx = np.random.choice(len(all_data.T), NUM_RANDOM_DATAPOINTS, replace=False)
n_datasets = len(np.unique(all_ds_idx))
reducer = umap.UMAP(random_state=rnd_state, verbose=True)
print("Embedding data...")
embedding = reducer.fit_transform(all_data[:, random_idx].T)
print("Embedding data...[Done]")
# colors = [COLOURS[int(i)] for i in all_ds_idx[random_idx]]
cmap = LinearSegmentedColormap.from_list("test", COLOURS[:25], N=n_datasets)

dat_2 = subplots_2.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=all_oxy[random_idx] * 100,
    s=2,
    alpha=1,
    vmin=0, vmax=100)
dat_2.set_rasterized(True)
cbar2 = plt.colorbar(dat_2, orientation="horizontal")
cbar2.set_label("Blood oxygenation sO$_2$ [%]", labelpad=-28, color="white")
subfigure_5.text(0.1, 0.92, "Mapping sO$_2$ onto UMAP", size=12, weight='bold')
subplots_2.set_ylim(-9, 10)
subplots_2.set_xlim(-10, 17)
subplots_2.axis("off")
subplots_2.set_aspect('equal')

subfigure_4.text(0.1, 0.915, "UMAP projection of datasets", fontweight="bold", fontsize=12)


def plot_for_index(axis, index):
    axis.axis("off")
    axis.set_title(datasets[index], y=-0.1)
    sc = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c="black",
        alpha=1,
        s=3)
    sc.set_rasterized(True)
    sc = axis.scatter(
        embedding[all_ds_idx[random_idx] == index, 0],
        embedding[all_ds_idx[random_idx] == index, 1],
        c=COLOURS[index],
        alpha=0.2,
        s=2)
    sc.set_rasterized(True)
    axis.set_aspect('equal')
    axis.set_ylim(-9, 10)
    axis.set_xlim(-10, 18)
    axis.set_anchor('S')


indexes = [1, 8, 22, 24]

plot_for_index(ax0, 1)
plot_for_index(ax1, 21)
plot_for_index(ax2, 22)
plot_for_index(ax3, 24)


plt.savefig(f"figure3.pdf", dpi=400)
plt.close()
