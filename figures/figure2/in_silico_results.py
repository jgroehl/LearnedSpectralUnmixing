from utils.io import preprocess_data
import matplotlib.gridspec as gridspec
import numpy as np
from paths import TRAINING_DATA_PATH
import glob
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap

COLOURS = ["#201923", "#fcff5d", "#7dfc00", "#0ec434", "#228c68",
           "#8ad8e8", "#235b54", "#29bdab", "#3998f5", "#37294f", "#277da7",
           "#3750db", "#f22020", "#991919", "#ffcba5", "#e68f66", "#c56133",
           "#96341c", "#632819", "#ffc413", "#f47a22", "#2f2aa0", "#b732cc",
           "#772b9d", "#f07cab", "#d30b94", "#edeff3", "#c3a5b4", "#946aa2", "#5d4c86"]

PATHS = glob.glob(TRAINING_DATA_PATH + "/*")

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

NUM_RANDOM_DATAPOINTS = 200000
random_idx = np.random.choice(len(all_data.T), NUM_RANDOM_DATAPOINTS, replace=False)

n_datasets = len(np.unique(all_ds_idx))
reducer = umap.UMAP()
print("Embedding data...")
embedding = reducer.fit_transform(all_data[:, random_idx].T)
print("Embedding data...[Done]")

fig = plt.figure(figsize=(18, 5), layout="constrained", dpi=50)
# plt.suptitle('UMAP projection of all datasets with semantic encoding', fontsize=14, fontweight="bold")

GridSpec = gridspec.GridSpec(ncols=12, nrows=1, figure= fig)

# colors = [COLOURS[int(i)] for i in all_ds_idx[random_idx]]
cmap = LinearSegmentedColormap.from_list("test", COLOURS[:25], N=n_datasets)

subfigure_2= fig.add_subfigure(GridSpec[0,0:4])
subfigure_2.text(0, 0.90, "A", size=30, weight='bold')
subplots_2= subfigure_2.subplots(1, 1)
subplots_2.set_title("Oxygenation encoding", fontweight="bold", fontsize=14)
dat_2 = subplots_2.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=all_oxy[random_idx] * 100,
    s=0.5,
    alpha=1,
    vmin=0, vmax=100)
cbar2 = plt.colorbar(dat_2)
cbar2.set_label("Oxygenation [%]", fontweight="bold", fontsize=12)
subplots_2.set_ylim(-7, 15)
subplots_2.set_xlim(-11, 16)
subplots_2.axis("off")
subplots_2.set_aspect('equal')


subfigure_1= fig.add_subfigure(GridSpec[0,4:9])
subfigure_1.text(0, 0.90, "B", size=30, weight='bold')
subplots_1= subfigure_1.subplots(1, 1)
subplots_1.set_title("Dataset encoding", fontweight="bold", fontsize=14)
dat = subplots_1.scatter(
    embedding[:, 0][:50000],
    embedding[:, 1][:50000],
    c=all_ds_idx[random_idx][:50000],
    s=5,
    alpha=1,
    cmap=cmap)
cbar1 = plt.colorbar(dat)
cbar1.set_ticks(np.linspace(0, n_datasets-1, 25))
cbar1.set_ticklabels(datasets)
subplots_1.set_ylim(-7, 15)
subplots_1.set_xlim(-11, 16)
subplots_1.axis("off")
subplots_1.set_aspect('equal')

subfigure_1= fig.add_subfigure(GridSpec[0,9:12])
((ax0, ax1), (ax2, ax3)) = subfigure_1.subplots(2, 2)
subfigure_1.suptitle("Individual training datasets", fontweight="bold", fontsize=14)
subfigure_1.text(0, 0.90, "C", size=30, weight='bold')

def plot_for_index(axis, index):

    axis.axis("off")
    axis.set_title(datasets[index], y=-0.1)
    axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c="black",
        alpha=1)
    axis.scatter(
        embedding[all_ds_idx[random_idx]==index, 0],
        embedding[all_ds_idx[random_idx]==index, 1],
        c=COLOURS[index],
        alpha=0.2,
        s=1)
    axis.set_aspect('equal')

plot_for_index(ax0, 1)
plot_for_index(ax1, 8)
plot_for_index(ax2, 12)
plot_for_index(ax3, 24)

plt.savefig("umap_projection.png", dpi=600)
plt.show()
plt.close()