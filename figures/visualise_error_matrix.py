from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.constants import ALL_MODELS
from paths import TEST_DATA_PATH
import matplotlib.pyplot as plt
import numpy as np
import json

with open(f"{TEST_DATA_PATH}/result_matrix.json", "r+") as json_file:
    result_matrix_dict = json.load(json_file)

# ALL
title= "ALL"
indices = np.arange(len(ALL_MODELS)).astype(int)

all_results_heatmap = np.zeros((len(ALL_MODELS), len(ALL_MODELS)))
all_results_heatmap_norm_train = np.zeros((len(ALL_MODELS), len(ALL_MODELS)))
all_results_heatmap_norm_test = np.zeros((len(ALL_MODELS), len(ALL_MODELS)))

node_sizes = []

for training_idx, model in enumerate(ALL_MODELS):
    for testing_idx, dataset in enumerate(ALL_MODELS):
        all_results_heatmap[training_idx, testing_idx] = result_matrix_dict[model][dataset]

    a = all_results_heatmap[training_idx, :]
    all_results_heatmap_norm_train[training_idx, :] = (a - min(a)) / (max(a) - min(a))

    node_sizes.append(np.mean(a) * 1000)

for testing_idx in range(len(ALL_MODELS)):
    a = all_results_heatmap[:, testing_idx]
    all_results_heatmap_norm_test[:, testing_idx] = (a - min(a)) / (max(a) - min(a))

f, (ax0) = plt.subplots(1, 1, figsize=(7, 6), layout="constrained")


all_results_heatmap = all_results_heatmap.T
all_results_heatmap = all_results_heatmap[indices, :][:, indices]
all_results_heatmap = all_results_heatmap.reshape((len(indices), len(indices)))
im = ax0.imshow(all_results_heatmap * 100)
ax0.set_yticks(np.arange(len(indices)), np.asarray(ALL_MODELS)[indices])
ax0.set_xticks(np.arange(len(indices)), np.asarray(ALL_MODELS)[indices], rotation=90)
ax0.set_xlabel("Training Data Set", fontweight="bold")
ax0.set_ylabel("Test Data Set", fontweight="bold")
divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = f.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label("Absolute sO$_2$ estimation error [%]")

plt.savefig(f"all_training_testcases_{title}.png")
plt.show()
