import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import iqr, linregress
from paths import TRAINING_DATA_PATH, TEST_DATA_PATH

MARKERSIZE = 15
flow_path = TEST_DATA_PATH + "/flow/"

ALL_MODELS = ['ALL', 'ACOUS', 'BASE', 'BG_0-100', 'BG_60-80', 'BG_H2O', 'HET_0-100', 'HET_60-80', 'ILLUM_5mm', 'ILLUM_POINT',
              'INVIS', 'INVIS_ACOUS', 'INVIS_SKIN', 'INVIS_SKIN_ACOUS', 'MSOT', 'MSOT_ACOUS', 'MSOT_ACOUS_SKIN',
              'MSOT_SKIN', 'RES_0.15', 'RES_0.15_SMALL', 'RES_0.6', 'RES_1.2', 'SKIN', 'SMALL', 'WATER_2cm',
              'WATER_4cm']

ALL_MODELS_2 = ['ALL', 'ACOUS', 'BASE', 'BG_0-100', 'BG_60-80', 'BG_H2O', 'HET_0-100', 'HET_60-80', 'ILLUM_5mm',
              'INVIS', 'INVIS_ACOUS', 'INVIS_SKIN', 'INVIS_SKIN_ACOUS', 'MSOT', 'MSOT_ACOUS', 'MSOT_ACOUS_SKIN',
              'MSOT_SKIN', 'RES_0.15', 'RES_0.15_SMALL', 'RES_0.6', 'RES_1.2', 'SKIN', 'SMALL', 'WATER_2cm',
              'WATER_4cm']

ALL_MODELS_3 = ['ALL', 'ACOUS', 'BASE', 'BG_0-100', 'BG_60-80', 'BG_H2O', 'HET_0-100', 'HET_60-80', 'ILLUM_5mm',
              'INVIS', 'INVIS_ACOUS', 'INVIS_SKIN', 'INVIS_SKIN_ACOUS', 'MSOT', 'MSOT_ACOUS',
              'MSOT_SKIN', 'RES_0.15', 'RES_0.15_SMALL', 'RES_0.6', 'RES_1.2', 'SKIN', 'SMALL', 'WATER_2cm',
              'WATER_4cm']



def plot_data(axis, models, title):

    flow_distances = np.load(flow_path + "/flow_2/all_distances.npz", allow_pickle=True)
    flow_distances = {key: flow_distances[key] for key in flow_distances}
    flow_results = np.load(flow_path + "/flow_2/all_results.npz", allow_pickle=True)
    flow_results = {key: flow_results[key] for key in flow_results}

    flow_distance_means = np.asarray([np.mean(flow_distances[model]) for model in models])
    best_dist_flow = np.argmin(flow_distance_means)
    worst_dist_flow = np.argmax(flow_distance_means)

    flow_model_results = [np.mean(np.abs(flow_results[model] - flow_results["reference"]) * 100) for model in
                          models]
    flow_model_results_error = [np.std(np.abs(flow_results[model] - flow_results["reference"]) * 100) / 10 for model in
                                models]

    axis.set_title(title, style="italic")
    axis.errorbar(flow_distance_means, flow_model_results, fmt=".", yerr=flow_model_results_error, alpha=0.5,
                ecolor="red", zorder=-20)
    axis.scatter(flow_distance_means, flow_model_results, c="gray", s=MARKERSIZE)
    slope, intercept, r_value, _, _ = linregress(flow_distance_means, flow_model_results)
    axis.plot([np.min(flow_distance_means), np.max(flow_distance_means)],
            [slope * (np.min(flow_distance_means)) + intercept, slope * (np.max(flow_distance_means)) + intercept],
            c="black",
            linestyle="dashed",
            label=f"fit (R={r_value:.2f})")
    axis.set_xlabel("$D_{JS}$ [a.u.]", fontweight="bold")
    axis.set_ylabel(r"$\epsilon sO_2$ [%]", fontweight="bold")
    axis.spines.right.set_visible(False)
    axis.spines.top.set_visible(False)
    axis.legend(loc="upper left")


def create_distribution_distance_figure():


    fig = plt.figure(layout="constrained", figsize=(10, 2.5))
    gs = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

    subfig2 = fig.add_subfigure(gs[0:1, 0:1])
    subfig3 = fig.add_subfigure(gs[0:1, 1:2])
    subfig4 = fig.add_subfigure(gs[0:1, 2:3])

    plot_data(subfig2.subplots(1, 1), ALL_MODELS, "All data")
    plot_data(subfig3.subplots(1, 1), ALL_MODELS_2, "Removing ILLUM_POINT")
    plot_data(subfig4.subplots(1, 1), ALL_MODELS_3,  "Removing MSOT_ACOUS_SKIN")

    MARKERSIZE = 15

    subfig2.text(0, 0.90, "A", size=20, weight='bold')
    subfig3.text(0, 0.90, "B", size=20, weight='bold')
    subfig4.text(0, 0.90, "C", size=20, weight='bold')

    plt.savefig("suppl_figure2.pdf", dpi=300)


if __name__ == "__main__":
    create_distribution_distance_figure()