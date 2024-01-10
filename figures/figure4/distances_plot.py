import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.constants import ALL_MODELS
from scipy.stats import iqr, linregress
from paths import TRAINING_DATA_PATH, TEST_DATA_PATH
from utils.distribution_distance import compute_jsd, compute_kld

baseline_path = TEST_DATA_PATH + "/baseline/"
mouse_path = TEST_DATA_PATH + "/mouse/"
CO2_path = TEST_DATA_PATH + "/cO2/"
forearm_path = TEST_DATA_PATH + "/forearm/"
flow_path = TEST_DATA_PATH + "/flow/"

# This has the outliers removed for the H2O dataset
ALL_MODELS_CLEANED = ['ALL', 'ACOUS', 'BASE', 'BG_0-100', 'BG_60-80', 'BG_H2O', 'HET_0-100', 'HET_60-80', 'ILLUM_5mm',
              'INVIS', 'INVIS_ACOUS', 'INVIS_SKIN', 'INVIS_SKIN_ACOUS', 'MSOT', 'MSOT_ACOUS',
              'MSOT_SKIN', 'RES_0.15', 'RES_0.15_SMALL', 'RES_0.6', 'RES_1.2', 'SKIN', 'SMALL', 'WATER_2cm',
              'WATER_4cm']


def compile_distance_measures(data_path):
    output_file = data_path + "/all_distances.npz"
    if os.path.exists(output_file):
        return output_file
    data_files = [data_path]

    all_wl = np.arange(700, 901, 5)
    results = {}

    for folder_path in data_files:
        print(folder_path)
        filename = folder_path.split("/")[-1].split("\\")[-1]
        print(filename)
        data = np.load(folder_path + "/" + filename + ".npz")
        spectra = data["spectra"]
        test_wl = data["wavelengths"]
        wl_mask = [x in test_wl for x in all_wl]
        num_wl = len(spectra)

        for _i in range(10):
            print(_i)

            test_spectra = spectra.reshape((num_wl, -1))
            test_spectra = spectra[:, np.random.choice(test_spectra.shape[1], 100000)]

            for train_path in glob.glob(TRAINING_DATA_PATH + "/*"):
                model_name = train_path.split("/")[-1].split("\\")[-1]
                print(model_name)
                data = np.load(train_path + f"/{model_name}_train.npz")
                train_spectra = data["spectra"][wl_mask, :]

                if model_name not in results:
                    results[model_name] = dict()
                if "JSD" not in results[model_name]:
                    results[model_name]["JSD"] = []
                results[model_name]["JSD"].append(compute_jsd(train_spectra, test_spectra))
                if "KLD" not in results[model_name]:
                    results[model_name]["KLD"] = []
                results[model_name]["KLD"].append(compute_kld(train_spectra, test_spectra))

    np.savez(output_file, **results, allow_pickle=True)
    return output_file


def compile_results(data_path):
    output_file = data_path + "/all_results.npz"
    if os.path.exists(output_file):
        return output_file
    results = {}
    for model in ALL_MODELS:
        results[model] = []

    data = np.load(data_path + "/baseline.npz")
    gt = np.squeeze(data["oxygenations"])

    for model in ALL_MODELS:
        print(model)
        model_result = np.squeeze(np.load(f"{data_path}/baseline_{model}_41.npz")["estimate"])
        abs_diff = np.abs(model_result - gt) * 100
        rel_diff = abs_diff / gt
        results[model] = np.asarray([np.median(abs_diff), iqr(abs_diff), np.median(rel_diff), iqr(rel_diff)])

    np.savez(output_file, **results, allow_pickle=True)
    return output_file


def create_distribution_distance_figure(data_path):
    distance_file = compile_distance_measures(data_path)
    results_path = compile_results(data_path)
    results = np.load(results_path, allow_pickle=True)
    results = {key: results[key] for key in results}
    distances = np.load(distance_file, allow_pickle=True)
    distances = {key: distances[key] for key in distances}

    dist_mean = []
    dist_std = []
    error_mean = []
    error_std = []
    for model in ALL_MODELS:
        dist_mean.append(np.mean(distances[model].item()["JSD"]))
        dist_std.append(np.std(distances[model].item()["JSD"]))
        error_mean.append(results[model][0])
        error_std.append(results[model][1] / 10)

    slope, intercept, r_value, _, _ = linregress(dist_mean, error_mean)

    fig = plt.figure(layout="constrained", figsize=(10, 4.5))
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[3, 1.5])

    subfig2 = fig.add_subfigure(gs[0:2, 0:1])
    subfig3 = fig.add_subfigure(gs[0:1, 1:2])
    subfig4 = fig.add_subfigure(gs[1:2, 1:2])

    a1 = subfig2.subplots(1, 1)
    MARKERSIZE = 15

    a1.set_title("In silico", style="italic")
    a1.plot([0.1, np.max(dist_mean) + 0.1],
             [slope * 0.1 + intercept, slope * (np.max(dist_mean) + 0.1) + intercept],
             c="black",
             linestyle="dashed",
             label=f"fit (R={r_value:.2f})")
    a1.errorbar(dist_mean, error_mean, fmt=".", xerr=dist_std, yerr=error_std, alpha=0.5, ecolor="red", zorder=-20)
    a1.scatter(dist_mean, error_mean, c="gray", s=MARKERSIZE)
    a1.scatter(np.mean(distances["BASE"].item()["JSD"]), results["BASE"][0], s=MARKERSIZE*5, marker="s", c="blue",
               label="BASE")
    a1.scatter(np.mean(distances["ILLUM_POINT"].item()["JSD"]), results["ILLUM_POINT"][0], s=MARKERSIZE*5, marker="*",
               c="red", label="ILLUM_POINT")
    a1.scatter(np.mean(distances["RES_0.15"].item()["JSD"]), results["RES_0.15"][0], s=MARKERSIZE*5, c="green",
               label="RES_0.15")
    a1.scatter(np.mean(distances["SMALL"].item()["JSD"]), results["SMALL"][0], s=MARKERSIZE*5, marker="h", c="purple",
               label="SMALL")
    a1.scatter(np.mean(distances["SKIN"].item()["JSD"]), results["SKIN"][0], s=MARKERSIZE*5,
               marker="D", c="orange", label="SKIN")
    a1.scatter(np.mean(distances["WATER_4cm"].item()["JSD"]), results["WATER_4cm"][0], s=MARKERSIZE*5,
               marker="P", c="#03A9F4", label="WATER_4cm")

    a1.legend(loc="lower right")

    a1.spines.right.set_visible(False)
    a1.spines.top.set_visible(False)
    a1.set_ylabel(r"Abs. $sO_2$ estimation error ($\epsilon sO_2$)[%]", fontweight="bold")
    a1.set_xlabel("Jensen-Shannon divergence ($D_{JS}$)", fontweight="bold")

    flow_d2o_distances = np.load(flow_path + "/flow_1/all_distances.npz", allow_pickle=True)
    flow_d2o_distances = {key: flow_d2o_distances[key] for key in flow_d2o_distances}
    flow_d2o_results = np.load(flow_path + "/flow_1/all_results.npz", allow_pickle=True)
    flow_d2o_results = {key: flow_d2o_results[key] for key in flow_d2o_results}
    flow_d2o_distance_means = np.asarray([np.mean(flow_d2o_distances[model]) for model in ALL_MODELS])
    best_dist_flow_d2o = np.argmin(flow_d2o_distance_means)
    worst_dist_flow_d2o = np.argmax(flow_d2o_distance_means)
    flow_d2o_model_results = [np.mean(np.abs(flow_d2o_results[model] - flow_d2o_results["reference"]) * 100) for model in ALL_MODELS]
    flow_d2o_model_results_error = [np.std(np.abs(flow_d2o_results[model] - flow_d2o_results["reference"]) * 100)/10 for model in
                          ALL_MODELS]

    flow_h2o_distances = np.load(flow_path + "/flow_2/all_distances.npz", allow_pickle=True)
    flow_h2o_distances = {key: flow_h2o_distances[key] for key in flow_h2o_distances}
    flow_h2o_results = np.load(flow_path + "/flow_2/all_results.npz", allow_pickle=True)
    flow_h2o_results = {key: flow_h2o_results[key] for key in flow_h2o_results}
    flow_h2o_distance_means = np.asarray([np.mean(flow_h2o_distances[model]) for model in ALL_MODELS_CLEANED])
    best_dist_flow_h2o = np.argmin(flow_h2o_distance_means)
    worst_dist_flow_h2o = np.argmax(flow_h2o_distance_means)
    flow_h2o_model_results = [np.mean(np.abs(flow_h2o_results[model] - flow_h2o_results["reference"]) * 100) for model
                              in ALL_MODELS_CLEANED]
    flow_h2o_model_results_error = [np.std(np.abs(flow_h2o_results[model] - flow_h2o_results["reference"]) * 100) / 10
                                    for model in
                                    ALL_MODELS_CLEANED]

    a2 = subfig3.subplots(1, 1)
    a2.set_title("In gello (D$_2$O)", style="italic")
    a2.errorbar(flow_d2o_distance_means, flow_d2o_model_results, fmt=".", yerr=flow_d2o_model_results_error, alpha=0.5, ecolor="red", zorder=-20)
    a2.scatter(flow_d2o_distance_means, flow_d2o_model_results, c="gray", s=MARKERSIZE)
    slope, intercept, r_value, _, _ = linregress(flow_d2o_distance_means, flow_d2o_model_results)
    a2.plot([np.min(flow_d2o_distance_means), np.max(flow_d2o_distance_means)],
             [slope * (np.min(flow_d2o_distance_means)) + intercept, slope * (np.max(flow_d2o_distance_means)) + intercept],
             c="black",
             linestyle="dashed",
             label=f"fit (R={r_value:.2f})")
    a2.set_xlabel("$D_{JS}$ [a.u.]", fontweight="bold")
    a2.set_ylabel(r"$\epsilon sO_2$ [%]", fontweight="bold")
    a2.spines.right.set_visible(False)
    a2.spines.top.set_visible(False)
    base_idx = np.argwhere(np.asarray(ALL_MODELS) == "BASE").item()
    a2.scatter(flow_d2o_distance_means[worst_dist_flow_d2o],
               flow_d2o_model_results[worst_dist_flow_d2o], s=2 * MARKERSIZE, marker="*", c="red")
    a2.scatter(flow_d2o_distance_means[base_idx],
               flow_d2o_model_results[base_idx], s=2 * MARKERSIZE, marker="s", c="blue")
    a2.scatter(flow_d2o_distance_means[best_dist_flow_d2o],
               flow_d2o_model_results[best_dist_flow_d2o], s=2 * MARKERSIZE, marker="h", c="purple")
    a2.legend(loc="lower right")

    print("Best flow D2O:", flow_d2o_distance_means[best_dist_flow_d2o], flow_d2o_model_results[best_dist_flow_d2o])
    print("Best flow D2O:", ALL_MODELS[best_dist_flow_d2o])
    print("Worst flow D2O:", flow_h2o_distance_means[worst_dist_flow_d2o], flow_d2o_model_results[worst_dist_flow_d2o])
    print("Worst flow D2O:", ALL_MODELS[worst_dist_flow_d2o])
    print("")


    a3 = subfig4.subplots(1, 1)
    a3.set_title("In gello (H$_2$O)", style="italic")
    a3.errorbar(flow_h2o_distance_means, flow_h2o_model_results, fmt=".", yerr=flow_h2o_model_results_error, alpha=0.5,
                ecolor="red", zorder=-20)
    a3.scatter(flow_h2o_distance_means, flow_h2o_model_results, c="gray", s=MARKERSIZE)
    slope, intercept, r_value, _, _ = linregress(flow_h2o_distance_means, flow_h2o_model_results)
    a3.plot([np.min(flow_h2o_distance_means), np.max(flow_h2o_distance_means)],
            [slope * (np.min(flow_h2o_distance_means)) + intercept,
             slope * (np.max(flow_h2o_distance_means)) + intercept],
            c="black",
            linestyle="dashed",
            label=f"fit (R={r_value:.2f})")
    a3.set_xlabel("$D_{JS}$ [a.u.]", fontweight="bold")
    a3.set_ylabel(r"$\epsilon sO_2$ [%]", fontweight="bold")
    a3.spines.right.set_visible(False)
    a3.spines.top.set_visible(False)
    base_idx = np.argwhere(np.asarray(ALL_MODELS_CLEANED) == "BASE").item()
    a3.scatter(flow_h2o_distance_means[worst_dist_flow_h2o],
               flow_h2o_model_results[worst_dist_flow_h2o], s=2 * MARKERSIZE,
               marker="D", c="orange")
    a3.scatter(flow_h2o_distance_means[base_idx],
               flow_h2o_model_results[base_idx], s=2 * MARKERSIZE, marker="s", c="blue")
    a3.scatter(flow_h2o_distance_means[best_dist_flow_h2o],
               flow_h2o_model_results[best_dist_flow_h2o], s=2 * MARKERSIZE,
               marker="P", c="#03A9F4")
    a3.legend(loc="lower right")

    print("Best flow H2O:", flow_h2o_distance_means[best_dist_flow_h2o], flow_h2o_model_results[best_dist_flow_h2o])
    print("Best flow H2O:", ALL_MODELS_CLEANED[best_dist_flow_h2o])
    print("Worst flow H2O:", flow_h2o_distance_means[worst_dist_flow_h2o], flow_h2o_model_results[worst_dist_flow_h2o])
    print("Worst flow H2O:", ALL_MODELS_CLEANED[worst_dist_flow_h2o])

    subfig2.text(0, 0.90, "A", size=20, weight='bold')
    subfig3.text(0, 0.90, "B", size=20, weight='bold')
    subfig4.text(0, 0.90, "C", size=20, weight='bold')

    plt.savefig("figure4.pdf", dpi=300)


if __name__ == "__main__":
    create_distribution_distance_figure(baseline_path)