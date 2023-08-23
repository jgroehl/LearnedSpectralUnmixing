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
forearm_path = TEST_DATA_PATH + "/forearm/"
flow_path = TEST_DATA_PATH + "/flow/"


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
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[3, 3])

    subfig2 = fig.add_subfigure(gs[0:2, 0:1])
    subfig3 = fig.add_subfigure(gs[0:1, 1:2])
    subfig4 = fig.add_subfigure(gs[1:2, 1:2])

    a1 = subfig2.subplots(1, 1)

    a1.set_title("Baseline Dataset")
    a1.plot([0, np.max(dist_mean) + 0.2],
             [intercept, slope * (np.max(dist_mean) + 0.2) + intercept],
             c="black",
             linestyle="dashed",
             label=f"fit (R={r_value:.2f})")
    a1.errorbar(dist_mean, error_mean, fmt="o", xerr=dist_std, yerr=error_std, alpha=0.5, ecolor="red", zorder=-20)
    a1.scatter(dist_mean, error_mean, c="gray")
    a1.scatter(np.mean(distances["BASE"].item()["JSD"]), results["BASE"][0], c="blue", label="BASE")
    a1.scatter(np.mean(distances["ILLUM_POINT"].item()["JSD"]), results["ILLUM_POINT"][0], c="red", label="ILLUM_POINT")
    a1.scatter(np.mean(distances["RES_0.15"].item()["JSD"]), results["RES_0.15"][0], c="green", label="RES_0.15")
    a1.scatter(np.mean(distances["ALL"].item()["JSD"]), results["ALL"][0], c="purple", label="ALL")

    a1.legend(loc="lower right")

    a1.spines.right.set_visible(False)
    a1.spines.top.set_visible(False)
    a1.set_ylabel("Absolute estimation error [%]", fontweight="bold")
    a1.set_xlabel("Jensen-Shannon divergence", fontweight="bold")

    flow_distances = np.load(flow_path + "/flow_1/all_distances.npz", allow_pickle=True)
    flow_distances = {key: flow_distances[key] for key in flow_distances}
    flow_results = np.load(flow_path + "/flow_1/all_results.npz", allow_pickle=True)
    flow_results = {key: flow_results[key] for key in flow_results}

    flow_distance_means = np.asarray([np.mean(flow_distances[model]) for model in ALL_MODELS])
    best_dist_flow = np.argmin(flow_distance_means)
    worst_dist_flow = np.argmax(flow_distance_means)

    flow_model_results = [np.mean(np.abs(flow_results[model] - flow_results["reference"]) * 100) for model in ALL_MODELS]
    flow_model_results_error = [np.std(np.abs(flow_results[model] - flow_results["reference"]) * 100)/10 for model in
                          ALL_MODELS]

    a2 = subfig3.subplots(1, 1)
    a2.set_title("Flow Phantom")
    a2.errorbar(flow_distance_means, flow_model_results, fmt="o", yerr=flow_model_results_error, alpha=0.5, ecolor="red", zorder=-20)
    a2.scatter(flow_distance_means, flow_model_results, c="gray")
    slope, intercept, r_value, _, _ = linregress(flow_distance_means, flow_model_results)
    a2.plot([np.min(flow_distance_means), np.max(flow_distance_means)],
             [slope * (np.min(flow_distance_means)) + intercept, slope * (np.max(flow_distance_means)) + intercept],
             c="black",
             linestyle="dashed",
             label=f"fit (R={r_value:.2f})")
    a2.set_xlabel("JSD [a.u.]", fontweight="bold")
    a2.set_ylabel("Delta sO$_2$ [%]", fontweight="bold")
    a2.spines.right.set_visible(False)
    a2.spines.top.set_visible(False)
    a2.scatter(flow_distance_means[worst_dist_flow],
               flow_model_results[worst_dist_flow], c="red", label=ALL_MODELS[worst_dist_flow])
    a2.scatter(flow_distance_means[best_dist_flow],
               flow_model_results[best_dist_flow], c="green", label=ALL_MODELS[best_dist_flow])
    a2.legend(loc="lower right")

    mouse_distances = np.load(mouse_path + "/all_distances.npz", allow_pickle=True)
    mouse_distances = {key: mouse_distances[key] for key in mouse_distances}
    mouse_distance_means = np.asarray([np.mean(mouse_distances[model]) for model in ALL_MODELS])
    best_dist_mouse = np.argmin(mouse_distance_means)
    worst_dist_mouse = np.argmax(mouse_distance_means)

    forearm_distances = np.load(forearm_path + "/all_distances.npz", allow_pickle=True)
    forearm_distances = {key: forearm_distances[key] for key in forearm_distances}
    forearm_distance_means = np.asarray([np.mean(forearm_distances[model]) for model in ALL_MODELS])
    best_dist_forearm = np.argmin(forearm_distance_means)
    worst_dist_forearm = np.argmax(forearm_distance_means)

    (a3) = subfig4.subplots(1, 1)

    positions = np.random.uniform(0.9, 1.1, size=np.shape(mouse_distance_means))
    positions[best_dist_forearm] = 1
    positions[worst_dist_forearm] = 1
    positions[best_dist_mouse] = 1

    a3.set_title("In Vivo Imaging")
    a3.set_ylabel("JSD [a.u.]", fontweight="bold")
    a3.scatter(positions, mouse_distance_means, c="gray")
    a3.spines.right.set_visible(False)
    a3.spines.top.set_visible(False)
    a3.set_xlim(0.75, 3)
    a3.set_xticks([1, 1.5], ["Mouse", "Forearm"])
    a3.scatter(positions[worst_dist_mouse],
               mouse_distance_means[worst_dist_mouse], c="red", label=ALL_MODELS[worst_dist_mouse])
    a3.scatter(positions[best_dist_mouse],
               mouse_distance_means[best_dist_mouse], c="green", label=ALL_MODELS[best_dist_mouse])

    a3.scatter(positions + 0.5, forearm_distance_means, c="gray")
    a3.scatter(positions[worst_dist_forearm] + 0.5,
               forearm_distance_means[worst_dist_forearm], c="red")
    a3.scatter(positions[best_dist_forearm] + 0.5,
               forearm_distance_means[best_dist_forearm], c="orange", label=ALL_MODELS[best_dist_forearm])

    a3.legend(loc="center right")

    subfig2.text(0, 0.90, "A", size=20, weight='bold')
    subfig3.text(0, 0.90, "B", size=20, weight='bold')
    subfig4.text(0, 0.90, "C", size=20, weight='bold')

    plt.savefig("distance_vs_results.png", dpi=300)


if __name__ == "__main__":
    create_distribution_distance_figure(baseline_path)