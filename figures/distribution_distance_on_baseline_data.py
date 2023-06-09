import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import ALL_MODELS
from scipy.stats import iqr, linregress
from paths import TRAINING_DATA_PATH, TEST_DATA_PATH
from utils.distribution_distance import compute_jsd, compute_kld


def compile_distance_measures(data_path):
    output_file = data_path + "/all_distances.npz"
    # if os.path.exists(output_file):
    #     return output_file
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

    f, (a0, a1) = plt.subplots(1, 2, width_ratios=[1, 3], layout="constrained")

    a0.violinplot(dist_mean)
    a0.spines.right.set_visible(False)
    a0.spines.top.set_visible(False)
    a0.set_ylabel("Jensen-Shannon divergence", fontweight="bold")
    a0.set_xticks([], [])
    a0.set_xlabel("Kernel density\nestimate", fontweight="bold")
    pos = np.random.normal(1, 0.01, size=np.shape(dist_mean))
    a0.scatter(pos, dist_mean, c="gray", alpha=0.75)
    a0.scatter(pos[1], np.mean(distances["BASE"].item()["JSD"]), c="blue", label="BASE")
    a0.scatter(pos[8], np.mean(distances["ILLUM_POINT"].item()["JSD"]), c="red", label="ILLUM_POINT")
    a0.scatter(pos[17], np.mean(distances["RES_0.15"].item()["JSD"]), c="green", label="RES_0.15")

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

    a1.legend(loc="lower right")

    a1.spines.right.set_visible(False)
    a1.spines.top.set_visible(False)
    a1.set_ylabel("Absolute estimation error [%]", fontweight="bold")
    a1.set_xlabel("Jensen-Shannon divergence", fontweight="bold")

    plt.savefig(data_path + "/distance_vs_results.png", dpi=300)


if __name__ == "__main__":
    create_distribution_distance_figure(fr"{TEST_DATA_PATH}\baseline")