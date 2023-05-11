import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import ALL_MODELS
from paths import TRAINING_DATA_PATH, TEST_DATA_PATH
from utils.distribution_distance import compute_jsd, compute_kld


def compile_distance_measures(mouse_data_path):
    output_file = mouse_data_path + "/all_distances.npz"
    if os.path.exists(output_file):
        return output_file
    mouse_data_files = []
    for folder_path in glob.glob(mouse_data_path + "/*"):
        if os.path.isdir(folder_path):
            mouse_data_files.append(folder_path)

    all_wl = np.arange(700, 901, 5)
    results = {}

    for folder_path in mouse_data_files:

        filename = folder_path.split("/")[-1].split("\\")[-1]
        data = np.load(folder_path + "/" + filename + ".npz")
        spectra = data["spectra"]
        test_wl = data["wavelengths"]
        wl_mask = [x in test_wl for x in all_wl]
        num_wl = len(spectra)
        test_spectra = spectra.reshape((num_wl, -1))

        for train_path in glob.glob(TRAINING_DATA_PATH + "/*"):
            model_name = train_path.split("/")[-1].split("\\")[-1]
            data = np.load(train_path + f"/{model_name}_train.npz")
            train_spectra = data["spectra"][wl_mask, :]

            if model_name not in results:
                results[model_name] = []
            results[model_name].append(compute_kld(train_spectra, test_spectra))

    np.savez(output_file, **results, allow_pickle=True)
    return output_file


def compile_mouse_results(data_path):
    output_file = data_path + "/all_results.npz"
    if os.path.exists(output_file):
        return output_file
    mouse_data_files = []
    for folder_path in glob.glob(data_path + "/*"):
        if os.path.isdir(folder_path):
            mouse_data_files.append(folder_path)

    results = {
        "LU": []
    }
    for model in ALL_MODELS:
        results[model] = []
    for folder_path in mouse_data_files:
        filename = folder_path.split("/")[-1].split("\\")[-1]
        data = np.load(folder_path + "/" + filename + ".npz")
        plt.figure()
        lu = data["lu"]
        mask = data["reference_mask"] == 6
        results["LU"].append(lu[mask])

        for model in ALL_MODELS:
            model_result = np.load(f"{data_path}/{filename}/{filename}_{model}.npz")["estimate"]
            model_result = np.reshape(model_result, np.shape(lu))
            results[model].append(model_result[mask])

    results["LU"] = np.asarray(results["LU"])
    for model in ALL_MODELS:
        results[model] = np.asarray(results[model])

    np.savez(output_file, **results, allow_pickle=True)
    return output_file


def load_mouse(mouse_path, best, worst):
    print(mouse_path)
    mouse_name = mouse_path.split("/")[-1].split("\\")[-1]
    print(mouse_name)
    mouse_data = np.load(mouse_path + "/" + mouse_name + ".npz")
    wavelengths = mouse_data["wavelengths"]
    spectra = mouse_data["spectra"]
    mask = mouse_data["reference_mask"]
    lu = mouse_data["lu"] * 100
    image = spectra[np.argwhere(wavelengths == 800)]
    image = np.squeeze(image)

    best = np.load(mouse_path + "/" + mouse_name + f"_{best}.npz")["estimate"].reshape(*np.shape(image)) * 100
    worst = np.load(mouse_path + "/" + mouse_name + f"_{worst}.npz")["estimate"].reshape(*np.shape(image)) * 100

    image[mask <= 1] = np.nan
    lu[mask <= 1] = np.nan
    best[mask <= 1] = np.nan
    worst[mask <= 1] = np.nan

    return np.squeeze(image), np.squeeze(lu), np.squeeze(best), np.squeeze(worst)


def create_mouse_figure(data_path, models):
    dist_path = compile_distance_measures(data_path)
    res_path = compile_mouse_results(data_path)
    results = np.load(res_path, allow_pickle=True)
    results = {key: results[key] for key in results}

    distances = np.load(dist_path, allow_pickle=True)
    distances = {key: distances[key] for key in distances}

    distance_means = np.asarray([np.mean(distances[model]) for model in ALL_MODELS])
    distance_std = np.asarray([np.std(distances[model]) for model in ALL_MODELS])
    all_means = np.asarray([np.mean([np.mean(entry) * 100 for entry in results[model]]) for model in ALL_MODELS])
    all_std = np.asarray([np.std([np.mean(entry) * 100 for entry in results[model]]) for model in ALL_MODELS])

    best_dist = np.argmin(distance_means)
    worst_dist = np.argmax(distance_means)
    distance_means_2 = distance_means.copy()
    distance_means_2[best_dist] = np.mean(distance_means)
    distance_means_2[worst_dist] = np.mean(distance_means)
    second_best_dist = np.argmin(distance_means_2)
    second_worst_dist = np.argmax(distance_means_2)

    models = ["LU"] + models + [ALL_MODELS[best_dist], ALL_MODELS[second_best_dist],
                                ALL_MODELS[worst_dist], ALL_MODELS[second_worst_dist]]

    means = [[np.mean(entry) * 100 for entry in results[model]] for model in models]

    fig = plt.figure(layout="constrained", figsize=(13, 12))
    subfigs = fig.subfigures(2, 1)
    (ax1, ax2) = subfigs[0].subplots(1, 2)
    ((ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10)) = subfigs[1].subplots(2, 4)

    ax1.fill_between([0.5, len(models) + 0.5], 94, 98, color="green", alpha=0.25)
    ax1.hlines(96, xmin=0.5, xmax=len(models) + 0.5, color="green")
    ax1.boxplot(means, labels=["Linear\nUnmixing", "Baseline", f"Best\n({ALL_MODELS[best_dist]})",
                               f"2nd best\n|\n({ALL_MODELS[second_best_dist]})",
                               f"Worst\n({ALL_MODELS[worst_dist]})",
                               f"2nd worst\n|\n({ALL_MODELS[second_worst_dist]})"])
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.set_ylabel("Estimated sO$_2$ [%]", fontweight="bold")
    ax1.set_xlabel("Training Data Set", fontweight="bold")

    ax2.errorbar(distance_means, 96 - all_means, fmt="o", xerr=distance_std, yerr=all_std, alpha=0.5, ecolor="red")
    ax2.scatter(distance_means, 96 - all_means, c="black")
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.fill_between([plt.gca().dataLim.min[0], plt.gca().dataLim.max[0]], -2, 2, color="green", alpha=0.25, zorder=-20)
    ax2.hlines(0, xmin=plt.gca().dataLim.min[0], xmax=plt.gca().dataLim.max[0], color="green", zorder=-20)
    ax2.set_ylabel("Distance from ideal arterial sO$_2$ [%]", fontweight="bold")
    ax2.set_xlabel("Kullback leibler divergence between the samples [a.u.]", fontweight="bold")

    def add_image(ax, img, data, title):
        ax.set_title(title)
        plt.colorbar(img)
        hist, bins = np.histogram(data, bins=50, range=[0, 100])
        ax.stairs(len(data) - (hist / np.max(hist) * 50), len(data)-(bins / 100 * len(data)), orientation="horizontal",
                  baseline=len(data), fill=True, alpha=0.5)
        ax.axis("off")

    image, lu, best, worst = load_mouse(data_path + "/Mouse_02", ALL_MODELS[best_dist], ALL_MODELS[worst_dist])
    add_image(ax3, ax3.imshow(image, cmap="magma"), image, "PA Signal @800 nm [a.u.]")
    add_image(ax4, ax4.imshow(lu, vmin=0, vmax=100), lu, "Linear Unmixing [%]")
    add_image(ax5, ax5.imshow(best, vmin=0, vmax=100), best, "Best KLD [%]")
    add_image(ax6, ax6.imshow(worst, vmin=0, vmax=100), worst, "Worst KLD [%]")

    image, lu, best, worst = load_mouse(data_path + "/Mouse_05", ALL_MODELS[best_dist], ALL_MODELS[worst_dist])
    add_image(ax7, ax7.imshow(image, cmap="magma"), image, None)
    add_image(ax8, ax8.imshow(lu, vmin=0, vmax=100), lu, None)
    add_image(ax9, ax9.imshow(best, vmin=0, vmax=100), best, None)
    add_image(ax10, ax10.imshow(worst, vmin=0, vmax=100), worst, None)



    plt.savefig(data_path + "/result.png", dpi=300)


if __name__ == "__main__":
    create_mouse_figure(fr"{TEST_DATA_PATH}\mouse/", ["BASE"])