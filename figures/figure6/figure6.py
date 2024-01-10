import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import ALL_MODELS
from paths import TRAINING_DATA_PATH, TEST_DATA_PATH
from utils.distribution_distance import compute_jsd
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar

RECOMPUTE = False
VENOUS_MASK = 5

mouse_path = TEST_DATA_PATH + "/mouse/"
forearm_path = TEST_DATA_PATH + "/forearm/"

print("Mouse")
mouse_distances = np.load(mouse_path + "/all_distances.npz", allow_pickle=True)
mouse_distances = {key: mouse_distances[key] for key in mouse_distances}
mouse_distance_means = np.asarray([np.mean(mouse_distances[model]) for model in ALL_MODELS])
best_dist_mouse = np.argmin(mouse_distance_means)
worst_dist_mouse = np.argmax(mouse_distance_means)
print("Best", ALL_MODELS[best_dist_mouse])
print("Worst", ALL_MODELS[worst_dist_mouse])

print("Forearm")
forearm_distances = np.load(forearm_path + "/all_distances.npz", allow_pickle=True)
forearm_distances = {key: forearm_distances[key] for key in forearm_distances}
forearm_distance_means = np.asarray([np.mean(forearm_distances[model]) for model in ALL_MODELS])
best_dist_forearm = np.argmin(forearm_distance_means)
print("Best", ALL_MODELS[best_dist_forearm])
worst_dist_forearm = np.argmax(forearm_distance_means)
print("Worst", ALL_MODELS[worst_dist_forearm])


def compile_distance_measures(data_path):
    output_file = data_path + "/all_distances.npz"
    if not RECOMPUTE and os.path.exists(output_file):
        return output_file
    data_files = []
    for folder_path in glob.glob(data_path + "/*"):
        if os.path.isdir(folder_path):
            data_files.append(folder_path)

    all_wl = np.arange(700, 901, 5)
    results = {}

    for folder_path in data_files:
        filename = folder_path.split("/")[-1].split("\\")[-1]
        print(filename)
        data = np.load(folder_path + "/" + filename + ".npz")
        spectra = data["spectra"]
        test_wl = data["wavelengths"]
        wl_mask = [x in test_wl for x in all_wl]
        test_spectra = spectra.reshape((len(test_wl), -1))

        for train_path in glob.glob(TRAINING_DATA_PATH + "/*"):
            model_name = train_path.split("/")[-1].split("\\")[-1]
            data = np.load(train_path + f"/{model_name}_train.npz")
            train_spectra = data["spectra"][wl_mask, :]

            if model_name not in results:
                results[model_name] = []
            results[model_name].append(compute_jsd(train_spectra, test_spectra))

    np.savez(output_file, **results, allow_pickle=True)
    return output_file


def compile_results(data_path, mouse):
    if not RECOMPUTE and os.path.exists(data_path + "/all_results_artery.npz"):
        return data_path + "/all_results_artery.npz", data_path + "/all_results_kidney.npz"
    mouse_data_files = []
    for folder_path in glob.glob(data_path + "/*"):
        if os.path.isdir(folder_path):
            mouse_data_files.append(folder_path)

    results = {
        "LU": []
    }
    results_2 = {
        "LU": []
    }
    for model in ALL_MODELS:
        results[model] = []
        results_2[model] = []
    for folder_path in mouse_data_files:
        filename = folder_path.split("/")[-1].split("\\")[-1]
        data = np.load(folder_path + "/" + filename + ".npz")
        wavelengths = data["wavelengths"]
        lu = data["lu"]
        spectra = data["spectra"]
        image = spectra[np.argwhere(wavelengths == 800)]
        image = np.squeeze(image)
        mask2 = None
        if mouse:
            mask = data["reference_mask"] == 6
            mask2 = data["reference_mask"] == VENOUS_MASK
            results_2["LU"].append(lu[mask2])
        else:
            mask = data["reference_mask"] == 1
        results["LU"].append(lu[mask])

        for model in ALL_MODELS:
            model_result = np.load(f"{data_path}/{filename}/{filename}_{model}_{len(wavelengths)}.npz")["estimate"].reshape(*np.shape(image))
            results[model].append(model_result[mask])
            if mouse:
                results_2[model].append(model_result[mask2])

    results["LU"] = np.asarray(results["LU"], dtype=object)
    for model in ALL_MODELS:
        results[model] = np.asarray(results[model], dtype=object)

    np.savez(data_path + "/all_results_artery.npz", **results, allow_pickle=True)

    if mouse:
        results_2["LU"] = np.asarray(results_2["LU"], dtype=object)
        for model in ALL_MODELS:
            results_2[model] = np.asarray(results_2[model], dtype=object)
        np.savez(data_path + "/all_results_kidney.npz", **results_2, allow_pickle=True)

    return data_path + "/all_results_artery.npz", data_path + "/all_results_kidney.npz"


def load_example(path, best, worst, model_1, model_2):
    name = path.split("/")[-1].split("\\")[-1]
    data = np.load(path + "/" + name + ".npz")
    wavelengths = data["wavelengths"]
    spectra = data["spectra"]
    mask = data["reference_mask"]
    lu = data["lu"] * 100
    image = spectra[np.argwhere(wavelengths == 800)]
    image = np.squeeze(image)

    best = np.load(path + "/" + name + f"_{best}_{len(wavelengths)}.npz")["estimate"].reshape(*np.shape(image)) * 100
    worst = np.load(path + "/" + name + f"_{worst}_{len(wavelengths)}.npz")["estimate"].reshape(*np.shape(image)) * 100
    model_1 = np.load(path + "/" + name + f"_{model_1}_{len(wavelengths)}.npz")["estimate"].reshape(*np.shape(image)) * 100
    model_2 = np.load(path + "/" + name + f"_{model_2}_{len(wavelengths)}.npz")["estimate"].reshape(*np.shape(image)) * 100

    return (np.squeeze(image), np.squeeze(lu), np.squeeze(best),
            np.squeeze(worst), np.squeeze(model_1),
            np.squeeze(model_2), np.squeeze(mask))


def load_data(path, models, mouse=False):
    dist_path = compile_distance_measures(path)
    res_path = compile_results(path, mouse)
    print(res_path)
    results = np.load(res_path[0], allow_pickle=True)
    results = {key: results[key] for key in results}
    distances = np.load(dist_path, allow_pickle=True)
    distances = {key: distances[key] for key in distances}
    distance_means = np.asarray([np.mean(distances[model]) for model in ALL_MODELS])

    results_kidney = None
    if mouse:
        results_kidney = np.load(res_path[1], allow_pickle=True)
        results_kidney = {key: results_kidney[key] for key in results_kidney}

    best_dist = np.argmin(distance_means)
    worst_dist = np.argmax(distance_means)
    distance_means_2 = distance_means.copy()
    distance_means_2[best_dist] = np.mean(distance_means)
    distance_means_2[worst_dist] = np.mean(distance_means)

    models = ["LU"] + models + [ALL_MODELS[best_dist], ALL_MODELS[worst_dist]]

    means = [[np.mean(entry) * 100 for entry in results[model]] for model in models]

    means_kidney = None
    if mouse:
        means_kidney = [[np.mean(entry) * 100 for entry in results_kidney[model]] for model in models]

        for (model, avg, avg_kid) in zip(models, means, means_kidney):
            print(model, np.mean(avg), np.mean(avg_kid))

    return models, means, means_kidney, best_dist, worst_dist


def create_forearm_figure(models):

    forearm_data_path = fr"{TEST_DATA_PATH}\forearm/"
    mouse_data_path = fr"{TEST_DATA_PATH}\mouse/"

    models_forearm, means_forearm, _, best_forearm, worst_forearm = load_data(forearm_data_path, models)
    models_mouse, means_mouse, means_mouse_kidney, best_mouse, worst_mouse = load_data(mouse_data_path,
                                                                                       models, mouse=True)

    fig, ((ax2, ax3, ax1),
          (ax4, ax5, ax6),
          (ax8, ax9, ax7),
          (ax10, ax11, ax12)) = plt.subplots(ncols=3, nrows=4, layout="constrained", figsize=(8, 8.5),
                                             gridspec_kw={"height_ratios": [1, 1, 1.8, 1.8]})

    np.random.seed(1336)
    positions = np.random.uniform(0.9, 1.1, size=np.shape(mouse_distance_means))
    positions[best_dist_forearm] = 1
    positions[worst_dist_forearm] = 1
    positions[best_dist_mouse] = 1

    ax3.set_title("$D_{JS}$ data analysis [a.u.]")
    ax3.spines.left.set_visible(False)
    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax3.set_yticks([], [])
    ax3.set_ylim(0.6, 1.2)
    ax3.scatter(forearm_distance_means, positions, c="gray", s=5)
    ax3.scatter(forearm_distance_means[worst_dist_forearm], positions[worst_dist_forearm], marker="*", c="red")
    ax3.scatter(forearm_distance_means[best_dist_forearm], positions[best_dist_forearm], marker="D", c="orange",
                label=ALL_MODELS[best_dist_forearm])
    ax3.scatter(forearm_distance_means[2], positions[2], marker="s", c="blue")
    ax3.legend(loc="lower right")
    ax3.tick_params(direction="in", labelsize=8)

    ax9.set_title("$D_{JS}$ data analysis [a.u.]")
    ax9.spines.right.set_visible(False)
    ax9.spines.left.set_visible(False)
    ax9.spines.top.set_visible(False)
    ax9.set_yticks([], [])
    ax9.set_ylim(0.6, 1.2)
    ax9.scatter(mouse_distance_means, positions, c="gray", s=5)
    ax9.scatter(mouse_distance_means[worst_dist_mouse], positions[worst_dist_mouse], marker="*", c="red",
                label=ALL_MODELS[worst_dist_mouse])
    ax9.scatter(mouse_distance_means[best_dist_mouse], positions[best_dist_mouse], marker="P", c="#03A9F4",
                label=ALL_MODELS[best_dist_mouse])
    ax9.scatter(mouse_distance_means[2], positions[2], marker="s", c="blue",
                label=ALL_MODELS[2])
    ax9.legend(loc="lower right")
    ax9.tick_params(direction="in", labelsize=8)

    ax1.text(0.5, 100, "C", size=22.5, weight='bold')
    ax2.text(0, 50, "A", size=22.5, weight='bold', color="white")
    ax3.text(0.465, 1.17, "B", size=22.5, weight='bold', color="black")
    ax4.text(0, 50, "D", size=22.5, weight='bold', color="white")
    ax5.text(0, 50, "E", size=22.5, weight='bold', color="white")
    ax6.text(0, 50, "F", size=22.5, weight='bold', color="white")
    ax7.text(0.5, 99, "I", size=22.5, weight='bold')
    ax8.text(0, 40, "G", size=22.5, weight='bold', color="white")
    ax9.text(0.38, 1.14, "H", size=22.5, weight='bold', color="black")
    ax10.text(10, 40, "J", size=22.5, weight='bold', color="white")
    ax11.text(0, 40, "K", size=22.5, weight='bold', color="white")
    ax12.text(0, 40, "L", size=22.5, weight='bold', color="white")

    model_names_forearm = np.copy(models_forearm)
    model_names_mouse = np.copy(models_mouse)

    ax1.set_title("sO$_2$ distribution (N=7)")
    ax1.fill_between([0.5, len(models_forearm) + 0.5], 90, 100, color="red", alpha=0.25)
    ax1.hlines(95, xmin=0.5, xmax=len(models_forearm) + 0.5, color="red")
    ax1.boxplot(means_forearm,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="red", color="red", alpha=0.5),
                medianprops=dict(linewidth=1, color='black'))
    ax1.tick_params(direction="in")
    models_forearm[0] = "LU"
    models_forearm[-1] = "Worst"
    models_forearm[-2] = "Best"
    ax1.set_xticks(np.arange(len(models_forearm)) + 1, models_forearm, fontsize=8)
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.set_ylabel("Est. sO$_2$ [%]", fontweight="bold", labelpad=-5)

    def add_image(ax, img, data, mask, title, color="red", mouse=False, colorbar=False):
        ax.set_title(title)
        if colorbar:
            cbar = plt.colorbar(img, location="left", pad=0)
            cbar.set_label("Est. sO$_2$ [%]", fontweight="bold", labelpad=-5)
        # data[data < 1e-5] = 1e-5
        # data[data > 100 - 1e-5] = 100 - 1e-5
        if mouse:
            ax.contour(mask == VENOUS_MASK, colors="blue")
            ax.contour(mask == 6, colors="red")
            ax.add_artist(ScaleBar(0.075, "mm", location="lower left", fixed_value=2,
                                   fixed_units="mm"))

        else:
            ax.contour(mask == 1, colors=color)
            ax.add_artist(ScaleBar(0.075, "mm", location="lower left", fixed_value=5,
                                   fixed_units="mm"))

        ydim, xdim = np.shape(image)

        WIDTH = xdim / 3
        BASELINE = ydim - 25
        LEFTSHIFT = 15
        HEIGHT = ydim / 5
        OUTLINE_COLOR = "black"

        if mouse:
            mean_kidney_sO2 = (np.nanmean(data[mask == 5]) / 100)
            mean_arterial_sO2 = (np.nanmean(data[mask == 6]) / 100)
            ax.plot([xdim - LEFTSHIFT - WIDTH + mean_kidney_sO2 * WIDTH,
                     xdim - LEFTSHIFT - WIDTH + mean_kidney_sO2 * WIDTH],
                    [BASELINE, BASELINE - HEIGHT], c="blue", linewidth=1.5)
        else:
            mean_arterial_sO2 = (np.mean(data[mask == 1]) / 100)

        hist, bins = np.histogram(data, bins=50, range=(0, 100))
        ax.plot([xdim - WIDTH - LEFTSHIFT, xdim - LEFTSHIFT],
                [BASELINE, BASELINE], c=OUTLINE_COLOR, linewidth=1)
        ax.stairs(BASELINE - (hist / np.max(hist) * HEIGHT),
                  (xdim - LEFTSHIFT - WIDTH + bins / 100 * WIDTH),
                  baseline=BASELINE, fill=True, color="lightgray")
        ax.stairs(BASELINE - (hist / np.max(hist) * HEIGHT),
                  (xdim - LEFTSHIFT - WIDTH + bins / 100 * WIDTH),
                  baseline=BASELINE, fill=False, color=OUTLINE_COLOR, linewidth=1)
        x_points = np.asarray(xdim - LEFTSHIFT - WIDTH + bins / 100 * WIDTH)

        for i in np.arange(0, 20, 2):
            ax.scatter(x_points, np.ones_like(x_points) * BASELINE + 2 + i,
                       c=((x_points-np.min(x_points))/(np.max(x_points)-np.min(x_points))),
                          s=2, marker="s", cmap="viridis")

        ax.add_artist(Rectangle((xdim - LEFTSHIFT - WIDTH - 2, BASELINE), WIDTH + 4, 22, fill=False, edgecolor="black"))

        if mouse:
            ax.text(xdim - LEFTSHIFT - WIDTH, BASELINE + 16, "0", color="white", fontsize=9)
            ax.text(xdim - LEFTSHIFT - WIDTH/2 - 20, BASELINE + 16, "sO$_2$", color="white", fontsize=9)
            ax.text(xdim - LEFTSHIFT - 35, BASELINE + 16, "100", color="white", fontsize=9)
        else:
            ax.text(xdim - LEFTSHIFT - WIDTH, BASELINE + 17, "0", color="white", fontsize=8)
            ax.text(xdim - LEFTSHIFT - WIDTH / 2 - 20, BASELINE + 17, "sO$_2$", color="white", fontsize=8)
            ax.text(xdim - LEFTSHIFT - 42, BASELINE + 17, "100", color="white", fontsize=8)

        ax.plot([xdim - LEFTSHIFT - WIDTH + mean_arterial_sO2 * WIDTH,
                 xdim - LEFTSHIFT - WIDTH + mean_arterial_sO2 * WIDTH],
                [BASELINE, BASELINE - HEIGHT], c="red", linewidth=1.5)

        ax.axis("off")



    image, lu, best, worst, model_1, model_2, mask = load_example(forearm_data_path + "/Forearm_07",
                                                                  ALL_MODELS[best_forearm], ALL_MODELS[worst_forearm],
                                                                  "BASE", "ALL")
    LOWER = 130
    UPPER = 350
    image = image[LOWER:UPPER, :]
    lu = lu[LOWER:UPPER, :]
    best = best[LOWER:UPPER, :]
    worst = worst[LOWER:UPPER, :]
    model_1 = model_1[LOWER:UPPER, :]
    model_2 = model_2[LOWER:UPPER, :]
    mask = mask[LOWER:UPPER, :]
    im = ax2.imshow(image/100, cmap="magma", vmin=-10, vmax=50)
    ax2.set_title("Forearm PAI [a.u.]")
    ax2.axis("off")
    ax2.add_artist(ScaleBar(0.075, "mm", location="lower left"))
    ax2.contour(mask == 1, colors="red")
    ax2.plot([], [], c="red", label="ARTERY")
    ax2.legend(loc="lower right", labelspacing=0, fontsize=9,
              borderpad=0.1, handlelength=1, handletextpad=0.4,
              labelcolor="white", framealpha=0)
    #add_image(ax3, ax3.imshow(lu, vmin=0, vmax=100), lu, mask, "Linear Unmixing")
    add_image(ax4, ax4.imshow(best, vmin=0, vmax=100), best, mask, f"Best: {model_names_forearm[-2]}")
    add_image(ax5, ax5.imshow(worst, vmin=0, vmax=100), worst, mask, f"Worst: {model_names_forearm[-1]}")
    add_image(ax6, ax6.imshow(model_1, vmin=0, vmax=100), model_1, mask, f"{model_names_forearm[-3]}")

    ax7.set_title("sO$_2$ distribution (N=7)")
    ax7.fill_between([0.3, len(models_mouse) + 0.7], 94, 98, color="red", alpha=0.25)
    ax7.hlines(96, xmin=0.3, xmax=len(models_mouse) + 0.7, color="red")

    ax7.fill_between([0.3, len(models_mouse) + 0.7], 60, 70, color="blue", alpha=0.25)
    ax7.hlines(65, xmin=0.3, xmax=len(models_mouse) + 0.7, color="blue")
    bp_1 = ax7.boxplot(means_mouse, widths=0.4, positions=np.arange(len(models_mouse)) + 0.8,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="red", color="red", alpha=0.5),
                       medianprops=dict(linewidth=1, color='black'))
    bp_2 = ax7.boxplot(means_mouse_kidney, widths=0.4, positions=np.arange(len(models_mouse)) + 1.2,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="blue", color="blue", alpha=0.5),
                       medianprops=dict(linewidth=1, color='black'))
    ax7.tick_params(direction="in")
    models_mouse[0] = "LU"
    models_mouse[-1] = "Worst"
    models_mouse[-2] = "Best"
    ax7.set_xticks(np.arange(len(models_mouse)) + 1, models_mouse, fontsize=8)
    ax7.spines.right.set_visible(False)
    ax7.spines.top.set_visible(False)
    ax7.set_ylabel("Est. sO$_2$ [%]", fontweight="bold", labelpad=-5)

    image, lu, best, worst, model_1, model_2, mask = load_example(mouse_data_path + "/Mouse_02",
                                                                  ALL_MODELS[best_mouse], ALL_MODELS[worst_mouse],
                                                                  "BASE", "ALL")

    lu = np.copy(lu)
    best = np.copy(best)
    worst = np.copy(worst)

    lu[mask < 2] = np.nan
    best[mask < 2] = np.nan
    worst[mask < 2] = np.nan
    model_1[mask < 2] = np.nan
    im = ax8.imshow(image/10, cmap="magma", vmin=5, vmax=45)
    ax8.set_title("Mouse PAI [a.u.]")
    ax8.axis("off")
    ax8.add_artist(ScaleBar(0.075, "mm", location="lower left"))
    ax8.contour(mask == 6, colors="red")
    ax8.contour(mask == 5, colors="blue")
    ax8.plot([], [], c="red", label="ARTERY")
    ax8.plot([], [], c="blue", label="SPINE")
    ax8.legend(loc="lower right", labelspacing=0, fontsize=9,
               borderpad=0.1, handlelength=1, handletextpad=0.4,
               labelcolor="white", framealpha=0)
    # cb = plt.colorbar(mappable=im, ax=ax8, location="right", pad=0)
    #ax9.imshow(image, cmap="magma", vmin=50, vmax=450)
    ax10.imshow(image, cmap="magma", vmin=50, vmax=450)
    ax11.imshow(image, cmap="magma", vmin=50, vmax=450)
    ax12.imshow(image, cmap="magma", vmin=50, vmax=450)
    #add_image(ax9, ax9.imshow(lu, vmin=0, vmax=100), lu, mask, "Linear Unmixing", mouse=True)
    add_image(ax10, ax10.imshow(best, vmin=0, vmax=100), best, mask, f"Best: {model_names_mouse[-2]}", mouse=True)
    add_image(ax11, ax11.imshow(worst, vmin=0, vmax=100), worst, mask, f"Worst: {model_names_mouse[-1]}", mouse=True)
    add_image(ax12, ax12.imshow(model_1, vmin=0, vmax=100), model_1, mask, f"{model_names_mouse[-3]}",
              mouse=True)

    plt.savefig("figure6.pdf", dpi=300)


if __name__ == "__main__":
    create_forearm_figure(["BASE"])
