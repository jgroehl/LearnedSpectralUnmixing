import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import ALL_MODELS
from paths import TRAINING_DATA_PATH, TEST_DATA_PATH
from utils.distribution_distance import compute_jsd
import matplotlib.gridspec as gridspec

RECOMPUTE = False

COLOURS = ["#201923", "#fcff5d", "#7dfc00", "#0ec434", "#228c68",
           "#8ad8e8", "#235b54", "#29bdab", "#3998f5", "#37294f", "#277da7",
           "#3750db", "#f22020", "#991919", "#ffcba5", "#e68f66", "#c56133",
           "#96341c", "#632819", "#ffc413", "#f47a22", "#2f2aa0", "#b732cc",
           "#772b9d", "#f07cab", "#d30b94",  "#c3a5b4", "#946aa2", "#5d4c86"]


def compile_distance_measures(data_path):
    data_files = []
    for folder_path in glob.glob(data_path + "/*"):
        if os.path.isdir(folder_path):
            data_files.append(folder_path)

    all_wl = np.arange(700, 901, 5)

    paths = []

    for folder_path in data_files:
        paths.append(folder_path + "/all_distances.npz")
        if os.path.exists(folder_path + "/all_distances.npz"):
            continue
        results = {}
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

        np.savez(folder_path + "/all_distances.npz", **results, allow_pickle=True)

    return paths


def compile_results(data_path):

    flow_data_files = []
    for folder_path in glob.glob(data_path + "/*"):
        if os.path.isdir(folder_path):
            flow_data_files.append(folder_path)

    paths = []

    for folder_path in flow_data_files:
        paths.append(folder_path + "/all_results.npz")
        if not RECOMPUTE and os.path.exists(folder_path + "/all_results.npz"):
            continue
        results = {
            "LU": []
        }
        for model in ALL_MODELS:
            results[model] = []
        filename = folder_path.split("/")[-1].split("\\")[-1]
        data = np.load(folder_path + "/" + filename + ".npz")

        wavelengths = data["wavelengths"][:-1]
        lu = data["lu"]
        spectra = data["spectra"]
        image = spectra[np.argwhere(wavelengths == 800)]
        image = np.squeeze(image)
        results["LU"].append(lu)
        results["timesteps"] = data["timesteps"]
        results["reference"] = data["oxygenations"]

        for model in ALL_MODELS:
            model_result = np.load(f"{data_path}/{filename}/{filename}_{model}_{len(wavelengths)}.npz")["estimate"].reshape(*np.shape(image))
            results[model].append(model_result)

        results["LU"] = np.asarray(results["LU"], dtype=object)
        for model in ALL_MODELS:
            results[model] = np.asarray(results[model], dtype=object)

        np.savez(folder_path + "/all_results.npz", **results, allow_pickle=True)

    return paths


def load_data(path, models):
    dist_path = compile_distance_measures(path)
    res_path = compile_results(path)
    print(dist_path)
    print(res_path)

    results = [np.load(r_p, allow_pickle=True) for r_p in res_path]
    results = [{key: res[key] for key in res} for res in results]
    distances = [np.load(d_p, allow_pickle=True) for d_p in dist_path]
    distances = [{key: dis[key] for key in dis} for dis in distances]

    distance_means = [np.asarray([np.mean(dis[model]) for model in ALL_MODELS]) for dis in distances]

    best_dist = [np.argmin(d_m) for d_m in distance_means]
    worst_dist = [np.argmax(d_m) for d_m in distance_means]

    models = [["LU"] + models + [ALL_MODELS[bd], ALL_MODELS[wd]] for bd, wd in zip(best_dist, worst_dist)]


    return models, results, best_dist, worst_dist


def create_flow_figure(models):

    flow_data_path = fr"{TEST_DATA_PATH}\flow/"

    models_flow, flow_estimates, best_flow, worst_flow = load_data(flow_data_path, models)

    fig = plt.figure(layout="constrained", figsize=(12, 5.5))
    gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    subfigs = [fig.add_subfigure(gs[0, 0]), fig.add_subfigure(gs[0, 1])]

    for subfig, models, estimates, best, worst in zip(subfigs, models_flow, flow_estimates, best_flow, worst_flow):

        ax = subfig.subplots(1, 1)
        timesteps = estimates["timesteps"]
        timesteps = timesteps - min(timesteps)
        timesteps = timesteps / 60
        ts = np.unique(timesteps)
        n_ts = len(ts)
        timesteps = timesteps.reshape((n_ts, -1))
        reference = estimates["reference"].reshape((n_ts, -1))

        def add_line(val, color, label):
            val = val.copy()
            val = val - reference
            val = val.astype(float)
            mean = np.mean(val, axis=1) * 100
            std = np.std(val, axis=1) * 100
            ax.plot(np.mean(reference, axis=1)*100, mean, color=color, label=label)
            ax.fill_between(np.mean(reference, axis=1)*100, mean-std, mean+std, color=color, alpha=0.3)

        add_line(reference, "green", "pO2 reference")

        for model in models:
            index = np.argwhere(np.asarray(ALL_MODELS + ["LU"]) == model).item()
            add_line(estimates[model].reshape((n_ts, -1)), COLOURS[index], model)

        ax.set_xlabel("Reference Oxygenation [%]", fontweight="bold", fontsize=12)
        ax.set_ylabel("Difference in oxygenation [%]", fontweight="bold", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-45, 45)

        ax.legend()

    subfigs[0].text(0, 0.90, "A", size=30, weight='bold')
    subfigs[1].text(0, 0.90, "B", size=30, weight='bold')

    plt.savefig("figure4.png", dpi=300)


if __name__ == "__main__":
    create_flow_figure(["BASE"])
