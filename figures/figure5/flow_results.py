import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import ALL_MODELS
from paths import TRAINING_DATA_PATH, TEST_DATA_PATH, MODEL_PATH
from utils.distribution_distance import compute_jsd
import matplotlib.gridspec as gridspec
from matplotlib_scalebar.scalebar import ScaleBar
from models import LSTMParams
from utils.io import load_test_data_as_tensorflow_datasets_with_wavelengths
from sklearn.ensemble import RandomForestRegressor as LSDMethod

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

    print("Loading and processing data.")
    example_data = np.load(fr"{TEST_DATA_PATH}\flow/flow_1/examples.npz")

    flow_data_for_lsd = np.load(fr"{TEST_DATA_PATH}\flow/flow_1/flow_1.npz")
    test_lsd_spectra = flow_data_for_lsd["spectra"][:-1, :]
    test_lsd_spectra = ((test_lsd_spectra - np.nanmean(test_lsd_spectra, axis=0)[np.newaxis, :]) /
                        np.nanstd(test_lsd_spectra, axis=0)[np.newaxis, :])

    data = example_data["recon"][-168:]
    # apply LSTM method to the data
    data = data.swapaxes(0, 1)
    data_shape = np.shape(data)
    data = data.reshape((11, -1))
    wavelengths = example_data["wavelengths"][:-1]
    np.savez("tmp_data.npz",
             spectra=data,
             wavelengths=example_data["wavelengths"])
    data_lstm = load_test_data_as_tensorflow_datasets_with_wavelengths("tmp_data.npz", example_data["wavelengths"][:-1])

    print("LSD estimates.")
    if os.path.exists("lsd_results.npz"):
        res = np.load("lsd_results.npz")
        lsd_result = res["lsd_result"]
        lsd_timestep_results = res["lsd_timestep_results"]
    else:
        train_data = np.load(TRAINING_DATA_PATH + "SMALL/SMALL_train.npz")
        train_spectra = train_data["spectra"]
        train_spectra = (train_spectra - np.nanmean(train_spectra, axis=0)[np.newaxis, :]) / np.nanstd(train_spectra, axis=0)[np.newaxis, :]
        oxy = train_data["oxygenation"]
        all_wl = np.arange(700, 901, 5)
        all_wl_mask = [wl in wavelengths for wl in all_wl]
        train_spectra = train_spectra[all_wl_mask, :]

        lsd_rf = LSDMethod(n_jobs=-1)
        lsd_rf.fit(train_spectra.T, oxy)

        flow_spectra = data[:-1, :]
        flow_spectra = (flow_spectra - np.nanmean(flow_spectra, axis=0)[np.newaxis, :]) / np.nanstd(flow_spectra,
                                                                                                    axis=0)[np.newaxis, :]
        lsd_result = lsd_rf.predict(flow_spectra.T)

        lsd_result = np.reshape(lsd_result, (1, data_shape[1], data_shape[2], data_shape[3], data_shape[4]))
        lsd_result = lsd_result.swapaxes(0, 1) * 100

        lsd_timestep_results = lsd_rf.predict(test_lsd_spectra.T)

        np.savez("lsd_results.npz",
                 lsd_result=lsd_result,
                 lsd_timestep_results=lsd_timestep_results)

    print("LSTM estimates.")
    if os.path.exists("lstm_result.npz"):
        res = np.load("lstm_result.npz")
        lstm_result = res["lstm_result"]
    else:
        model_params = LSTMParams.load(MODEL_PATH + f"SMALL_LSTM_10.h5")
        model_params.compile()
        results = []
        for i in range(len(data_lstm) // 200000):
            result = model_params(data_lstm[i * 200000:(i + 1) * 200000])
            results.append(result.numpy())
        i = len(data_lstm) // 200000
        results.append(model_params(data_lstm[i * 200000:]).numpy())
        lstm_result = np.vstack(results)
        lstm_result = np.reshape(lstm_result, (1, data_shape[1], data_shape[2], data_shape[3], data_shape[4]))
        lstm_result = lstm_result.swapaxes(0, 1) * 100
        np.savez("lstm_result.npz",
                 lstm_result=lstm_result)


    flow_data_path = fr"{TEST_DATA_PATH}\flow/"

    models_flow, flow_estimates, best_flow, worst_flow = load_data(flow_data_path, models)

    fig = plt.figure(layout="constrained", figsize=(12, 5.5))
    gs = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)

    subfig = fig.add_subfigure(gs[1:3, 0])
    legend = True

    models, estimates, best, worst = models_flow[0], flow_estimates[0], best_flow[0], worst_flow[0]

    ax = subfig.subplots(1, 1)
    timesteps = estimates["timesteps"]
    timesteps = timesteps - min(timesteps)
    timesteps = timesteps / 60
    ts = np.unique(timesteps)
    n_ts = len(ts)
    timesteps = timesteps.reshape((n_ts, -1))
    reference = estimates["reference"].reshape((n_ts, -1))

    def add_line(val, color, label, linestyle="solid"):
        val = val.copy()
        #val = val - reference
        val = val.astype(float)
        mean = np.mean(val, axis=1) * 100
        ax.plot(ts, mean, color=color, label=label, linestyle=linestyle)

    def add_scatter(val, color, label, linestyle="solid"):
        val = val.copy()
        #val = val - reference
        val = val.astype(float)
        mean = np.mean(val, axis=1) * 100
        # np.mean(reference, axis=1)*100
        ax.scatter(ts, mean, color=color, label=label, linestyle=linestyle, s=3)

    add_line(reference, "black", "sO$_2$ reference (Severinghaus)", linestyle="dashed")

    mae_lstm = np.median(np.abs(estimates["SMALL"].reshape((n_ts, -1)) - reference)) * 100
    mae_lsd = np.median(np.abs(lsd_timestep_results.reshape((n_ts, -1)) - reference)) * 100

    add_scatter(lsd_timestep_results.reshape((n_ts, -1)), "orange",
                fr"Learned Spectral Decolouring ($\epsilon$sO$_2$: {mae_lsd:.1f}%)")
    add_scatter(estimates["SMALL"].reshape((n_ts, -1)), "purple",
                fr"LSTM-based method ($\epsilon$sO$_2$: {mae_lstm:.1f}%)")

    ax.set_ylabel("sO$_2$ reference [%]", fontweight="bold", fontsize=12)
    ax.set_xlabel("Time [min]", fontweight="bold", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.set_ylim(-20, 20)
    if legend:
        ax.legend(loc="upper right", labelspacing=1, framealpha=0)
    legend = not legend

    subfig.text(-0.005, 0.92, "B", size=30, weight='bold')

    subfig_images = fig.add_subfigure(gs[0, 0:2])
    ex_signal = example_data["recon"]
    ex_sO2 = example_data["sO2"][-168:] * 100
    ex_sO2[0, :, 0:10, 0:10, :] = np.nan
    lstm_result[0, :, 0:10, 0:10, :] = np.nan
    lsd_result[0, :, 0:10, 0:10, :] = np.nan

    ex_sO2[:, :, -5:, :, :] = np.nan
    lstm_result[:, :, -5:, :, :] = np.nan
    lsd_result[:, :, -5:, :, :] = np.nan

    ex_timesteps = example_data["timesteps"]
    print(np.shape(ex_timesteps))
    ex_wl = example_data["wavelengths"]
    mask = example_data["mask"]
    #reference = flow_estimates[0]["reference"]
    print(np.shape(reference))
    WL_IDX = 0
    axes = subfig_images.subplots(1, 6)

    def add_img(ax, img, time, cbar=False, scalebar=False):
        im = ax.imshow(img[time]/100, vmin=0, vmax=20, cmap="magma")
        ax.contour(mask, colors="red")
        ax.axis("off")
        ax.set_title(f"t={np.mean(timesteps[time]):.0f} min", y=-0.15, fontweight="bold")
        ax.text(2, 38, f"sO2 ref.: {np.mean(reference[time])*100:.0f}%", color="white")
        if cbar:
            cb = plt.colorbar(mappable=im, ax=ax)
            cb.set_label(f"PAI @{ex_wl[WL_IDX]:.0f}nm [a.u.]", fontweight="bold")
        if scalebar:
            ax.add_artist(ScaleBar(0.075, "mm"))

    def add_sO2(ax, img, time, cbar=False, scalebar=False, method="LU", title=False):
        im = ax.imshow(img[time], vmin=0, vmax=100, cmap="viridis")
        ax.contour(mask, colors="red")
        ax.axis("off")
        if title:
            ax.set_title(f"t={np.mean(timesteps[time]):.0f} min", y=-0.15, fontweight="bold")
        ax.text(2, 38, f"{method} est.: {np.mean(img[time][mask==0]):.0f}%", color="black")
        if cbar:
            cb = plt.colorbar(mappable=im, ax=ax)
            cb.set_label(f"{method} sO$_2$ [%]", fontweight="bold")
        if scalebar:
            ax.add_artist(ScaleBar(0.075, "mm"))

    add_img(axes[0], ex_signal[:, WL_IDX, :, :, 0], 0, scalebar=True)
    add_img(axes[1], ex_signal[:, WL_IDX, :, :, 0], 80)
    add_img(axes[2], ex_signal[:, WL_IDX, :, :, 0], 160, cbar=True)

    add_sO2(axes[3], ex_sO2[:, 0, :, :, 0], 0, scalebar=True)
    add_sO2(axes[4], ex_sO2[:, 0, :, :, 0], 80)
    add_sO2(axes[5], ex_sO2[:, 0, :, :, 0], 160, cbar=True)

    axes[0].text(0, 8, "A", size=30, weight='bold', color="white")
    axes[3].text(0, 8, "C", size=30, weight='bold')

    lsd_images = fig.add_subfigure(gs[1, 1])
    axes = lsd_images.subplots(1, 3)
    add_sO2(axes[0], lsd_result[:, 0, :, :, 0], 0, scalebar=True, method="LSD")
    add_sO2(axes[1], lsd_result[:, 0, :, :, 0], 80, method="LSD")
    add_sO2(axes[2], lsd_result[:, 0, :, :, 0], 160, cbar=True, method="LSD")
    axes[0].text(0, 8, "D", size=30, weight='bold')

    lstm_images = fig.add_subfigure(gs[2, 1])
    axes = lstm_images.subplots(1, 3)
    add_sO2(axes[0], lstm_result[:, 0, :, :, 0], 0, scalebar=True, method="LSTM", title=True)
    add_sO2(axes[1], lstm_result[:, 0, :, :, 0], 80, method="LSTM", title=True)
    add_sO2(axes[2], lstm_result[:, 0, :, :, 0], 160, cbar=True, method="LSTM", title=True)
    axes[0].text(0, 8, "E", size=30, weight='bold')

    plt.savefig("figure5.pdf", dpi=300)


if __name__ == "__main__":
    create_flow_figure(["BASE"])
