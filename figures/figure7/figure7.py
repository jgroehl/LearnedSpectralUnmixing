import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import ALL_MODELS
from paths import TRAINING_DATA_PATH, TEST_DATA_PATH
from utils.distribution_distance import compute_jsd
import matplotlib.gridspec as gridspec

RECOMPUTE = False
MASK_LABELS = {
    1: "BACKGROUND",
    2: "BODY",
    3: "SPLEEN",
    4: "KIDNEY",
    5: "SPINE",
    6: "ARTERY"
}
MASK_INDEXES = [2, 3, 4, 5, 6]
PATH = r"H:\learned spectral unmixing\test_final_mp\CO2/"


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
        spectra = data["spectra"]   # IS [POS, WL, X, Y]
        test_wl = data["wavelengths"]
        wl_mask = [x in test_wl for x in all_wl]
        spectra = np.swapaxes(spectra, 0, 1)  # SWAP TO [WL, POS, X, Y]
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


def compile_results(data_path):
    if not RECOMPUTE and os.path.exists(data_path + "/all_results.npz"):
        return data_path + "/all_results.npz"
    mouse_data_files = []
    for folder_path in glob.glob(data_path + "/*"):
        if os.path.isdir(folder_path):
            mouse_data_files.append(folder_path)

    results = dict()

    for mask_index in MASK_INDEXES:
        print(MASK_LABELS[mask_index])
        results[MASK_LABELS[mask_index]] = dict()
        results[MASK_LABELS[mask_index]]["LU"] = []
        for model in ALL_MODELS:
            results[MASK_LABELS[mask_index]][model] = []
        for folder_path in mouse_data_files:
            filename = folder_path.split("/")[-1].split("\\")[-1]
            data = np.load(folder_path + "/" + filename + ".npz")
            wavelengths = data["wavelengths"]
            lu = data["lu"]
            spectra = data["spectra"]  # [POS, WL, X, Y]
            image = spectra[:, np.argwhere(wavelengths == 800), :, :]  # [POS, X, Y]
            image = np.squeeze(image)
            mask = data["reference_mask"] == mask_index
            results[MASK_LABELS[mask_index]]["LU"].append(lu[mask])

            for model in ALL_MODELS:
                model_result = np.load(f"{data_path}/{filename}/{filename}_{model}_{len(wavelengths)}.npz")["estimate"].reshape(*np.shape(image))
                results[MASK_LABELS[mask_index]][model].append(model_result[mask])

        results[MASK_LABELS[mask_index]]["LU"] = np.asarray(results[MASK_LABELS[mask_index]]["LU"], dtype=object)
        for model in ALL_MODELS:
            results[MASK_LABELS[mask_index]][model] = np.asarray(results[MASK_LABELS[mask_index]][model], dtype=object)

    np.savez(data_path + "/all_results.npz", **results, allow_pickle=True)
    return data_path + "/all_results.npz"


dist_path = compile_distance_measures(PATH)
res_path = compile_results(PATH)


results = np.load(res_path, allow_pickle=True)
results = {key: results[key] for key in results}
distances = np.load(dist_path, allow_pickle=True)
distances = {key: distances[key] for key in distances}

