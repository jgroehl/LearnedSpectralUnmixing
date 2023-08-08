from utils.io import load_test_data_as_tensorflow_datasets,\
                     load_test_data_as_tensorflow_datasets_with_wavelengths
from models import LSTMParams
import numpy as np
from paths import MODEL_PATH, TEST_DATA_PATH
import glob
import os


def evaluate(_dataset, _data_path, wl):
    for model_path in glob.glob(MODEL_PATH + f"*_{wl}.h5"):
        model_name = model_path.split("/")[-1].split("\\")[-1].split(f"_LSTM_{wl}")[0]
        print("\t", model_path)
        model_params = LSTMParams.load(model_path)
        model_params.compile()
        results = []
        for i in range(len(_dataset)//200000):
            result = model_params(_dataset[i*200000:(i+1)*200000])
            results.append(result.numpy())
        i = len(_dataset)//200000
        results.append(model_params(_dataset[i * 200000:]).numpy())
        result = np.vstack(results)
        np.savez(_data_path.replace(".npz", f"_{model_name}_{wl}.npz"),
                 estimate=result)


def evaluate_baseline_methods(_data_path):
    for model_path in glob.glob(MODEL_PATH + f"/BASE_*.h5"):
        print(model_path)
        wl = model_path.split("/")[-1].split("\\")[-1].split(f"_")[-1].split(".")[0]
        wavelengths = np.linspace(700, 900, int(wl)).astype(int)
        wavelengths = (np.around(wavelengths / 5, decimals=0) * 5).astype(int)
        wavelengths = list(wavelengths)
        print(wavelengths)
        _dataset = load_test_data_as_tensorflow_datasets_with_wavelengths(_data_path + "/baseline.npz", wavelengths)
        model_name = model_path.split("/")[-1].split("\\")[-1].split(f"_LSTM_{wl}")[0]
        print("\t", model_path)
        model_params = LSTMParams.load(model_path)
        model_params.compile()
        results = []
        for i in range(len(_dataset)//200000):
            result = model_params(_dataset[i*200000:(i+1)*200000])
            results.append(result.numpy())
        i = len(_dataset)//200000
        results.append(model_params(_dataset[i * 200000:]).numpy())
        result = np.vstack(results)
        np.savez(f"{_data_path}/baseline_est_{model_name}_{wl}.npz",
                 estimate=result)


def evaluate_distance_from_target_wavelengths(_data_path, perform_correction):
    model_path = MODEL_PATH + f"/BASE_LSTM_20.h5"
    model_name = model_path.split("/")[-1].split("\\")[-1].split(f"_LSTM_20")[0]
    print("\t", model_path)
    model_params = LSTMParams.load(model_path)
    model_params.compile()

    for num_wl in [3, 5, 10, 15, 18, 19, 20, 21, 23, 25, 30, 40, 41]:
        wavelengths = np.linspace(700, 900, num_wl).astype(int)
        wavelengths = (np.around(wavelengths / 5, decimals=0) * 5).astype(int)
        wavelengths = list(wavelengths)
        print(wavelengths)
        _dataset = load_test_data_as_tensorflow_datasets_with_wavelengths(_data_path + "/baseline.npz", wavelengths)
        if perform_correction:
            _dataset = _dataset * (20/num_wl)
        results = []
        for i in range(len(_dataset)//200000):
            result = model_params(_dataset[i*200000:(i+1)*200000])
            results.append(result.numpy())
        i = len(_dataset)//200000
        results.append(model_params(_dataset[i * 200000:]).numpy())
        result = np.vstack(results)
        corr = "corr" if perform_correction else ""
        np.savez(f"{_data_path}/baseline_dist{corr}_{model_name}_{len(wavelengths)}.npz",
                 estimate=result)


# for folder_path in glob.glob(TEST_DATA_PATH + "/mouse/*"):
#     if not os.path.isdir(folder_path):
#         continue
#     print(folder_path)
#     folder = folder_path.split("/")[-1].split("\\")[-1]
#     data_path = folder_path + "/" + folder + ".npz"
#     wavelengths = np.load(data_path)["wavelengths"]
#     dataset = load_test_data_as_tensorflow_datasets(data_path)
#     evaluate(dataset, data_path, len(wavelengths))
#
#
# data_path = TEST_DATA_PATH + "/baseline/baseline.npz"
# wavelengths = list(np.arange(700, 901, 5))
dataset = load_test_data_as_tensorflow_datasets_with_wavelengths(data_path, wavelengths)
# evaluate(dataset, data_path, len(wavelengths))
#
#
# evaluate_baseline_methods(TEST_DATA_PATH + "/baseline/")

evaluate_distance_from_target_wavelengths(TEST_DATA_PATH + "/baseline/", perform_correction=True)

# results = dict()
# for folder_path in glob.glob(TEST_DATA_PATH + "/forearm/*"):
#     if not os.path.isdir(folder_path):
#         continue
#     print(folder_path)
#     folder = folder_path.split("/")[-1].split("\\")[-1]
#     data_path = folder_path + "/" + folder + ".npz"
#     wavelengths = np.load(data_path)["wavelengths"]
#     dataset = load_test_data_as_tensorflow_datasets(data_path)
#     evaluate(dataset, data_path, len(wavelengths))
