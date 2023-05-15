from utils.io import load_test_data_as_tensorflow_datasets,\
                     load_test_data_as_tensorflow_datasets_with_wavelengths
from models import LSTMParams
import numpy as np
import glob
import os

MODEL_PATH = r"H:\learned spectral unmixing\models_LSTM/"
TEST_DATA_PATH = r"H:\learned spectral unmixing\test_final/"


def evaluate(_dataset, _data_path, wl):
    for model_path in glob.glob(MODEL_PATH + f"*_{wl}.h5"):
        model_name = model_path.split("/")[-1].split("\\")[-1].split(f"_LSTM_{wl}")[0]
        print("\t", model_path)
        model_params = LSTMParams.load(model_path)
        model_params.compile()
        results = []
        for i in range(len(dataset)//200000):
            result = model_params(_dataset[i*200000:(i+1)*200000])
            results.append(result.numpy())
        i = len(dataset)//200000
        results.append(model_params(_dataset[i * 200000:]).numpy())
        result = np.vstack(results)
        np.savez(_data_path.replace(".npz", f"_{model_name}.npz"),
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


# data_path = TEST_DATA_PATH + "/baseline/baseline.npz"
# wavelengths = list(np.arange(700, 901, 5))
# dataset = load_test_data_as_tensorflow_datasets_with_wavelengths(data_path, wavelengths)
# evaluate(dataset, data_path, len(wavelengths))

results = dict()
for folder_path in glob.glob(TEST_DATA_PATH + "/forearm/*"):
    if not os.path.isdir(folder_path):
        continue
    print(folder_path)
    folder = folder_path.split("/")[-1].split("\\")[-1]
    data_path = folder_path + "/" + folder + ".npz"
    wavelengths = np.load(data_path)["wavelengths"]
    dataset = load_test_data_as_tensorflow_datasets(data_path)
    evaluate(dataset, data_path, len(wavelengths))
