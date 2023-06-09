from utils.io import load_full_wl_test_data_as_tensorflow_datasets_oxy
from models import LSTMParams
import numpy as np
from paths import MODEL_PATH, TEST_DATA_PATH, TRAINING_DATA_PATH
from utils.constants import ALL_MODELS
import glob
import json


def evaluate(_dataset, _model):
        results = []
        for i in range(len(_dataset)//200000):
            result = _model(_dataset[i * 200000:(i + 1) * 200000])
            results.append(result.numpy())
        i = len(_dataset)//200000
        results.append(_model(_dataset[i * 200000:]).numpy())
        result = np.vstack(results)
        return result


results = dict()

for model_path in glob.glob(MODEL_PATH + f"*_41.h5"):
    model_name = model_path.split("/")[-1].split("\\")[-1].split(f"_LSTM_41")[0]
    print("Loading model", model_name)
    results[model_name] = dict()
    model_params = LSTMParams.load(model_path)
    model_params.compile()

    for model in ALL_MODELS:
        print("\t", "Evaluating on", model)
        train_data_path = f"{TRAINING_DATA_PATH}/{model}/{model}_train.npz"
        data, oxy = load_full_wl_test_data_as_tensorflow_datasets_oxy(train_data_path)
        res = np.squeeze(evaluate(data, model_params))
        oxy = np.squeeze(oxy)
        results[model_name][model] = np.mean(np.abs(res-oxy))
        print("\t\t", "MAE:", f"{results[model_name][model] * 100:.2f}")

with open(f"{TEST_DATA_PATH}/result_matrix.json", "w+") as json_file:
    json.dump(results, json_file)

