from models.lstm import LSTMParams, get_model
from utils.io import load_data_as_tensorflow_datasets
import glob
import os

PATH = r"H:\learned spectral unmixing\training_processed/"
NUM_WAVELENGTHS = [5, 6, 10, 20, 30, 40, 3]

for n_wl in NUM_WAVELENGTHS:
    for file in glob.glob(PATH + "*"):
        model = get_model()
        base_filename = file.split("/")[-1].split("\\")[-1]
        if os.path.exists(f"H:/learned spectral unmixing/models_LSTM/{base_filename}_LSTM_{n_wl}.h5"):
            print("Skipping", base_filename)
            continue
        print("Running", base_filename)

        model_params = LSTMParams(name=base_filename, wl=n_wl)
        train_ds, val_ds = load_data_as_tensorflow_datasets(PATH + "/" + base_filename + "/" + base_filename + "_train.npz",
                                                            n_wl)
        model_params.compile(model)
        model_params.fit(train_ds, val_ds, model)

