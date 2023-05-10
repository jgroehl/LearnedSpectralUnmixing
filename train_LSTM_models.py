from models.lstm import LSTMParams, get_model
from utils.io import load_data_as_tensorflow_datasets
import glob

PATH = r"H:\learned spectral unmixing\training_processed/"
NUM_WAVELENGTHS = [5, 6]

for n_wl in NUM_WAVELENGTHS:
    for file in glob.glob(PATH + "*")[3:]:
        model = get_model()
        base_filename = file.split("/")[-1].split("\\")[-1]
        print(base_filename)

        model_params = LSTMParams(name=base_filename, wl=n_wl)
        train_ds, val_ds = load_data_as_tensorflow_datasets(PATH + "/" + base_filename + "/" + base_filename + "_train.npz",
                                                            n_wl)
        model_params.compile(model)
        model_params.fit(train_ds, val_ds, model)

