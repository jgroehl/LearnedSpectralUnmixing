from models.lstm import LSTMParams, get_model
from utils.io import load_data_as_tensorflow_datasets
import glob
import argparse
from paths import TRAINING_DATA_PATH, MODEL_PATH
import os


parser = argparse.ArgumentParser(
    prog="LSTM_TRAINER",
    description="Use this function to train the LSTM networks from a batch script"
)

parser.add_argument("model_name")
parser.add_argument("wavelength", type=int)


args = parser.parse_args()
model_name = args.model_name
n_wl = args.wavelength

# This line allows for incomplete model name matching
file = glob.glob(f"{TRAINING_DATA_PATH}/{model_name}*")[0]
model = get_model()
base_filename = file.split("/")[-1].split("\\")[-1]

if os.path.exists(f"{MODEL_PATH}/{base_filename}_LSTM_{n_wl}.h5"):
    print("Skipping", base_filename, n_wl)
    exit(0)

print("Running", base_filename)
model_params = LSTMParams(name=base_filename, wl=n_wl)
train_ds, val_ds = load_data_as_tensorflow_datasets(TRAINING_DATA_PATH + "/" + base_filename + "/" +
                                                    base_filename + "_train.npz",
                                                    n_wl, batch_size=10000,
                                                    load_strategy="equal")
model_params.compile(model)
model_params.fit(train_ds, val_ds, model)
