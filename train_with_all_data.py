from models.lstm import LSTMParams, get_model
from utils.io import load_data_as_tensorflow_datasets
import glob
from paths import TRAINING_DATA_PATH

files = glob.glob(f"{TRAINING_DATA_PATH}/*")
num_files = len(files)
WAVELENGTHS = 41

train_data = None
test_data = None

for f_idx, file in enumerate(files):
    base_filename = file.split("/")[-1].split("\\")[-1]
    print(f"Loading ({f_idx+1}/{num_files})", base_filename, "...")
    spectra_train, spectra_test = load_data_as_tensorflow_datasets(file + f"/{base_filename}_train.npz", WAVELENGTHS)
    if train_data is None:
        train_data = spectra_train
        test_data = spectra_test
    else:
        train_data.concatenate(spectra_train)
        test_data.concatenate(spectra_test)
    print(f"Loading ({f_idx+1}/{num_files})", base_filename, "...[Done]")

model = get_model()
model_params = LSTMParams(name="ALL", wl=WAVELENGTHS)
model_params.compile(model)
model_params.fit(train_data, test_data, model)
