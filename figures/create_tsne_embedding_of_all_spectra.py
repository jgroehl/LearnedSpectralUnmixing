from utils.io import preprocess_data
import numpy as np
from paths import TRAINING_DATA_PATH
import glob

PATHS = glob.glob(TRAINING_DATA_PATH + "/*")

all_data = []
all_oxy = []
all_ds_idx = []

for ds_idx, path in enumerate(PATHS):
    base_filename = path.split("/")[-1].split("\\")[-1]
    spectra, oxy = preprocess_data(f"{path}/{base_filename}_train.npz", 41)
    all_data.append(spectra)
    all_oxy.append(oxy)
    all_ds_idx.append(np.ones_like(oxy) * ds_idx)

all_data = np.hstack(all_data)
all_oxy = np.hstack(all_oxy)
all_ds_idx = np.hstack(all_ds_idx)

print(np.shape(all_data))
print(np.shape(all_oxy))
print(np.shape(all_ds_idx))