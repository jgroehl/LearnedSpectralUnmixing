import numpy as np
from paths import TRAINING_DATA_PATH

PATH = TRAINING_DATA_PATH + "/ALL/ALL.npz"

spectra = np.load(PATH)["spectra"]
oxy = np.load(PATH)["oxy"]

np.savez(PATH,
         spectra=spectra,
         oxygenation=oxy)