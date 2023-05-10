from utils.io import preprocess_data
import numpy as np

PATH = r"H:\learned spectral unmixing\training_processed\ACOUS/ACOUS_train.npz"
spectra, oxy = preprocess_data(PATH, 21)

print(np.shape(spectra))
