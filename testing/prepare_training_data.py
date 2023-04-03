from utils.io import load_spectra_file
import matplotlib.pyplot as plt
import numpy as np
import glob

NUM_SELECTED_SPECTRA = 300000

PATH = r"H:\learned spectral unmixing\training_processed/"

for file in glob.glob(PATH + "*"):
    base_filename = file.split("/")[-1].split("\\")[-1]
    print(base_filename)
    data = load_spectra_file(PATH + base_filename + "/" + base_filename + ".npz")
    all_oxygenations = data[1]
    all_spectra = data[3]

    random_indices = np.random.choice(len(all_oxygenations), NUM_SELECTED_SPECTRA)

    selected_oxygenations = all_oxygenations[random_indices]
    selected_spectra = all_spectra[:, random_indices]

    np.savez(PATH + base_filename + "/" + base_filename + "_train.npz",
             spectra=selected_spectra,
             oxygenation=selected_oxygenations)
