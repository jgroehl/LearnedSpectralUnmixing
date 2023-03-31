from utils.io import load_spectra_file
import matplotlib.pyplot as plt

PATH = r"H:\learned spectral unmixing\training_processed\ACOUS\ACOUS.npz"

data = load_spectra_file(PATH)

plt.scatter(data[1]*100, data[2]*100, alpha=0.01)
plt.ylim(0, 100)
plt.show()