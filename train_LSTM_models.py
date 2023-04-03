from models import OximetryLSTM
from utils.io import load_data_as_tensorflow_datasets
import glob

PATH = r"H:\learned spectral unmixing\training_processed/"

for file in glob.glob(PATH + "*"):
    base_filename = file.split("/")[-1].split("\\")[-1]
    print(base_filename)

    model = OximetryLSTM(name=base_filename + "_LSTM")
    train_ds, val_ds = load_data_as_tensorflow_datasets(PATH + "/" + base_filename + "/" + base_filename + "_train.npz", 10)
    model.compile()
    model.fit(train_ds, val_ds)
