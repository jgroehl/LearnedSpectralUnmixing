from models import OximetryLSTM
import glob

TRAIN_DATA_PATH = r"H:\learned spectral unmixing\training_processed/"
TEST_DATA_PATH = r"H:\learned spectral unmixing\test_data_processed/"

for data in glob.glob(TEST_DATA_PATH + "*"):
    print(data)
    # Load test data
    for train_data_path in glob.glob(TRAIN_DATA_PATH + "*"):
        base_filename = train_data_path.split("/")[-1].split("\\")[-1]
        print("\t", base_filename)
        model = OximetryLSTM.load(base_filename + "_LSTM")
