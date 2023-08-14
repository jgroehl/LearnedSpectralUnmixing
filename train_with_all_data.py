from models.lstm import LSTMParams, get_model
import tensorflow as tf
from utils.io import preprocess_data
import glob
import numpy as np
from paths import TRAINING_DATA_PATH

files = glob.glob(TRAINING_DATA_PATH + "/*")
WAVELENGTHS = 15


spectra, oxy = preprocess_data(f"{TRAINING_DATA_PATH}/ALL/ALL_train.npz", WAVELENGTHS)

# use 5% of training data for validation (15k randomly chosen spectra/oxy pairs)
training_samples = np.random.choice(len(oxy), int(len(oxy)), replace=False)
test_samples = training_samples[int(0.95 * len(oxy)):]
training_samples = training_samples[:int(0.95 * len(oxy))]

all_train_data = spectra[:, training_samples]
all_test_data = spectra[:, test_samples]
all_train_oxy = oxy[training_samples]
all_test_oxy = oxy[test_samples]

all_train_data = np.swapaxes(all_train_data, 0, 1)
all_test_data = np.swapaxes(all_test_data, 0, 1)

all_train_data = all_train_data.reshape((len(all_train_data), len(all_train_data[0]), 1))
all_test_data = all_test_data.reshape((len(all_test_data), len(all_test_data[0]), 1))
all_train_oxy = all_train_oxy.reshape((len(all_train_oxy), 1))
all_test_oxy = all_test_oxy.reshape((len(all_test_oxy), 1))

ds_train = tf.data.Dataset.from_tensor_slices((all_train_data, all_train_oxy))
ds_validation = tf.data.Dataset.from_tensor_slices((all_test_data, all_test_oxy))

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(buffer_size=len(all_train_oxy))
ds_train = ds_train.batch(1024, drop_remainder=True)

ds_validation = ds_validation.batch(1024, drop_remainder=True)
ds_validation = ds_validation.cache()

num_files = len(files)

print(ds_train.cardinality().numpy())

model = get_model()
model_params = LSTMParams(name="ALL", wl=WAVELENGTHS)
model_params.compile(model)
model_params.fit(ds_train, ds_validation, model)
