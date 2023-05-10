from models.lstm import LSTMParams, get_model
from utils.io import load_data_as_tensorflow_datasets
import tensorflow as tf

TRAIN_SAMPLES = 64
for i in range(4):
    model = get_model()
    model_params = LSTMParams(f"DUMMY_{i}")

    train_spectra = tf.random.normal((TRAIN_SAMPLES, 41, 1))
    train_oxy = tf.random.normal((TRAIN_SAMPLES, 1))
    val_spectra = tf.random.normal((32, 41, 1))
    val_oxy = tf.random.normal((32, 1))

    train_ds = tf.data.Dataset.from_tensor_slices((train_spectra, train_oxy))
    val_ds = tf.data.Dataset.from_tensor_slices((val_spectra, val_oxy))

    batch_size = 32

    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=TRAIN_SAMPLES)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)

    val_ds = val_ds.batch(batch_size, drop_remainder=True)
    val_ds = val_ds.cache()

    model_params.compile(model)
    model_params.fit(train_ds, val_ds, model)