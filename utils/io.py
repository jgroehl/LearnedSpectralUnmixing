import numpy as np
import tensorflow as tf
from utils.linear_unmixing import linear_unmixing


def load_data_as_tensorflow_datasets(file_paths, num_wavelengths, batch_size=1024):

    spectra, oxy = preprocess_data(file_paths, num_wavelengths)
    print(np.shape(spectra))
    spectra = np.swapaxes(spectra, 0, 1)
    print(np.shape(spectra))
    spectra = spectra.reshape((len(spectra), len(spectra[0]), 1))

    oxy = oxy.reshape((len(oxy), 1))
    threshold = int(0.95 * len(oxy))  # use 5% of training data for validation (15k randomly chosen spectra/oxy pairs)

    training_spectra = tf.convert_to_tensor(spectra[:threshold, :, :])
    val_spectra = tf.convert_to_tensor(spectra[threshold:, :, :])
    training_oxy = tf.convert_to_tensor(oxy[:threshold, :])
    val_oxy = tf.convert_to_tensor(oxy[threshold:, :])

    ds_train = tf.data.Dataset.from_tensor_slices((training_spectra, training_oxy))
    ds_validation = tf.data.Dataset.from_tensor_slices((val_spectra, val_oxy))

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size=len(training_oxy))
    ds_train = ds_train.batch(batch_size, drop_remainder=True)

    ds_validation = ds_validation.batch(batch_size, drop_remainder=True)
    ds_validation = ds_validation.cache()

    return ds_train, ds_validation


def preprocess_data(file_path, num_wavelengths) -> tuple:
    data = np.load(file_path)
    spectra = data["spectra"]

    if len(np.shape(spectra)) > 2:
        spectra = spectra.reshape((-1, np.shape(spectra)[0]))
        spectra = np.swapaxes(spectra, 0, 1)

    oxygenations = None

    if "oxygenation" in data:
        oxygenations = data["oxygenation"]

    spectra_mask = np.zeros_like(spectra)
    # completely randomised spectral data for training
    spectra_mask[:num_wavelengths, :] = 1
    rng = np.random.default_rng()
    rng.shuffle(spectra_mask, axis=0)
    nan_spectra = spectra.copy()
    nan_spectra[spectra_mask == 0] = np.nan
    spectra = (spectra - np.nanmean(nan_spectra, axis=0)[np.newaxis, :]) / np.nanstd(nan_spectra, axis=0)[np.newaxis, :]
    spectra[spectra_mask == 0] = 0
    return spectra, oxygenations


def load_test_data_as_tensorflow_datasets(file_paths, batch_size=1024):
    spectra = preprocess_test_data(file_paths)
    spectra = np.swapaxes(spectra, 0, 1)
    spectra = spectra.reshape((len(spectra), len(spectra[0]), 1))

    spectra = tf.convert_to_tensor(spectra)
    return spectra


def load_test_data_as_tensorflow_datasets_with_wavelengths(file_path, wavelengths=None):
    data = np.load(file_path)
    spectra = data["spectra"]
    data_wl = data["wavelengths"]

    if wavelengths:
        wl_mask = [wl in wavelengths for wl in data_wl]
        inv_wl_mask = np.invert(wl_mask)
        spectra[inv_wl_mask, :] = np.nan
        spectra = (spectra - np.nanmean(spectra, axis=0)[np.newaxis, :]) / np.nanstd(spectra, axis=0)[np.newaxis, :]
        full_spectra = np.zeros((41, len(spectra[0])))
        full_spectra[wl_mask, :] = spectra[wl_mask, :]
        spectra = full_spectra

    spectra = np.swapaxes(spectra, 0, 1)
    spectra = spectra.reshape((len(spectra), len(spectra[0]), 1))

    spectra = tf.convert_to_tensor(spectra)
    return spectra


def preprocess_test_data(file_path):
    data = np.load(file_path)
    spectra = data["spectra"]
    wavelengths = data["wavelengths"]
    wavelengths = np.around(wavelengths / 5, decimals=0) * 5  # Round wavelengths to the nearest 5
    input_wavelengths = np.arange(700, 901, 5)
    wl_mask = np.isin(input_wavelengths, wavelengths)

    if len(np.shape(spectra)) > 2:
        spectra = np.swapaxes(spectra, 0, 1)
        spectra = np.swapaxes(spectra, 1, 2)
        spectra = spectra.reshape((np.shape(spectra)[0]**2, -1))
        spectra = np.swapaxes(spectra, 0, 1)

    spectra = (spectra - np.nanmean(spectra, axis=0)[np.newaxis, :]) / np.nanstd(spectra, axis=0)[np.newaxis, :]
    print(np.shape(spectra))
    full_spectra = np.zeros((41, len(spectra[0])))
    print(np.shape(full_spectra))
    full_spectra[wl_mask, :] = spectra
    return full_spectra


def load_spectra_file(file_path: str, load_all_data: bool = False) -> tuple:
    data = np.load(file_path, allow_pickle=True)
    wavelengths = data["wavelengths"]
    oxygenations = data["oxygenations"]
    spectra = data["spectra"]
    distances = data["distances"]
    depths = data["depths"]
    melanin_concentration = data["melanin_concentration"]
    background_oxygenation = data["background_oxygenation"]
    if "timesteps" in data:
        timesteps = data["timesteps"]
    else:
        timesteps = None

    if not load_all_data:
        # Signal intensity @ 800nm
        selector = spectra[21, :] / np.max(spectra[21, :]) > 0.10

        # Enforce that at least 10% of the data is used for training
        if np.sum(selector) / len(selector) < 0.1:
            print("Less than 10% of training data would be used. Using top 10% signals at 800nm.")

        spectra = spectra[:, selector]
        oxygenations = oxygenations[selector]

        if str(distances) != "None":
            distances = distances[selector]
        if str(depths) != "None":
            depths = depths[selector]
        if str(melanin_concentration) != "None":
            melanin_concentration = melanin_concentration[selector]
        if str(background_oxygenation) != "None":
            background_oxygenation = background_oxygenation[selector]
        if "timesteps" in data:
            timesteps = data["timesteps"]

    if "tumour_mask" in data:
        tumour_mask = data["tumour_mask"]
    else:
        tumour_mask = None
    if "reference_mask" in data:
        reference_mask = data["reference_mask"]
    else:
        reference_mask = None
    if "mouse_body_mask" in data:
        mouse_body_mask = data["mouse_body_mask"]
    else:
        mouse_body_mask = None
    if "background_mask" in data:
        background_mask = data["background_mask"]
    else:
        background_mask = None
    if "lu" in data:
        lu = data["lu"]
    else:
        _reshaped_spectra = spectra.reshape([1, len(spectra), 1, 1, -1])
        sO2 = linear_unmixing(_reshaped_spectra, wavelengths)
        lu = sO2.raw_data.reshape((-1, ))

    return (wavelengths, oxygenations, lu, spectra, melanin_concentration,
            background_oxygenation, distances, depths, timesteps,
            tumour_mask, reference_mask,
            mouse_body_mask, background_mask)
