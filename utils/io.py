import numpy as np
import tensorflow as tf
from utils.linear_unmixing import linear_unmixing


def load_data_as_tensorflow_datasets(file_paths, num_wavelengths, batch_size=1024, load_strategy=None):

    spectra, oxy = preprocess_data(file_paths, num_wavelengths, strategy=load_strategy)
    print(np.shape(spectra))
    spectra = np.swapaxes(spectra, 0, 1)
    print(np.shape(spectra))
    spectra = spectra.reshape((len(spectra), len(spectra[0]), 1))

    oxy = oxy.reshape((len(oxy), 1))
    # use 5% of training data for validation (15k randomly chosen spectra/oxy pairs)
    training_samples = np.random.choice(len(oxy), int(0.95 * len(oxy)), replace=False)

    training_spectra = tf.convert_to_tensor(spectra[training_samples, :, :])
    val_spectra = tf.convert_to_tensor(spectra[np.invert(training_samples), :, :])
    training_oxy = tf.convert_to_tensor(oxy[training_samples, :])
    val_oxy = tf.convert_to_tensor(oxy[np.invert(training_samples), :])

    ds_train = tf.data.Dataset.from_tensor_slices((training_spectra, training_oxy))
    ds_validation = tf.data.Dataset.from_tensor_slices((val_spectra, val_oxy))

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size=len(training_oxy))
    ds_train = ds_train.batch(batch_size, drop_remainder=True)

    ds_validation = ds_validation.batch(batch_size, drop_remainder=True)
    ds_validation = ds_validation.cache()

    return ds_train, ds_validation


def preprocess_data(file_path, num_wavelengths, strategy=None) -> tuple:
    data = np.load(file_path)
    spectra = data["spectra"]

    if strategy is None:
        strategy = "random"

    if len(np.shape(spectra)) > 2:
        spectra = spectra.reshape((-1, np.shape(spectra)[0]))
        spectra = np.swapaxes(spectra, 0, 1)

    oxygenations = None

    if "oxygenation" in data:
        oxygenations = data["oxygenation"]

    total_wl = len(spectra)

    def get_equal_spectra_mask():
        spectra_mask = np.zeros_like(spectra)
        num_spectra = spectra.shape[1]
        step_size = int(np.floor(total_wl / num_wavelengths))
        for idx in range(num_wavelengths):
            num_entries = step_size
            if (idx + 1) * step_size > total_wl:
                num_entries = total_wl - idx * step_size
            sub_mask = np.zeros((num_entries, num_spectra))
            sub_mask[0, :] = 1
            [np.random.shuffle(x) for x in sub_mask.T]
            spectra_mask[idx * step_size: idx * step_size + num_entries, :] = sub_mask
        return spectra_mask

    def get_random_spectra_mask():
        spectra_mask = np.zeros_like(spectra)
        spectra_mask[:num_wavelengths, :] = 1
        [np.random.shuffle(x) for x in spectra_mask.T]
        return spectra_mask

    def apply_spectra_mask(_spectra_mask, _spectra):
        nan_spectra = _spectra.copy()
        nan_spectra[_spectra_mask == 0] = np.nan
        _spectra = (_spectra - np.nanmean(nan_spectra, axis=0)[np.newaxis, :]) / np.nanstd(nan_spectra, axis=0)[
                                                                               np.newaxis, :]
        _spectra[_spectra_mask == 0] = 0
        return _spectra

    if strategy == "equal" and num_wavelengths < total_wl/2:
        spectra = apply_spectra_mask(get_equal_spectra_mask(), spectra)
    elif strategy == "mixed":
        equal_mask = get_equal_spectra_mask()
        random_mask = get_random_spectra_mask()

        spectra = np.hstack([spectra, spectra])
        mask = np.hstack([equal_mask, random_mask])
        oxygenations = np.hstack([oxygenations, oxygenations])
        spectra = apply_spectra_mask(mask, spectra)

    else:
        spectra = apply_spectra_mask(get_random_spectra_mask(), spectra)

    return spectra, oxygenations


def load_test_data_as_tensorflow_datasets(file_paths, batch_size=1024):
    spectra = preprocess_test_data(file_paths)
    spectra = np.swapaxes(spectra, 0, 1)
    spectra = spectra.reshape((len(spectra), len(spectra[0]), 1))

    spectra = tf.convert_to_tensor(spectra)
    return spectra


def load_full_wl_test_data_as_tensorflow_datasets_oxy(file_paths):
    spectra, oxy = preprocess_data(file_paths, 41)
    spectra = np.swapaxes(spectra, 0, 1)
    spectra = spectra.reshape((len(spectra), len(spectra[0]), 1))
    oxy = oxy.reshape((len(oxy), 1))

    oxy = tf.convert_to_tensor(oxy)
    spectra = tf.convert_to_tensor(spectra)
    return spectra, oxy


def load_test_data_as_tensorflow_datasets_with_wavelengths(file_path, wavelengths=None):
    data = np.load(file_path)
    spectra = data["spectra"]
    data_wl = data["wavelengths"]
    all_wl = np.arange(700, 901, 5)

    if wavelengths is not None:
        wl_mask = [wl in wavelengths for wl in data_wl]
        all_wl_mask = [wl in wavelengths for wl in all_wl]
        inv_wl_mask = np.invert(wl_mask)
        spectra[inv_wl_mask, :] = np.nan
        spectra = (spectra - np.nanmean(spectra, axis=0)[np.newaxis, :]) / np.nanstd(spectra, axis=0)[np.newaxis, :]
        full_spectra = np.zeros((41, len(spectra[0])))
        full_spectra[all_wl_mask, :] = spectra[wl_mask, :]
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

    if len(np.shape(spectra)) == 3:  # [WL, X, Y]
        spectra = np.swapaxes(spectra, 0, 1)  # [X, WL, Y]
        spectra = np.swapaxes(spectra, 1, 2)  # [X, Y, WL]
        spectra = spectra.reshape((np.shape(spectra)[0]**2, -1))  # [X * Y, WL]
        spectra = np.swapaxes(spectra, 0, 1)   # [WL, X * Y]

    if len(np.shape(spectra)) == 4:  # [POS, WL, X, Y]
        spectra = np.swapaxes(spectra, 1, 2)  # [POS, X, WL, Y]
        spectra = np.swapaxes(spectra, 2, 3)  # [POS, X, Y, WL]
        spectra = spectra.reshape((-1, len(wavelengths)))  # [POS * X * Y, WL]
        spectra = np.swapaxes(spectra, 0, 1)  # [WL, POS * X * Y]

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
