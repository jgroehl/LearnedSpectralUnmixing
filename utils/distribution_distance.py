import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


def validate_input(a, b):
    if a.shape[0] > a.shape[1] or b.shape[0] > b.shape[1]:
        raise AssertionError("0-axis must be the wavelength and 1-axis must be the samples")

    if len(a) != len(b):
        raise AssertionError("The number of wavelengths must be the same for both input samples")


def compute_kld(a, b):
    # Normalise the data
    a = (a - np.mean(a, axis=0)[np.newaxis, :]) / np.std(a, axis=0)[np.newaxis, :]
    b = (b - np.mean(b, axis=0)[np.newaxis, :]) / np.std(b, axis=0)[np.newaxis, :]

    # Compute discrete KLD from marginal histograms
    kld = 0
    for wl_idx in range(len(a)):
        marginal_p, _ = np.histogram(a[wl_idx], bins=np.arange(-3, 3, 6 / 100))
        marginal_q, _ = np.histogram(b[wl_idx], bins=np.arange(-3, 3, 6 / 100))
        marginal_p = marginal_p + 0.00001
        marginal_q = marginal_q + 0.00001
        kld += entropy(marginal_p, marginal_q, base=2)
    kld = kld / len(a)
    return kld


def compute_jsd(a, b):
    # Normalise the data
    a = (a - np.mean(a, axis=0)[np.newaxis, :]) / np.std(a, axis=0)[np.newaxis, :]
    b = (b - np.mean(b, axis=0)[np.newaxis, :]) / np.std(b, axis=0)[np.newaxis, :]

    # Compute discrete KLD from marginal histograms
    jsd = 0
    for wl_idx in range(len(a)):
        marginal_p, _ = np.histogram(a[wl_idx], bins=np.arange(-3, 3, 6 / 100))
        marginal_q, _ = np.histogram(b[wl_idx], bins=np.arange(-3, 3, 6 / 100))
        marginal_p = marginal_p + 0.00001
        marginal_q = marginal_q + 0.00001
        jsd += jensenshannon(marginal_p, marginal_q, base=2)
    jsd = jsd / len(a)
    return jsd
