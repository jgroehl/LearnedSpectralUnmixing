import numpy as np


def compute_ensemble_average(estimates, distances, smaller_is_better=False):
    distances = distances.copy()
    distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    if smaller_is_better:
        distances = 1 - distances
    distances = distances ** 2
    print(distances)
    print(estimates)
    estimates = np.mean(estimates, axis=0) #/ np.sum(distances)
    print(estimates)
    return estimates
