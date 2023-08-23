from paths import EXAMPLE_PATH
import simpa as sp
import matplotlib.pyplot as plt
import numpy as np

images = [
    "Baseline_10000.hdf5",
    "Skin_10000.hdf5",
    "Water_4cm_10000.hdf5"
]

for image in images:
    p0 = sp.load_data_field(EXAMPLE_PATH + image,
                            sp.Tags.DATA_FIELD_INITIAL_PRESSURE,
                            800)

    plt.imshow(p0[:, int(len(p0)/2), :].T, vmin=0, vmax=3000)
    plt.gca().axis("off")
    plt.savefig(image.replace("hdf5", "png"))
    plt.close()
