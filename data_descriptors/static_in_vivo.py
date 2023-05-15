import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

PATH = r"H:\learned spectral unmixing\test_final\forearm/"

for folder_path in glob.glob(PATH + "*"):
    if not os.path.isdir(folder_path):
        continue
    folder = folder_path.split("/")[-1].split("\\")[-1]

    print(folder)
    data = np.load(folder_path + "/" + folder + ".npz")
    spectra = data["spectra"]
    print(np.shape(spectra))
    lu = data["lu"]
    mask = data["reference_mask"]
    wavelengths = data["wavelengths"]
    print(wavelengths)

    artery_mask = mask == 1
    if "Mouse" in folder:
        artery_mask = mask == 6

    s800nm = np.squeeze(spectra[wavelengths == 800])
    artery_mask[s800nm < np.percentile(s800nm[artery_mask], 40)] = 0

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.title("PA Image @ 800 nm")
    plt.imshow(np.squeeze(spectra[wavelengths == 800]))
    plt.contour(artery_mask, colors="r")
    plt.subplot(2, 2, 2)
    plt.title("Segmentation Mask")
    plt.imshow(artery_mask)
    plt.subplot(2, 2, 3)
    plt.plot(wavelengths, np.mean(spectra[:, artery_mask], axis=1))
    plt.subplot(2, 2, 4)
    values = lu[artery_mask].copy()
    values[values < 0] = 0
    values[values > 1] = 1
    plt.violinplot(values)
    plt.boxplot(values)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(folder_path + "/data.png")
    plt.close()

    spectra = spectra / np.max(spectra) * 255

    imgs = [Image.fromarray(img) for img in spectra]

    for idx, img in enumerate(imgs):
        I1 = ImageDraw.Draw(img)
        # Add Text to an image
        I1.text((28, 36), f"{wavelengths[idx]} nm")
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(folder_path + "/data.gif", save_all=True, append_images=imgs[1:], duration=500, loop=0)
