import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import ALL_MODELS
from paths import TRAINING_DATA_PATH, TEST_DATA_PATH
from utils.distribution_distance import compute_jsd
import matplotlib.gridspec as gridspec
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import distance_transform_edt

RECOMPUTE = False
MASK_LABELS = {
    1: "BACKGROUND",
    2: "BODY",
    3: "SPLEEN",
    4: "KIDNEY",
    5: "SPINE",
    6: "ARTERY"
}
COLOURS = ["black", "lightgray", "pink", "purple", "orange", "red"]
MASK_INDEXES = [2, 3, 4, 5, 6]
PATH = r"H:\learned spectral unmixing\test_final_mp\CO2/"


def compile_distance_measures(data_path):
    output_file = data_path + "/all_distances.npz"
    if os.path.exists(output_file):
        return output_file
    data_files = []
    for folder_path in glob.glob(data_path + "/*"):
        if os.path.isdir(folder_path):
            data_files.append(folder_path)

    all_wl = np.arange(700, 901, 5)
    results = {}

    for folder_path in data_files:
        filename = folder_path.split("/")[-1].split("\\")[-1]
        print(filename)
        data = np.load(folder_path + "/" + filename + ".npz")
        spectra = data["spectra"]   # IS [POS, WL, X, Y]
        test_wl = data["wavelengths"]
        wl_mask = [x in test_wl for x in all_wl]
        spectra = np.swapaxes(spectra, 0, 1)  # SWAP TO [WL, POS, X, Y]
        test_spectra = spectra.reshape((len(test_wl), -1))

        for train_path in glob.glob(TRAINING_DATA_PATH + "/*"):
            model_name = train_path.split("/")[-1].split("\\")[-1]
            data = np.load(train_path + f"/{model_name}_train.npz")
            train_spectra = data["spectra"][wl_mask, :]

            if model_name not in results:
                results[model_name] = []
            results[model_name].append(compute_jsd(train_spectra, test_spectra))

    np.savez(output_file, **results, allow_pickle=True)
    return output_file


def compile_results(data_path):
    if not RECOMPUTE and os.path.exists(data_path + "/all_results.npz"):
        return data_path + "/all_results.npz"
    mouse_data_files = []
    for folder_path in glob.glob(data_path + "/*"):
        if os.path.isdir(folder_path):
            mouse_data_files.append(folder_path)

    results = dict()

    for mask_index in MASK_INDEXES:
        print(MASK_LABELS[mask_index])
        results[MASK_LABELS[mask_index]] = dict()
        results[MASK_LABELS[mask_index]]["LU"] = [[], [], [], [], [], []]
        for model in ALL_MODELS:
            results[MASK_LABELS[mask_index]][model] = [[], [], [], [], [], []]
        for folder_path in mouse_data_files:
            filename = folder_path.split("/")[-1].split("\\")[-1]
            data = np.load(folder_path + "/" + filename + ".npz")
            wavelengths = data["wavelengths"]
            lu = data["lu"]
            spectra = data["spectra"]  # [POS, WL, X, Y]
            image = spectra[:, np.argwhere(wavelengths == 800), :, :]  # [POS, X, Y]
            image = np.squeeze(image)
            mask = data["reference_mask"] == mask_index
            dist_mask = distance_transform_edt(data["reference_mask"] > 1)

            # Only consider everything up to 3mm depth
            mask = (mask & (dist_mask < 40))

            if mask_index == 6:
                # Except for the arota as it lies deeper
                mask[data["reference_mask"] == 6] = 1

            for i in range(6):
                results[MASK_LABELS[mask_index]]["LU"][i].append(lu[i][mask[i]])

                for model in ALL_MODELS:
                    model_result = np.load(f"{data_path}/{filename}/{filename}_{model}_{len(wavelengths)}.npz")["estimate"].reshape(*np.shape(image))
                    results[MASK_LABELS[mask_index]][model][i].append(np.reshape(model_result[i][mask[i]], (-1, 1)))

        results[MASK_LABELS[mask_index]]["LU"] = np.asarray(results[MASK_LABELS[mask_index]]["LU"], dtype=object)
        for model in ALL_MODELS:
            results[MASK_LABELS[mask_index]][model] = np.asarray(results[MASK_LABELS[mask_index]][model], dtype=object)

    np.savez(data_path + "/all_results.npz", **results, allow_pickle=True)
    return data_path + "/all_results.npz"


dist_path = compile_distance_measures(PATH)
res_path = compile_results(PATH)


results = np.load(res_path, allow_pickle=True)
results = {key: results[key] for key in results}
distances = np.load(dist_path, allow_pickle=True)
distances = {key: distances[key] for key in distances}


def content(site, algo):
    res = []
    for i in range(6):
        res.append(np.vstack(results[site].item()[algo][i]))
    bl = np.vstack(res[0:3]) * 100
    post = np.vstack(res[3:6]) * 100

    bl_val = np.nanmean(bl)
    bl_std = np.nanstd(bl)
    post_val = np.nanmean(post)
    diff = post_val-bl_val

    return rf"{bl_val:.0f}$\pm${bl_std:.0f} & {'' if diff < 0 else '+'}{diff:.0f}"


print(r"\begin{table}[]\centering\begin{tabular}{r|cc|cc|cc}")
print(r"& \multicolumn{2}{c}{LU} & \multicolumn{2}{c}{BASE} & \multicolumn{2}{c}{WATER\_4cm} \\")
print(r" Organ & BL & $\Delta sO_2$ & BL & $\Delta sO_2$ & BL & $\Delta sO_2$ \\")
print(r"\hline")
print(rf" Body & {content('BODY', 'LU')} & {content('BODY', 'BASE')} & {content('BODY', 'WATER_4cm')} \\")
print(rf" Spleen & {content('SPLEEN', 'LU')} & {content('SPLEEN', 'BASE')} & {content('SPLEEN', 'WATER_4cm')} \\")
print(rf" Kidney & {content('KIDNEY', 'LU')} & {content('KIDNEY', 'BASE')} & {content('KIDNEY', 'WATER_4cm')} \\")
print(rf" Spine & {content('SPINE', 'LU')} & {content('SPINE', 'BASE')} & {content('SPINE', 'WATER_4cm')} \\")
print(rf" Aorta & {content('ARTERY', 'LU')} & {content('ARTERY', 'BASE')} & {content('ARTERY', 'WATER_4cm')} \\")
print(r"\end{tabular}\caption{Caption}\label{tab:co2_results}\end{table}")

EX_DATA = "CO2_3"
example_dataset_path = PATH + f"/{EX_DATA}/"

data = np.load(example_dataset_path + EX_DATA + ".npz")
ex_lu = data["lu"]
ex_spectra = data["spectra"]
ex_mask = data["reference_mask"]


def load_dl_result(path):
    return np.load(path)["estimate"].reshape(*np.shape(ex_lu))


ex_BASE = load_dl_result(example_dataset_path + EX_DATA + "_BASE_10.npz")
ex_WATER_4cm = load_dl_result(example_dataset_path + EX_DATA + "_WATER_4cm_10.npz")
ex_wl = data["wavelengths"]
WL_IDX = 5

fig = plt.figure(layout="constrained", figsize=(12, 6))
gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig,
                       height_ratios=[1.04, 1])
subfig_1 = fig.add_subfigure(gs[0, 0])
subfig_2 = fig.add_subfigure(gs[1, 0])

(ax1, ax2, ax3, ax4) = subfig_1.subplots(1, 4)
(ax5, ax6, ax7, ax8) = subfig_2.subplots(1, 4)


def show_img(ax, img, slice_idx, title=False):
    if title:
        ax.set_title(f"PAI @{ex_wl[WL_IDX]:.0f}nm [a.u.]")
    im = ax.imshow(img[slice_idx]/100, cmap="magma", vmin=0, vmax=6)
    plt.colorbar(mappable=im, ax=ax, location="left")
    for idx in [2, 3, 4, 5, 6]:
        ax.plot([], [], color=COLOURS[idx-1], label=MASK_LABELS[idx])
        ax.contour(ex_mask[slice_idx]==idx, colors=COLOURS[idx-1])
    ax.axis("off")
    ax.legend(ncol=1, loc="lower left", labelspacing=0, fontsize=9,
              borderpad=0.1, handlelength=1, handletextpad=0.4,
              labelcolor="white", framealpha=0)
    ax.add_artist(ScaleBar(0.075, "mm", location="lower right"))


def show_oxy(ax, img, method, slice_idx, cbar=False, title=False):
    if title:
        ax.set_title(f"{method} [%]")
    ax.imshow(ex_spectra[slice_idx, WL_IDX] / 100, cmap="magma", vmin=0, vmax=6)
    # dist_mask = distance_transform_edt(ex_mask[slice_idx] > 1)
    img[slice_idx][ex_spectra[slice_idx, WL_IDX] < 0] = np.nan
    # print(np.nanmean(img[slice_idx][ex_mask[slice_idx]==3]))
    im = ax.imshow(img[slice_idx] * 100, cmap="viridis", vmin=0, vmax=100)
    if cbar:
        plt.colorbar(mappable=im, ax=ax, location="left")
    for idx in [2, 3, 4, 5, 6]:
        ax.contour(ex_mask[slice_idx] == idx, colors=COLOURS[idx - 1])
    ax.axis("off")
    ax.add_artist(ScaleBar(0.075, "mm", location="lower right"))


ex_lu[ex_mask<2] = np.nan
ex_BASE[ex_mask<2] = np.nan
ex_WATER_4cm[ex_mask<2] = np.nan

#
# def apply_threshold_for_mask(mask, threshold):
#     ex_lu[3:6][(ex_spectra[3:6, 5] < threshold) & (ex_mask[3:6]==mask)] = np.nan
#     ex_BASE[3:6][(ex_spectra[3:6, 5] < threshold) & (ex_mask[3:6]==mask)] = np.nan
#     ex_WATER_4cm[3:6][(ex_spectra[3:6, 5] < threshold) & (ex_mask[3:6]==mask)] = np.nan
#
#
# apply_threshold_for_mask(6, 100)
# apply_threshold_for_mask(5, 200)
# apply_threshold_for_mask(4, 300)
# apply_threshold_for_mask(3, 200)
# apply_threshold_for_mask(2, 200)

show_img(ax1, ex_spectra[:, WL_IDX, :, :], 0, title=True)
show_oxy(ax2, ex_lu, "Linear Unmixing", 0, cbar=True, title=True)
show_oxy(ax3, ex_BASE, "BASE", 0, title=True)
show_oxy(ax4, ex_WATER_4cm, "WATER_4cm", 0, title=True)

show_img(ax5, ex_spectra[:, WL_IDX, :, :], 5)
show_oxy(ax6, ex_lu, "Linear Unmixing", 5, cbar=True)
show_oxy(ax7, ex_BASE, "BASE", 5)
show_oxy(ax8, ex_WATER_4cm, "WATER_4cm", 5)

ax1.text(5, 45, "A", size=30, weight='bold', color="white")
ax2.text(5, 45, "B", size=30, weight='bold', color="white")
ax3.text(5, 45, "C", size=30, weight='bold', color="white")
ax4.text(5, 45, "D", size=30, weight='bold', color="white")

ax5.text(5, 45, "E", size=30, weight='bold', color="white")
ax6.text(5, 45, "F", size=30, weight='bold', color="white")
ax7.text(5, 45, "G", size=30, weight='bold', color="white")
ax8.text(5, 45, "H", size=30, weight='bold', color="white")

plt.savefig("figure7.png", dpi=300)