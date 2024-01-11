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
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu

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

            for time_point in range(6):
                results[MASK_LABELS[mask_index]]["LU"][time_point].append(lu[time_point][mask[time_point]])

                for model in ALL_MODELS:
                    model_result = np.load(f"{data_path}/{filename}/{filename}_{model}_{len(wavelengths)}.npz")["estimate"].reshape(*np.shape(image))
                    results[MASK_LABELS[mask_index]][model][time_point].append(np.reshape(model_result[time_point][mask[time_point]], (-1, 1)))

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


def content(organ, algo):
    res = []
    stats_results = []
    for time_point in range(6):
        r = results[organ].item()[algo][time_point]
        stats_results.append([np.nanmean(r[subject]) for subject in range(6)])
        res.append(np.vstack(r))

    bl = np.vstack(res[0:3]) * 100
    post = np.vstack(res[3:6]) * 100

    bl_stats = np.asarray(np.mean(stats_results[0:3], axis=0)) * 100
    post_stats = np.asarray(np.mean(stats_results[3:6], axis=0)) * 100

    bl_val = np.nanmean(bl)
    bl_std = np.nanstd(bl)
    post_val = np.nanmean(post)
    diff = post_val-bl_val

    stats, p_val = mannwhitneyu(bl_stats.reshape((-1, )), post_stats.reshape((-1, )))

    p_val_marker = "n.s."

    if p_val < 0.05:
        p_val_marker = '*'
    if p_val < 0.01:
        p_val_marker = '**'
    if p_val < 0.001:
        p_val_marker = '***'
    if p_val < 0.0001:
        p_val_marker = '****'

    return rf"{bl_val:.0f}$\pm${bl_std:.0f} & {'' if diff < 0 else '+'}{diff:.0f} ({p_val_marker})"


print(r"\begin{table}[]\centering\begin{tabular}{r|cc|cc|cc}")
print(r"& \multicolumn{2}{c}{LU} & \multicolumn{2}{c}{BASE} & \multicolumn{2}{c}{WATER\_4cm} \\")
print(r" Organ & Before & $\Delta sO_2$ & Before & $\Delta sO_2$ & Before & $\Delta sO_2$ \\")
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
ex_mask = np.squeeze(data["reference_mask"])


def load_dl_result(path):
    return np.load(path)["estimate"].reshape(*np.shape(ex_lu))


ex_BASE = load_dl_result(example_dataset_path + EX_DATA + "_BASE_10.npz")
ex_WATER_4cm = load_dl_result(example_dataset_path + EX_DATA + "_WATER_4cm_10.npz")
ex_wl = data["wavelengths"]
WL_IDX = 5

fig = plt.figure(layout="constrained", figsize=(12, 6))
gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig,
                       height_ratios=[1, 1])
subfig_1 = fig.add_subfigure(gs[0, 0])
subfig_2 = fig.add_subfigure(gs[1, 0])

(ax1, ax2, ax3, ax4) = subfig_1.subplots(1, 4)
(ax5, ax6, ax7, ax8) = subfig_2.subplots(1, 4)


def show_img(ax, img, slice_idx, title=False):
    ax.text(50, 20, f"PAI @{ex_wl[WL_IDX]:.0f}nm", color="white", fontweight="bold")
    im = ax.imshow(img[slice_idx]/100, cmap="magma", vmin=0, vmax=6)
    #plt.colorbar(mappable=im, ax=ax, location="left")
    for idx in [2, 3, 4, 5, 6]:
        ax.plot([], [], color=COLOURS[idx-1], label=MASK_LABELS[idx])
        ax.contour(ex_mask[slice_idx]==idx, colors=COLOURS[idx-1])
    ax.axis("off")
    ax.legend(ncol=1, loc="lower right", labelspacing=0, fontsize=9,
              borderpad=0.1, handlelength=1, handletextpad=0.4,
              labelcolor="white", framealpha=0)
    ax.add_artist(ScaleBar(0.075, "mm", location="lower left"))


def show_oxy(ax, img, method, slice_idx, title=False):
    ax.text(50, 20, f"{method}", color="white", fontweight="bold")
    ax.imshow(ex_spectra[slice_idx, WL_IDX] / 100, cmap="magma", vmin=0, vmax=6)
    # dist_mask = distance_transform_edt(ex_mask[slice_idx] > 1)
    img[slice_idx][ex_spectra[slice_idx, WL_IDX] < 70] = np.nan
    # print(np.nanmean(img[slice_idx][ex_mask[slice_idx]==3]))
    im = ax.imshow(img[slice_idx] * 100, cmap="viridis", vmin=0, vmax=100)
    ax.contour((ex_mask[slice_idx] > 1), colors=COLOURS[1])
    for idx in [3, 4, 5]:
        dist_mask = distance_transform_edt(ex_mask[slice_idx] > 1)
        ax.contour(((ex_mask[slice_idx] == idx) & (dist_mask < 40)), colors=COLOURS[idx - 1])
    for idx in [6]:
        ax.contour((ex_mask[slice_idx] == idx), colors=COLOURS[idx - 1])
    ax.axis("off")
    ax.add_artist(ScaleBar(0.075, "mm", location="lower left"))

    data = np.squeeze(img[slice_idx]) * 100
    ydim, xdim = np.shape(data)

    WIDTH = xdim / 3
    BASELINE = ydim - 25
    LEFTSHIFT = 15
    HEIGHT = ydim / 10
    OUTLINE_COLOR = "black"

    mean_kidney_sO2 = (np.nanmean(data[ex_mask[slice_idx] == 5]) / 100)
    mean_arterial_sO2 = (np.nanmean(data[ex_mask[slice_idx] == 6]) / 100)
    ax.plot([xdim - LEFTSHIFT - WIDTH + mean_kidney_sO2 * WIDTH,
             xdim - LEFTSHIFT - WIDTH + mean_kidney_sO2 * WIDTH],
            [BASELINE - 1, BASELINE - HEIGHT], c="orange", linewidth=1.5)
    ax.plot([xdim - LEFTSHIFT - WIDTH + mean_arterial_sO2 * WIDTH,
             xdim - LEFTSHIFT - WIDTH + mean_arterial_sO2 * WIDTH],
            [BASELINE - 1, BASELINE - HEIGHT], c="red", linewidth=1.5)

    hist, bins = np.histogram(data, bins=50, range=(0, 100))
    ax.plot([xdim - WIDTH - LEFTSHIFT, xdim - LEFTSHIFT],
            [BASELINE, BASELINE], c=OUTLINE_COLOR, linewidth=1)
    ax.stairs(BASELINE - (hist / np.max(hist) * HEIGHT),
              (xdim - LEFTSHIFT - WIDTH + bins / 100 * WIDTH),
              baseline=BASELINE, fill=True, color="lightgray")
    ax.stairs(BASELINE - (hist / np.max(hist) * HEIGHT),
              (xdim - LEFTSHIFT - WIDTH + bins / 100 * WIDTH),
              baseline=BASELINE, fill=False, color=OUTLINE_COLOR, linewidth=1)
    x_points = np.asarray(xdim - LEFTSHIFT - WIDTH + bins / 100 * WIDTH)

    for i in np.arange(0, 20, 2):
        ax.scatter(x_points, np.ones_like(x_points) * BASELINE + 2 + i,
                   c=((x_points - np.min(x_points)) / (np.max(x_points) - np.min(x_points))),
                   s=2, marker="s", cmap="viridis")

    ax.add_artist(Rectangle((xdim - LEFTSHIFT - WIDTH - 2, BASELINE), WIDTH + 4, 22, fill=False, edgecolor="black"))

    ax.text(xdim - LEFTSHIFT - WIDTH, BASELINE + 16, "0", color="white", fontsize=9)
    ax.text(xdim - LEFTSHIFT - WIDTH / 2 - 20, BASELINE + 16, "sO$_2$", color="white", fontsize=9)
    ax.text(xdim - LEFTSHIFT - 28, BASELINE + 16, "100", color="white", fontsize=9)


ex_lu[ex_mask < 2] = np.nan
ex_BASE[ex_mask < 2] = np.nan
ex_WATER_4cm[ex_mask < 2] = np.nan

show_img(ax1, ex_spectra[:, WL_IDX, :, :], 0, title=True)
show_oxy(ax2, ex_lu, "Linear Unmixing", 0, title=True)
show_oxy(ax3, ex_BASE, "BASE", 0, title=True)
show_oxy(ax4, ex_WATER_4cm, "WATER_4cm", 0, title=True)

show_img(ax5, ex_spectra[:, WL_IDX, :, :], 5)
show_oxy(ax6, ex_lu, "Linear Unmixing", 5)
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

plt.savefig("figure7.pdf", dpi=300)
