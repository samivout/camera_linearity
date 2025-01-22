"""
Module for visualizing the output of various methods and functions in the package.
"""

import matplotlib.pyplot as plt
from typing import Optional
import general_functions as gf
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.odr import *
from image_set import ImageSet
from global_settings import GlobalSettings as gs
import read_data as rd
from typing import Dict
from cupy_wrapper import get_array_libraries

np, cp, using_cupy = get_array_libraries()
cnp = cp if using_cupy else np


def plot_noise_profiles_3d(mean_data_arr):
    data_step = int(gs.DATAPOINTS / gs.BITS)
    x0 = gs.MIN_DN
    x1 = gs.MAX_DN
    y0 = gs.MIN_DN
    y1 = gs.MAX_DN

    for c in range(gs.NUM_OF_CHS):
        mean_data_channel = mean_data_arr[:, :, c]
        x = np.linspace(0, 1, num=gs.BITS)
        y = np.linspace(0, 1, num=gs.BITS)
        x = x[x0:x1]
        y = y[y0:y1]

        X, Y = np.meshgrid(x, y)

        mean_data_channel = normalize_rows_by_sum(mean_data_channel)
        sampled_data = mean_data_channel[:, ::data_step]
        data_to_plot = sampled_data[x0:x1, y0:y1]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, data_to_plot, rstride=1, cstride=1, cmap='viridis',
                        edgecolor='none')
        ax.view_init(45, -30)
        plt.savefig(gs.OUTPUT_PATH.joinpath(f'3d_Profiles{c}.png'))
        plt.clf()

    return


def plot_noise_profiles_2d(mean_data_array: np.ndarray, number_of_profiles: int, lower_bound: int, upper_bound: int):

    bound_diff = upper_bound - lower_bound
    if number_of_profiles >= bound_diff:
        row_step = 1
    else:
        row_step = int(bound_diff / number_of_profiles)

    sampled_mean_data = mean_data_array[lower_bound:upper_bound:row_step, ::gs.DATAPOINT_MULTIPLIER, :]
    x_range = np.linspace(0, gs.MAX_DN, gs.BITS)

    for c in range(gs.NUM_OF_CHS):

        normalized_data = normalize_rows_by_sum(sampled_mean_data[:, :, c])

        for i in range(1, number_of_profiles):

            normalized_row = normalized_data[i, :]
            mode_index = np.argmax(normalized_row)
            mode = normalized_row[mode_index]
            plt.xlim(lower_bound, upper_bound)
            plt.plot(x_range, normalized_row)
            plt.vlines(mode_index, 0, mode)

        plt.savefig(gs.OUTPUT_PATH.joinpath(f'Profiles{c}.png'), dpi=300)
        plt.clf()

    return


def plot_ICRF(ICRF_array, name):
    """
    Simple function for plotting an inverse camera response function with all channels.
    :param ICRF_array: NumPy array in the shape of (gs.BITS, gs.NUM_OF_CHS).
    :param name: name used for saving the plot.
    """
    x_range = np.linspace(0, 1, gs.DATAPOINTS)
    plt.ylabel('Normalized exposure X (arb. units)')
    plt.xlabel('Normalized brightness B (arb. units)')
    for c in range(gs.NUM_OF_CHS):
        plt.plot(x_range, ICRF_array[:, c], color=gs.CH_CHARS[c])
    plt.savefig(gs.OUTPUT_PATH.joinpath(name), dpi=300)
    plt.clf()


def normalize_rows_by_max(mean_data_arr):
    mean_data_arr = mean_data_arr / np.amax(mean_data_arr, axis=1, keepdims=True)

    return mean_data_arr


def normalize_rows_by_sum(mean_data_arr):
    mean_data_arr = mean_data_arr / mean_data_arr.sum(axis=1, keepdims=True)

    return mean_data_arr


def print_mean_data_mode(mean_data_array):
    modes = np.zeros((gs.BITS, gs.NUM_OF_CHS), dtype=int)

    for c in range(gs.NUM_OF_CHS):
        for i in range(gs.MAX_DN):
            noise_profile = mean_data_array[i, ::4, c]
            modes[i, c] = np.argmax(noise_profile)

    np.savetxt(gs.OUTPUT_PATH.joinpath('modes.txt'), modes, fmt='%i')

    return


def plot_PCA():
    for i, file in enumerate(gs.PCA_FILES):

        PCA_array = rd.read_data_from_txt(file)
        image_name = str(file)
        image_name = image_name.replace('.txt', '.png')
        datapoints, components = np.shape(PCA_array)
        x_range = np.linspace(0, 1, num=datapoints)

        for component in range(components):
            plt.plot(x_range, PCA_array[:, component])

        plt.savefig(gs.OUTPUT_PATH.joinpath(image_name), dpi=300)
        plt.clf()


def plot_dorf_PCA():
    data_array = rd.read_data_from_txt('dorf_PCA.txt')
    x_range = data_array[:, 0]

    for i in range(2, 7):
        plt.plot(x_range, data_array[:, i])
    plt.savefig(gs.OUTPUT_PATH.joinpath('dorf_PCA.png'), dpi=300)
    plt.clf()

    plt.plot(x_range, data_array[:, 1])
    plt.savefig(gs.OUTPUT_PATH.joinpath('dorf_mean_ICRF.png'), dpi=300)
    plt.clf()


def plot_ICRF_PCA():
    """ The inverse camera response function calculated in terms of the mean
    ICRF, PCA vectors and the PCA coefficients that are subject to optimization.
    Dimensions of the PCA_array and PCA_params must match so that matrix
    multiplication produces an array of equal dimensions to mean_ICRF.

    Args:
        mean_ICRF: Numpy array containing the [0,1] normalized mean irradiance
            datapoints.
        PCA_array: Numpy array containing the principal component vectors.
        PCA_params: The PCA coefficients for each vector in PCA_array, these
            the values subject to optimization.

    Return:
        The new iteration of the inverse camera response function.
    """
    PCA_BLUE = [-0.10908332, -1.7415221, 0.66263865, -0.23043307, 0.15340393]
    PCA_GREEN = [-0.56790662, -0.44675708, 0.08047224, 0.16562418, -0.0744729]
    PCA_RED = [0.38280571, -1.45670034, 0.27022986, 0.43637866, -0.34930558]
    PCA_Components = [PCA_BLUE, PCA_GREEN, PCA_RED]
    x_range = np.linspace(0, 1, gs.BITS)
    for i in range(len(gs.MEAN_DATA_FILES)):

        initial_function = x_range ** 4
        PCA_file_name = gs.PCA_FILES[i]
        PCA_array = rd.read_data_from_txt(PCA_file_name)

        product = np.matmul(PCA_array, PCA_Components[i])
        iterated_ICRF = initial_function + product

        if i == 0:
            plt.plot(x_range, iterated_ICRF, color='b')
        if i == 1:
            plt.plot(x_range, iterated_ICRF, color='g')
        if i == 2:
            plt.plot(x_range, iterated_ICRF, color='r')

    plt.savefig(gs.OUTPUT_PATH.joinpath('ICRF_manual_plot.png'), dpi=300)

    return


def mean_data_plot():
    x_range = np.linspace(0, 1, gs.BITS)
    dx = 1 / (gs.BITS - 1)
    mean_data_array = np.zeros((gs.BITS, gs.DATAPOINTS, gs.NUM_OF_CHS), dtype=int)
    mean_ICRF_array = np.zeros((gs.DATAPOINTS, gs.NUM_OF_CHS), dtype=float)

    for i in range(len(gs.MEAN_DATA_FILES)):
        mean_file_name = gs.MEAN_DATA_FILES[i]
        mean_ICRF_file_name = gs.MEAN_ICRF_FILES[i]

        mean_data_array[:, :, i] = rd.read_data_from_txt(mean_file_name)
        mean_ICRF_array[:, i] = rd.read_data_from_txt(mean_ICRF_file_name)

    ICRF_calibrated = rd.read_data_from_txt(gs.ICRF_CALIBRATED_FILE)
    ICRF_diff = np.zeros_like(ICRF_calibrated)

    for c in range(gs.NUM_OF_CHS):

        if c == 0:
            color = 'blue'
        elif c == 1:
            color = 'green'
        else:
            color = 'red'

        ICRF_diff[:, c] = np.gradient(ICRF_calibrated[:, c], dx)
        plt.plot(x_range, ICRF_diff[:, c], color=color)

    np.savetxt(gs.OUTPUT_PATH.joinpath('ICRF_diff.txt'), ICRF_diff)
    plt.savefig(gs.OUTPUT_PATH.joinpath('ICRF_diff.png'), dpi=300)
    plt.clf()

    plot_ICRF(mean_ICRF_array, 'mean_ICRF.png')
    plot_ICRF(ICRF_calibrated, 'ICRF_calibrated.png')
    plot_noise_profiles_2d(mean_data_array)
    plot_noise_profiles_3d(mean_data_array)
    print_mean_data_mode(mean_data_array)


def calculate_and_plot_mean_ICRF(filepath: Optional[Path] = None):
    if filepath is None:
        filepath = gf.get_filepath_dialog('Choose ICRF file')

    path = filepath.parent
    name = filepath.name.replace('.txt', 'png')
    ICRF = rd.read_data_from_txt(filepath, str(filepath.parent))

    mean_ICRF = np.mean(ICRF, axis=1)
    np.savetxt(path.joinpath('mean_ICRF.txt'), mean_ICRF)
    x_range = np.linspace(0, 1, gs.DATAPOINTS)

    plt.ylabel('Normalized irradiance')
    plt.xlabel('Normalized brightness')
    plt.plot(x_range, mean_ICRF)
    plt.savefig(path.joinpath(name), dpi=300)
    plt.clf()

    return


def calculate_mean_ICRF(filepath_1: Optional[Path] = None,
                        filepath_2: Optional[Path] = None):
    if filepath_1 is None:
        filepath_1 = gf.get_filepath_dialog('Choose ICRF file')

    if filepath_2 is None:
        filepath_2 = gf.get_filepath_dialog('Choose ICRF file')

    ICRF_1 = rd.read_data_from_txt(filepath_1, str(filepath_1.parent))
    ICRF_2 = rd.read_data_from_txt(filepath_1, str(filepath_2.parent))

    ICRF_mean = (ICRF_1 + ICRF_2) / 2

    np.savetxt(gs.OUTPUT_PATH.joinpath('combined_ICRF.txt'), ICRF_mean)

    return


def plot_two_ICRF_and_calculate_RMSE(filepath1: Optional[Path] = None,
                                     filepath2: Optional[Path] = None):
    if filepath1 is None:
        filepath1 = gf.get_filepath_dialog('Choose ICRF file')

    if filepath2 is None:
        filepath2 = gf.get_filepath_dialog('Choose ICRF file')

    path = filepath2.parent
    name = 'ICRF_RMSE.png'

    ICRF1 = rd.read_data_from_txt(filepath1, str(filepath1.parent))
    ICRF2 = rd.read_data_from_txt(filepath2, str(filepath2.parent))

    RMSE = np.sqrt(np.mean((ICRF1 - ICRF2) ** 2))

    x_range = np.linspace(0, 1, gs.DATAPOINTS)
    plot_title = f'RMSE: {RMSE:.4f}'

    plt.title(plot_title)
    plt.ylabel('Normalized irradiance')
    plt.xlabel('Normalized brightness')
    plt.plot(x_range, ICRF1, c='r')
    plt.plot(x_range, ICRF2, c='b')
    plt.savefig(path.joinpath(name), dpi=300)
    plt.clf()


def smoothen_ICRF(ICRF_path: Optional[Path] = None):
    x_range = np.linspace(0, 1, gs.BITS)
    dx = 2 / (gs.BITS - 1)

    if ICRF_path is None:
        ICRF_path = gf.get_filepath_dialog('Choose ICRF file')

    ICRF = rd.read_data_from_txt(ICRF_path.name, ICRF_path.parent)
    ICRF_smoothed = np.zeros_like(ICRF)
    ICRF_smoothed_diff = np.zeros_like(ICRF)

    fig, axes = plt.subplots(1, gs.NUM_OF_CHS, figsize=(20, 5))

    for c, ax in enumerate(axes):

        if c == 0:
            color = 'Blue'
        elif c == 1:
            color = 'Green'
        else:
            color = 'Red'

        ICRF_smoothed[:, c] = savgol_filter(ICRF[:, c], 6, 2)
        ICRF_smoothed[0, c] = 0
        ICRF_smoothed[-1, c] = 1
        ICRF_smoothed_diff[:, c] = np.gradient(ICRF_smoothed[:, c], dx)
        RMSE = np.sqrt(np.mean(ICRF_smoothed[:, c] - ICRF[:, c]) ** 2)
        ax.plot(x_range, ICRF_smoothed[:, c], color='red')
        ax.plot(x_range, ICRF[:, c], color='blue', alpha=0.6)
        ax.set_title(f'{color}: RMSE = {RMSE: .4f}')

    np.savetxt(gs.OUTPUT_PATH.joinpath('ICRF_smoothed.txt'), ICRF_smoothed)
    plt.savefig(gs.OUTPUT_PATH.joinpath('ICRF_smoothed.png'), dpi=300)
    plt.clf()

    for c in range(gs.NUM_OF_CHS):

        if c == 0:
            color = 'Blue'
        elif c == 1:
            color = 'Green'
        else:
            color = 'Red'

        plt.plot(x_range, ICRF_smoothed_diff[:, c], color=color)

    plt.axvline(gs.LOWER_LIN_LIM / gs.MAX_DN, color='0', alpha=0.5)
    plt.axvline(gs.UPPER_LIN_LIM / gs.MAX_DN, color='0', alpha=0.5)
    plt.savefig(gs.OUTPUT_PATH.joinpath('ICRF_smoothed_diff.png'), dpi=300)

    return


def plot_channels_separately(im0: ImageSet, title: Optional[str] = "Pixel values (arb. units)",
                             color_map: Optional[str] = "inferno", use_std: Optional[bool] = False):
    """
    Plots the channels of an ImageSet object's acquired image or uncertainty image separately in a row with a given
    plt colormap and given title.

    Args:
        im0: ImageSet to plot.
        title: Title name given for each image. The name is prefixed by the channel's name.
        color_map: A plt colormap. Defaults to inferno.
        use_std: Whether to plot the acquired image or uncertainty image.
    """
    fig, axes = plt.subplots(1, gs.NUM_OF_CHS, figsize=(20, 5))

    if im0.measurand.std is not None and use_std:
        image = im0.measurand.std
    else:
        image = im0.measurand.val

    for c, ax in enumerate(axes):
        channel = ax.imshow(image[:, :, c], cmap=color_map)
        fig.colorbar(channel, ax=ax)
        ax.set_axis_off()
        ax.set_title(fr'{gs.CH_NAMES[c]} {title}', fontsize=14)

    fig.tight_layout(pad=1.2)
    plt.savefig(im0.path.parent.joinpath(f'{im0.path.name.replace(".tif", ".png")}'), dpi=300)
    plt.clf()

    return


def linear_function(B, x):
    return B[0] + B[1]*x


def create_linearity_plots(stats: Dict, save_path: Path, fit_line: bool, ylabel: str,
                           symbol: str):

    x = stats['ratios']
    fig, axes = plt.subplots(1, gs.NUM_OF_CHS, figsize=(20, 5))

    for c, ax in enumerate(axes):

        color = gs.CH_NAMES[c]
        cc = gs.CH_CHARS[c]

        y = stats['means'][:, c]
        y_std = stats['stds'][:, c]
        y_err = stats['errors'][:, c]

        if fit_line:
            linear_model = Model(linear_function)
            fit = RealData(x, y, sy=y_std)
            odr = ODR(fit, linear_model, beta0=[0., 0.])
            odr_output = odr.run()
            line = linear_function(odr_output.beta, x)
            '''
            plot_title = f'{cc}: A={odr_output.beta[0]:.4f} $\\pm$ {odr_output.sd_beta[0]:.4f},' \
                         f'B={odr_output.beta[1]:.4f} $\\pm$ {odr.output.sd_beta[1]:.4f}\n' \
                         f'$\\overline{{{symbol}}}_{{{cc}}}$={mean:.4f} $\\pm$ {mean_std: .4f}, $\\sigma_{{{cc},\\overline{{{symbol}}}}}$={std:.4f}'
            '''
        else:
            pass
            '''
            plot_title = f'{cc}: $\\overline{{{symbol}}}_{{{cc}}}$={mean:.4f} $\\pm$ {mean_std: .4f}, $\\sigma_{{{cc},\\overline{{{symbol}}}}}$={std:.4f}'
            '''

        ax.errorbar(x, y, yerr=(y_std / 5), elinewidth=1,
                    c=color, marker=None, linestyle='none', markersize=3, alpha=0.5, label=fr'$\sigma_{{{cc}, {symbol}}}$')
        ax.errorbar(x, y, yerr=y_err, elinewidth=1,
                    c='0', marker='x', linestyle='none', markersize=3, alpha=1, label=fr'$\delta {symbol}_{cc}$')
        ax.legend(loc='best')
        if fit_line:
            ax.plot(x, line, c='black')
        # ax.set_title(plot_title, fontsize=14)

    axes[0].set(ylabel=ylabel)
    axes[1].set(xlabel=r'Exposure time ratio $t_s/t_l$')
    axes[0].yaxis.label.set_size(16)
    axes[1].xaxis.label.set_size(16)
    plt.savefig(save_path, dpi=300)
    plt.clf()


def plot_histograms(histogram_dictionary: Dict, save_path: Path, file_name: str):

    for channel_key in histogram_dictionary.keys():

        hist, bin_edges = histogram_dictionary[channel_key]
        width = float(abs(bin_edges[1] - bin_edges[0]))
        hist = hist / cnp.sum(hist)

        if isinstance(hist, cp.ndarray):
            hist, bin_edges = cp.asnumpy(hist), cp.asnumpy(bin_edges)

        plt.bar(bin_edges[:-1], hist, width=width, fc=gs.CH_CHARS[channel_key], ec=None)
        full_path = save_path.joinpath(f'{file_name} {gs.CH_NAMES[channel_key]}.png')
        plt.savefig(full_path, dpi=300)
        plt.clf()

    return


def plot_kde(kde_dictionary: Dict, save_path: Path, file_name: str):

    for channel_key in kde_dictionary.keys():

        kde, x_range = kde_dictionary[channel_key]
        kde = kde / cnp.sum(kde)

        plt.plot(x_range, kde, c=gs.CH_CHARS[channel_key], label='KDE', linewidth=3)
        plt.legend(loc='best')
        full_path = save_path.joinpath(f'{file_name} {gs.CH_NAMES[channel_key]}.png')
        plt.savefig(full_path, dpi=300)
        plt.clf()

    return


if __name__ == "__main__":
    pass
