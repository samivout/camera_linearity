"""
Module for visualizing the output of various methods and functions in the package.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.odr import *
from image_set import ImageSet
from global_settings import GlobalSettings as gs
from typing import Dict, Union, Optional
import numpy as np


def plot_noise_profiles_3d(noise_data_array: np.ndarray, file_name: Union[Path, str],
                           save_path: Optional[Union[Path, str]]):
    """
    Function for creating a 3-d plot of the noise profiles based on camera noise data.
    Args:
        noise_data_array: Array of noise data, should be in shape (gs.BITS, gs.BITS).
        save_path: path where to save the figure.
        file_name: name of the saved file.
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if isinstance(file_name, str):
        file_name = Path(file_name)

    data_step = int(gs.DATAPOINTS / gs.BITS)
    x0 = gs.MIN_DN
    x1 = gs.MAX_DN
    y0 = gs.MIN_DN
    y1 = gs.MAX_DN

    for c in range(gs.NUM_OF_CHS):
        mean_data_channel = noise_data_array[:, :, c]
        x = np.linspace(0, 1, num=gs.BITS)
        y = np.linspace(0, 1, num=gs.BITS)
        x = x[x0:x1]
        y = y[y0:y1]

        X, Y = np.meshgrid(x, y)

        mean_data_channel = _normalize_rows_by_sum(mean_data_channel)
        sampled_data = mean_data_channel[:, ::data_step]
        data_to_plot = sampled_data[x0:x1, y0:y1]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, data_to_plot, rstride=1, cstride=1, cmap='viridis',
                        edgecolor='none')
        ax.view_init(45, -30)

        used_suffix = file_name.suffix
        channel_file_name = file_name.suffix.replace(used_suffix, f' {gs.CH_NAMES[c]}{used_suffix}')
        plt.savefig(save_path.joinpath(channel_file_name), dpi=300)
        plt.clf()

    return


def plot_noise_profiles_2d(noise_data_array: np.ndarray, number_of_profiles: int, lower_bound: int, upper_bound: int,
                           file_name: Union[Path, str], save_path: Optional[Union[Path, str]]):
    """
    Plot a given number of noise profiles based on the noise data of a camera between given lower and upper bounds.
    Args:
        noise_data_array: Array of noise data, should be in shape (gs.BITS, gs.BITS).
        number_of_profiles: the number of profiles to plot, meaning the number of rows to plot from noise data.
        lower_bound: lower bound of the range to plot.
        upper_bound: upper bound of the range to plot.
        save_path: path where to save the figure.
        file_name: name of the saved file.
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if isinstance(file_name, str):
        file_name = Path(file_name)

    bound_diff = upper_bound - lower_bound
    if number_of_profiles >= bound_diff:
        row_step = 1
    else:
        row_step = int(bound_diff / number_of_profiles)

    sampled_mean_data = noise_data_array[lower_bound:upper_bound:row_step, ::gs.DATAPOINT_MULTIPLIER, :]
    x_range = np.linspace(0, gs.MAX_DN, gs.BITS)

    for c in range(gs.NUM_OF_CHS):

        normalized_data = _normalize_rows_by_sum(sampled_mean_data[:, :, c])

        for i in range(1, number_of_profiles):

            normalized_row = normalized_data[i, :]
            mode_index = np.argmax(normalized_row)
            mode = normalized_row[mode_index]
            plt.xlim(lower_bound, upper_bound)
            plt.plot(x_range, normalized_row)
            plt.vlines(mode_index, 0, mode)

        used_suffix = file_name.suffix
        channel_file_name = file_name.suffix.replace(used_suffix, f' {gs.CH_NAMES[c]}{used_suffix}')
        plt.savefig(save_path.joinpath(channel_file_name), dpi=300)
        plt.clf()

    return


def plot_ICRF(ICRF_array: np.ndarray, file_name: Union[Path, str], save_path: Optional[Union[Path, str]]):
    """
    Simple function for plotting an inverse camera response function with all channels.
    Args:
        ICRF_array: NumPy array in the shape of (gs.BITS, gs.NUM_OF_CHS).
        save_path: path where to save the figure.
        file_name: name of the saved file.
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if isinstance(file_name, str):
        file_name = Path(file_name)

    x_range = np.linspace(0, 1, gs.DATAPOINTS)
    plt.ylabel('Normalized exposure X (arb. units)')
    plt.xlabel('Normalized brightness B (arb. units)')
    for c in range(gs.NUM_OF_CHS):
        plt.plot(x_range, ICRF_array[:, c], color=gs.CH_CHARS[c])
    plt.savefig(save_path.joinpath(file_name), dpi=300)
    plt.clf()


def _normalize_rows_by_max(mean_data_arr):
    mean_data_arr = mean_data_arr / np.amax(mean_data_arr, axis=1, keepdims=True)

    return mean_data_arr


def _normalize_rows_by_sum(mean_data_arr):
    mean_data_arr = mean_data_arr / mean_data_arr.sum(axis=1, keepdims=True)

    return mean_data_arr


def plot_image_set_channels_separately(im0: ImageSet, title: Optional[str] = "Pixel values (arb. units)",
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


def _linear_function(B, x):
    return B[0] + B[1]*x


def create_linearity_plots(stats: Dict, save_path: Path, fit_line: bool, ylabel: str, symbol: str):
    """
    Function for creating linearity plots based on data obtained from the linearity analysis of an ExposureSeries object.

    Args:
        stats: Dictionary of stats returned from ExposureSeries instance method .collect_exposure_pair_stats().
        save_path: path where to save the figure.
        fit_line: whether to fit a line in to the linearity data.
        ylabel: label text for the y-axis.
        symbol: a symbol for the plotted quantity, used in title.
    """
    x = stats['ratios']
    fig, axes = plt.subplots(1, gs.NUM_OF_CHS, figsize=(20, 5))

    for c, ax in enumerate(axes):

        color = gs.CH_NAMES[c]
        cc = gs.CH_CHARS[c]

        y = stats['means'][:, c]
        y_std = stats['stds'][:, c]
        y_err = stats['errors'][:, c]

        if fit_line:
            linear_model = Model(_linear_function)
            fit = RealData(x, y, sy=y_std)
            odr = ODR(fit, linear_model, beta0=[0., 0.])
            odr_output = odr.run()
            line = _linear_function(odr_output.beta, x)
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
    """
    Function for plotting a histogram based on data from Measurand instance method compute_channel_histogram().
    Args:
        histogram_dictionary: the data dictionary from compute_channel_histogram().
        save_path: path to save the figure to.
        file_name: name of the file(s) to save, channel name is appended to the name.
    """
    for channel_key in histogram_dictionary.keys():

        hist, bin_edges = histogram_dictionary[channel_key]
        width = float(abs(bin_edges[1] - bin_edges[0]))
        hist = hist / np.sum(hist)

        plt.bar(bin_edges[:-1], hist, width=width, fc=gs.CH_CHARS[channel_key], ec=None)
        full_path = save_path.joinpath(f'{file_name} {gs.CH_NAMES[channel_key]}.png')
        plt.savefig(full_path, dpi=300)
        plt.clf()

    return


def plot_kde(kde_dictionary: Dict, save_path: Path, file_name: str):
    """
    Function for plotting a kernel density estimate on data from Measurand instance method
    compute_kernel_density_estimate().
    Args:
        kde_dictionary: the data dictionary from compute_kernel_density_estimate().
        save_path: path to save the figure to.
        file_name: name of the file(s) to save, channel name is appended to the name.
    """
    for channel_key in kde_dictionary.keys():

        kde, x_range = kde_dictionary[channel_key]
        kde = kde / np.sum(kde)

        plt.plot(x_range, kde, c=gs.CH_CHARS[channel_key], label='KDE', linewidth=3)
        plt.legend(loc='best')
        full_path = save_path.joinpath(f'{file_name} {gs.CH_NAMES[channel_key]}.png')
        plt.savefig(full_path, dpi=300)
        plt.clf()

    return


if __name__ == "__main__":
    pass
