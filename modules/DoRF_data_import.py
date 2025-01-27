"""
Module for processing the camera response function data provided by the authors of the database of camera response
functions (DoRF).
"""
from pathlib import Path
from typing import Optional
from sklearn.decomposition import PCA
from global_settings import GlobalSettings as gs
from cupy_wrapper import get_array_libraries
import read_data as rd
np, cp, using_cupy = get_array_libraries()
cnp = cp if using_cupy else np


def _read_dorf_data(file_path: Path, include_gamma: bool, color_split: bool):
    """
    Load numerical data from a .txt file of the given name from the data directory. The dorfCurves.txt contains
    measured irradiance vs. digital number data for various cameras. In this function all the data is read in and split
    into separate Numpy float arrays for each color channel.

    Args:
        file_path: the name of the .txt file containing the dorf data.

    Return:
        list of numpy float arrays, one for each color channel.
    """
    red_curves = np.zeros((1, gs.DORF_DATAPOINTS), dtype=float)
    blue_curves = np.zeros((1, gs.DORF_DATAPOINTS), dtype=float)
    green_curves = np.zeros((1, gs.DORF_DATAPOINTS), dtype=float)
    number_of_lines = 0
    is_red = False
    is_green = False
    is_blue = False
    with open(file_path) as f:
        for line in f:
            text = line.rstrip().casefold()
            number_of_lines += 1
            if (number_of_lines + 5) % 6 == 0:
                if text.endswith('red') or text[-1] == 'r' or text[-2] == 'r':
                    is_red = True
                    continue
                elif text.endswith('green') or text[-1] == 'g' or text[-2] == 'g':
                    is_green = True
                    continue
                elif text.endswith('blue') or text[-1] == 'b' or text[-2] == 'b':
                    is_blue = True
                    continue
                else:
                    if not include_gamma:
                        is_red = False
                        is_green = False
                        is_blue = False
                    else:
                        is_red = True
                        is_green = True
                        is_blue = True

            if not color_split:
                is_red = True
                is_green = True
                is_blue = True

            if number_of_lines % 6 == 0:
                line_to_arr = np.fromstring(text, dtype=float, sep=' ')
                if is_red:
                    red_curves = np.vstack([red_curves, line_to_arr])
                    is_red = False
                if is_green:
                    green_curves = np.vstack([green_curves, line_to_arr])
                    is_green = False
                if is_blue:
                    blue_curves = np.vstack([blue_curves, line_to_arr])
                    is_blue = False

    f.close()

    # Remove the initial row of zeros from the arrays.
    red_curves = np.delete(red_curves, 0, 0)
    green_curves = np.delete(green_curves, 0, 0)
    blue_curves = np.delete(blue_curves, 0, 0)

    list_of_curves = [blue_curves, green_curves, red_curves]

    return list_of_curves


def _invert_and_interpolate_data(list_of_curves, new_datapoints):
    """
    Invert the camera response functions obtained from the dorfCurves.txt file. Numpy interpolation is used to obtain
    the same digital value datapoints for all curves, as originally the evenly spaced datapoints were in the irradiance
    domain.

    Args:
        list_of_curves: list containing all the CRFs, separated based on color channel, in Numpy float arrays. Original
            data is spaced into 1024 points evenly from 0 to 1.
        new_datapoints: number of datapoints to be used in the digital value range.

    Return:
        list of numpy float arrays, one for each color channel,
        containing the inverted camera response functions, or ICRFs.
    """
    list_of_processed_curves = []
    x_old = np.linspace(0, 1, gs.DORF_DATAPOINTS)
    x_new = np.linspace(0, 1, new_datapoints)

    for index, arr in enumerate(list_of_curves):
        rows = arr.shape[0]
        y_new = np.zeros(new_datapoints, dtype=float)

        for i in range(rows):
            y = arr[i]
            y_inv = np.interp(x_old, y, x_old)

            if gs.DORF_DATAPOINTS != new_datapoints:

                interpolated_row = np.interp(x_new, x_old, y_inv)
                y_new = np.vstack([y_new, interpolated_row])

        y_new = np.delete(y_new, 0, 0)
        list_of_processed_curves.append(y_new)

    return list_of_processed_curves


def _calculate_mean_curve(list_of_curves):
    """
    Calculate the mean function from a collection of CRFs or ICRFs

    Args:
        list_of_curves: list containing the Numpy float arrays of the CRFs or ICRFs from which a mean function will be
        calculated for each channel.

    Return:
        list containing the Numpy float arrays for the mean CRFs or ICRFs for each color channel.
    """

    for index, curves in enumerate(list_of_curves):

        list_of_curves[index] = np.mean(curves, axis=0)

    return list_of_curves


def _calculate_principal_components(covariance_array: np.ndarray):
    """
    Calculates the principal components from the given covariance array.
    Args:
        covariance_array: The covariance array for which to compute the principal components.

    Returns:
        Array of the principal components.
    """
    PCA_array = PCA(n_components=gs.NUM_OF_PCA_PARAMS)
    PCA_array.fit(covariance_array)
    result = PCA_array.transform(covariance_array)

    # Scale to unit vector and shift to start and end at y-value zero.
    for n in range(gs.NUM_OF_PCA_PARAMS):
        norm = np.linalg.norm(result[:, n])
        result[:, n] /= norm
        result[:, n] -= result[0, n]

    return result


def _calculate_covariance_matrix(data_array, mean_data_array):
    """
    Calculate the covariance matrix according to the article 'What is the space
    of camera response functions' using vectorized operations.

    Args:
        data_array: The ICRF data obtained from the original DoRF file for each channel separately.
        mean_data_array: Mean ICRF data for each channel calculated from the collection of ICRFs in the original
            DoRF file.

    Returns:
        The covariance matrix calculated for the given data.
    """
    # Center the data by subtracting the mean
    centered_data = data_array - mean_data_array

    # Calculate covariance matrix using matrix multiplication
    # Transpose centered_data to compute dot product
    covariance_matrix = centered_data.T @ centered_data

    return covariance_matrix


def analyze_principal_components():
    """
    Main function to be called outside the module, used to run the process of obtaining principal components for the
    ICRF data for each channel separately.
    """
    for i in range(len(gs.ICRF_FILES)):
        file_name = gs.ICRF_FILES[i]
        mean_file_name = gs.MEAN_ICRF_FILES[i]
        ICRF_array = rd.read_data_from_txt(file_name, use_cupy=True)
        mean_ICRF_array = rd.read_data_from_txt(mean_file_name, use_cupy=True)

        covariance_matrix = _calculate_covariance_matrix(ICRF_array,
                                                         mean_ICRF_array)

        covariance_matrix = cp.asnumpy(covariance_matrix)
        PCA_array = _calculate_principal_components(covariance_matrix)

        np.savetxt(gs.DATA_PATH.joinpath(gs.PCA_FILES[i]), PCA_array)

    return


def process_CRF_data(include_gamma: Optional[bool] = False, color_split: Optional[bool] = True):
    """
    Main function to be called outside the module, used for obtaining the CRFs from dorfCurves.txt, invert them and
    determine a mean ICRF for each color channel separately.

    Args:
        include_gamma: bool whether to include non-color-specific response functions in the data. Defaults to False.
    """
    data_file_path = gs.DATA_PATH.joinpath(gs.DORF_FILE)
    list_of_curves = _read_dorf_data(data_file_path, include_gamma, color_split)
    processed_curves = _invert_and_interpolate_data(list_of_curves, gs.DATAPOINTS)
    list_of_mean_curves = processed_curves.copy()
    list_of_mean_curves = _calculate_mean_curve(list_of_mean_curves)

    for i in range(len(gs.ICRF_FILES)):

        np.savetxt(gs.DATA_PATH.joinpath(gs.ICRF_FILES[i]), processed_curves[i])
        np.savetxt(gs.DATA_PATH.joinpath(gs.MEAN_ICRF_FILES[i]), list_of_mean_curves[i])

    return


if __name__ == "__main__":
    pass
