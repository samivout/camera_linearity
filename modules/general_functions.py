"""
Module for various functions that can be or are used in multiple places.
"""
from pathlib import Path
from typing import Optional
from scipy.interpolate import interp1d
import cv2 as cv
from global_settings import GlobalSettings as gs
import math
import numpy as np
from array_wrapper import ArrayType, get_array_lib, cast_to_array


def is_broadcastable(shape1: tuple[int, ...], shape2: tuple[int, ...]):

    if not shape1 or not shape2:
        raise ValueError('Shapes cannot be empty')

    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def choose_evenly_spaced_points(array: ArrayType, step_x: int, step_y: Optional[int] = None):
    """
    Select points evenly in a Numpy array.
    Args:
        array: Input array
        step_x: step size for columns
        step_y: step size for rows

    Returns: Sampled array
    """
    # Calculate the step size between points
    if step_y is None:
        step_y = step_x

    # Select the evenly spaced points
    points = array[::step_x, ::step_y, ...]

    return points


def predict_output_shape(input_shape: tuple[int, int], step_x: int, step_y: Optional[int] = None)\
        -> tuple[int, int]:
    """
    Predict the shape of the output array from the choose_evenly_spaced_points function.

    Args:
        input_shape: A tuple (rows, columns) representing the shape of the input array.
        step_x: Step size for columns.
        step_y: Step size for rows. If None, it will be set equal to step_x.

    Returns:
        A tuple (rows_out, cols_out) representing the shape of the output array.
    """
    if step_y is None:
        step_y = step_x

    rows, cols = input_shape
    rows_out = (rows + step_x - 1) // step_x  # Equivalent to ceil(rows / step_x)
    cols_out = (cols + step_y - 1) // step_y  # Equivalent to ceil(cols / step_y)

    return rows_out, cols_out


def interpolate_data(clean_data_arr: np.ndarray):
    """
    Function to interpolate additional values between datapoints if the number of desired datapoints and the bit depth
    of the original images mismatch.
    Args:
        clean_data_arr:

    Returns:
        The interpolated data array.
    """
    if gs.BITS == gs.DATAPOINTS:
        return clean_data_arr

    interpolated_data = np.zeros((gs.BITS, gs.DATAPOINTS), dtype=float)

    for i in range(gs.BITS):
        x = np.linspace(0, 1, num=gs.BITS)
        y = clean_data_arr[i, :]

        f = interp1d(x, y)

        x_new = np.linspace(0, 1, num=gs.DATAPOINTS)
        interpolated_data[i, :] = f(x_new)

    return interpolated_data


def map_linearity_limits(lower_limit: Optional[int], upper_limit: Optional[int], ICRF: Optional[ArrayType]):
    """
    Maps the initial non-linear limit values to linear values using the given ICRF.

    Args:
        lower_limit: Integer determining how far from the lower edge of values the limit should be.
        upper_limit: Integer determining how far from the upper edge of values the limit should be.
        ICRF: the inverse camera response function used to map the limit values.

    Returns:
        The mapped lower and upper limits.
    """
    arr_lib = get_array_lib(ICRF)

    if lower_limit is None:
        lower = arr_lib.array([gs.LOWER_LIN_LIM] * gs.NUM_OF_CHS, dtype="float64")
    else:
        lower = arr_lib.array([lower_limit] * gs.NUM_OF_CHS, dtype="float64")

    if upper_limit is None:
        upper = arr_lib.array([gs.UPPER_LIN_LIM] * gs.NUM_OF_CHS, dtype="float64")
    else:
        upper = arr_lib.array([gs.MAX_DN - upper_limit] * gs.NUM_OF_CHS, dtype="float64")

    if ICRF is None:
        lower /= gs.MAX_DN
        upper /= gs.MAX_DN
    else:
        for c in range(gs.NUM_OF_CHS):
            lower[c] = ICRF[int(lower[c]), c]
            upper[c] = ICRF[int(upper[c]), c]

    return lower, upper


def weighted_avg_and_std(values: ArrayType, weights: Optional[ArrayType]):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    arr_lib = get_array_lib(values)

    average = arr_lib.average(values, weights=weights)
    variance = arr_lib.average((values - average) ** 2, weights=weights)

    return average, math.sqrt(variance)


def nanaverage(values: ArrayType, weights: ArrayType, axis: int | tuple[int, ...]):
    """
    Calculate the weighted average along the specified axis, ignoring NaN values in both A and weights.

    Args:
        values: NumPy or CuPy array, input values.
        weights: NumPy or CuPy array, matching type of values. Weights corresponding to the values in A.
        axis: The axis along which the weighted average is computed.

    Returns:
        Weighted average along the specified axis, ignoring NaN values.
    """
    arr_lib = get_array_lib(values)

    # Mask NaN values in both A and weights
    valid_mask = ~arr_lib.isnan(values) & ~arr_lib.isnan(weights)

    # Calculate the weighted sum of valid elements
    weighted_sum = arr_lib.nansum(values * weights * valid_mask, axis=axis)

    # Calculate the sum of the valid weights
    valid_weights_sum = arr_lib.nansum(valid_mask * weights, axis=axis)

    # Avoid division by zero by checking if valid_weights_sum is zero
    result = weighted_sum / valid_weights_sum
    result[valid_weights_sum == 0] = arr_lib.nan  # Set to NaN where there are no valid weights

    return result


def weighted_percentile(values: ArrayType, percentiles: Optional[ArrayType] = None,
                        weights: Optional[ArrayType] = None):
    """
    Calculates percentiles associated with a (possibly weighted) array.
    Args:
        values: The input array from which to calculate percents.
        percentiles: The percentiles to calculate (0.0 - 100.0).
        weights: The weights to assign to values array. Equal weighting if None is specified.

    Returns:
        The values associated with the specified percentiles.
    """
    arr_lib = get_array_lib(values)
    if percentiles is None:
        percentiles = arr_lib.array([75, 25])

    # Standardize and sort based on values in a
    percentiles = arr_lib.array(percentiles) / 100.0
    if weights is None:
        weights = arr_lib.ones(values.size)
    idx = arr_lib.argsort(values)
    a_sort = values[idx]
    w_sort = weights[idx]

    # Get the cumulative sum of weights
    ecdf = arr_lib.cumsum(w_sort)

    # Find the percentile index positions associated with the percentiles
    p = percentiles * (weights.sum() - 1)

    # Find the bounding indices (both low and high)
    idx_low = arr_lib.searchsorted(ecdf, p, side='right')
    idx_high = arr_lib.searchsorted(ecdf, p + 1, side='right')
    idx_high[idx_high > ecdf.size - 1] = ecdf.size - 1

    # Calculate the weights
    weights_high = p - arr_lib.floor(p)
    weights_low = 1.0 - weights_high

    # Extract the low/high indexes and multiply by the corresponding weights
    x1 = arr_lib.take(a_sort, idx_low) * weights_low
    x2 = arr_lib.take(a_sort, idx_high) * weights_high

    # Return the average
    return arr_lib.add(x1, x2)


def video_frame_generator(video_path: Path):
    """
    Generator function that yields frames from a video file at given path frame by frame.
    Args:
        video_path: Path to the video file.

    Yields:
        NumPy ndarray or None. None is yielded if no frames remain.
    """
    video = cv.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError(f'Unable to open video file at {video_path}')

    try:
        while True:
            ret, frame = video.read()

            if not ret:
                yield None
                break

            yield frame

    finally:
        video.release()


def read_ICRF_file(file_path: Path, return_derivative: Optional[bool] = True, use_cupy: Optional[bool] = False):
    """
    Utility function to read ICRF files into memory.
    Args:
        file_path: absolute path to the ICRF file.
        return_derivative: whether to return the derivative along with the main ICRF value array.
        use_cupy: whether to return a CuPy or NumPy array.
    Returns:
        Tuple, with first element containing the ICRF array, and second containing None or the derivative.
    """
    ICRF = np.loadtxt(file_path, dtype=float)
    if not return_derivative:
        ICRF = cast_to_array(ICRF, use_cupy=use_cupy)
        return ICRF, None

    ICRF_diff = np.zeros_like(ICRF)
    dx = 2 / (gs.BITS - 1)
    for c in range(gs.NUM_OF_CHS):
        ICRF_diff[:, c] = np.gradient(ICRF[:, c], dx)

    ICRF = cast_to_array(ICRF, use_cupy=use_cupy)
    ICRF_diff = cast_to_array(ICRF, use_cupy=use_cupy)

    return ICRF, ICRF_diff


def read_txt_to_array(file_name: str, path: Optional[str] = None, use_cupy: Optional[bool] = True):
    """
    Load numerical data from a .txt file of given name. Defaults to data
    directory but optionally can use other paths.

    Args:
        file_name: name of the file to load.
        path: path to the file, optional.
        use_cupy: whether to load data into a CuPy or NumPy array.

    Returns: numpy array of the txt file.
    """
    if path is None:
        load_path = gs.DATA_PATH
    else:
        load_path = Path(path)

    data_array = np.loadtxt(load_path.joinpath(file_name), dtype=float)

    if use_cupy:
        data_array = cast_to_array(data_array, use_cupy=use_cupy)

    return data_array


if __name__ == "__main__":
    pass
