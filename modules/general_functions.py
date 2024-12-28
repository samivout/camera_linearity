"""
Module for various functions that can be or are used in multiple places.
"""
from typing import Optional
from scipy.interpolate import interp1d
import cv2 as cv
from global_settings import *
import math

from cupy_wrapper import get_array_libraries

np, cp, using_cupy = get_array_libraries()
cnp = cp if using_cupy else np


def is_broadcastable(shape1: tuple[int, ...], shape2: tuple[int, ...]):

    if not shape1 or not shape2:
        raise ValueError('Shapes cannot be empty')

    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def choose_evenly_spaced_points(array: cnp.ndarray, step_x: int, step_y: Optional[int] = None):
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
    if BITS == DATAPOINTS:
        return clean_data_arr

    interpolated_data = np.zeros((BITS, DATAPOINTS), dtype=float)

    for i in range(BITS):
        x = np.linspace(0, 1, num=BITS)
        y = clean_data_arr[i, :]

        f = interp1d(x, y)

        x_new = np.linspace(0, 1, num=DATAPOINTS)
        interpolated_data[i, :] = f(x_new)

    return interpolated_data


def map_linearity_limits(lower_limit: Optional[int], upper_limit: Optional[int], ICRF: Optional[cp.ndarray]):
    """
    Maps the initial non-linear limit values to linear values using the given ICRF.

    Args:
        lower_limit: Integer determining how far from the lower edge of values the limit should be.
        upper_limit: Integer determining how far from the upper edge of values the limit should be.
        ICRF: the inverse camera response function used to map the limit values.

    Returns:
        The mapped lower and upper limits.
    """
    if lower_limit is None:
        lower = cp.array([LOWER_LIN_LIM] * CHANNELS, dtype="float64")
    else:
        lower = cp.array([lower_limit] * CHANNELS, dtype="float64")

    if upper_limit is None:
        upper = cp.array([UPPER_LIN_LIM] * CHANNELS, dtype="float64")
    else:
        upper = cp.array([MAX_DN - upper_limit] * CHANNELS, dtype="float64")

    if ICRF is None:
        lower /= MAX_DN
        upper /= MAX_DN
    else:
        for c in range(CHANNELS):
            lower[c] = ICRF[int(lower[c]), c]
            upper[c] = ICRF[int(upper[c]), c]

    return lower, upper


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    if isinstance(values, cnp.ndarray):
        average = float(cnp.average(values, weights=weights))
        variance = float(cnp.average((values - average) ** 2, weights=weights))
    else:
        average = cnp.average(values, weights=weights)
        variance = cnp.average((values - average) ** 2, weights=weights)

    return average, math.sqrt(variance)


def nanaverage(A: cnp.ndarray, weights: cnp.ndarray, axis: int | tuple[int, ...]):
    """
    Calculate the weighted average along the specified axis, ignoring NaN values in both A and weights.

    Args:
        A: CuPy array, input values.
        weights: CuPy array, weights corresponding to the values in A.
        axis: The axis along which the weighted average is computed.

    Returns:
        Weighted average along the specified axis, ignoring NaN values.
    """
    # Mask NaN values in both A and weights
    valid_mask = ~cnp.isnan(A) & ~cnp.isnan(weights)

    # Calculate the weighted sum of valid elements
    weighted_sum = cnp.nansum(A * weights * valid_mask, axis=axis)

    # Calculate the sum of the valid weights
    valid_weights_sum = cnp.nansum(valid_mask * weights, axis=axis)

    # Avoid division by zero by checking if valid_weights_sum is zero
    result = weighted_sum / valid_weights_sum
    result[valid_weights_sum == 0] = cnp.nan  # Set to NaN where there are no valid weights

    return result


def weighted_percentile(a: cp.ndarray, q: Optional[cp.ndarray] = cp.array([75, 25]), w: Optional[cp.ndarray] = None):
    """
    Calculates percentiles associated with a (possibly weighted) array.
    Args:
        a: The input array from which to calculate percents.
        q: The percentiles to calculate (0.0 - 100.0).
        w: The weights to assign to values of a. Equal weighting if None is specified.

    Returns:
        The values associated with the specified percentiles.
    """
    # Standardize and sort based on values in a
    q = cp.array(q) / 100.0
    if w is None:
        w = cp.ones(a.size)
    idx = cp.argsort(a)
    a_sort = a[idx]
    w_sort = w[idx]

    # Get the cumulative sum of weights
    ecdf = cp.cumsum(w_sort)

    # Find the percentile index positions associated with the percentiles
    p = q * (w.sum() - 1)

    # Find the bounding indices (both low and high)
    idx_low = cp.searchsorted(ecdf, p, side='right')
    idx_high = cp.searchsorted(ecdf, p + 1, side='right')
    idx_high[idx_high > ecdf.size - 1] = ecdf.size - 1

    # Calculate the weights
    weights_high = p - cp.floor(p)
    weights_low = 1.0 - weights_high

    # Extract the low/high indexes and multiply by the corresponding weights
    x1 = cp.take(a_sort, idx_low) * weights_low
    x2 = cp.take(a_sort, idx_high) * weights_high

    # Return the average
    return cp.add(x1, x2)


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


if __name__ == "__main__":
    pass
