"""
Module containing the base class for Measurands and the main Measurand classes. The Measurand classes implements most of
the mathematical and statistical operations of the package. The class maintains two arrays of same shape,
one representing the value of a measurement and the other representing the uncertainty of the measurement. The last
dimension of the array is assumed to contain independent data on each slice. For example in the case of images with
shape (height, width, channels), the third dimension is assumed independent. The class can handle general data in
arbitrary shapes, but operations between Measurands need to adhere to NumPy broadcasting rules. For the sake of not
making things too complex, the shapes of the .val and .std are required to be the same.
"""
from typing import Optional, Union, List
from typing import Union
from typing import List
from global_settings import GlobalSettings as gs
from cupyx.scipy.ndimage import median_filter as cp_median_filter
from scipy.stats import gaussian_kde
import numpy as np

from measurand import AbstractMeasurand, ScalarType

try:
    import cupy as cp
except ImportError:
    raise ImportError("CuPy is not installed.")

ScalarType = Union[int, float]


class CupyMeasurand(AbstractMeasurand):
    """
    CuPy version of the base AbstractMeasurand class. CupyMeasurand allows assigning NumPy arrays into val and std
    instance attributes by doing a silent conversion into a CuPy array. The equivalent case is not true for
    NumpyMeasurand and instead leads to a ValueError, following CuPy convention of not allowing implicit conversion.
    """

    lib = cp
    backend = "cupy"
    fn_median_filter = cp_median_filter
    ArrayType = cp.ndarray
    InputType = Union[ScalarType, ArrayType]

    def __init__(self, val: Optional[InputType] = None, std: Optional[InputType] = None):

        super().__init__(val, std)
        if val is not None and not isinstance(val, self.InputType):
            raise TypeError('Invalid value type.')
        if std is not None and not isinstance(std, self.InputType):
            raise TypeError('Invalid std type')

        if isinstance(val, ScalarType):
            val = cp.array([val], dtype=cp.dtype('float64'))

        if isinstance(std, ScalarType):
            std = cp.array([std], dtype=cp.dtype('float64'))

        if val is not None and std is not None and val.shape != std.shape:
            raise ValueError('Value and std shapes must match.')

        self._val = val
        self._std = std
        self._initialized = True

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value: Optional[Union[np.ndarray, cp.ndarray]]):
        if value is not None and isinstance(value, np.ndarray):
            self._val = cp.array(value)
        elif value is None or isinstance(value, cp.ndarray):
            self._val = value
        else:
            raise TypeError(f"val must be an array or None, got {type(value)} instead.")

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value: Optional[Union[np.ndarray, cp.ndarray]]):
        if value is not None and isinstance(value, np.ndarray):
            self.std = cp.array(value)
        elif value is None or isinstance(value, cp.ndarray):
            self.std = value
        else:
            raise TypeError(f"std must be an array or None, got {type(value)} instead.")

    def compute_kernel_density_estimate(self, data_points: int, included_range: Optional[tuple[float, float]] = None,
                                        channels: Optional[List[int]] = None, use_std: Optional[bool] = False):
        """
        Computes a kernel density estimate (essentially a "better" histogram).
        Args:
            data_points: number of data points to use.
            included_range: the range of values to include.
            channels: on which indices of the last dimension to perform the computations on. One dictionary key is
                used for each included channel, containing the results in a (result, range) tuple.
            use_std: whether to use .std values as an inverse weight in the computation.

        Returns:
            A dictionary containing the KDE as a (result, range) tuple, with the channel index acting as a key.
        """
        if channels is None:
            channels = [c for c in range(gs.NUM_OF_CHS)]

        estimates = {}

        for c in channels:

            channel_values = self.val[..., c]
            finite_mask = cp.isfinite(channel_values)

            if use_std:
                stds = self.std[..., c]
                non_zeros = stds != 0
                finite_mask = cp.logical_and(finite_mask, non_zeros)
                weights = stds[finite_mask]
                weights = 1 / weights
                channel_values = channel_values[finite_mask]
            else:
                weights = None
                channel_values = channel_values[finite_mask]

            channel_values = cp.asnumpy(channel_values)
            if weights is not None:
                weights = cp.asnumpy(weights)

            if included_range is None:
                x_range = np.linspace(np.min(channel_values), np.max(channel_values), num=data_points)
            else:
                x_range = np.linspace(included_range[0], included_range[1], num=data_points)

            gkde = gaussian_kde(channel_values, 'silverman', weights=weights)
            gkde_result = gkde.evaluate(x_range)

            estimates[c] = (gkde_result, x_range)

        return estimates
