"""
Module containing the base class for Measurands and the main Measurand classes. The Measurand classes implements most of
the mathematical and statistical operations of the package. The class maintains two arrays of same shape,
one representing the value of a measurement and the other representing the uncertainty of the measurement. The last
dimension of the array is assumed to contain independent data on each slice. For example in the case of images with
shape (height, width, channels), the third dimension is assumed independent. The class can handle general data in
arbitrary shapes, but operations between Measurands need to adhere to NumPy broadcasting rules. For the sake of not
making things too complex, the shapes of the .val and .std are required to be the same.
"""
import math
import general_functions as gf
import copy
from abc import ABC, abstractmethod

from cupy_wrapper import get_array_libraries
from typing import Optional
from typing import Union
from typing import List
from global_settings import GlobalSettings as gs
from scipy.stats import gaussian_kde
from scipy.ndimage import median_filter as np_median_filter

np, cp, using_cupy = get_array_libraries()
cnp = cp if using_cupy else np

ScalarType = Union[int, float]


class AbstractMeasurand(ABC):

    lib = np                                # Default class variables to NumPy.
    backend = "numpy"
    fn_median_filter = np_median_filter
    ArrayType = np.ndarray
    InputType = Union[ArrayType, ScalarType]

    def __init__(self, val, std: Optional = None):
        """Initialize with an array-like object."""
        if self.lib is None:
            raise NotImplementedError("Subclasses must define 'lib' (np or cp)")
        if self.InputType is None:
            raise NotImplementedError("Subclasses must define input type to match the array library.")

        self._val = val
        self._std = std
        self._channels = None

    """
    Base class for the Measurand, enforcing an interface for managing a measured value and its associated uncertainty.
    """

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value: Optional[ArrayType]):
        if value is not None and not isinstance(value, self.ArrayType):
            raise TypeError(f"val must be an array or None, got {type(value)} instead.")
        self._val = value

        number_of_dims = len(self.lib.shape(self._val))
        self._channels = self.lib.arange(0, number_of_dims, step=1)

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value: Optional[ArrayType]):
        if value is not None and not isinstance(value, self.ArrayType):
            raise TypeError(f"std must be an array or None, got {type(value)} instead.")
        self._std = value

    @property
    def channels(self):
        """
        Read-only property, which is based on the shape of the value (and std) instance attributes.
        Returns:
            The number of channels in the value (and std) array.
        """
        return self._channels

    @channels.setter
    def channels(self, new_channels: Optional[ArrayType]):
        """Read-only property, raises an error if attempting to modify."""
        raise AttributeError("Channels is a read-only attribute, based on the shape of val array.")

    def __repr__(self):

        value_shape = self.val.shape if self.val is not None else 'None'
        values = self.val if self.val is not None else 'None'
        std_shape = self.std.shape if self.std is not None else 'None'

        return f'Measurand(value.shape= {value_shape}, std.shape= {std_shape}, values={values})'

    def __copy__(self):

        return self.__class__(self.val, self.std)

    def __deepcopy__(self, memo):

        value, std = None, None
        if self.val is not None:
            value = copy.deepcopy(self.val, memo)
        if self.std is not None:
            std = copy.deepcopy(self.std, memo)

        return self.__class__(value, std)

    def __add__(self, other: Union[InputType, 'AbstractMeasurand']):

        normalized_other, use_std = self._normalize_input(other)
        x1, x2 = self.val, normalized_other.val

        if not gf.is_broadcastable(x1.shape, x2.shape):
            raise ValueError('Measurands are not broadcastable.')

        sum_result = x1 + x2

        if not use_std:
            return self.__class__(val=sum_result, std=None)

        std1, std2 = self.std, normalized_other.std

        if std1 is None:
            std1 = self.lib.zeros_like(x1)
        if std2 is None:
            std2 = self.lib.zeros_like(x2)

        sum_std = self.lib.sqrt((std1 ** 2) + (std2 ** 2))

        return self.__class__(val=sum_result, std=sum_std)

    def __sub__(self, other: Union[InputType, 'AbstractMeasurand']):

        normalized_other, use_std = self._normalize_input(other)
        x1, x2 = self.val, normalized_other.val

        if not gf.is_broadcastable(x1.shape, x2.shape):
            raise ValueError('Measurands are not broadcastable.')

        subtraction_result = x1 - x2

        if not use_std:
            return self.__class__(val=subtraction_result, std=None)

        std1, std2 = self.std, normalized_other.std
        if std1 is None:
            std1 = self.lib.zeros_like(x1)
        if std2 is None:
            std2 = self.lib.zeros_like(x2)

        subtraction_std = self.lib.sqrt((std1 ** 2) + (std2 ** 2))
        return self.__class__(val=subtraction_result, std=subtraction_std)

    def __neg__(self):

        negation_result = self.lib.negative(self.val)
        if self.std is not None:
            if not isinstance(self.std, ScalarType):
                negation_std = self.std.copy()
            else:
                negation_std = self.std
        else:
            negation_std = None

        return self.__class__(val=negation_result, std=negation_std)

    def __truediv__(self, other: Union[InputType, 'AbstractMeasurand']):

        normalized_other, use_std = self._normalize_input(other)
        x1, x2 = self.val, normalized_other.val

        if not gf.is_broadcastable(x1.shape, x2.shape):
            raise ValueError('Measurands are not broadcastable.')

        division_result = x1 / x2

        if not use_std:
            return self.__class__(val=division_result, std=None)

        std1, std2 = self.std, normalized_other.std
        if std1 is None:
            std1 = self.lib.zeros_like(x1)
        if std2 is None:
            std2 = self.lib.zeros_like(x2)

        u1 = std1 / x2
        u2 = (x1 * std2) / (x2 ** 2)
        division_std = self.lib.sqrt(u1 ** 2 + u2 ** 2)

        return self.__class__(val=division_result, std=division_std)

    def __mul__(self, other: Union[InputType, 'AbstractMeasurand']):

        normalized_other, use_std = self._normalize_input(other)
        x1, x2 = self.val, normalized_other.val

        if not gf.is_broadcastable(x1.shape, x2.shape):
            raise ValueError('Measurands are not broadcastable.')

        multiplication_result = x1 * x2

        if not use_std:
            return self.__class__(val=multiplication_result, std=None)

        std1, std2 = self.std, normalized_other.std
        if std1 is None:
            std1 = self.lib.zeros_like(x1)
        if std2 is None:
            std2 = self.lib.zeros_like(x2)

        multiplication_std = self.lib.sqrt((x1 * std2) ** 2 + (x2 * std1) ** 2)

        return self.__class__(val=multiplication_result, std=multiplication_std)

    def __rmul__(self, other: Union[InputType, 'AbstractMeasurand']):

        return self * self.__class__(other)

    def __pow__(self, other: Union[InputType, 'AbstractMeasurand']):

        normalized_other, use_std = self._normalize_input(other)
        x1, x2 = self.val, normalized_other.val

        if not gf.is_broadcastable(x1.shape, x2.shape):
            raise ValueError('Measurands are not broadcastable.')

        exponentation_result = x1 ** x2

        if not use_std:
            return self.__class__(val=exponentation_result, std=None)

        std1, std2 = self.std, normalized_other.std
        if std1 is None:
            std1 = self.lib.zeros_like(x1)
        if std2 is None:
            std2 = self.lib.zeros_like(x2)

        u1 = (x2 * x1 ** (x2 - 1))
        u2 = (self.lib.log(x1) * x1 ** x2)

        exponentation_std = self.lib.sqrt((u1 * std1) ** 2 + (u2 * std2) ** 2)

        return self.__class__(val=exponentation_result, std=exponentation_std)

    def log_e(self):

        x1 = self.val

        use_std = False
        if self.std is not None:
            use_std = True

        log_result = self.lib.log(x1)

        if not use_std:
            return self.__class__(val=log_result, std=None)

        std1 = self.std

        log_std = std1 / self.lib.log(x1)

        return self.__class__(val=log_result, std=log_std)

    def log_10(self):

        x1 = self.val

        use_std = False
        if self.std is not None:
            use_std = True

        log_result = self.lib.log10(x1)

        if not use_std:
            return self.__class__(val=log_result, std=None)

        std1 = self.std

        log_std = std1 / (x1 * (self.lib.log(5) + self.lib.log(2)))

        return self.__class__(val=log_result, std=log_std)

    def _normalize_input(self, other: Union[InputType, 'AbstractMeasurand']):
        """
        Internal method of the class for normalizing the inputs of different operations, enabling the direct use of
        scalars and NumPy/CuPy arrays with Measurand objects.
        Args:
            other: a valid MeasurandType, see top of class for definition.

        Returns:
            The normalized value of the input argument and whether to utilize uncertainty propagation.
        """
        use_std = False
        if isinstance(other, self.__class__):
            normalized_other = other
        elif type(other) in self.InputType:
            normalized_other = self.__class__(other)
        else:
            raise TypeError('Invalid other type.')

        if self.std is not None or normalized_other.std is not None:
            use_std = True

        return normalized_other, use_std

    def zeros_like_measurand(self):
        """
        Initializes and returns a new measurand object with .val and .std of same size as self, but elements set to zero.
        Returns: a new measurand with same size, filled with zeros.
        """
        new_val = None
        if self.val is not None:
            new_val = self.lib.zeros_like(self.val)
        new_std = None
        if self.std is not None:
            new_std = self.lib.zeros_like(self.std)

        return self.__class__(new_val, new_std)

    def compute_dimension_statistics(self, axis: Optional[None | int | tuple[int, ...]] = None):
        """
        Computes statistics using the given axis, which follows NumPy conventions. Based on the availability of .std
        weighted statistics are computed, using the uncertainties as inverse weights. NaNs are ignored.
        Args:
            axis: optional axis argument along which to perform the computations. Follows NumPy conventions.

        Returns:
            A dictionary of the statistics, containing mean, standard deviation and uncertainty of the mean under their
            respective keys.
        """
        use_std = False
        if self.std is not None:
            use_std = True

        values = self.val
        if use_std:
            stds = self.std

        std_mean = None
        if not use_std:
            value_mean = self.lib.nanmean(values, axis=axis)
            value_std = self.lib.nanstd(values, axis=axis)
        else:
            weights = 1 / stds
            sum_of_weights = self.lib.nansum(weights, axis=axis)
            value_mean = self.lib.nansum(values * weights, axis=axis) / sum_of_weights
            value_std = self.lib.sqrt(self.lib.nansum(weights * (values - value_mean) ** 2, axis=axis) / sum_of_weights)
            std_mean = self.lib.nanmean(stds, axis=axis)

        result_dictionary = {"mean": value_mean, "std": value_std, "error": std_mean}

        return result_dictionary

    def extract(self, dims: Optional[int | List[int]] = None, axis: Optional[int] = None):
        """
        Extracts the desired slices along the given axis from the source array.
        Args:
            dims: list of the slice indices to extract.
            axis: along which axis to extract, follows NumPy conventions.

        Returns:
            A new Measurand object with the extracted arrays as .val and .std values.
        """
        target_dims = None
        if type(dims) is int:
            target_dims = [dims]
        if type(dims) is list:
            target_dims = dims

        std = None
        value = self.lib.take(self.val, target_dims, axis=axis)
        if self.std is not None:
            std = self.lib.take(self.std, target_dims, axis=axis)

        return self.__class__(value, std)

    def apply_thresholds(self, lower: Optional[List[float | None]] = None,
                         upper: Optional[List[float | None]] = None):
        """
        Applies the given thresholds to the array by setting values outside to NaN.

        The thresholds are applied to the last dimension of the array, which is treated as the independent axis.
        For example, in an array with shape (height, width, channels), thresholds are applied per channel.

        Args:
            lower: List of lower thresholds, one for each slice of the independent axis. Use None for any slice to skip
                   applying the lower threshold for that slice. If None, skip all lower thresholds.
            upper: List of upper thresholds, one for each slice of the independent axis. Use None for any slice to skip
                   applying the upper threshold for that slice. If None, skip all upper thresholds.

        Returns:
            None. The function modifies `self.value` in place.
        """
        # Number of dependent dimensions (all except the last one)
        number_of_dependent_axes = self.val.ndim - 1
        independent_axis_size = self.val.shape[-1]  # Size of the last dimension

        # Default thresholds if none are provided
        if lower is None:
            lower = [None] * independent_axis_size
        if upper is None:
            upper = [None] * independent_axis_size

        # Check that the length of thresholds matches the size of the independent axis
        if len(lower) != independent_axis_size or len(upper) != independent_axis_size:
            raise ValueError("The length of 'lower' and 'upper' must match the size of the independent axis.")

        value = self.val  # Main data array

        # Convert thresholds to arrays, replacing None with -inf/inf for proper broadcasting
        lower = self.lib.array([l if l is not None else -cnp.inf for l in lower], dtype=value.dtype)
        upper = self.lib.array([u if u is not None else cnp.inf for u in upper], dtype=value.dtype)

        # Reshape thresholds to match the dependent dimensions
        # Example: If self.value.shape = (height, width, channels), reshape to (1, 1, channels)
        lower = lower.reshape((1,) * number_of_dependent_axes + lower.shape)
        upper = upper.reshape((1,) * number_of_dependent_axes + upper.shape)

        # Build the mask for all slices along the independent axis
        mask = (value < lower) | (value > upper)

        # Apply the mask to the `value` array
        value[mask] = self.lib.nan
        self.val = value

        # Optionally apply the mask to the uncertainty image
        if self.std is not None:
            std = self.std  # Uncertainty array
            std[mask] = self.lib.nan
            self.std = std

    def compute_channel_histogram(self, bins: int, included_range: Optional[tuple[float, float]] = None,
                                  channels: Optional[List[int]] = None, use_std: Optional[bool] = False):
        """
        Computes the histogram for each slice in the independent dimension. channels argument can be used to define the
        channels the computation is performed on, if None then all channels are used. use_std argument specifies if the
        uncertainties are used as inverse weights for the histogram.
        Args:
            bins: how many bins to use.
            included_range: tuple that sets the minimum and maximum values that are included in the histogram.
            channels: which channels are used in the computation.
            use_std: whether to use uncertainties as inverse weights.

        Returns:
            Dictionary containing a (histogram values, bin edges) tuple for each channel index acting as the key value.
        """
        if channels is None:
            channels = [c for c in range(gs.NUM_OF_CHS)]

        histograms = {}

        for c in channels:

            channel_values = self.val[..., c]
            finite_mask = self.lib.isfinite(channel_values)

            if use_std:
                stds = self.std[..., c]
                non_zeros = stds != 0
                finite_mask = self.lib.logical_and(finite_mask, non_zeros)
                weights = stds[finite_mask]
                weights = 1 / weights
                channel_values = channel_values[finite_mask]
            else:
                weights = None
                channel_values = channel_values[finite_mask]

            histogram = self.lib.histogram(channel_values, bins=bins, range=included_range, weights=weights)
            histograms[c] = histogram

        return histograms

    def linearize(self, ICRF: ArrayType, ICRF_diff: Optional[ArrayType] = None):
        """
        The main linearization method. Different private implementation used based on array shape.
        Args:
            ICRF: the mapping function.
            ICRF_diff: the derivative of the mapping function.

        Returns:
            A new Measurand object with the linearized values.
        """
        shape = self.lib.shape(self.val)
        if shape[-1] >= 2:
            return self._linearize_channel(ICRF, ICRF_diff)
        else:
            return self._linearize_single(ICRF, ICRF_diff)

    def _linearize_channel(self, ICRF: ArrayType, ICRF_diff: Optional[ArrayType] = None):
        """
        The linearization implementation for multiple channels.
        Args:
            ICRF: the mapping function.
            ICRF_diff: the derivative of the mapping function.

        Returns:
            A new Measurand object with the linearized values.
        """

        use_std = False
        if self.std is not None and ICRF_diff is not None:
            use_std = True

        if not self.lib.issubdtype(self.val.dtype, self.lib.integer):
            integer_values = self.lib.around(self.val * gs.MAX_DN).astype(self.lib.dtype('uint8'))
        else:
            integer_values = self.val.copy()

        result = ICRF[integer_values[..., None], self.channels]

        if not use_std:
            return self.__class__(result, None)

        result_std = ICRF_diff[integer_values[..., None], self.channels] * self.std

        return self.__class__(result, result_std)

    def _linearize_single(self, ICRF: ArrayType, ICRF_diff: Optional[ArrayType] = None):
        """
        The linearization implementation for a single channel.
        Args:
            ICRF: the mapping function.
            ICRF_diff: the derivative of the mapping function.

        Returns:
            A new Measurand object with the linearized values.
        """
        use_std = False
        if self.std is not None and ICRF_diff is not None:
            use_std = True

        if not self.lib.issubdtype(self.val.dtype, self.lib.integer):
            integer_values = self.lib.around(self.val * gs.MAX_DN).astype(self.lib.dtype('uint8'))
        else:
            integer_values = self.val.copy()
        result = ICRF[integer_values]

        if not use_std:
            return self.__class__(result, None)

        result_std = ICRF_diff[integer_values] * self.std

        return self.__class__(result, result_std)

    def filter_larger_than_by_map(self, map: 'AbstractMeasurand', threshold_value: float):

        large_indices = map.val > threshold_value
        convolved_val_array = self.fn_median_filter(self.val, size=(gs.MEDIAN_FILTER_KERNEL_SIZE, gs.MEDIAN_FILTER_KERNEL_SIZE),
                                                axes=(0, 1), mode='reflect')

        convolved_val_array[~large_indices] = self.val

        convolved_std_array = None
        if self.std is not None:
            convolved_std_array = self.fn_median_filter(self.std, size=(gs.MEDIAN_FILTER_KERNEL_SIZE, gs.MEDIAN_FILTER_KERNEL_SIZE),
                                            axes=(0, 1), mode='reflect')
            convolved_std_array[~large_indices] = self.std

        return self.__class__(convolved_val_array, convolved_std_array)

    def normalize_by_map(self, map: 'AbstractMeasurand'):

        def flat_field_mean(flat_field):
            """
            Calculates the mean brightness of an image inside a centered ROI.

            Returns:
                list of mean image brightness inside ROI for each channel.
            """

            # Define ROI for calculating flat field spatial mean
            ROI_dx = math.floor(gs.IM_SIZE_X * gs.FF_MID_PERCENTAGE)
            ROI_dy = math.floor(gs.IM_SIZE_Y * gs.FF_MID_PERCENTAGE)
            ROI_start_index = (math.floor(1 / gs.FF_MID_PERCENTAGE) - 1) / 2  # Should be an odd number to center on image.

            # Calculate ROI bounds
            x_start, x_end = ROI_start_index * ROI_dx, (ROI_start_index + 1) * ROI_dx
            y_start, y_end = ROI_start_index * ROI_dy, (ROI_start_index + 1) * ROI_dy

            # Slice ROI for all channels and compute the mean across spatial dimensions
            return self.lib.mean(flat_field[x_start:x_end, y_start:y_end, :], axis=(0, 1))

        # Determine flat field means
        flat_field_means = flat_field_mean(map.val)
        flat_field_stds = flat_field_mean(map.std)

        # Compute input image uncertainty
        u_acq_term = (self.std ** 2) / (map.val ** 2)
        u_acq_term *= flat_field_means ** 2

        # Compute flat field uncertainty
        u_ff_term = (self.val ** 2) / (map.val ** 4)
        u_ff_term *= map.std ** 2
        u_ff_term *= flat_field_means ** 2

        # Compute flat field mean uncertainty
        u_ffm_term = (self.val ** 2) / (map.val ** 2)
        u_ffm_term *= flat_field_stds ** 2

        # Update uncertainty with the combined terms
        ret_std = self.lib.sqrt(u_acq_term + u_ff_term + u_ffm_term)

        # Flat field correction (broadcast multiplication for channel-specific scaling)
        ret_val = (self.val / map.val) * flat_field_means

        return self.__class__(ret_val, ret_std)

    def apply_gaussian_weight(self: 'AbstractMeasurand'):
        """
        Weighting function used to give priority to pixels that are more towards the center of the [0,1] range of digital
        values. The form is a general gaussian, for which the peak is set to 1.

        Returns:
            the weighted value and the corresponding derivative value.
        """
        # Normal distribution: np.e ** (-0.5 * ((x-0.5) / 0.1) ** 2)
        y = self.lib.e ** (-30 * (self.val - 0.5) ** 2)
        dydx = -2 * 30 * (self.val - 0.5) * y

        return y, dydx

    @staticmethod
    def compute_difference(x: 'AbstractMeasurand', y: 'AbstractMeasurand', multiplier: float):
        """
        Implements computing the (scaled) difference of two Measurands, including uncertainty propagation.
        Args:
            x: first measurand.
            y: second measurand.
            multiplier: a multiplier applied to the second argument.

        Returns:
            Two new Measurand objects with the first containing the absolute difference and the second containing the
            relative difference.
        """
        cls = x.__class__
        scale_term = multiplier * y.val
        abs_diff = x.val - scale_term
        rel_diff = abs_diff / scale_term

        x_val, y_val = x.val, y.val
        x_std, y_std = 0, 0
        use_std = False
        if x.std is not None:
            use_std = True
            x_std = x.std
        if y.std is not None:
            use_std = True
            y_std = y.std

        abs_std = None
        rel_std = None

        if use_std:
            abs_std = cnp.sqrt(x_std ** 2 + (multiplier * y_std) ** 2)
            rel_std = cnp.sqrt((x_std / (multiplier * y_val)) ** 2 + ((y_std * x_val) / (multiplier * y_val ** 2)) ** 2)

        return cls(abs_diff, abs_std), cls(rel_diff, rel_std)

    @staticmethod
    def interpolate(x0: 'AbstractMeasurand', x1: 'AbstractMeasurand', y0: float, y1: float, y: float):

        cls = x0.__class__
        use_std = False
        if x0.std is not None or x1.std is not None:
            use_std = True

        res = (x0.val * (y1 - y) + x1.val * (y - y0)) / (y1 - y0)

        if not use_std:
            return cls(val=res, std=None)

        if x0.std is None:
            x0_std = 0
        else:
            x0_std = x0.std
        if x1.std is None:
            x1_std = 0
        else:
            x1_std = x1.std

        res_std = cnp.sqrt(x0_std * ((y1 - y) / (y1 - y0)) ** 2 + x1_std * ((y - y0) / (y1 - y0)) ** 2)

        return cls(res, res_std)


class NumpyMeasurand(AbstractMeasurand):
    """
    NumPy version of the base AbstractMeasurand class.
    """

    lib = np
    backend = "numpy"
    fn_median_filter = np_median_filter
    ArrayType = np.ndarray
    InputType = Union[ScalarType, ArrayType]

    def __init__(self, val: Optional[InputType] = None, std: Optional[InputType] = None):

        super().__init__(val, std)
        if val is not None and not isinstance(val, self.InputType):
            raise TypeError('Invalid value type.')
        if std is not None and not isinstance(std, self.InputType):
            raise TypeError('Invalid std type')

        if isinstance(val, ScalarType):
            val = np.array([val], dtype=np.dtype('float64'))

        if isinstance(std, ScalarType):
            std = np.array([std], dtype=np.dtype('float64'))

        if val is not None and std is not None and val.shape != std.shape:
            raise ValueError('Value and std shapes must match.')

        self._val = val
        self._std = std
        self._initialized = True

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
            finite_mask = np.isfinite(channel_values)

            if use_std:
                stds = self.std[..., c]
                non_zeros = stds != 0
                finite_mask = np.logical_and(finite_mask, non_zeros)
                weights = stds[finite_mask]
                weights = 1 / weights
                channel_values = channel_values[finite_mask]
            else:
                weights = None
                channel_values = channel_values[finite_mask]

            if included_range is None:
                x_range = np.linspace(np.min(channel_values), np.max(channel_values), num=data_points)
            else:
                x_range = np.linspace(included_range[0], included_range[1], num=data_points)

            gkde = gaussian_kde(channel_values, 'silverman', weights=weights)
            gkde_result = gkde.evaluate(x_range)

            estimates[c] = (gkde_result, x_range)

        return estimates
