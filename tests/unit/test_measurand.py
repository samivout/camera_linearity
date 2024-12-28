"""
Unit tests for the Measurand class. This class is highly dependent on NumPy/CuPy, so these are not mocked.
"""
import pytest
from copy import deepcopy
from hypothesis import given, settings
from hypothesis import strategies as st
from measurand import Measurand

from cupy_wrapper import get_array_libraries

np, cp, using_cupy = get_array_libraries()
cnp = cp if using_cupy else np


def generate_ICRFs(max_dims=5, max_DN=256):

    ICRFs = cnp.empty((max_DN, max_dims))
    ICRF_diffs = cnp.empty((max_DN, max_dims))

    for c in range(max_dims):
        ICRFs[:, c] = cnp.linspace(0, 1, max_DN) ** (c + 1)
        ICRF_diffs[:, c] = cnp.gradient(ICRFs[:, c], 2 / (max_DN - 1))

    return ICRFs, ICRF_diffs


@st.composite
def broadcastable_arrays(draw, max_dims=5, max_side=10, dtype=cnp.float64):
    """Generates two broadcastable n-dimensional arrays."""

    # Generate number of dimensions for each array
    ndims1 = draw(st.integers(1, max_dims))
    ndims2 = draw(st.integers(1, max_dims))

    # Generate shapes for each array
    base_shape1 = draw(st.lists(st.integers(1, max_side), min_size=ndims1, max_size=ndims1))
    base_shape2 = draw(st.lists(st.integers(1, max_side), min_size=ndims2, max_size=ndims2))

    # Pad shapes to make them broadcastable (extend with 1's to align dimensions)
    max_len = max(ndims1, ndims2)
    padded_shape1 = [1] * (max_len - ndims1) + base_shape1
    padded_shape2 = [1] * (max_len - ndims2) + base_shape2

    # Now make sure shapes are compatible by broadcasting them
    for i in range(len(padded_shape1)):
        if padded_shape1[i] != padded_shape2[i]:
            # If dimensions are not the same, one of them should be 1 for broadcasting to work
            if padded_shape1[i] != 1 and padded_shape2[i] != 1:
                # If both dimensions are not 1, make one of them 1 for broadcasting
                if padded_shape1[i] > padded_shape2[i]:
                    padded_shape2[i] = 1
                else:
                    padded_shape1[i] = 1

    # Generate the arrays using the broadcastable shapes
    array1 = cnp.random.rand(*padded_shape1).astype(dtype)
    array1 /= cnp.max(array1)
    array2 = cnp.random.rand(*padded_shape2).astype(dtype)
    array2 /= cnp.max(array2)

    return array1, array2


@st.composite
def broadcastable_measurands(draw, max_dims, max_side, dtype):
    value1, value2 = draw(broadcastable_arrays(max_dims=max_dims, max_side=max_side, dtype=dtype))

    std1 = draw(st.one_of(st.none(), st.just(value1.copy() * 0.1)))
    std2 = draw(st.one_of(st.none(), st.just(value2.copy() * 0.1)))

    measurand1 = Measurand(value1, std1)
    measurand2 = Measurand(value2, std2)

    return measurand1, measurand2


@st.composite
def broadcastable_measurands_and_ICRFs(draw, max_dims, max_side, dtype):
    value1, value2 = draw(broadcastable_arrays(max_dims=max_dims, max_side=max_side, dtype=dtype))

    std1 = draw(st.one_of(st.none(), st.just(value1.copy() * 0.1)))
    std2 = draw(st.one_of(st.none(), st.just(value2.copy() * 0.1)))

    measurand1 = Measurand(value1, std1)
    measurand2 = Measurand(value2, std2)

    channels = measurand1.val.shape[-1]

    ICRF, ICRF_diff = generate_ICRFs(max_dims=channels, max_DN=256)

    return measurand1, measurand2, ICRF, ICRF_diff


@st.composite
def measurand_and_limits(draw, max_dims, max_side, dtype):
    measurand1, _ = draw(broadcastable_measurands(max_dims, max_side, dtype))

    list_size = measurand1.val.shape[-1]
    threshold = draw(st.floats(min_value=0.25, max_value=0.75))

    lower_limits = draw(st.lists(
        st.one_of(
            st.floats(min_value=0.0, max_value=threshold),
            st.just(None)
        ),
        min_size=list_size,
        max_size=list_size
    ))
    upper_limits = draw(st.lists(
        st.one_of(
            st.floats(min_value=threshold, max_value=1.0),
            st.just(None)
        ),
        min_size=list_size,
        max_size=list_size
    ))

    return measurand1, lower_limits, upper_limits


class TestMeasurandInitialization:

    def test_initialize_with_float_value(self):
        """Test initializing Measurand with a float value and no std"""
        measurand = Measurand(10.0)
        assert isinstance(measurand.val, cnp.ndarray)
        assert measurand.val == cnp.array(10.0, dtype=cnp.dtype('float64'))
        assert measurand.std is None

    def test_initialize_with_float_value_and_std(self):
        """Test initializing Measurand with a float value and a float std"""
        measurand = Measurand(10.0, 1.0)
        assert isinstance(measurand.val, cnp.ndarray)
        assert measurand.val == cnp.array(10.0, dtype=cnp.dtype('float64'))
        assert isinstance(measurand.std, cnp.ndarray)
        assert measurand.std == cnp.array(1.0, dtype=cnp.dtype('float64'))

    def test_initialize_with_array_value(self):
        """Test initializing Measurand with an array value and no std"""
        value = cnp.array([10.0, 20.0])
        measurand = Measurand(value)
        assert isinstance(measurand.val, cnp.ndarray)
        assert cnp.array_equal(measurand.val, value)
        assert measurand.std is None

    def test_initialize_with_array_value_and_std(self):
        """Test initializing Measurand with an array value and an array std"""
        value = cnp.array([10.0, 20.0])
        std = cnp.array([1.0, 2.0])
        measurand = Measurand(value, std)
        assert isinstance(measurand.val, cnp.ndarray)
        assert cnp.array_equal(measurand.val, value)
        assert isinstance(measurand.std, cnp.ndarray)
        assert cnp.array_equal(measurand.std, std)

    def test_initialize_with_invalid_value_type(self):
        """Test initializing Measurand with an invalid value type"""
        with pytest.raises(TypeError, match="Invalid value type"):
            Measurand("invalid_val", 1.0)  # Invalid std type, should be float or array

    def test_initialize_with_invalid_std_type(self):
        """Test initializing Measurand with an invalid std type"""
        with pytest.raises(TypeError, match="Invalid std type"):
            Measurand(10.0, "invalid_std")  # Invalid std type, should be float or array

    def test_initialize_with_none_std(self):
        """Test initializing Measurand with no std (std=None)"""
        measurand = Measurand(10.0, None)
        assert isinstance(measurand.val, cnp.ndarray)
        assert measurand.val == cnp.array(10.0, dtype=cnp.dtype('float64'))
        assert measurand.std is None


class TestMeasurandAddition:

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_commutativity(self, measurands):
        measurand1, measurand2 = measurands
        result1 = measurand1 + measurand2
        result2 = measurand2 + measurand1
        assert cnp.allclose(result1.val, result2.val, atol=1.e-8)
        if measurand1.std is not None or measurand2.std is not None:
            assert cnp.allclose(result1.std, result2.std, atol=1.e-8)
        else:
            assert result1.std is None and result2.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_associativity(self, measurands):
        measurand1, measurand2 = measurands
        measurand3 = deepcopy(measurand1)
        measurand1_2 = measurand1 + measurand2
        measurand2_3 = measurand2 + measurand3
        result1 = measurand1_2 + measurand3
        result2 = measurand2_3 + measurand1
        assert cnp.allclose(result1.val, result2.val, atol=1.e-8)
        if measurand1.std is not None or measurand2.std is not None:
            assert cnp.allclose(result1.std, result2.std, atol=1.e-8)
        else:
            assert result1.std is None and result2.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_identity(self, measurands):
        measurand1, _ = measurands
        result = measurand1 + 0
        assert cnp.allclose(result.val, measurand1.val, atol=1.e-8)
        if measurand1.std is not None:
            assert cnp.allclose(result.std, measurand1.std, atol=1.e-8)
        else:
            assert result.std is None


class TestMeasurandSubtraction:

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_commutativity(self, measurands):
        measurand1, measurand2 = measurands
        result1 = measurand1 - measurand2
        result2 = measurand2 - measurand1
        assert cnp.allclose(result1.val, -1 * result2.val, atol=1.e-8)
        if measurand1.std is not None or measurand2.std is not None:
            assert cnp.allclose(result1.std, result2.std, atol=1.e-8)
        else:
            assert result1.std is None and result2.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_self_subtraction(self, measurands):
        measurand1, _ = measurands
        result = measurand1 - measurand1
        assert cnp.allclose(result.val, cnp.zeros_like(result.val), atol=1.e-8)
        if measurand1.std is not None:
            assert cnp.all(result.std >= measurand1.std)
        else:
            assert result.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_identity(self, measurands):
        measurand1, _ = measurands
        result = measurand1 - 0
        assert cnp.allclose(result.val, measurand1.val, atol=1.e-8)
        if measurand1.std is not None:
            assert cnp.allclose(result.std, measurand1.std, atol=1.e-8)
        else:
            assert result.std is None


class TestMeasurandDivision:

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_inversion(self, measurands):
        measurand1, measurand2 = measurands
        result1 = measurand1 / measurand2
        result2 = measurand2 / measurand1
        assert cnp.allclose(result1.val, 1 / result2.val, atol=1.e-8)
        if measurand1.std is not None or measurand2.std is not None:
            assert result1.std is not None
            assert result2.std is not None
        else:
            assert result1.std is None and result2.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_distributivity(self, measurands):
        measurand1, measurand2 = measurands
        measurand3 = deepcopy(measurand1)
        result1 = (measurand1 + measurand2) / measurand3
        result2 = measurand1 / measurand3 + measurand2 / measurand3
        assert cnp.allclose(result1.val, result2.val, atol=1.e-8)
        if measurand1.std is not None or measurand2.std is not None:
            assert result1.std is not None
            assert result2.std is not None
        else:
            assert result1.std is None and result2.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_identity(self, measurands):
        measurand1, _ = measurands
        result = measurand1 / 1
        assert cnp.allclose(result.val, measurand1.val, atol=1.e-8)
        if measurand1.std is not None:
            assert cnp.allclose(result.std, measurand1.std, atol=1.e-8)
        else:
            assert result.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_unity_property(self, measurands):
        measurand1, _ = measurands
        result = measurand1 / measurand1
        expected_val = cnp.ones_like(measurand1.val)
        assert cnp.allclose(result.val, expected_val, atol=1.e-8)
        if measurand1.std is not None:
            assert result.std is not None
        else:
            assert result.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_division_by_zero(self, measurands):
        measurand1, _ = measurands
        result = measurand1 / 0
        expected_val = cnp.ones_like(measurand1.val) * cnp.inf
        assert cnp.allclose(result.val, expected_val, atol=1.e-8)
        if measurand1.std is not None:
            assert result.std is not None
        else:
            assert result.std is None


class TestMeasurandMultiplication:

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_commutativity(self, measurands):
        measurand1, measurand2 = measurands
        result1 = measurand1 * measurand2
        result2 = measurand2 * measurand1
        assert cnp.allclose(result1.val, result2.val, atol=1.e-8)
        if measurand1.std is not None or measurand2.std is not None:
            assert cnp.allclose(result1.std, result2.std, atol=1.e-8)
        else:
            assert result1.std is None and result2.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_associativity(self, measurands):
        measurand1, measurand2 = measurands
        measurand3 = deepcopy(measurand1)
        measurand1_2 = measurand1 * measurand2
        measurand2_3 = measurand2 * measurand3
        result1 = measurand1_2 * measurand3
        result2 = measurand2_3 * measurand1
        assert cnp.allclose(result1.val, result2.val, atol=1.e-8)
        if measurand1.std is not None or measurand2.std is not None:
            assert cnp.allclose(result1.std, result2.std, atol=1.e-8)
        else:
            assert result1.std is None and result2.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_distributivity(self, measurands):
        measurand1, measurand2 = measurands
        measurand3 = deepcopy(measurand1)
        result1 = measurand1 * (measurand2 + measurand3)
        result2 = (measurand1 * measurand2) + (measurand1 * measurand3)
        assert cnp.allclose(result1.val, result2.val, atol=1.e-8)
        if measurand1.std is not None or measurand2.std is not None:
            assert result1.std is not None
            assert result2.std is not None
        else:
            assert result1.std is None and result2.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_identity(self, measurands):
        measurand1, _ = measurands
        result = measurand1 * 1
        assert cnp.allclose(result.val, measurand1.val, atol=1.e-8)
        if measurand1.std is not None:
            assert cnp.allclose(result.std, measurand1.std, atol=1.e-8)
        else:
            assert result.std is None

    @settings(deadline=None)
    @given(broadcastable_measurands(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_broadcastable_measurand_zero_property(self, measurands):
        measurand1, _ = measurands
        result = measurand1 * 0
        expected_val = cnp.zeros_like(measurand1.val)
        assert cnp.allclose(result.val, expected_val, atol=1.e-8)
        if measurand1.std is not None:
            expected_std = cnp.zeros_like(measurand1.std)
            assert cnp.allclose(result.std, expected_std, atol=1.e-8)
        else:
            assert result.std is None


class TestNormalizeInput:

    def test_other_is_measurand(self):
        # Create two Measurand instances
        m1 = Measurand(10.0)
        m2 = Measurand(20.0)

        normalized_other, use_std = m1._normalize_input(m2)

        assert normalized_other == m2
        assert use_std is False  # Neither has std, so use_std should be False

    def test_other_is_measurand_with_std(self):
        # Create Measurand instances with std
        m1 = Measurand(10.0, 1.0)
        m2 = Measurand(20.0, 2.0)

        normalized_other, use_std = m1._normalize_input(m2)

        assert normalized_other == m2
        assert use_std is True  # One of the Measurands has a std, so use_std should be True

    def test_other_is_float(self):
        m = Measurand(10.0, 1.0)  # Measurand with std
        normalized_other, use_std = m._normalize_input(20.0)  # Passing float as other

        assert isinstance(normalized_other, Measurand)
        assert normalized_other.val == 20.0
        assert normalized_other.std is None  # std is not provided
        assert use_std is True  # Since m has a std, use_std should be True

    def test_other_is_cnp_array(self):
        m = Measurand(10.0, 1.0)  # Measurand with std
        normalized_other, use_std = m._normalize_input(cnp.array([1, 2, 3]))  # Passing numpy array as other

        assert isinstance(normalized_other, Measurand)
        assert cnp.array_equal(normalized_other.val, cnp.array([1, 2, 3]))
        assert normalized_other.std is None  # std is not provided
        assert use_std is True  # Since m has a std, use_std should be True

    def test_invalid_other_type_raises_type_error(self):
        m = Measurand(10.0)

        with pytest.raises(TypeError, match="Invalid other type."):
            m._normalize_input("invalid_string")  # Invalid type for 'other'

    def test_other_is_measurand_with_std_and_use_std(self):
        # Test Measurand instance with std
        m1 = Measurand(10.0, 1.0)
        m2 = Measurand(20.0, 2.0)

        normalized_other, use_std = m1._normalize_input(m2)

        assert normalized_other == m2
        assert use_std is True  # Both have std, use_std should be True

    def test_other_is_cnp_array_and_use_std(self):
        m = Measurand(10.0, 1.0)  # Measurand with std
        normalized_other, use_std = m._normalize_input(cnp.array([1, 2, 3]))  # Passing numpy array as other

        assert isinstance(normalized_other, Measurand)
        assert cnp.array_equal(normalized_other.val, cnp.array([1, 2, 3]))
        assert normalized_other.std is None  # std is not provided
        assert use_std is True  # Since m has a std, use_std should be True


class TestMeasurandLinearize:

    @settings(deadline=None)
    @given(broadcastable_measurands_and_ICRFs(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_linearize(self, params):

        measurand, _, ICRF, ICRF_diff = params

        channels = ICRF.shape[-1]

        # Lazy sectioning for the arguments based on whether there is one channel or more.
        if channels == 1:
            linearized_measurand = measurand.linearize(ICRF[:, 0], ICRF_diff[:, 0])
        else:
            linearized_measurand = measurand.linearize(ICRF, ICRF_diff)

        # Each slice of the final dimension should contain only values from the ICRF of equivalent index. Similar logic
        # is also used in the uncertainty computation, which is a trivial multiplication on top of this logic, no need
        # to test it separately.
        for c in range(channels):
            assert cnp.isin(linearized_measurand.val[..., c], ICRF[..., c]).all() == cnp.array(True)


class TestMeasurandApplyThreshold:

    @settings(deadline=None)
    @given(measurand_and_limits(max_dims=5, max_side=10, dtype=cnp.float64))
    def test_apply_thresholds_regression(self, params):
        """
        A simpler previously working version of the thresholding method is used to test against the current version.

        """
        measurand, lower_limits, upper_limits = params
        test_measurand = deepcopy(measurand)
        test_measurand_copy = deepcopy(test_measurand)
        test_measurand.apply_thresholds(lower_limits, upper_limits)

        channels = test_measurand_copy.val.shape[-1]
        ndims = test_measurand_copy.val.ndim

        if ndims < 2:
            for c in range(channels):
                if lower_limits[c] is not None:
                    lower = lower_limits[c]
                else:
                    lower = -cnp.inf
                if upper_limits[c] is not None:
                    upper = upper_limits[c]
                else:
                    upper = cnp.inf

                mask = (test_measurand_copy.val[c] < lower) | (test_measurand_copy.val[c] > upper)
                if mask:
                    test_measurand_copy.val[c] = cnp.nan
                    if test_measurand_copy.std is not None:
                        test_measurand_copy.std[c] = cnp.nan
        else:
            for c in range(channels):
                if lower_limits[c] is not None:
                    lower = lower_limits[c]
                else:
                    lower = -cnp.inf
                if upper_limits[c] is not None:
                    upper = upper_limits[c]
                else:
                    upper = cnp.inf
                vals = test_measurand_copy.val[..., c]
                mask = (vals < lower) | (vals > upper)
                vals[mask] = cnp.nan
                if test_measurand_copy.std is not None:
                    stds = test_measurand_copy.std[..., c]
                    stds[mask] = cnp.nan

        assert cnp.allclose(test_measurand.val, test_measurand_copy.val, equal_nan=True)
        if measurand.std is not None:
            assert cnp.allclose(test_measurand.std, test_measurand_copy.std, equal_nan=True)
