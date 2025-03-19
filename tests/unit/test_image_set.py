"""
Unit tests for the ImageSet class. OpenCV, Measurand class and paths are mocked. NumPy/CuPy is not mocked as it is such
a core part of the package.

TODO: most test are done, but some calibration functions and correction methods lack formal testing.
"""
import pytest
import math
from pathlib import Path
from unittest.mock import patch, MagicMock
from unittest.mock import call
from measurand import AbstractMeasurand
from image_set import ImageSet
from image_set import _features_from_file_name
from hypothesis import strategies as st
from conftest import USE_CUPY, xp, np


@pytest.fixture
def mock_measurand_factory():
    """Fixture to create a mock Measurand object."""

    def _create_mocks(n):
        list_of_mocks = []
        for i in range(n):
            mock = MagicMock(spec=AbstractMeasurand)
            if USE_CUPY:
                mock.backend = "cupy"
            else:
                mock.backend = "numpy"
            list_of_mocks.append(mock)

        if len(list_of_mocks) == 1:
            return list_of_mocks[0]
        else:
            return list_of_mocks

    return _create_mocks


@pytest.fixture
def random_array(request):
    """
    Pytest fixture to generate a NumPy array with random values in the specified range.
    Accepts 'shape', 'min_value', and 'max_value' as parameters.
    """
    # Extract parameters from the request
    shape = request.param.get("shape")
    min_value = request.param.get("min_value", 0)  # Default to 0 if not provided
    max_value = request.param.get("max_value", 1)  # Default to 1 if not provided

    # Validate shape
    if not isinstance(shape, tuple) or not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError("Shape must be a tuple of positive integers.")

    # Validate min and max values
    if not (isinstance(min_value, (int, float)) and isinstance(max_value, (int, float))):
        raise ValueError("min_value and max_value must be numeric.")
    if min_value >= max_value:
        raise ValueError("min_value must be less than max_value.")

    # Generate the random array in the specified range
    random_values = xp.random.rand(*shape)
    scaled_values = random_values * (max_value - min_value) + min_value
    return scaled_values


@pytest.fixture
def generate_ICRFs(request):
    exponent = request.param
    if not isinstance(exponent, float) and not isinstance(exponent, int):
        raise TypeError('Exponent must be int or float.')

    ICRFs = xp.empty((256, 3))
    ICRF_diffs = xp.empty((256, 3))

    for c in range(3):
        ICRFs[:, c] = xp.linspace(0, 1, 256) ** exponent
        ICRF_diffs[:, c] = xp.gradient(ICRFs[:, c], 2 / (256 - 1))

    return ICRFs, ICRF_diffs


def file_name_strategy():
    base_name = st.sampled_from(['image', 'sample', 'test', '1111', 'xxxx', 'YYYY'])
    illumination = st.one_of(st.none(), st.sampled_from(['bf', 'df']))
    magnification = st.one_of(st.none(), st.sampled_from(['1x', '10x', '20x', '50x']))
    exposure = st.one_of(st.none(), st.sampled_from(['1ms', '20ms', '100ms', '59.9ms']))
    has_std = st.booleans()

    file_name_elements = [base_name, illumination, magnification, exposure]

    file_name = ''
    for element in file_name_elements:
        if element is not None:
            file_name = file_name + ' ' + element

    file_name = f'{file_name}.tif'
    if has_std:
        std_file_name = file_name.replace('.tif', ' STD.tif')
    else:
        std_file_name = None

    return file_name, std_file_name


class TestImageSetInitialization:

    def test_imageset_init_with_no_args(self):
        image_set = ImageSet()
        assert image_set.measurand is not None
        assert image_set.measurand.val is None
        assert image_set.measurand.std is None
        assert image_set.path is None
        assert image_set.features is None

    def test_imageset_init_with_mock_measurand(self, mock_measurand_factory):
        """Test ImageSet initialization with a mock Measurand."""
        mock_measurand = mock_measurand_factory(1)
        image_set = ImageSet(measurand=mock_measurand)

        # Ensure the mock was correctly assigned
        assert image_set.measurand == mock_measurand

    def test_multiple_from_path_mock_glob(self):
        # Create a mock Path object
        mock_path = MagicMock(spec=Path)

        # Mock return value for path.glob("*.tif")
        mock_files = [
            Path("image1.tif"),
            Path("image2_STD.tif"),  # This should be excluded by the method
            Path("image3.tif"),
        ]
        mock_path.glob.return_value = mock_files

        # Call the method under test
        image_sets = ImageSet.multiple_from_path(mock_path)

        # Assertions
        assert len(image_sets) == 2  # Only image1.tif and image3.tif should be included
        assert isinstance(image_sets[0], ImageSet)
        assert image_sets[0].path == Path("image1.tif")
        assert image_sets[1].path == Path("image3.tif")

        # Verify that glob was called with "*.tif"
        mock_path.glob.assert_called_once_with("*.tif")


class TestImageSetMockMeasurand:

    @pytest.mark.parametrize("generate_ICRFs", [2.0], indirect=True)
    def test_imageset_linearize(self, generate_ICRFs, mock_measurand_factory):

        ICRF, ICRF_diff = generate_ICRFs
        mock_measurand_1, mock_measurand_2 = mock_measurand_factory(2)

        mock_measurand_1.linearize.return_value = mock_measurand_2

        image_set = ImageSet(measurand=mock_measurand_1)

        result = image_set.linearize(ICRF, ICRF_diff)

        mock_measurand_1.linearize.assert_called_once_with(ICRF, ICRF_diff)
        assert result.measurand == mock_measurand_2

    @patch("measurand.AbstractMeasurand.compute_difference")
    def test_imageset_compute_difference(self, mock_compute_difference, mock_measurand_factory, tmp_path):

        mock_measurand_1, mock_measurand_2, mock_measurand_3, mock_measurand_4 = mock_measurand_factory(4)

        image_set_1 = ImageSet(measurand=mock_measurand_1, file_path=tmp_path.joinpath(r'20ms test_image.tif'))
        image_set_2 = ImageSet(measurand=mock_measurand_2, file_path=tmp_path.joinpath(r'50ms test_image.tif'))

        mock_compute_difference.return_value = (mock_measurand_3, mock_measurand_4)

        absolute_set, relative_set = ImageSet.compute_difference(image_set_1, image_set_2)

        # Assert that the mocked compute difference method is called once with the given expected arguments.
        expected_call = call(mock_measurand_1, mock_measurand_2, 20 / 50)
        assert any((call_.args[0], expected_call.args[0]) and (call_.args[1], expected_call.args[1]) and
                   math.isclose(call_.args[2], expected_call.args[2], rel_tol=1e-6)
                   for call_ in mock_compute_difference.call_args_list
                   )

        # Assert that the resulting ImageSet objects have the desired mock Measurands set into the measurand attributes.
        assert absolute_set.measurand == mock_measurand_3
        assert relative_set.measurand == mock_measurand_4

    @patch("measurand.AbstractMeasurand.interpolate")
    def test_imageset_exposure_interpolation(self, mock_interpolate, mock_measurand_factory, tmp_path):

        mock_measurand_1, mock_measurand_2, mock_measurand_3 = mock_measurand_factory(3)

        image_set_1 = ImageSet(measurand=mock_measurand_1, file_path=tmp_path.joinpath(r'20ms test_image.tif'))
        image_set_2 = ImageSet(measurand=mock_measurand_2, file_path=tmp_path.joinpath(r'50ms test_image.tif'))
        interpolation_point = 0.04

        mock_interpolate.return_value = mock_measurand_3

        result = ImageSet.exposure_interpolation(image_set_1, image_set_2, interpolation_point)

        mock_interpolate.assert_called_once_with(mock_measurand_1, mock_measurand_2, 0.02, 0.05, interpolation_point)
        assert result.measurand == mock_measurand_3

    def test_imageset_extract(self, mock_measurand_factory):

        mock_measurand_1, mock_measurand_2 = mock_measurand_factory(2)
        mock_measurand_1.extract.return_value = mock_measurand_2

        image_set = ImageSet(measurand=mock_measurand_1)

        result = image_set.extract()

        mock_measurand_1.extract.assert_called_once()
        assert result.measurand == mock_measurand_2


class TestImageSetIO:

    def test_load_value_image_8bit(self, mock_measurand_factory):
        dummy_image = xp.ones((3, 3, 3), dtype=xp.uint8) * 128
        mock_measurand = mock_measurand_factory(1)

        with patch('cv2.imread', return_value=dummy_image) as mock_imread:
            # Initialize the ImageSet object
            img_set = ImageSet(file_path=Path("dummy/path/image.tif"))

            # Mock the measurand after instantiation
            img_set.measurand = mock_measurand
            img_set.load_value_image(bit64=False)

            # Assertions
            xp.testing.assert_array_equal(img_set.measurand.val, dummy_image.astype(xp.float64) / 255)

    def test_load_value_image_64bit(self, mock_measurand_factory):
        dummy_image = xp.ones((3, 3, 3), dtype=np.uint8) * 128
        mock_measurand = mock_measurand_factory(1)

        with patch('cv2.imread', return_value=dummy_image) as mock_imread:
            # Initialize the ImageSet object
            img_set = ImageSet(file_path=Path("dummy/path/image.tif"))

            # Mock the measurand after instantiation
            img_set.measurand = mock_measurand
            img_set.load_value_image(bit64=True)

            # Assertions
            xp.testing.assert_allclose(img_set.measurand.val, dummy_image.astype(xp.float64))

    def test_load_std_image_not_found(self, mock_measurand_factory):

        dummy_path = Path("dummy/path/image.tif")
        dummy_std_path = Path("dummy/path/image STD.tif")
        mock_measurand = mock_measurand_factory(1)

        with patch('cv2.imread', return_value=None) as mock_imread:
            with patch.object(ImageSet, 'calculate_numerical_STD') as mock_calculate_numerical_STD:
                # Initialize the ImageSet object
                img_set = ImageSet(file_path=dummy_path)

                # Mock the measurand after instantiation
                img_set.measurand = mock_measurand

                img_set.load_std_image(bit64=True)
                mock_imread.assert_called_once()
                mock_calculate_numerical_STD.assert_called_once()

    def test_load_std_image_found(self, mock_measurand_factory):

        dummy_path = Path("dummy/path/image.tif")
        dummy_image = np.ones((3, 3, 3), dtype=np.float64)
        mock_measurand = mock_measurand_factory(1)

        with patch('cv2.imread', return_value=dummy_image) as mock_imread:
            # Initialize the ImageSet object
            img_set = ImageSet(file_path=dummy_path)

            # Mock the measurand after instantiation
            img_set.measurand = mock_measurand

            img_set.load_std_image(bit64=True)
            mock_imread.assert_called_once()
            xp.testing.assert_allclose(img_set.measurand.std, xp.array(dummy_image))

    @pytest.mark.parametrize("generate_ICRFs", [2.0], indirect=True)
    def test_calculate_numerical_STD(self, generate_ICRFs, mock_measurand_factory):

        ICRF, _ = generate_ICRFs
        channels = ICRF.shape[-1]
        dummy_path = Path("dummy/path/image.tif")
        dummy_image = xp.ones((3, 3, 3), dtype=xp.float64) / 100
        mock_measurand = mock_measurand_factory(1)
        # The STD_array has the exact same format as ICRFs, so we can use the ICRF fixture here.
        with patch('general_functions.read_txt_to_array', return_value=ICRF) as mock_read_data_from_txt:
            image_set = ImageSet(dummy_path)
            image_set.measurand = mock_measurand
            image_set.measurand.val = dummy_image
            result = image_set.calculate_numerical_STD()

            mock_read_data_from_txt.assert_called_once()
            for c in range(channels):
                assert xp.isin(result[..., c], ICRF[..., c]).all() == xp.array(True)

    def test_calculate_numerical_STD_not_found(self):

        dummy_path = Path("dummy/path/image.tif")
        with patch('general_functions.read_txt_to_array', side_effect=FileNotFoundError) as mock_read_data_from_txt:
            image_set = ImageSet(dummy_path)
            result = image_set.calculate_numerical_STD()
            mock_read_data_from_txt.assert_called_once()
            assert result is None


class TestSupportFunctions:

    @pytest.mark.parametrize("file_path, expected_features", [
        (Path("bf 40x 100ms sample.tif"),
         {"illumination": "bf", "magnification": "40x", "exposure": 0.1, "subject": "sample"}),
        (Path("df 20x 200ms test_subject.tif"),
         {"illumination": "df", "magnification": "20x", "exposure": 0.2, "subject": "test_subject"}),
        (Path("40x 500ms subject1.tif"), {"illumination": "", "magnification": "40x", "exposure": 0.5, "subject": "subject1"}),
        (Path("bf 1000ms.tif"), {"illumination": "bf", "magnification": "", "exposure": 1.0, "subject": ""}),
        (Path("sample 10x.tif"), {"illumination": "", "magnification": "10x", "exposure": 0, "subject": "sample"}),
    ])
    def test_features_from_file_name(self, file_path, expected_features):
        assert _features_from_file_name(file_path) == expected_features

    def test_is_exposure_match(self):
        # Test 1: Both ImageSets with matching features (except 'exposure')
        image_set1 = ImageSet(features={"illumination": "bf", "magnification": "40x", "subject": "sample"})
        image_set2 = ImageSet(features={"illumination": "bf", "magnification": "40x", "subject": "sample"})

        assert image_set1.is_exposure_match(image_set2) is True

        # Test 2: One ImageSet with None features
        image_set1 = ImageSet(features=None)
        image_set2 = ImageSet(features={"illumination": "bf", "magnification": "40x", "subject": "sample"})

        assert image_set1.is_exposure_match(image_set2) is False

        # Test 3: Both ImageSets with different features (except 'exposure')
        image_set1 = ImageSet(features={"illumination": "bf", "magnification": "40x", "subject": "sample"})
        image_set2 = ImageSet(features={"illumination": "df", "magnification": "20x", "subject": "test"})

        assert image_set1.is_exposure_match(image_set2) is False

        # Test 4: Both ImageSets with the same features, but different 'exposure' values
        image_set1 = ImageSet(
            features={"illumination": "bf", "magnification": "40x", "subject": "sample", "exposure": 0.1})
        image_set2 = ImageSet(
            features={"illumination": "bf", "magnification": "40x", "subject": "sample", "exposure": 0.2})

        # Even though the 'exposure' values differ, the method ignores it
        assert image_set1.is_exposure_match(image_set2) is True

        # Test 5: Both ImageSets with None features
        image_set1 = ImageSet(features=None)
        image_set2 = ImageSet(features=None)

        assert image_set1.is_exposure_match(image_set2) is False  # No features to compare, should return False
