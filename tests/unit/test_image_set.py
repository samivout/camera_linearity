"""
Unit tests for the ImageSet class. OpenCV, Measurand class and paths are mocked. NumPy/CuPy is not mocked as it is such
a core part of the package.

TODO: most test are done, but some calibration functions and correction methods lack formal testing.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from image_set import ImageSet
from image_set import _features_from_file_name
from hypothesis import strategies as st

from cupy_wrapper import get_array_libraries

np, cp, using_cupy = get_array_libraries()
cnp = cp if using_cupy else np


@pytest.fixture
def mock_measurand():
    """Fixture to create a mock Measurand object."""
    mock = Mock() # Mock the result of linearize
    mock.interpolate.return_value = Mock()
    mock.compute_difference.return_value = Mock()
    return mock


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
    random_values = cnp.random.rand(*shape)
    scaled_values = random_values * (max_value - min_value) + min_value
    return scaled_values


@pytest.fixture
def generate_ICRFs(request):

    exponent = request.param
    if not isinstance(exponent, float) and not isinstance(exponent, int):
        raise TypeError('Exponent must be int or float.')

    ICRFs = cnp.empty((256, 3))
    ICRF_diffs = cnp.empty((256, 3))

    for c in range(3):
        ICRFs[:, c] = cnp.linspace(0, 1, 256) ** exponent
        ICRF_diffs[:, c] = cnp.gradient(ICRFs[:, c], 2 / (256 - 1))

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



class MockMeasurand:
    def __init__(self, val, std=None):
        self.val = val
        self.std = std


class TestImageSetInitialization:

    def test_imageset_init_with_no_args(self):

        image_set = ImageSet()
        assert image_set.measurand is not None
        assert image_set.measurand.val is None
        assert image_set.measurand.std is None
        assert image_set.path is None
        assert image_set.features is None
        assert image_set.channels is None

    def test_imageset_init_with_mock_measurand(self, mock_measurand):
        """Test ImageSet initialization with a mock Measurand."""
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
    def test_imageset_linearize(self, generate_ICRFs):

        ICRF, ICRF_diff = generate_ICRFs
        mock_measurand = MagicMock()
        mock_measurand.linearize.return_value = "mocked_linearized_measurand"

        image_set = ImageSet(measurand=mock_measurand)

        result = image_set.linearize(ICRF, ICRF_diff)
        mock_measurand.linearize.assert_called_once_with(ICRF, ICRF_diff)

        assert result.measurand == "mocked_linearized_measurand"

    @patch("measurand.Measurand.compute_difference")
    def test_imageset_compute_difference(self, mock_compute_difference, mock_measurand, tmp_path):

        image_set_1 = ImageSet(measurand=mock_measurand, file_path=tmp_path.joinpath(r'20ms test_image.tif'))
        image_set_2 = ImageSet(measurand=mock_measurand, file_path=tmp_path.joinpath(r'50ms test_image.tif'))

        mock_abs_measurand = MockMeasurand(val=50, std=3)
        mock_rel_measurand = MockMeasurand(val=0.25, std=0.02)
        mock_compute_difference.return_value = (mock_abs_measurand, mock_rel_measurand)

        absolute_set, relative_set = ImageSet.compute_difference(image_set_1, image_set_2)

        assert absolute_set.measurand.val == mock_abs_measurand.val
        assert absolute_set.measurand.std == mock_abs_measurand.std
        assert relative_set.measurand.val == mock_rel_measurand.val
        assert relative_set.measurand.std == mock_rel_measurand.std

    @patch("measurand.Measurand.interpolate")
    def test_imageset_exposure_interpolation(self, mock_interpolate, mock_measurand, tmp_path):
        image_set_1 = ImageSet(measurand=mock_measurand, file_path=tmp_path.joinpath(r'20ms test_image.tif'))
        image_set_2 = ImageSet(measurand=mock_measurand, file_path=tmp_path.joinpath(r'50ms test_image.tif'))

        mock_interpolated_measurand = MockMeasurand(val=50, std=3)
        mock_interpolate.return_value = mock_interpolated_measurand

        result = ImageSet.exposure_interpolation(image_set_1, image_set_2, 0.04)

        assert result.measurand.val == mock_interpolated_measurand.val
        assert result.measurand.std == mock_interpolated_measurand.std

    def test_imageset_extract(self):

        mock_measurand = MagicMock()
        mock_measurand.extract.return_value = "mocked_extracted_measurand"

        image_set = ImageSet(measurand=mock_measurand)

        result = image_set.extract()
        mock_measurand.extract.assert_called_once()

        assert result.measurand == "mocked_extracted_measurand"


class TestImageSetIO:

    def test_load_value_image_8bit(self):
        dummy_image = np.ones((3, 3, 3), dtype=np.uint8) * 128

        with patch('cv2.imread', return_value=dummy_image) as mock_imread:
            # Initialize the ImageSet object
            img_set = ImageSet(file_path=Path("dummy/path/image.tif"))

            # Mock the measurand after instantiation
            img_set.measurand = MagicMock()

            img_set.load_value_image(bit64=False, use_cupy=False)

            # Assertions
            np.testing.assert_array_equal(img_set.measurand.val, dummy_image.astype(np.float64) / 255)
            assert img_set.channels == [0, 1, 2]  # Simulated 3D array

    def test_load_value_image_64bit(self):
        dummy_image = np.ones((3, 3, 3), dtype=np.uint8) * 128

        with patch('cv2.imread', return_value=dummy_image) as mock_imread:
            # Initialize the ImageSet object
            img_set = ImageSet(file_path=Path("dummy/path/image.tif"))

            # Mock the measurand after instantiation
            img_set.measurand = MagicMock()

            img_set.load_value_image(bit64=True, use_cupy=False)

            # Assertions
            np.testing.assert_allclose(img_set.measurand.val, dummy_image.astype(np.float64))
            assert img_set.channels == [0, 1, 2]  # Simulated 3D array

    def test_load_value_image_cupy(self):

        if using_cupy:
            dummy_image = np.ones((3, 3, 3), dtype=np.uint8) * 128
            dummy_path = Path("dummy/path/image.tif")

            with patch('cv2.imread', return_value=dummy_image) as mock_imread:
                # Initialize the ImageSet object
                img_set = ImageSet(file_path=dummy_path)

                # Mock the measurand after instantiation
                img_set.measurand = MagicMock()

                img_set.load_value_image(bit64=True, use_cupy=True)
                mock_imread.assert_called_once_with(str(dummy_path), -1)

                cp.testing.assert_allclose(img_set.measurand.val, dummy_image.astype(np.float64))
        else:
            pass

    def test_load_std_image_not_found(self):

        dummy_path = Path("dummy/path/image.tif")
        dummy_std_path = Path("dummy/path/image STD.tif")

        with patch('cv2.imread', return_value=None) as mock_imread:
            with patch.object(ImageSet, 'calculate_numerical_STD') as mock_calculate_numerical_STD:

                # Initialize the ImageSet object
                img_set = ImageSet(file_path=dummy_path)

                # Mock the measurand after instantiation
                img_set.measurand = MagicMock()

                img_set.load_std_image(bit64=True, use_cupy=True)
                mock_imread.assert_called_once()
                mock_calculate_numerical_STD.assert_called_once()

    def test_load_std_image_found(self):

        dummy_path = Path("dummy/path/image.tif")
        dummy_image = np.ones((3, 3, 3), dtype=np.float64)

        with patch('cv2.imread', return_value=dummy_image) as mock_imread:

            # Initialize the ImageSet object
            img_set = ImageSet(file_path=dummy_path)

            # Mock the measurand after instantiation
            img_set.measurand = MagicMock()

            img_set.load_std_image(bit64=True, use_cupy=True)
            mock_imread.assert_called_once()
            cnp.testing.assert_allclose(img_set.measurand.std, cnp.array(dummy_image))

    @pytest.mark.parametrize("generate_ICRFs", [2.0], indirect=True)
    def test_calculate_numerical_STD(self, generate_ICRFs):

        ICRF, _ = generate_ICRFs
        channels = ICRF.shape[-1]
        dummy_path = Path("dummy/path/image.tif")
        dummy_image = cnp.ones((3, 3, 3), dtype=cnp.float64) / 100
        # The STD_array has the exact same format as ICRFs, so we can use the ICRF fixture here.
        with patch('read_data.read_data_from_txt', return_value=ICRF) as mock_read_data_from_txt:

            image_set = ImageSet(dummy_path)
            image_set.measurand = MagicMock()
            image_set.measurand.val = dummy_image
            result = image_set.calculate_numerical_STD()

            mock_read_data_from_txt.assert_called_once()
            for c in range(channels):
                assert cnp.isin(result[..., c], ICRF[..., c]).all() == cnp.array(True)

    def test_calculate_numerical_STD_not_found(self):

        dummy_path = Path("dummy/path/image.tif")
        with patch('read_data.read_data_from_txt', side_effect=FileNotFoundError) as mock_read_data_from_txt:
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
        (Path("40x 500ms subject1.tif"), {"magnification": "40x", "exposure": 0.5, "subject": "subject1"}),
        (Path("bf 1000ms.tif"), {"illumination": "bf", "exposure": 1.0, "subject": ""}),
        (Path("sample 10x.tif"), {"magnification": "10x", "subject": "sample"}),
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
