"""
Unit tests for the ExposureSeries class. The ImageSet and Measurand classes are mocked in these tests.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from exposure_series import ExposureSeries
from exposure_series import ExposurePair
from image_set import ImageSet


# Test case using pytest and mocks
@pytest.fixture
def mock_image_sets():
    # Create mock instances of ImageSet with paths and features
    mock_image_set_1 = MagicMock(spec=ImageSet)
    mock_image_set_1.features = {"exposure": 100}
    mock_image_set_2 = MagicMock(spec=ImageSet)
    mock_image_set_2.features = {"exposure": 200}
    mock_image_set_3 = MagicMock(spec=ImageSet)
    mock_image_set_3.features = {"exposure": 50}

    return mock_image_set_1, mock_image_set_2, mock_image_set_3


class TestExposureSeriesInitialization:

    def test_initialization_with_all_args(self, mock_image_sets, tmp_path):
        # Given: Mock data for ImageSets
        mock_image_set_1, mock_image_set_2, _ = mock_image_sets
        directory_path = tmp_path.joinpath("/mock/directory")

        # When: Initialize ExposureSeries with all parameters
        exposure_series = ExposureSeries(
            merged_image_set=mock_image_set_1,
            directory_path=directory_path,
            input_image_sets=[mock_image_set_1, mock_image_set_2]
        )

        # Then: Check initialization
        assert exposure_series.merged_image_set == mock_image_set_1
        assert exposure_series.directory_path == directory_path
        assert exposure_series.input_image_sets == [mock_image_set_1, mock_image_set_2]
        assert exposure_series.exposure_pairs is None

    def test_initialization_with_only_merged_image_set(self, mock_image_sets):
        # Given: Mock data for a single ImageSet
        mock_image_set_1, _, _ = mock_image_sets

        # When: Initialize ExposureSeries with only merged_image_set
        exposure_series = ExposureSeries(merged_image_set=mock_image_set_1)

        # Then: Check initialization
        assert exposure_series.merged_image_set == mock_image_set_1
        assert exposure_series.input_image_sets == []
        assert exposure_series.directory_path is None
        assert exposure_series.exposure_pairs is None

    def test_initialization_with_directory_path(self, mock_image_sets, tmp_path):
        # Given: A file path
        mock_image_set_1, _, _ = mock_image_sets
        file_path = tmp_path.joinpath("/mock/directory/some_file.tif")

        # When: Initialize ExposureSeries with directory_path set to a file
        exposure_series = ExposureSeries(directory_path=file_path)

        # Then: Check directory_path is set to the parent directory
        assert exposure_series.directory_path == file_path.parent

    def test_initialization_with_none(self, mock_image_sets):
        # When: Initialize ExposureSeries with no arguments (defaults)
        exposure_series = ExposureSeries()

        # Then: Check that attributes are set to their defaults
        assert exposure_series.merged_image_set is None
        assert exposure_series.input_image_sets == []
        assert exposure_series.directory_path is None
        assert exposure_series.exposure_pairs is None

    def test_initialization_with_directory_path_as_directory(self, mock_image_sets):
        # Given: A directory path
        mock_image_set_1, _, _ = mock_image_sets
        directory_path = Path("/mock/directory")

        # When: Initialize ExposureSeries with directory_path set to a directory
        exposure_series = ExposureSeries(directory_path=directory_path)

        # Then: Check directory_path is set correctly (no change for directory)
        assert exposure_series.directory_path == directory_path


class TestExposureSeriesFromImageSet:

    @patch('image_set.ImageSet.multiple_from_path')
    def test_from_image_set(self, mock_multiple_from_path, mock_image_sets):

        mock_image_set_1, mock_image_set_2, mock_image_set_3 = mock_image_sets

        # Mocking the `multiple_from_path` method to return mock image sets.
        mock_multiple_from_path.return_value = [mock_image_set_1, mock_image_set_2, mock_image_set_3]

        # Create a mock reference image set with arbitrary path and features.
        reference_image_set = MagicMock(spec=ImageSet)
        reference_image_set.path = Path("/fake/path/reference_image_set.tif")
        reference_image_set.features = {"exposure": 150}
        reference_image_set.is_exposure_match.return_value = True

        # Call the class method `from_image_set`.
        exposure_series = ExposureSeries.from_image_set(reference_image_set)

        # Assertions
        # The input_image_sets should be sorted by exposure in ascending order
        assert exposure_series.input_image_sets == [mock_image_set_3, mock_image_set_1, mock_image_set_2]
        # The path should point to the parent of the reference image_set
        assert exposure_series.directory_path == reference_image_set.path.parent

        # Assert is_exposure_match is called for each image_set
        mock_multiple_from_path.assert_called_with(reference_image_set.path.parent)
        for mock_image_set in mock_image_sets:
            reference_image_set.is_exposure_match.assert_any_call(mock_image_set)
            assert reference_image_set.is_exposure_match.call_count == len(mock_image_sets)

        # Assert sorting by exposure
        for i, _ in enumerate(exposure_series.input_image_sets):
            if i == len(exposure_series.input_image_sets) - 1:
                break
            assert (exposure_series.input_image_sets[i].features["exposure"] <=
                    exposure_series.input_image_sets[i+1].features["exposure"])
