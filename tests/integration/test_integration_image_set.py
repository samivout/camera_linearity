import time
import pytest
from image_set import ImageSet
from conftest import USE_CUPY, xp


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


class TestImageSetInitialization:

    @pytest.mark.parametrize("random_array", [{"shape": (100, 100, 3)}], indirect=True)
    def test_imageset_init_with_val_and_std(self, random_array):
        values = random_array
        stds = random_array * 0.1

        image_set = ImageSet(value=values, std=stds)
        assert xp.all(image_set.measurand.val == values)
        assert xp.all(image_set.measurand.std == stds)


class TestImageSetIO:

    @pytest.mark.parametrize("random_array", [{"shape": (100, 100, 3)}], indirect=True)
    def test_imageset_save_and_load_8bit(self, random_array, tmp_path):

        file_name = "1.0ms test_image BF 5x.tif"
        full_path = tmp_path.joinpath(file_name)
        std = random_array * 0.1

        image_set = ImageSet(file_path=full_path, value=random_array, std=std)
        image_set.save_8bit(save_path=full_path)

        time.sleep(2)

        other_image_set = ImageSet(file_path=full_path)
        other_image_set.load_value_image(bit64=False)
        other_image_set.load_std_image(bit64=False)

        assert xp.allclose(image_set.measurand.val, other_image_set.measurand.val, atol=0.5/255)
        assert xp.allclose(image_set.measurand.std, other_image_set.measurand.std)

    @pytest.mark.parametrize("random_array", [{"shape": (100, 100, 3)}], indirect=True)
    def test_imageset_save_and_load_64bit(self, random_array, tmp_path):

        file_name = "1.0ms test_image BF 5x.tif"
        full_path = tmp_path.joinpath(file_name)
        std = random_array * 0.1

        image_set = ImageSet(file_path=full_path, value=random_array, std=std)
        image_set.save_64bit(save_path=full_path)

        time.sleep(2)

        other_image_set = ImageSet(file_path=full_path)
        other_image_set.load_value_image(bit64=True)
        other_image_set.load_std_image(bit64=True)

        assert xp.allclose(image_set.measurand.val, other_image_set.measurand.val)
        assert xp.allclose(image_set.measurand.std, other_image_set.measurand.std)