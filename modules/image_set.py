"""
In an arbitrary order the name should contain (exposure time)ms, (illumination
type as bf or df), (magnification)x and (your image name). Each descriptor
should be separated by a space and within a descriptor there should be no white
space. For example: '5ms BF sample_1 50x.tif'. Additionally, if the image is an
uncertainty image, it should contain a separate 'STD' descriptor in it. Only
.tif support for now. Flat field images should have a name 'flat' in them
and dark frames should have 'dark' in them. TODO: implement common file types.
"""
import cv2 as cv
import re
import math
from pathlib import Path
from typing import Optional, List, Dict, Union
from cupyx.scipy.ndimage import median_filter
from measurand import Measurand, NumPyMeasurand
from global_settings import GlobalSettings as gs
import read_data as rd
from cupy_wrapper import get_array_libraries

np, cp, cupy_available = get_array_libraries()
cnp = cp if cupy_available else np


class ImageSet(object):

    def __init__(self, file_path: Optional[str | Path] = None, value: Optional[cnp.ndarray] = None,
                 std: Optional[cnp.ndarray] = None, features: Optional[Dict] = None,
                 measurand: Optional[Union[Measurand, NumPyMeasurand]] = None, channels: Optional[List[int]] = None,
                 use_cupy: Optional[bool] = True):

        if isinstance(file_path, str):
            self.path = Path(file_path)
        else:
            self.path = file_path

        if measurand is not None:
            self._measurand = measurand
            if isinstance(measurand, NumPyMeasurand):
                self._use_cupy = False
            elif isinstance(measurand, Measurand):
                self._use_cupy = True
        else:
            if use_cupy and cupy_available:
                self._measurand = Measurand(value, std)
                self._use_cupy = True
            else:
                self._measurand = NumPyMeasurand(value, std)
                self._use_cupy = False

        if features is not None:
            self.features = features
        elif file_path is not None:
            self.features = _features_from_file_name(self.path)
        else:
            self.features = None
        self.channels = channels
        self.is_HDR = False

    @property
    def measurand(self):
        return self._measurand

    @measurand.setter
    def measurand(self, new_measurand):

        if self._use_cupy:
            expected_type = Measurand
        else:
            expected_type = NumPyMeasurand

        if isinstance(new_measurand, expected_type):
            self._measurand = new_measurand
        else:
            raise ValueError(f'Expected type {expected_type}, got {type(new_measurand)} instead.')

    def to_numpy(self):
        """
        Convert this ImageSet to use NumPy.
        """
        if isinstance(self.measurand, Measurand):
            self._use_cupy = False
            self._measurand = self._measurand.to_numpy()

    def to_cupy(self):
        """
        Convert this ImageSet to use CuPy
        """
        if isinstance(self.measurand, NumPyMeasurand):
            self._use_cupy = True
            self._measurand = self._measurand.to_cupy()

    def linearize(self, ICRF: cnp.ndarray, ICRF_diff: Optional[cnp.ndarray] = None):
        """
        Calls the linearize method of the underlying Measurand class, yielding a new Measurand object with the linerized
        values. Constructs a new ImageSet object.
        Args:
            ICRF: the inverse camera response function.
            ICRF_diff: the derivative of the ICRF.

        Returns:
            A new ImageSet object with a new Measurand, whose values are the linearized values of the source.
        """
        new_measurand = self.measurand.linearize(ICRF, ICRF_diff)

        return ImageSet(file_path=self.path, features=self.features, measurand=new_measurand)

    def get_file_path_without_exposure(self):
        if self.path is not None:
            return self.path.parent.joinpath(
                f"{self.features['subject']} {self.features['illumination']} {self.features['magnification']}.tif")
        return None

    def is_exposure_match(self, other: 'ImageSet'):
        """
        Determines whether the other ImageSet is an exposure match to this ImageSet. They are a match if all features,
        apart from exposure time, are equal.
        Args:
            other: the other ImageSet.

        Returns:
            bool representing if they are a match or not.
        """
        if self.features is None or other.features is None:
            return False

        is_match = True
        for key in self.features.keys():
            if key == "exposure":
                continue
            if self.features[key] != other.features[key]:
                is_match = False
                break

        return is_match

    def get_flat_field(self, list_of_flat_fields: Optional[List['ImageSet']] = None):

        if list_of_flat_fields is None:
            list_of_flat_fields = ImageSet.multiple_from_path(gs.DEFAULT_FLAT_PATH)
        for flat_set in list_of_flat_fields:
            if self.features['illumination'] == flat_set.features['illumination'] and self.features['magnification'] == \
                    flat_set.features['magnification']:
                return flat_set

        return None

    def get_dark_field(self, list_of_dark_fields: Optional[List['ImageSet']] = None):
        """
        Function to determine an appropriate dark frame for a given image. Either finds the perfect match in terms of
        exposure time or finds the closest longer exposure time dark frame and scales it down to the input image's exposure
        time.
        Args:
            list_of_dark_fields: list of available dark frames.

        Returns:
            A matching or scaled dark frame.
        """
        if list_of_dark_fields is None:
            list_of_dark_fields = ImageSet.multiple_from_path(gs.DEFAULT_DARK_PATH)

        target_exposure = self.features['exposure']

        if target_exposure >= gs.DARK_THRESHOLD:
            lesser_exp = False
            greater_exp = False
            greater_index = 0

            for i, darkSet in enumerate(list_of_dark_fields):

                if darkSet.features['exposure'] < target_exposure:
                    lesser_exp = True

                if darkSet.features['exposure'] > target_exposure:
                    greater_exp = True
                    greater_index = i

                if target_exposure == darkSet.features['exposure']:
                    darkSet.load_value_image()
                    return darkSet

                if lesser_exp is True and greater_exp is True:
                    greater_dark = list_of_dark_fields[greater_index]
                    greater_dark.load_value_image()
                    scaled_dark_set = greater_dark.scale_to_exposure(target_exposure)

                    return scaled_dark_set

        return None

    def extract(self, channels: Optional[int | List[int]] = None):
        """
        Create a new instance of an ImageSet object with the specified channels, if any.

        Args:
            channels: integer or list of integers representing the channel(s). None to copy no channels.

        Returns:
            A new ImageSet object with copied data from the base ImageSet.
        """
        new_measurand = self.measurand.extract(dims=channels, axis=-1)

        return ImageSet(file_path=self.path, features=self.features, measurand=new_measurand, channels=channels)

    def load_value_image(self, bit64: Optional[bool] = False):
        """
        Load the acquired image of the ImageSet object into memory. You can specify whether to load it as an 8-bits per
        channel image or a 64-bit float per channel image.

        Args:
            bit64: whether the image should be in 64-bit float form or not.
        """

        if not bit64:
            value = cv.imread(str(self.path)).astype(np.float64) / gs.MAX_DN
        else:
            value = cv.imread(str(self.path), cv.IMREAD_UNCHANGED)
        number_of_dims = len(np.shape(value))
        self.channels = list(np.arange(0, number_of_dims, step=1))
        if self._use_cupy:
            self.measurand.val = cp.asarray(value)
            self.channels = cp.asarray(self.channels)

    def single_channel_to_multiple(self):

        value_image = cv.imread(str(self.path), cv.IMREAD_UNCHANGED).astype(np.float64)
        if self._use_cupy:
            value_image = cp.array(value_image)

        number_of_dims = len(cnp.shape(value_image))

        if number_of_dims == 3:
            self.measurand.val = value_image
        if number_of_dims == 2:
            value_image = value_image[:, :, cnp.newaxis]
            self.measurand.val = cnp.concatenate((value_image, value_image, value_image), axis=2)
        self.channels = [0, 1, 2]

    def load_std_image(self, STD_data: Optional[np.ndarray] = None, bit64: Optional[bool] = False):
        """
        Loads the error image of an ImageSet object to memory.
        Args:
            bit64: whether the image to load is already in float or not
            STD_data: Numpy array representing the STD data of pixel values.
        """
        std_path = str(self.path).removesuffix('.tif') + ' STD.tif'
        std_array = cv.imread(std_path, cv.IMREAD_UNCHANGED)
        if std_array is None:
            std_array = self.calculate_numerical_STD(STD_data)

        if std_array is None:
            return

        if self._use_cupy:
            std_array = cp.asarray(std_array)

        self.measurand.std = std_array

    def scale_to_exposure(self, target_exp: float):
        """
        Function to scale image by exposure time.

        Args:
            target_exp: exposure time at which to perform the interpolation.

        Returns:
            Scaled ImageSet.
        """
        new_measurand = self.measurand
        new_features = self.features
        new_features['exposure'] = target_exp
        exposure = self.features['exposure']

        new_measurand = (target_exp / exposure) * new_measurand

        return ImageSet(file_path=self.path, measurand=new_measurand, features=new_features)

    def save_64bit(self, save_path: Optional[Path] = None, is_HDR: Optional[bool] = False,
                   separate_channels: Optional[bool] = False):
        """
        Saves an ImageSet object's acquired image and error image to disk to given
        path into separate BGR channels in 32-bit format.

        Args:
            is_HDR: whether to add HDR to the end of the filename or not.
            save_path: Full absolute path to save location, including filename.
            separate_channels: whether to save each channel to a separate file.
        """
        if save_path is None:
            file_path = self.path.parent.joinpath('64bit', self.path.name)
        else:
            file_path = save_path

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        file_path = str(file_path)

        if is_HDR:
            acq_file_suffix = ' HDR.tif'
            std_file_suffix = ' HDR STD.tif'
        else:
            acq_file_suffix = '.tif'
            std_file_suffix = ' STD.tif'

        if isinstance(self.measurand.val, cp.ndarray):
            acq = cp.asnumpy(self.measurand.val)
        else:
            acq = self.measurand.val
        if self.measurand.std is not None and isinstance(self.measurand.std, cp.ndarray):
            std = cp.asnumpy(self.measurand.std)
        else:
            std = self.measurand.std

        if not separate_channels:

            bit64_image = acq.astype(np.dtype('float64'))
            cv.imwrite(file_path.removesuffix('.tif') + acq_file_suffix, bit64_image)

            if self.measurand.std is not None:
                bit64_image = std.astype(np.dtype('float64'))
                cv.imwrite(file_path.removesuffix('.tif') + std_file_suffix, bit64_image)

        else:
            for c in range(gs.NUM_OF_CHS):

                bit64_image = acq[:, :, c]
                cv.imwrite(file_path.removesuffix('.tif')
                           + acq_file_suffix.replace('.tif', f' {gs.CH_NAMES[c]}.tif'), bit64_image)

                if self.measurand.std is not None:
                    bit64_image = std[:, :, c]
                    cv.imwrite(file_path.removesuffix('.tif')
                               + std_file_suffix.replace('.tif', f' {gs.CH_NAMES[c]}.tif'), bit64_image)

    def save_8bit(self, save_path: Optional[Path] = None, force_8_bit: Optional[bool] = False):
        """
        Saves an ImageSet object's acquired image and error image to disk to given path in an 8-bits per channel format.

        Args:
            save_path: Dir path where to save images, name is supplied by the ImageSet object
            force_8_bit: Scales image to [0, 1] range without any application of gamma compression for 8-bit saving.
        """
        if save_path is None:
            file_path = self.path.parent.joinpath('8bit', self.path.name)
        else:
            file_path = save_path

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        file_path = str(file_path)

        save_stds = None
        if isinstance(self.measurand.val, cp.ndarray):
            save_values = cp.asnumpy(self.measurand.val)
        elif isinstance(self.measurand.val, np.ndarray):
            save_values = self.measurand.val.copy
        else:
            raise TypeError('ImageSet measurand value has unsupported type.')
        if self.measurand.std is not None:
            if isinstance(self.measurand.std, cp.ndarray):
                save_stds = cp.asnumpy(self.measurand.std)
            elif isinstance(self.measurand.std, np.ndarray):
                save_stds = self.measurand.std.copy
            else:
                raise TypeError('ImageSet measurand std has unsupported type.')

        bit8_image = save_values
        max_float = np.amax(bit8_image)

        if max_float > 1:
            bit8_image /= max_float

        bit8_image = (np.around(bit8_image * gs.MAX_DN)).astype(np.dtype('uint8'))
        cv.imwrite(file_path, bit8_image)

        if save_stds is not None:
            bit8_image = save_stds
            if force_8_bit:
                max_float = np.amax(bit8_image)
                if max_float > 1:
                    bit8_image /= max_float
                bit8_image = (np.around(bit8_image * gs.MAX_DN)).astype(np.dtype('uint8'))
            cv.imwrite(file_path.removesuffix('.tif') + ' STD.tif', bit8_image)

    def calculate_numerical_STD(self, STD_data: Optional[cnp.ndarray] = None):
        """
        Calculates an uncertainty estimate for a given acquired image.
        TODO: rewrite to use proper warnings when file not found.

        Args:
            STD_data: previously gathered camera noise distribution data.

        Returns:
            A numerical estimate of the uncertainty of an acquired image.
        """
        if STD_data is None:
            try:
                STD_data = rd.read_txt_to_array(gs.STD_FILE_NAME, use_cupy=self._use_cupy)
            except FileNotFoundError:
                print('Could not load STD data for numerical estimation.')
                return None

        numerical_measurand = self.measurand.linearize(ICRF=STD_data)

        return numerical_measurand.val

    def bad_pixel_filter(self: 'ImageSet', darkSet: 'ImageSet'):
        """
        Replace hot pixels with surrounding median value.

        Args:
            acqSet: ImageSet object of image being corrected.
            darkSet: ImageSet object of dark frame used to map bad pixels.

        Returns:
            ImageSet with corrected image.
        """

        def filter_hot_positions(acq, dark, convolve):
            hot_indices = dark > gs.HOT_PIXEL_THRESHOLD
            acq[hot_indices] = convolve[hot_indices]
            return acq

        print(f'Bad pixel filter for image: {self.path} and dark: {darkSet.path}')
        convolved_image = cp.zeros_like(self.measurand.val, dtype=cp.dtype('float64'))
        for c in range(gs.NUM_OF_CHS):
            convolved_image[:, :, c] = median_filter(self.measurand.val[:, :, c],
                                                     (gs.MEDIAN_FILTER_KERNEL_SIZE, gs.MEDIAN_FILTER_KERNEL_SIZE),
                                                     mode='reflect')

        acq = filter_hot_positions(self.measurand.val, darkSet.measurand.val, convolved_image)

        if self.measurand.std is not None:
            for c in range(gs.NUM_OF_CHS):
                convolved_image[:, :, c] = median_filter(self.measurand.std[:, :, c],
                                                         (gs.MEDIAN_FILTER_KERNEL_SIZE, gs.MEDIAN_FILTER_KERNEL_SIZE),
                                                         mode='reflect')

            self.measurand.std = filter_hot_positions(acq, darkSet.measurand.val, convolved_image)

        self.measurand.val = acq

    def flat_field_correction(self: 'ImageSet', flatSet: 'ImageSet'):
        """
        Correct the flat-field and fixed-pattern effects of an acquired image by using a flat-field image. Also calculates
        an uncertainty image.

        Args:
            imageSet: the acquired image subject to flat-field correction.
            flatSet: the flat-field image used for correction.

        Returns:
            The corrected image as an ImageSet object.
        """

        def flat_field_mean(flat_field):
            """
            Calculates the mean brightness of an image inside a centered ROI.

            Returns:
                list of mean image brightness inside ROI for each channel.
            """
            flat_field_means = []

            # Define ROI for calculating flat field spatial mean
            ROI_dx = math.floor(gs.IM_SIZE_X * gs.FF_MID_PERCENTAGE)
            ROI_dy = math.floor(gs.IM_SIZE_Y * gs.FF_MID_PERCENTAGE)
            ROI_start_index = (math.floor(
                1 / gs.FF_MID_PERCENTAGE) - 1) / 2  # Should be an odd number to center on image.

            # Calculate ROI bounds
            x_start, x_end = ROI_start_index * ROI_dx, (ROI_start_index + 1) * ROI_dx
            y_start, y_end = ROI_start_index * ROI_dy, (ROI_start_index + 1) * ROI_dy

            # Slice ROI for all channels and compute the mean across spatial dimensions
            return cnp.mean(flat_field[x_start:x_end, y_start:y_end, :], axis=(0, 1))

        if flatSet.measurand.val is None:
            flatSet.load_value_image()
        if flatSet.measurand.std is None:
            flatSet.load_std_image()

        # Determine flat field means
        flat_field_means = flat_field_mean(flatSet.measurand.val)
        flat_field_stds = flat_field_mean(flatSet.measurand.std)

        # Simplify variable names
        acq = self.measurand.val
        flat = flatSet.measurand.val
        u_acq = self.measurand.std
        u_ff = flatSet.measurand.std

        # Compute input image uncertainty
        u_acq_term = (u_acq ** 2) / (flat ** 2)
        u_acq_term *= flat_field_means ** 2

        # Compute flat field uncertainty
        u_ff_term = (acq ** 2) / (flat ** 4)
        u_ff_term *= u_ff ** 2
        u_ff_term *= flat_field_means ** 2

        # Compute flat field mean uncertainty
        u_ffm_term = (acq ** 2) / (flat ** 2)
        u_ffm_term *= flat_field_stds ** 2

        # Update uncertainty with the combined terms
        self.measurand.std = cnp.sqrt(u_acq_term + u_ff_term + u_ffm_term)

        # Flat field correction (broadcast multiplication for channel-specific scaling)
        self.measurand.val = (acq / flat) * flat_field_means

    def show_image(self):
        """
        A support function to quickly display the value image.
        """
        if self.measurand.val is None:
            raise ValueError('No image to show.')

        cv.namedWindow(self.path.name, cv.WINDOW_NORMAL)
        cv.imshow(self.path.name, self.measurand.val)
        cv.waitKey(0)
        cv.destroyAllWindows()

        return

    @staticmethod
    def compute_difference(short_exposure_set: 'ImageSet', long_exposure_set: 'ImageSet'):

        ratio = short_exposure_set.features["exposure"] / long_exposure_set.features["exposure"]

        absolute_measurand, relative_measurand = Measurand.compute_difference(short_exposure_set.measurand,
                                                                              long_exposure_set.measurand, ratio)

        absolute_set = ImageSet(file_path=short_exposure_set.path, features=short_exposure_set.features,
                                measurand=absolute_measurand)
        relative_set = ImageSet(file_path=short_exposure_set.path, features=short_exposure_set.features,
                                measurand=relative_measurand)

        return absolute_set, relative_set

    @staticmethod
    def exposure_interpolation(short_exposure_set: 'ImageSet', long_exposure_set: 'ImageSet', exp: float):
        """
        Simple linear interpolation of two frames by exposure time. Utilizes the linear interpolation method of the
        underlying Measurand class.

        Args:
            short_exposure_set: Lower exposure ImageSet object.
            long_exposure_set: Higher exposure ImageSet object.
            exp: exposure time at which to perform the interpolation.

        Returns:
            Interpolated ImageSet.
        """
        if not isinstance(exp, float):
            raise TypeError('Interpolation point has unsupported type.')

        x0 = short_exposure_set.measurand
        x1 = long_exposure_set.measurand
        exp0 = short_exposure_set.features['exposure']
        exp1 = long_exposure_set.features['exposure']

        if exp > exp1 or exp < exp0:
            raise ValueError('Interpolation point is not between the reference values.')

        new_measurand = Measurand.interpolate(x0, x1, exp0, exp1, exp)

        return ImageSet(features=short_exposure_set.features, measurand=new_measurand)

    @classmethod
    def multiple_from_path(cls, path: Path):
        """
        Initialize the ImageSet objects based on image files in the given path, but without loading the acq or std
        images into memory.

        Args:
            path: the path from which to initialize the ImageSet objects.

        Returns:
            A list containing the ImageSet objects.
        """
        list_of_ImageSets = []
        image_files = path.glob("*.tif")
        for file in image_files:
            if not ("STD" in file.name):
                imageSet = cls(file_path=file)
                list_of_ImageSets.append(imageSet)

        return list_of_ImageSets


def calibrate_flats():
    """
    Function to calibrate flat frames, i.e. bias subtraction.
    :return:
    """
    list_of_original_dark_frames = ImageSet.multiple_from_path(gs.DEFAULT_DARK_PATH)
    list_of_original_dark_frames.sort(key=lambda image_set: image_set.features['exposure'])
    list_of_original_flat_fields = ImageSet.multiple_from_path(gs.UNCALIBRATED_FLAT_PATH)

    bias = list_of_original_dark_frames[0]
    bias.load_value_image()
    bias.load_std_image()

    for flat_field in list_of_original_flat_fields:
        flat_field.load_value_image()
        flat_field.load_std_image()
        flat_field.measurand = flat_field.measurand - bias.measurand
        flat_field.save_8bit(gs.DEFAULT_FLAT_PATH)


def calibrate_dark_frames():
    """
    Function that handles calibration of raw dark frames, i.e. bias subtraction.
    :return:
    """
    list_of_original_dark_frames = ImageSet.multiple_from_path(gs.UNCALIBRATED_DARK_PATH)
    list_of_original_dark_frames.sort(key=lambda image_set: image_set.features['exposure'])
    bias = list_of_original_dark_frames[0]
    bias.load_value_image()
    bias.load_std_image()

    for i, dark_frame in enumerate(list_of_original_dark_frames):
        dark_frame.load_value_image()
        dark_frame.load_std_image()
        dark_frame.measurand = dark_frame.measurand - bias.measurand
        dark_frame.save_8bit(gs.DEFAULT_DARK_PATH)


def _features_from_file_name(file_path: Path):
    """
    Constructs a feature dictionary from the file name of the image.
    Args:
        file_path: path to the image file.

    Returns:
        A dictionary containing the features under the given keys.
    """
    feature_dict = {}
    file_name_array = file_path.name.removesuffix('.tif').split()

    for element in file_name_array:
        if element.casefold() == 'bf' or element.casefold() == 'df':
            feature_dict["illumination"] = element
        elif re.match("^[0-9]+.*[xX]$", element):
            feature_dict["magnification"] = element
        elif re.match("^[0-9]+.*ms$", element):
            feature_dict["exposure"] = float(element.removesuffix('ms')) / 1000
        else:
            feature_dict["subject"] = element

    return feature_dict


if __name__ == "__main__":
    pass
