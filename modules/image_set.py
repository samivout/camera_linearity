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
from pathlib import Path
from typing import Optional, List, Dict, Union
from measurand import AbstractMeasurand
from measurand_factory import Measurand, measurand_to_numpy, measurand_to_cupy, CUPY_AVAILABLE, ArrayType
from global_settings import GlobalSettings as gs
import read_data as rd
from cupy_wrapper import get_array_libraries
import numpy as np


class ImageSet(object):

    def __init__(self, file_path: Optional[str | Path] = None, value: Optional[ArrayType] = None,
                 std: Optional[ArrayType] = None, features: Optional[Dict] = None,
                 measurand: Optional[AbstractMeasurand] = None, use_cupy: Optional[bool] = True):

        if isinstance(file_path, str):
            self.path = Path(file_path)
        else:
            self.path = file_path

        # Measurand assignment. use_cupy parameter is overridden if a measurand parameter is given.
        if measurand is not None:
            self._measurand = measurand
            if measurand.backend == "numpy":
                self._use_cupy = False
            else:
                self._use_cupy = True
        else:
            self._measurand = Measurand(value, std, use_cupy)
            self._use_cupy = use_cupy

        if features is not None:
            self.features = features
        elif file_path is not None:
            self.features = _features_from_file_name(self.path)
        else:
            self.features = None
        self.is_HDR = False

    @property
    def measurand(self):
        return self._measurand

    @measurand.setter
    def measurand(self, new_measurand: AbstractMeasurand):

        if self.measurand is not None:
            if self._use_cupy:
                expected_backend = "cupy"
            else:
                expected_backend = "numpy"

            if new_measurand.backend == expected_backend:
                self._measurand = new_measurand
            else:
                raise ValueError(f'Expected type {expected_backend}, got {type(new_measurand)} instead.')
        else:
            self._measurand = new_measurand
            if new_measurand.backend == "numpy":
                self._use_cupy = False
            else:
                self._use_cupy = True

    def to_numpy(self):
        """
        Convert this ImageSet to use NumPy.
        """
        self._measurand = measurand_to_numpy(self.measurand)
        self._use_cupy = False

    def to_cupy(self):
        """
        Convert this ImageSet to use CuPy
        """
        self._measurand = measurand_to_cupy(self.measurand)
        self._use_cupy = True

    def linearize(self, ICRF: ArrayType, ICRF_diff: Optional[ArrayType] = None):
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

        return ImageSet(file_path=self.path, features=self.features, measurand=new_measurand)

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
        self.measurand.val = value

    def load_std_image(self, STD_data: Optional[ArrayType] = None, bit64: Optional[bool] = False):
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

        return ImageSet(file_path=self.path, features=new_features, measurand=new_measurand)

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

        if self.measurand.backend == "numpy":
            tmp_measurand = self.measurand
        else:
            tmp_measurand = measurand_to_numpy(self.measurand)

        val = tmp_measurand.val
        std = tmp_measurand.std

        if not separate_channels:

            bit64_image = val.astype(np.dtype('float64'))
            cv.imwrite(file_path.removesuffix('.tif') + acq_file_suffix, bit64_image)

            if self.measurand.std is not None:
                bit64_image = std.astype(np.dtype('float64'))
                cv.imwrite(file_path.removesuffix('.tif') + std_file_suffix, bit64_image)

        else:
            for c in range(gs.NUM_OF_CHS):

                bit64_image = val[:, :, c]
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

        if self.measurand.backend == "numpy":
            tmp_measurand = self.measurand.__deepcopy__()
        else:
            tmp_measurand = measurand_to_numpy(self.measurand)

        val = tmp_measurand.val
        std = tmp_measurand.std

        max_float = np.amax(val)

        if max_float > 1:
            val /= max_float

        val = (np.around(val * gs.MAX_DN)).astype(np.dtype('uint8'))
        cv.imwrite(file_path, val)

        if std is not None:
            if force_8_bit:
                max_float = np.amax(std)
                if max_float > 1:
                    std /= max_float
                std = (np.around(std * gs.MAX_DN)).astype(np.dtype('uint8'))
            cv.imwrite(file_path.removesuffix('.tif') + ' STD.tif', std)

    def calculate_numerical_STD(self, STD_data: Optional[ArrayType] = None):
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

    def bad_pixel_filter(self: 'ImageSet', darkSet: 'ImageSet', threshold_value: Optional[float]):
        """
        Replace hot pixels with surrounding median value.

        Args:
            acqSet: ImageSet object of image being corrected.
            darkSet: ImageSet object of dark frame used to map bad pixels.
            threshold_value: threshold for considering a pixel as a hot pixel.

        Returns:
            ImageSet with corrected image.
        """

        new_measurand = self.measurand.filter_larger_than_by_map(darkSet.measurand, threshold_value)
        return ImageSet(file_path=self.path, measurand=new_measurand)

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

        if flatSet.measurand.val is None:
            flatSet.load_value_image()
        if flatSet.measurand.std is None:
            flatSet.load_std_image()

        new_measurand = self.measurand.normalize_by_map(flatSet.measurand)

        return ImageSet(file_path=self.path, measurand=new_measurand)

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
