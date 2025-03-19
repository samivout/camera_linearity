"""
Module containing the ExposureSeries and ExposurePair classes. These implement the functionality to analyze the
linearity of a stack of images captured at different exposure times. Also implements the functionality to merge the
stack into an HDR image.
"""
from pathlib import Path

import general_functions
import general_functions as gf
from image_set import ImageSet
from typing import Optional
from typing import List
from typing import Dict
from global_settings import GlobalSettings as gs
from array_wrapper import cast_to_array, ArrayType


class ExposurePair(object):
    """
    Class for managing the pairs of ImageSet object's as part of the ExposureSeries class. Main functionality is to
    make it easy for maintaining information of computed statistics.
    """
    def __init__(self, short_exposure: ImageSet, long_exposure: ImageSet):
        self.short_exposure = short_exposure
        self.long_exposure = long_exposure

        self.exposure_ratio = short_exposure.features["exposure"] / long_exposure.features["exposure"]
        self.absolute_difference = None
        self.relative_difference = None
        self.absolute_stats = None
        self.relative_stats = None

    def compute_difference(self):
        """
        Method for computing and storing into attributes the relative and absolute differences between the ImageSets
        composing the ExposurePair.
        """
        self.absolute_difference, self.relative_difference = (
            ImageSet.compute_difference(self.short_exposure, self.long_exposure))

    def compute_stats(self, axis: Optional[None | int | tuple[int, ...]] = None,
                      release_memory_after: Optional[bool] = True):
        """
        Runs statistics computations on the relative and absolute difference of the ExposurePair and stores the results
        into attributes.
        Args:
            axis: axis for the statistics computation, follows NumPy conventions.
            release_memory_after: Whether to delete the difference images of the ExposurePair after computing the stats.
        """
        self.absolute_stats = self.absolute_difference.measurand.compute_dimension_statistics(axis=axis)
        self.relative_stats = self.relative_difference.measurand.compute_dimension_statistics(axis=axis)
        if release_memory_after:
            self.absolute_difference = None
            self.relative_difference = None

    def process_linearity_distribution(self, bins: int, included_range: Optional[tuple[float, float]] = None,
                                       channels: List[int] = None, use_std: Optional[bool] = False):
        """
        Computes a histogram for the difference ImageSets of the ExposurePair. The histograms are stored as a
        (bins, bin_edges) tuple into a dictionary with the channel indices acting as keys.
        Args:
            bins: number of bins to use.
            included_range: the range of values to include.
            channels: the channels to perform the computation on.
            use_std: whether to use uncertainty images as weights in the histogram computation or not.

        Returns:
            A dictionary containing a (bins, bin_edges) tuple for each desired channel, with the channel index acting as
            a key.
        """
        absolute_histogram = self.absolute_difference.measurand.compute_channel_histogram(bins, included_range,
                                                                                          channels, use_std)
        relative_histogram = self.relative_difference.measurand.compute_channel_histogram(bins, included_range,
                                                                                          channels, use_std)

        return absolute_histogram, relative_histogram


class ExposureSeries(object):
    """
    Class that maintains and uses collections of ImageSets for measuring linearity and constructing HDR images.
    ExposureSeries are formed from a collection of ImageSets, whose features are the same, except exposure time.
    """
    def __init__(self, merged_image_set: Optional[ImageSet] = None, directory_path: Optional[Path] = None,
                 input_image_sets: Optional[List[ImageSet]] = None, use_cupy: Optional[bool] = True):

        self.merged_image_set = None
        if merged_image_set is not None:
            self.merged_image_set = merged_image_set

        self.input_image_sets = []
        if input_image_sets is not None:
            self.input_image_sets = input_image_sets

        # Path.is_file() checks for existence, we don't want that.
        if isinstance(directory_path, Path) and directory_path.suffix != "":
            self.directory_path = directory_path.parent
        else:
            self.directory_path = directory_path

        self.exposure_pairs = None

        if not input_image_sets:
            self._use_cupy = use_cupy
        else:
            self._use_cupy = input_image_sets[0].use_cupy

    @property
    def use_cupy(self):
        return self._use_cupy

    @use_cupy.setter
    def use_cupy(self, new_value):
        """Read-only property, raises an error if attempting to modify."""
        raise AttributeError("use_cupy is a read-only attribute, managing the state of the used array backend.")

    @classmethod
    def from_image_set(cls, reference_image_set: ImageSet, directory_path: Optional[Path] = None):
        """
        Constructs and ExposureSeries object based on a reference ImageSet object. It searches the given path or the
        parent directory of the reference ImageSet for exposure matches. Matched ImageSets are collected into the
        input_image_sets attribute of a new ExposureSeries object.
        Args:
            reference_image_set: ImageSet for which to find exposure matches.
            directory_path: Path to search for images, overrides the parent directory of the reference ImageSet for
                searching images.

        Returns:
            An ExposureSeries object with matched ImageSets collected as the input_image_sets.
        """
        input_image_sets = []

        if directory_path is None:
            search_path = reference_image_set.path.parent
        else:
            search_path = directory_path

        list_of_found_image_sets = ImageSet.multiple_from_path(search_path)
        for image_set in list_of_found_image_sets:
            if reference_image_set.is_exposure_match(image_set):
                input_image_sets.append(image_set)

        input_image_sets.sort(key=lambda imageSet: imageSet.features["exposure"])

        return cls(directory_path=search_path, input_image_sets=input_image_sets)

    @classmethod
    def from_dir_path(cls, directory_path: Path):
        """
        Creates a list of ExposureSeries objects from the given path. Utilizes the .from_multiple_image_sets class
        method of ExposureSeries.
        Args:
            directory_path: path to the directory from which to search for images to create ExposureSeries objects.

        Returns:
            A list of ExposureSeries objects.
        """
        list_of_image_sets = ImageSet.multiple_from_path(directory_path)
        list_of_exposure_series = ExposureSeries.from_multiple_image_sets(list_of_image_sets)

        return list_of_exposure_series

    @classmethod
    def from_multiple_image_sets(cls, list_of_image_sets: List[ImageSet]):
        """
        Generates a list of ExposureSeries objects from a list of ImageSets by collecting ImageSets that are an exposure
        match under one ExposureSeries.
        Args:
            list_of_image_sets: A list of ImageSet objects.

        Returns:
            List of ExposureSeries object.
        """
        list_of_sublists = []
        list_of_exposure_series = []

        for image_set in list_of_image_sets:

            # Check if list_of_sublists is empty. If yes, create first sublist and
            # automatically add the first ImageSet object to it.
            if not list_of_sublists:
                sublist = [image_set]
                list_of_sublists.append(sublist)
                continue

            number_of_sublists = len(list_of_sublists)
            for i in range(number_of_sublists):

                sublist = list_of_sublists[i]
                reference_image_set = sublist[0]
                if reference_image_set.is_exposure_match(image_set):
                    sublist.append(image_set)
                    break
                if number_of_sublists - 1 - i == 0:
                    additional_list = [image_set]
                    list_of_sublists.append(additional_list)
                    break

        for sublist in list_of_sublists:
            sublist.sort(key=lambda image_set: image_set.features['exposure'])
            list_of_exposure_series.append(cls(input_image_sets=sublist))

        return list_of_exposure_series

    def load_value_images(self, bit_64: Optional[bool] = False):
        """
        Preloads all value images of the managed input images into memory.
        Args:
            bit_64: whether to load images at bit-depth of 64 or 8.
        """
        image_set: ImageSet
        for image_set in self.input_image_sets:

            image_set.load_value_image(bit64=bit_64)

    def load_std_images(self, bit_64: Optional[bool] = False):
        """
        Preloads all std images of the managed input images into memory.
        Args:
            bit_64: whether to load images at bit-depth of 64 or 8.
        """
        image_set: ImageSet
        for image_set in self.input_image_sets:
            image_set.load_std_image(bit64=bit_64)

    def linearize(self, ICRF: ArrayType, ICRF_diff: Optional[ArrayType] = None, release_memory: Optional[bool] = False):
        """
        Calls the linearize method of all managed input ImageSets witht the given ICRF and its derivative. This
        operation constructs a new ExposureSeries object. release_memory argument can be used to dynamically release
        memory by deleting the value and std images of the source ExposureSet's input ImageSets as they are linearized.
        Args:
            ICRF: the inverse camera response function as an array, shape (gs.BITS, gs.NUM_OF_CHS)
            ICRF_diff: the derivateive of the ICRF, shape (gs.BITS, gs.NUM_OF_CHS)
            release_memory: whether to delete the value and std images of the source ImageSets to release memory.

        Returns:
            New ExposureSeries object with the input ImageSets of the original having been linearized.
        """
        new_input_image_sets = []

        if self.input_image_sets is not None:
            input_image_set: ImageSet
            for input_image_set in self.input_image_sets:
                new_input_image_sets.append(input_image_set.linearize(ICRF, ICRF_diff))
                if release_memory:
                    input_image_set.measurand.val = None
                    input_image_set.measurand.std = None

        return ExposureSeries(merged_image_set=self.merged_image_set, directory_path=self.directory_path,
                              input_image_sets=new_input_image_sets)

    def extract(self, channels: Optional[int | List[int]] = None, release_memory: Optional[bool] = False):
        """
        Calls the extract method on the input ImageSets, using the given channels. Constructs a new ExposureSeries
        object with the input ImageSets being the extracted versions of the source ImageSets. Memory can be dynamically
        released using the release_memory argument. This deletes the value and std images of the source ImageSets as
        they are extracted.
        Args:
            channels: which channels to extract from the source ImageSets
            release_memory: whether to dynamically release memory as the extractions are performed on the source
                ImageSets.

        Returns:
            New ExposureSeries object with the input ImageSets being the extracted versions of the source ImageSets.
        """
        new_input_image_sets = []
        new_merged_image_set = None

        if self.merged_image_set is not None:
            new_merged_image_set = self.merged_image_set.extract(channels)

        if self.input_image_sets is not None:
            input_image_set: ImageSet
            for input_image_set in self.input_image_sets:
                new_input_image_sets.append(input_image_set.extract(channels))
                if release_memory:
                    input_image_set.measurand.val = None
                    input_image_set.measurand.std = None

        return ExposureSeries(merged_image_set=new_merged_image_set, directory_path=self.directory_path,
                              input_image_sets=new_input_image_sets)

    def initialize_exposure_pairs(self):
        """
        Initializes the ExposurePairs for the ExposureSeries in preparation of analyzing the linearity. It iterates
        through all possible combinations of input ImageSets and accepts only valid ImageSets. Currently, the condition
        is that the ImageSets' exposure times can't be too different from each other.
        TODO: refactor the validity value into an input argument or global setting.
        """
        valid_pairs = []

        for i, x in enumerate(self.input_image_sets):
            for j, y in enumerate(self.input_image_sets):

                if i >= j:
                    continue

                ratio = x.features["exposure"] / y.features["exposure"]
                if ratio < 0.1:
                    continue

                valid_pairs.append(ExposurePair(x, y))

        self.exposure_pairs = valid_pairs

    def _construct_merged_image_set_path(self, path: Optional[Path]):
        """
        Constructs a path for the merged HDR image from its features. TODO: rework into a more usable method.
        Args:
            path: sets the parent directory of the ImageSet, overriding the original.
        """
        if path is not None:
            self.merged_image_set.path = path
        elif self.input_image_sets:
            self.merged_image_set.path = self.input_image_sets[0].get_file_path_without_exposure()

    def _precalculate_sum_of_weights(self, list_of_dark_fields: List[ImageSet],
                                     dark_threshold: Optional[float] = gs.DARK_THRESHOLD):
        """
        Apply the weighting function to all the pixels in each image and compute a sum of weights image and its square.
        By precomputing the divisor of the HDR image merging computation, we can run the calculation considerably faster.
        Args:
            list_of_dark_fields: list of dark frames.

        Returns:
            A sum of weights array and its element-wise square.
        """
        first_image_set = self.input_image_sets[0]
        if first_image_set.measurand.val is None:
            first_image_set.load_value_image()
        sum_of_weights = first_image_set.measurand.zeros_like_measurand()

        image_set: ImageSet
        for image_set in self.input_image_sets:

            image_set.load_value_image()
            dark_set = image_set.get_dark_field(list_of_dark_fields)
            if dark_set is not None:
                image_set.bad_pixel_filter(dark_set, dark_threshold)
            sum_of_weights += image_set.measurand.apply_gaussian_weight()[0]
            image_set.measurand.val = None

        squared_sum_of_weight = sum_of_weights ** 2

        return sum_of_weights, squared_sum_of_weight

    def _compute_HDR_image_set(self, list_of_dark_fields: List[ImageSet],
                               sum_of_weights: ArrayType, square_sum_of_weights: ArrayType,
                               ICRF: ArrayType, ICRF_diff: ArrayType):
        """
        Function to calculate an HDR image from multiple exposures. Uses dark frames for bad pixel filtering, ICRF
        for linearization and the precalculated weigths.

        Args:
            list_of_dark_fields: list of dark frames.
            sum_of_weights: precalculated sum of weights.
            square_sum_of_weights: precalculated square of the sum of weights.
            ICRF: the inverse camera response function.
            ICRF_diff: the derivative of the ICRF.

        Returns:
            The merged HDR image as an ImageSet object.
        """
        first_image_set = self.input_image_sets[0]
        if first_image_set.measurand.val is not None:
            first_image_set.load_value_image()

        HDR_measurand = first_image_set.measurand.zeros_like_measurand()
        HDR_image_path = self.input_image_sets[0].get_file_path_without_exposure()

        image_set: ImageSet
        for image_set in self.input_image_sets:

            dark_set = image_set.get_dark_field(list_of_dark_fields)

            image_set.load_value_image()
            image_set.load_std_image()

            if dark_set is not None:
                image_set.bad_pixel_filter(dark_set)

            w, dw = image_set.measurand.apply_gaussian_weight()
            image_set.measurand = image_set.measurand.linearize(ICRF, ICRF_diff)
            g = image_set.measurand.val
            dg = image_set.measurand.std
            t = image_set.features['exposure']

            HDR_measurand.val += (w * g) / (sum_of_weights * t)
            HDR_measurand.std += (((dw * g + w * dg) / sum_of_weights - (dw * w * g) / square_sum_of_weights) * dg / t) ** 2

            image_set.measurand.val = None
            image_set.measurand.std = None

        HDR_measurand.std = HDR_measurand.std ** (1 / 2)
        HDR_image_set = ImageSet(file_path=HDR_image_path, measurand=HDR_measurand)

        return HDR_image_set

    def process_HDR_image(self, ICRF: Optional[ArrayType] = None):
        """
        Main function for merging the input images into HDR images.

        Args:
            ICRF: The utilized ICRF. If no ICRF is given, the default ICRF is loaded instead.
        """
        if ICRF is None:
            ICRF, ICRF_diff = general_functions.read_txt_to_array(gs.ICRF_CALIBRATED_FILE)

        dark_list = ImageSet.multiple_from_path(gs.DEFAULT_DARK_PATH)

        sum_of_weights, square_sum_of_weights = self._precalculate_sum_of_weights(dark_list)
        HDR_imageSet = self._compute_HDR_image_set(dark_list, sum_of_weights, square_sum_of_weights,
                                                   ICRF, ICRF_diff)

        flat_set = HDR_imageSet.get_flat_field()
        if flat_set is not None:
            HDR_imageSet.flat_field_correction(flat_set)

        self.merged_image_set = HDR_imageSet

    def process_linearity(self, ICRF: ArrayType, linearity_limit: Optional[int] = None,
                          use_std: Optional[bool] = False):
        """
        Main method responsible for getting the linearity statistics for the exposure series.
        Args:
            ICRF: the inverse camera response function used to map limits and linearize images.
            linearity_limit: the limits that define how far from data range edges pixel values should be for inclusion.
            use_std: whether to use uncertainty analysis or not.

        Returns:
            the linearity statistics for the exposure series in absolute and relative scales.
        """

        lower, upper = gf.map_linearity_limits(linearity_limit, linearity_limit, ICRF)

        for image_set in self.input_image_sets:
            if image_set.measurand.val is None:
                image_set.load_value_image()
            if image_set.measurand.std is None and use_std:
                image_set.load_std_image()
            image_set.measurand.apply_thresholds(lower, upper)

        exposure_pair: ExposurePair
        for exposure_pair in self.exposure_pairs:
            exposure_pair.compute_difference()
            exposure_pair.compute_stats(axis=(0, 1), release_memory_after=True)

    def collect_exposure_pair_stats(self, return_cupy: Optional[bool] = False):
        """
        Method that manages the collection of stats from each ExposurePair under this ExposureSeries.
        Args:
            return_cupy: whether to return the results as CuPy or NumPy arrays.

        Returns:
            Dictionary of 2-d arrays with first dimension corresponding each ExposurePair and the second dimension
            corresponding to each channel.
        """
        relative_results = {'ratios': [], 'means': [], 'stds': [], 'errors': []}
        absolute_results = {'ratios': [], 'means': [], 'stds': [], 'errors': []}

        exposure_pair: ExposurePair
        for exposure_pair in self.exposure_pairs:
            absolute_results['ratios'].append(exposure_pair.exposure_ratio)
            absolute_results['means'].append(exposure_pair.absolute_stats['mean'])
            absolute_results['stds'].append(exposure_pair.absolute_stats['std'])
            absolute_results['errors'].append(exposure_pair.absolute_stats['error'])

            relative_results['ratios'].append(exposure_pair.exposure_ratio)
            relative_results['means'].append(exposure_pair.relative_stats['mean'])
            relative_results['stds'].append(exposure_pair.relative_stats['std'])
            relative_results['errors'].append(exposure_pair.relative_stats['error'])

        relative_results = _to_2d_array(relative_results, return_cupy)
        absolute_results = _to_2d_array(absolute_results, return_cupy)

        return absolute_results, relative_results


def _to_2d_array(dictionary: Dict, use_cupy: bool):
    """
    Helper function to convert dictionary entries of nested lists into 2d CuPy or NumPy arrays.
    Args:
        dictionary: input dictionary with nested lists as entries.
        use_cupy: whether to return a CuPy or NumPy array.

    Returns:
        Dictionary with its entries converted into 2-d arrays.
    """
    for key in dictionary.keys():

        value = dictionary[key]
        value = cast_to_array(value, use_cupy)
        dictionary[key] = value

    return dictionary


if __name__ == "__main__":
    pass
