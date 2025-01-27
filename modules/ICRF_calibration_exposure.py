"""
Module for solving the inverse camera response function. The solving is based on optimizing an energy function by using
the differential evolution solver of SciPy. The energy function being evaluated measures the linearity of the pixel
values at the same position in a stack of images captured at different exposure times.
"""
from pathlib import Path
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver # Accessed to enable manual interrupt.
from image_set import ImageSet
from exposure_series import ExposureSeries
import general_functions as gf
from typing import Optional
from joblib import delayed, parallel
from global_settings import GlobalSettings as gs
import read_data as rd

from cupy_wrapper import get_array_libraries

np, cp, using_cupy = get_array_libraries()
cnp = cp if using_cupy else np


def _inverse_camera_response_function(mean_ICRF, PCA_array, PCA_params, use_mean_ICRF):
    """
    The inverse camera response function calculated in terms of the mean ICRF, PCA vectors and the PCA coefficients that
    are subject to optimization.Dimensions of the PCA_array and PCA_params must match so that matrix multiplication
    produces an array of equal dimensions to mean_ICRF.

    Args:
        mean_ICRF: Numpy array containing the [0,1] normalized mean irradiance datapoints.
        PCA_array: Numpy array containing the principal component vectors.
        PCA_params: The PCA coefficients for each vector in PCA_array, these the values subject to optimization.

    Return:
        The new iteration of the inverse camera response function.
    """
    if isinstance(PCA_params, np.ndarray):
        temp = cnp.asarray(PCA_params)
    if not use_mean_ICRF:
        mean_ICRF = cnp.linspace(0, 1, gs.BITS) ** temp[0]
        product = cnp.matmul(PCA_array, temp[1:])
    else:
        product = cnp.matmul(PCA_array, temp)

    iterated_ICRF = mean_ICRF + product

    return iterated_ICRF


def weighted_avg_and_std(values: cnp.ndarray, weights: cnp.ndarray):
    """
    Computes the weighted average of an array, while rejecting non-finite values.
    Args:
        values: the array on which to compute the weighted average.
        weights: the array used as the weight in the computation.

    Returns:
        The weighted average.
    """
    finite_indices = cnp.logical_and(cnp.isfinite(values), weights != 0)
    if not cnp.any(finite_indices):
        return cnp.nan
    weights = 1 / weights[finite_indices]
    average = cnp.average(values[finite_indices], weights=weights)

    return average


def analyze_linearity(image_value_stack, image_std_stack, lower: int, upper: int,
                      use_relative: bool, exposure_values):
    """
    Analyze the linearity of images taken at different exposures.
    Args:
        list_of_exposure_series: Optionally pass input_images from previous calculations.
        use_relative: whether to utilize relative or absolute pixel values.
        lower:
        upper:
        use_std:

    Returns:
        An array consisting of the results of the linearity analysis.
    """
    if image_value_stack.ndim != 3:
        raise ValueError("image_stack must be a 3D CuPy array with shape (X, Y, N).")

    if exposure_values.ndim != 1 or exposure_values.size != image_value_stack.shape[2]:
        raise ValueError("exposure_values must be a 1D CuPy array matching the third dimension of image_stack.")

    use_std = False
    if image_std_stack is not None:
        use_std = True

    # Dimensions
    X, Y, N = image_value_stack.shape
    pair_indices = cnp.triu_indices(N, k=1)
    ignored_indices = cnp.tril_indices(N, k=0)

    # Mask values outside the threshold
    mask = (image_value_stack < lower) | (image_value_stack > upper)
    masked_stack = cnp.where(mask, cnp.nan, image_value_stack)

    # Compute exposure ratios for all pairs (N, N)
    exposure_ratios = exposure_values[:, None] / exposure_values[None, :]
    exposure_ratios[ignored_indices] = cnp.nan

    # Expand ratios to match image dimensions: (X, Y, N, N)
    ratio_stack = cnp.expand_dims(exposure_ratios, axis=(0, 1))

    # Broadcast the image stack for pairwise operations: (X, Y, N, N)
    image_stack_i = cnp.expand_dims(masked_stack, axis=3)
    image_stack_j = cnp.expand_dims(masked_stack, axis=2)

    # Compute scaled images
    scaled_image = image_stack_j * ratio_stack

    # Compute linearity differences
    linear_measurand = image_stack_i - scaled_image

    if use_relative:
        linear_measurand /= scaled_image

    # Compute absolute differences: (X, Y, N, N)
    abs_differences = cnp.abs(linear_measurand)

    if use_std:
        image_std_stack_i = cnp.expand_dims(image_std_stack, axis=3)
        image_std_stack_j = cnp.expand_dims(image_std_stack, axis=2)

        if use_relative:
            linear_measurand_std = cnp.sqrt((image_std_stack_i / scaled_image) ** 2 + ((image_stack_i * image_std_stack_j) / (ratio_stack * image_stack_j ** 2) ) ** 2)
        else:
            linear_measurand_std = cnp.sqrt(image_std_stack_i ** 2 + (ratio_stack * image_std_stack_j) ** 2)

    # Average differences across spatial dimensions: (N, N)
    if use_std:
        finite_indices = cnp.logical_and(cnp.isfinite(abs_differences), linear_measurand_std != 0)
        weights = cnp.where(finite_indices, 1 / linear_measurand_std, cnp.nan)
        results = gf.nanaverage(abs_differences, weights, axis=(0, 1))

    else:
        results = cnp.nanmean(abs_differences, axis=(0, 1))

    # Return the upper triangle of the results matrix as a 1D array

    results_vector = results[pair_indices]

    return results_vector


def _energy_function(PCA_params, mean_ICRF, PCA_array, image_value_stack, image_std_stack, lower, upper, use_mean,
                     exposure_values):
    """ The energy function, whose value is subject to minimization. Iterates
    ICRF on a global scale.

    Args:
        PCA_params: The PCA coefficients for each vector in PCA_array, these the values subject to optimization.
        mean_ICRF: Numpy array containing the [0,1] normalized mean irradiance datapoints.
        PCA_array: Numpy array containing the principal component vectors.

    Return:
        The mean skewness of value of all the distributions as a float.
    """
    use_std = False
    if image_std_stack is not None:
        use_std = True

    ICRF_ch = _inverse_camera_response_function(mean_ICRF, PCA_array, PCA_params, use_mean)
    ICRF_diff_ch = None
    ICRF_ch += 1 - ICRF_ch[-1]
    ICRF_ch[0] = 0

    if use_std:
        dx = 2 / (gs.BITS - 1)
        ICRF_diff_ch = cnp.gradient(ICRF_ch, dx)

    if cnp.max(ICRF_ch) > 1 or cnp.min(ICRF_ch) < 0:
        energy = cnp.inf
        return energy

    if not cnp.all(ICRF_ch[1:] > ICRF_ch[:-1]):
        energy = cnp.inf
        return energy

    mapped_lower = ICRF_ch[lower]
    mapped_upper = ICRF_ch[upper]

    iterated_image_value_stack = image_value_stack.copy()
    if use_std:
        iterated_image_std_stack = image_std_stack.copy()
    else:
        iterated_image_std_stack = None

    iterated_image_value_stack = ICRF_ch[iterated_image_value_stack]

    linearity_data = analyze_linearity(iterated_image_value_stack, iterated_image_std_stack, mapped_lower,
                                       mapped_upper, True, exposure_values)

    energy = cnp.nanmean(linearity_data)
    if cnp.isnan(energy):
        energy = cnp.Inf

    energy = float(energy)
    return energy


def _initial_energy_function(x, list_of_exposure_series, channel, lower, upper):
    """
    Function for evaluating the energy function at the initial conditions of the optimization
    process.
    Args:
        x:
        list_of_exposure_series:
        channel:

    Returns:

    """
    initial_function = cnp.linspace(0, 1, gs.BITS) ** x
    dx = 2 / (gs.BITS - 1)
    initial_function_diff = cnp.gradient(initial_function, dx)

    list_of_single_channel_exposure_series = []
    exposure_series: ExposureSeries
    for exposure_series in list_of_exposure_series:
        current_series = exposure_series.extract(channel)
        current_series = current_series.linearize(initial_function, initial_function_diff)
        list_of_single_channel_exposure_series.append(current_series)

    linearity_data = analyze_linearity(list_of_single_channel_exposure_series, lower, upper, use_relative=True)
    energy = cnp.nanmean(linearity_data)
    if cnp.isnan(energy):
        energy = cnp.Inf

    return energy


def interpolate_ICRF(ICRF_array):
    if gs.BITS == gs.DATAPOINTS:
        return ICRF_array

    x_new = cnp.linspace(0, 1, num=gs.BITS)
    x_old = cnp.linspace(0, 1, num=gs.DATAPOINTS)
    interpolated_ICRF = cnp.zeros((gs.BITS, gs.NUM_OF_CHS), dtype=float)

    for c in range(gs.NUM_OF_CHS):
        y_old = ICRF_array[:, c]
        interpolated_ICRF[:, c] = cnp.interp(x_new, x_old, y_old)

    return interpolated_ICRF


def initialize_channel_image_stacks(image_path: Path, use_std: bool, data_spacing: int | tuple[int, int]):
    """
    Initializes the images used in the optimization process by stacking the channels of each image into one array.
    Args:
        image_path: path from which to construct the ImageSets.
        use_std: whether to use uncertainty images of not.
        data_spacing: an integer for determining how sparse the utilized data should be. Larger values result in sparser
            data.

    Returns:

    """
    if type(data_spacing) is tuple:
        x_step = data_spacing[0]
        y_step = data_spacing[1]
    else:
        x_step = data_spacing
        y_step = data_spacing

    list_of_input_image_sets = ImageSet.multiple_from_path(image_path)
    list_of_input_image_sets.sort(key=lambda imageSet: imageSet.features["exposure"])
    number_of_images = len(list_of_input_image_sets)
    exposure_values = []

    first_image = list_of_input_image_sets[0]
    first_image.load_value_image(bit64=True)
    if use_std:
        first_image.load_std_image()

    initial_rows, initial_cols, channels = first_image.measurand.val.shape
    original_number_of_elements = initial_rows * initial_cols

    final_rows, final_cols = gf.predict_output_shape((initial_rows, initial_cols), x_step, y_step)
    final_number_of_elements = final_rows * final_cols
    print(f'Original elements: {original_number_of_elements}, Final elements: {final_number_of_elements}, '
          f'ratio: {final_number_of_elements / original_number_of_elements}')

    channel_image_value_stacks = [cnp.empty((final_rows, final_cols, number_of_images),
                                            dtype=first_image.measurand.val[0].dtype) for _ in range(channels)]

    if use_std:
        channel_image_std_stacks = [cnp.empty((final_rows, final_cols, number_of_images),
                                              dtype=first_image.measurand.std[0].dtype) for _ in range(channels)]
    else:
        channel_image_std_stacks = [None for _ in range(channels)]

    for n, image_set in enumerate(list_of_input_image_sets):
        exposure_values.append(image_set.features['exposure'])
        image_set.load_value_image(bit64=True)
        if use_std:
            image_set.load_std_image()

        image_set.measurand.val = gf.choose_evenly_spaced_points(image_set.measurand.val, x_step, y_step)
        if use_std:
            image_set.measurand.std = gf.choose_evenly_spaced_points(image_set.measurand.std, x_step, y_step)

        for c in range(channels):
            channel_image_value_stacks[c][:, :, n] = image_set.measurand.val[:, :, c]
            if use_std:
                channel_image_std_stacks[c][:, :, n] = image_set.measurand.std[:, :, c]

        del image_set.measurand.val
        del image_set.measurand.std

    exposure_values = cnp.array(exposure_values)

    return channel_image_value_stacks, channel_image_std_stacks, exposure_values


def calibration(lower_PCA_limit: float, upper_PCA_limit: float,
                initial_function: Optional[cp.ndarray] = None,
                data_spacing: Optional[int | tuple[int, int]] = 150,
                data_limits: Optional[tuple[int, int]] = (5, 250),
                use_std: Optional[bool] = False,
                image_path: Optional[Path] = gs.DEFAULT_IMG_SRC_PATH,
                energy_limit: Optional[float] = 0,
                rng_seed: Optional[int] = 7):
    """ The main function running the ICRF calibration process that is called
    from outside the module.

       Args:
           lower_PCA_limit: a lower limit for the PCA coefficient values.
           upper_PCA_limit: an upper limit for the PCA coefficient values.
           initial_function: base function from which iteration starts.
           data_limits: tuple of ints representing the limits for including pixels in linearity analysis.
           data_spacing: used to determine the amount of pixels used in linearity analysis.
           use_std: whether to include uncertainty analysis in the process.
           image_path: the path from which to utilize images in the calibration process.
           energy_limit: limit for the energy function value at which to stop the iteration. Defaults to zero.
           rng_seed: the seed of the solver if one needs consistent operation. Use none for random process.

       Return:
            ICRF_array: a Numpy float array containing the optimized ICRFs of each channel.
            initial_energy_array: a Numpy float array containing the initial energies of each channel.
            final_energy_array: a Numpy float array containing the final energies of each channel.
       """
    ICRF = cp.zeros((gs.DATAPOINTS, gs.NUM_OF_CHS), dtype=float)
    final_energy_array = cp.zeros(gs.NUM_OF_CHS, dtype=float)
    initial_energy_array = cp.zeros(gs.NUM_OF_CHS, dtype=float)

    limits = []
    x0 = []
    if initial_function is None:
        use_mean_ICRF = True
    else:
        use_mean_ICRF = False  # With this option we utilize an additional optimization parameter,
        limits.append([1, 8])  # which is the exponent of a general exponential a^x. These additions
        x0.append(3)  # to the limits and x0 lists correspond to the additional parameter.

    for i in range(gs.NUM_OF_PCA_PARAMS):
        limits.append([lower_PCA_limit, upper_PCA_limit])
        x0.append(0)

    channel_image_value_stacks, channel_image_std_stacks, exposure_values = (
        initialize_channel_image_stacks(image_path, use_std, data_spacing))

    def solve_channel(PCA_file_name: str, mean_ICRF_file_name: str, channel: int, seed,
                      image_value_stack, image_std_stack, exposure_values):

        PCA_array = rd.read_txt_to_array(PCA_file_name)
        if use_mean_ICRF:
            mean_ICRF_array = rd.read_txt_to_array(mean_ICRF_file_name)
        else:
            mean_ICRF_array = initial_function

        args = (
            mean_ICRF_array, PCA_array, image_value_stack, image_std_stack, data_limits[0], data_limits[1],
            use_mean_ICRF, exposure_values)

        number_of_iterations = 0
        # Access DifferentialEvolutionSolver directly to stop iteration if solution has converged or energy function
        # value is under given limit.
        with DifferentialEvolutionSolver(
                _energy_function, limits, args=args,
                strategy='currenttobest1bin', tol=0.01, x0=x0, mutation=(0, 1.95), recombination=0.4,
                init='sobol', seed=seed) as solver:  # seed=1995

            for step in solver:
                number_of_iterations += 1
                step = next(solver)  # Returns a tuple of xk and func evaluation
                func_value = step[1]  # Retrieves the func evaluation
                if number_of_iterations % 20 == 0:
                    print(
                        f'Channel {channel} value: {func_value} on step {number_of_iterations}')
                if solver.converged() or number_of_iterations == 1000 or func_value < energy_limit:
                    break

        result = solver.x
        return_array = _inverse_camera_response_function(mean_ICRF_array, PCA_array, result, use_mean_ICRF)
        del solver

        print(f'Channel {channel} result: f{result}, number of iterations: {number_of_iterations}')

        return return_array

    seeds = [rng_seed + c for c in range(gs.NUM_OF_CHS)]

    # Parallelize the solving of the channels.
    results = parallel.Parallel(n_jobs=gs.NUM_OF_CHS)(
        delayed(solve_channel)(gs.PCA_FILES[c], gs.MEAN_ICRF_FILES[c], c, seeds[c],
                               channel_image_value_stacks[c], channel_image_std_stacks[c], exposure_values)
        for c in range(gs.NUM_OF_CHS))

    for c in range(gs.NUM_OF_CHS):
        ICRF[:, c] = results[c]
        ICRF[:, c] += 1 - ICRF[-1, c]  # The ICRF might be shifted on the y-axis, so we adjust it back to [0,1] here.
        ICRF[0, c] = 0

    # Clipping values just in case. Shouldn't be needed as the ICRF should be continuously increasing between [0,1]
    # without going outside that interval.
    ICRF[ICRF < 0] = 0
    ICRF[ICRF > 1] = 1

    ICRF_interpolated = interpolate_ICRF(ICRF)

    pixel_ratio = 0

    return ICRF_interpolated, initial_energy_array, final_energy_array, pixel_ratio


if __name__ == "__main__":
    pass
