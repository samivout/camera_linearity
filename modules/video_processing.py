"""
Module for processing video files. Main functionality is computing mean and std frames of video files. 
"""
import numpy as np
import cv2 as cv
import general_functions as gf
from typing import Optional, List, Union
from global_settings import GlobalSettings as gs


def clean_data_edges(base_data_arr: np.ndarray):
    """
    Function to clean the edges of the noise data distributions. For each distribution start at the expected middle
    value and loop in both positive and negative directions, while replacing unexpected dips in values by the average
    of its neighbours. This process is followed by looping from the minimum and maximum values towards the center, while
    setting the distribution values so that it is monotonically increasing towards the center.
    Args:
        base_data_arr: The original noise data array.

    Returns:
        Array containing the cleaned noise distributions.
    """

    for i in range(gs.BITS):
        dist = base_data_arr[i, :]

        # Center index for processing
        center = i

        # --- Smooth from center towards the minimum edge ---
        m = center - 1
        while m > gs.MIN_DN:
            if dist[m] == 0 and dist[m - 1] == 0:
                dist[:m] = 0
                break
            if dist[m - 1] >= dist[m] or dist[m + 1] <= dist[m]:
                dist[m] = (dist[m - 1] + dist[m + 1]) // 2  # Smooth using neighbors
            m -= 1

        # --- Smooth from center towards the maximum edge ---
        m = center + 1
        while m < gs.MAX_DN:
            if dist[m] == 0 and dist[m + 1] == 0:
                dist[m:] = 0
                break
            if dist[m + 1] >= dist[m] or dist[m - 1] <= dist[m]:
                dist[m] = (dist[m - 1] + dist[m + 1]) // 2  # Smooth using neighbors
            m += 1

        # --- Ensure monotonicity from minimum edge to center ---
        m = gs.MIN_DN + 1
        while m < center:
            if dist[m] == 0 and dist[m - 1] != 0 and dist[m + 1] != 0:
                dist[m] = dist[m - 1]  # Fill gaps with previous value
            elif dist[m] == dist[m + 1] and dist[m] != 0:
                dist[m + 1] += 1  # Prevent flat spots
                m -= 1  # Revisit the modified index
            m += 1

        # --- Ensure monotonicity from maximum edge to center ---
        m = gs.MAX_DN - 1
        while m > center:
            if dist[m] == 0 and dist[m - 1] != 0 and dist[m + 1] != 0:
                dist[m] = dist[m + 1]  # Fill gaps with next value
            elif dist[m] == dist[m - 1] and dist[m] != 0:
                dist[m - 1] += 1  # Prevent flat spots
                m += 1  # Revisit the modified index
            m -= 1

        # Update the original array with cleaned distribution
        base_data_arr[i, :] = dist

    return base_data_arr


def compute_noise_profiles(video_files: List[Path]):
    """
    Function for computing the noise profiles of a camera based on videos of a static scene. Uses a precomputed mean
    frame of the video for faster computation.
    Args:
        video_files: path to the video files to utilize in the computation.

    Returns:
        Noise profiles as NumPy array, shape (gs.BITS, gs.BITS, gs.NUM_OF_CHS), and mean frame of the videos.
    """
    noise_profiles = np.zeros((gs.BITS, gs.BITS, gs.NUM_OF_CHS), dtype=int)

    mean_frame = welford_algorithm(video_files, None, False)['mean']

    for video_file in video_files:

        frame_generator = gf.video_frame_generator(video_file)
        for frame in frame_generator:

            if frame is None:
                break

            for c in range(gs.NUM_OF_CHS):

                frame_channel = frame[..., c].flatten()
                mean_channel = mean_frame[..., c].flatten()

                np.add.at(noise_profiles[:, :, c], (mean_channel, frame_channel), 1)

    return noise_profiles, mean_frame


def _calculate_STD(mean_data_array: np.ndarray):
    """
    Function for computing the expected standard deviation for each signal level based on the mean data.
    Args:
        mean_data_array: NumPy array containing the mean data in shape (gs.BITS, gs.BITS)

    Returns:
        NumPy array in shape in shape (gs.BITS,) containing the standard deviation of each signal level.
    """
    STD_array = np.zeros(gs.MAX_DN + 1, dtype=float)

    for i in range(gs.MAX_DN + 1):

        bin_edges = np.linspace(0, 1, num=gs.DATAPOINTS, dtype=float)
        hist = mean_data_array[i, :]
        nonzeros = np.nonzero(hist)
        hist = hist[nonzeros]
        bin_edges = bin_edges[nonzeros]
        counts = np.sum(hist)
        mean = np.sum(hist * bin_edges)/counts
        squared_variances = np.power((bin_edges - mean), 2) * hist
        STD = math.sqrt(np.sum(squared_variances)/counts)
        STD_array[i] = STD

    return STD_array


def process_STD_data(pass_result: Optional[bool] = True):
    """
    Main function for managing the STD data calculation process.
    Args:
        pass_result: whether to return the result or not.

    Returns:
        Conditionally returns the STD data array, shape (gs.BITS, gs.BITS, gs.NUM_OF_CHS).
    """
    mean_data_array = np.zeros((gs.MAX_DN + 1, gs.DATAPOINTS, gs.NUM_OF_CHS), dtype=int)
    STD_data = np.zeros((gs.MAX_DN + 1, gs.NUM_OF_CHS), dtype=float)
    for i in range(len(gs.MEAN_DATA_FILES)):

        mean_file_name = gs.MEAN_DATA_FILES[i]
        mean_data_array[:, :, i] = rd.read_txt_to_array(mean_file_name)
        STD_data[:, i] = _calculate_STD(mean_data_array[:, :, i])

    np.savetxt(data_directory.joinpath(gs.STD_FILE_NAME), STD_data)

    if pass_result:
        return STD_data

    return


def welford_algorithm(file_paths: Union[Path, List[Path]], ICRF: Optional[np.ndarray], use_std: Optional[bool] = False):
    """
    Implementation of the Welford algorithm for calculating the mean and standard deviation frame from all the frames of
    a single video or videos in a directory. Streams the frames into memory one by one via generator. If ICRF is given,
    each frame is linearized upon loading to memory, providing technically a more accurate estimation of the mean frame
    and standard deviation frame.
    Args:
        file_paths: path to the video file or directory containing video fiels.
        ICRF: The inverse camera response function as NumPy array, should be shaped as (gs.BITS, gs.NUM_OF_CHS) and match the
            bitdepth and number of channels of the video.
        use_std:
            Whether to calculate the standard deviation or not.
    Returns:
        Dictionary containing the mean frame and standard deviation frame.
    """
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    video = cv.VideoCapture(str(file_paths[0]))
    video_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    mean = np.zeros((video_height, video_width, gs.NUM_OF_CHS), dtype=np.dtype('float64'))
    m2 = None
    if use_std:
        m2 = np.zeros((video_height, video_width, gs.NUM_OF_CHS), dtype=np.dtype('float64'))

    total_frame_count = 0
    for file_path in file_paths:

        frame_generator = gf.video_frame_generator(file_path)

        for frame in frame_generator:

            if frame is None:
                break

            total_frame_count += 1

            if ICRF:
                frame = ICRF[frame, np.arange(gs.NUM_OF_CHS)]
            else:
                frame = (frame/gs.MAX_DN).astype(np.dtype('float64'))

            delta = frame - mean
            mean = mean + delta / total_frame_count
            if use_std:
                m2 = m2 + delta * (frame - mean)

    mean = mean * gs.MAX_DN
    mean = (np.around(mean)).astype(np.dtype('uint8'))

    if use_std:
        m2 = np.sqrt(m2 / (total_frame_count - 1)) / np.sqrt(total_frame_count)
        m2 = (np.around(m2)).astype(np.dtype('uint8'))

    ret = {'mean': mean, 'std': m2}

    return ret


def process_video(video_path: Path, ICRF: Optional[np.ndarray] = None, use_std: Optional[bool] = True):
    """
    Function for managing the process of computing a mean and std frame of a single video file.
    Args:
        video_path: path to a single video file.
        ICRF: Inverse camera reseponse function as NumPy array, with shape (gs.BITS, gs.NUM_OF_CHS) that match the video.
        use_std: whether to compute standard deviation frame or not.
    """

    ret = welford_algorithm(video_path, ICRF, use_std)

    for key in ret:
        if ret[key] is not None:
            save_path = str(video_path.parent.joinpath(video_path.name.replace('.avi', f'.{key}.tif')))
            cv.imwrite(save_path, ret[key])


def process_directory(dir_path: Path, ICRF: Optional[np.ndarray] = None, separately: Optional[bool] = True):
    """
    Function for managing the processing of a directory of video files. The videos can be each processed separately or
    a single mean and std frame can be computed for them all.
    Args:
        dir_path: path to the directory containing the video files.
        ICRF: the inverse camera response function, shape (gs.BITS, gs.NUM_OF_CHS) should match the video data.
        separately: whether to process the files separately.
    """
    video_files = list(dir_path.glob("*.avi"))

    if not separately:
        ret = welford_algorithm(video_files, ICRF)

        for key in ret:
            if ret[key] is not None:
                save_path = str(dir_path.joinpath(f'total_{key}.tif'))
                cv.imwrite(save_path, ret[key])

    else:
        for path in video_files:
            print(f'Starting video file {path}')
            ret = welford_algorithm(path, ICRF)
            print('Finished file')

            for key in ret:
                if ret[key] is not None:
                    save_dir = path.parent.joinpath(key)
                    save_dir.mkdir(exist_ok=True)
                    if key == 'std':
                        save_path = str(save_dir.joinpath(path.name.replace('.avi', f' STD.tif')))
                    else:
                        save_path = str(save_dir.joinpath(path.name.replace('.avi', f'.tif')))
                    cv.imwrite(save_path, ret[key])

    return


if __name__ == "__main__":
    pass
