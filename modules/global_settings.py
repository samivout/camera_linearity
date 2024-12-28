"""
Module for reading and importing global settings from config.ini file in the project data directory.
TODO: reorganize the global constants into more logical sets. Refactor some things to instance attributes of ImageSet.
"""
import read_data as rd
from pathlib import Path

# From read_data.py
data_directory = rd.data_directory
module_directory = rd.current_directory

# From HDR_full_process.py
IM_SIZE_X = rd.read_config_single('image size x')
IM_SIZE_Y = rd.read_config_single('image size y')
PIXEL_COUNT = IM_SIZE_Y * IM_SIZE_Y
DEFAULT_ACQ_PATH = Path(rd.read_config_single('acquired images path'))
FLAT_PATH = Path(rd.read_config_single('flat fields path'))
OG_FLAT_PATH = Path(rd.read_config_single('original flat fields path'))
DARK_PATH = Path(rd.read_config_single('dark frames path'))
OG_DARK_PATH = Path(rd.read_config_single('original dark frames path'))
OUT_PATH = Path(rd.read_config_single('corrected output path'))
ICRF_CALIBRATED_FILE = rd.read_config_single('calibrated ICRFs')
CHANNELS = rd.read_config_single('channels')
CHANNEL_NAMES = rd.read_config_list('channel names')
CHANNEL_CHAR = [element[0] for element in CHANNEL_NAMES]
BIT_DEPTH = rd.read_config_single('bit depth')
BITS = 2 ** BIT_DEPTH
MAX_DN = BITS - 1
MIN_DN = 0
DATAPOINTS = rd.read_config_single('final datapoints')
DATA_MULTIPLIER = DATAPOINTS / BITS
DATAPOINT_MULTIPLIER = rd.read_config_single('datapoint multiplier')
STD_FILE_NAME = rd.read_config_single('STD data')
AVERAGED_FRAMES = rd.read_config_single('averaged frames')

# From camera_data_tools.py
MEAN_DATA_FILES = rd.read_config_list('camera mean data')
BASE_DATA_FILES = rd.read_config_list('camera base data')

# From process_CRF_database.py
DORF_FILE = rd.read_config_single('source DoRF data')
DORF_DATAPOINTS = rd.read_config_single('original DoRF datapoints')
ICRF_FILES = rd.read_config_list('ICRFs')
MEAN_ICRF_FILES = rd.read_config_list('mean ICRFs')

# From ICRF_calibration_exposure.py
OUTPUT_DIRECTORY = module_directory.parent.joinpath('output')
NUM_OF_PCA_PARAMS = rd.read_config_single('number of principal components')
PCA_FILES = rd.read_config_list('principal components')
ACQ_PATH = Path(rd.read_config_single('acquired images path'))

# From image_correction.py
DARK_THRESHOLD = rd.read_config_single('dark threshold')
FF_MID_PERCENTAGE = rd.read_config_single('flat field middle zone percentage')
HOT_PIXEL_THRESHOLD = rd.read_config_single('hot pixel threshold')
MEDIAN_FILTER_KERNEL_SIZE = rd.read_config_single('median filter kernel size')

# From calibration_benchmark.py
EVALUATION_HEIGHTS = rd.read_config_list('evaluation heights')
NUMBER_OF_HEIGHTS = len(EVALUATION_HEIGHTS)
LOWER_PCA_LIM = rd.read_config_single('lower PC coefficient limit')
UPPER_PCA_LIM = rd.read_config_single('upper PC coefficient limit')
IN_PCA_GUESS = rd.read_config_list('initial guess')

# From camera_noise_distribution.py
VIDEO_PATH = Path(rd.read_config_single('video path'))

# New
LOWER_LIN_LIM = rd.read_config_single('lower linearity limit')
UPPER_LIN_LIM = rd.read_config_single('upper linearity limit')

CH_STR = {0: 'Blue', 1: 'Green', 2: 'Red'}
PIXEL_SIZE = {'5x': 0.9235, '10x': 0.4617, '20x': 0.2309, '50x': 0.0923,
              '1000x': 0.05464480874, '3000x': 0.01724137931, '8000x': 0.006756756757}
PIXEL_SIZE_U = {'5x': 0.0088, '10x': 0.0044, '20x': 0.0022, '50x': 0.0009,
                '1000x': 0.002732240437, '3000x': 0.0008620689655, '8000x': 0.0003378378379}
CH_BG_LVL = {0: 0.14, 1: 0.27, 2: 0.18}
