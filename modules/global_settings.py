"""
Module for reading and importing global settings from config.ini file in the project data directory.
TODO: Refactor some things to instance attributes of ImageSet.
"""
import read_config as rd
from pathlib import Path


class GlobalSettings:
    # Paths for data storage and module execution
    DATA_PATH = rd.data_directory
    MODULE_PATH = rd.current_directory
    OUTPUT_PATH = MODULE_PATH.parent.joinpath('output')

    # Image size and related parameters
    IM_SIZE_X = rd.read_config_single('image size x')
    IM_SIZE_Y = rd.read_config_single('image size y')
    PIXEL_COUNT = IM_SIZE_Y * IM_SIZE_Y

    # File paths for image calibration and processing
    DEFAULT_IMG_SRC_PATH = Path(rd.read_config_single('acquired images path'))
    DEFAULT_FLAT_PATH = Path(rd.read_config_single('flat fields path'))
    DEFAULT_DARK_PATH = Path(rd.read_config_single('dark frames path'))
    UNCALIBRATED_FLAT_PATH = Path(rd.read_config_single('original flat fields path'))
    UNCALIBRATED_DARK_PATH = Path(rd.read_config_single('original dark frames path'))
    ICRF_CALIBRATED_FILE = rd.read_config_single('calibrated ICRFs')

    # Channel-related settings
    NUM_OF_CHS = rd.read_config_single('channels')
    CH_NAMES = rd.read_config_list('channel names')
    CH_CHARS = [element[0] for element in CH_NAMES]
    CH_STR = {0: 'Blue', 1: 'Green', 2: 'Red'}  # Mapping of channel indices to names

    # Bit depth and pixel intensity values
    BIT_DEPTH = rd.read_config_single('bit depth')
    BITS = 2 ** BIT_DEPTH
    MAX_DN = BITS - 1
    MIN_DN = 0

    # Data processing and statistical parameters
    DATAPOINTS = rd.read_config_single('final datapoints')
    DATAPOINT_MULTIPLIER = rd.read_config_single('datapoint multiplier')
    STD_FILE_NAME = rd.read_config_single('STD data')

    # Preprocessed camera data files
    MEAN_DATA_FILES = rd.read_config_list('camera mean data')
    BASE_DATA_FILES = rd.read_config_list('camera base data')

    # Reference and calibration data files
    DORF_FILE = rd.read_config_single('source DoRF data')
    DORF_DATAPOINTS = rd.read_config_single('original DoRF datapoints')
    ICRF_FILES = rd.read_config_list('ICRFs')
    MEAN_ICRF_FILES = rd.read_config_list('mean ICRFs')

    # Principal Component Analysis (PCA) settings
    NUM_OF_PCA_PARAMS = rd.read_config_single('number of principal components')
    PCA_FILES = rd.read_config_list('principal components')
    IN_PCA_GUESS = rd.read_config_list('initial guess')

    # Image correction and filtering thresholds
    DARK_THRESHOLD = rd.read_config_single('dark threshold')
    FF_MID_PERCENTAGE = rd.read_config_single('flat field middle zone percentage')
    HOT_PIXEL_THRESHOLD = rd.read_config_single('hot pixel threshold')
    MEDIAN_FILTER_KERNEL_SIZE = rd.read_config_single('median filter kernel size')

    # Linearity limits for pixel intensity correction
    LOWER_LIN_LIM = rd.read_config_single('lower linearity limit')
    UPPER_LIN_LIM = rd.read_config_single('upper linearity limit')

    # Pixel size mapping for different magnifications
    PIXEL_SIZE = {
        '5x': 0.9235, '10x': 0.4617, '20x': 0.2309, '50x': 0.0923,
        '1000x': 0.05464480874, '3000x': 0.01724137931, '8000x': 0.006756756757
    }
    PIXEL_SIZE_U = {
        '5x': 0.0088, '10x': 0.0044, '20x': 0.0022, '50x': 0.0009,
        '1000x': 0.002732240437, '3000x': 0.0008620689655, '8000x': 0.0003378378379
    }

    # Background levels for each color channel
    CH_BG_LVL = {0: 0.14, 1: 0.27, 2: 0.18}
