"""
Module for reading data from .ini files and .txt files. Handles all the read operations outside of ImageSet.
"""
import configparser
from typing import Optional
from pathlib import Path

from cupy_wrapper import get_array_libraries

np, cp, using_cupy = get_array_libraries()
cnp = cp if using_cupy else np

current_directory = Path(__file__).parent.resolve()
root_directory = current_directory.parent
data_directory = root_directory.joinpath('data')


def read_config_list(key):
    """
    Read list config data from the config.ini file in the data directory.

    :param key: The keyword for a particular config list entry.

    :return: Returns a list of strings, ints or floats.
    """
    config = configparser.ConfigParser()
    config.read(data_directory.joinpath('config.ini'))
    sections = config.sections()
    data_list = []
    for section in sections:

        if key in config[section]:

            data_list = config[section][key].split(',')

            if section == 'Float data':

                data_list = [float(element) for element in data_list]

            elif section == 'Integer data':

                data_list = [int(element) for element in data_list]

    return data_list


def read_config_single(key):
    """
    Read single config data from the config.ini file in the data directory.

    :param key: The keyword for a particular config entry.

    :return: Returns a single string, int, or float value.
    """
    config = configparser.ConfigParser()
    config.read(data_directory.joinpath('config.ini'))
    sections = config.sections()
    single_item = ''
    for section in sections:

        if key in config[section]:

            single_item = config[section][key]

            if section == 'Float data':

                single_item = float(single_item)

            if section == 'Integer data':

                single_item = int(single_item)

    return single_item


def read_txt_to_array(file_name: str, path: Optional[str] = None, use_cupy: Optional[bool] = True):
    """
    Load numerical data from a .txt file of given name. Defaults to data
    directory but optionally can use other paths.

    Args:
        file_name: name of the file to load.
        path: path to the file, optional.
        use_cupy: whether to load data into a CuPy or NumPy array.

    Returns: numpy array of the txt file.
    """
    if path is None:
        load_path = data_directory
    else:
        load_path = Path(path)

    if use_cupy and using_cupy:
        data_array = cp.loadtxt(load_path.joinpath(file_name), dtype=float)
    else:
        data_array = np.loadtxt(load_path.joinpath(file_name), dtype=float)

    return data_array


if __name__ == "__main__":
    pass
