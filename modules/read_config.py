"""
Module for reading data from .ini files and .txt files. Handles all the read operations outside ImageSet.
"""
import configparser
from pathlib import Path

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


if __name__ == "__main__":
    pass
