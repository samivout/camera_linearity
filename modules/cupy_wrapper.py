"""
A wrapper module for setting whether to prefer using CuPy or NumPy based on availability.
TODO: create an override based on global settings.
"""


def get_array_libraries():
    """
    Getter for setting up use of NumPy and possibly CuPy. Common functionalities are supposed to prefer CuPy over NumPy.

    Returns:
        imported libraries and
    """
    import numpy as np
    try:
        import cupy as cp
        using_cupy = True
    except ImportError:
        cp = None
        using_cupy = False

    return np, cp, using_cupy


def get_filter_functions():
    """
    Getter for setting up use of filter functions on either CuPy or SciPy, based on availability of CuPy.
    Returns:
        Either CuPy or SciPy implementation of the gaussian and median filters.
    """
    try:
        from cupyx.scipy.ndimage import gaussian_filter
        from cupyx.scipy.ndimage import median_filter
    except ImportError:
        from scipy.ndimage import gaussian_filter
        from scipy.ndimage import median_filter

    return gaussian_filter, median_filter
