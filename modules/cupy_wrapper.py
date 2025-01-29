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
