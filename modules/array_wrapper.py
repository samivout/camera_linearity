from typing import Optional, Union
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    ArrayType = Union[np.ndarray, cp.ndarray]
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
    ArrayType = np.ndarray


def cast_to_array(generic_value, use_cupy: Optional[bool] = False):
    """
    Wrapper function to convert generics to either a CuPy array or NumPy array based on availability and user input.
    Args:
        generic_value: a type that can be cast to a CuPy array.
        use_cupy: whether to return a CuPy or NumPy array.
    Returns:
        CuPy array of the generic, provided that CuPy is available. Otherwise, returns input as is.
    """
    if not CUPY_AVAILABLE or not use_cupy:
        return np.array(generic_value)

    return cp.array(generic_value)


def get_array_lib(input_array: Optional[ArrayType]):
    """
    Utility function to get matching array library type with the input array. Defaults to NumPy when no input is given
    or the input is not a CuPy array.
    Args:
        input_array: array to match the library with.

    Returns:
        reference to the appropriate array library.
    """
    if isinstance(input_array, cp.ndarray):
        return cp
    else:
        return np
