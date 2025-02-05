from typing import Optional, Union
from measurand import NumpyMeasurand
import numpy as np

try:
    import cupy as cp
    from cupy_measurand import CupyMeasurand
    CUPY_AVAILABLE = True
    ArrayType = Union[np.ndarray, cp.ndarray]
except ImportError:
    CuPyMeasurand = NumpyMeasurand
    cp = np
    CUPY_AVAILABLE = False
    ArrayType = np.ndarray


def Measurand(val: Optional = None, std: Optional = None, use_cupy=True):
    """Factory function to return the appropriate Measurand class."""
    if use_cupy and CUPY_AVAILABLE:
        return CupyMeasurand(val, std)
    return NumpyMeasurand(val, std)


def generic_to_array(generic_value, use_cupy: Optional[bool] = False):
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


def measurand_to_cupy(numpy_measurand: NumpyMeasurand):
    """
     Returns a new instance of the measurand converted to utilize CuPy.
     Returns:
         A CuPy version of this measurand.
     """
    if not CUPY_AVAILABLE:
        return numpy_measurand

    if numpy_measurand.val is not None:
        ret_val = cp.array(numpy_measurand.val)
    else:
        ret_val = None
    if numpy_measurand.std is not None:
        ret_std = cp.array(numpy_measurand.std)
    else:
        ret_std = None

    return CupyMeasurand(val=ret_val, std=ret_std)


def measurand_to_numpy(cupy_measurand: CupyMeasurand):
    """
    Returns a new instance of the measurand converted to utilize NumPy.
    Returns:
        A NumPy version of this measurand.
    """
    if not CUPY_AVAILABLE:
        return cupy_measurand

    if cupy_measurand.val is not None:
        ret_val = cp.asnumpy(cupy_measurand.val)
    else:
        ret_val = None
    if cupy_measurand.std is not None:
        ret_std = cp.asnumpy(cupy_measurand.std)
    else:
        ret_std = None

    return NumpyMeasurand(val=ret_val, std=ret_std)
