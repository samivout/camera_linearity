from typing import Optional
from measurand import NumPyMeasurand
import numpy as np

try:
    import cupy as cp
    from cupy_measurand import CuPyMeasurand
    CUPY_AVAILABLE = True
except ImportError:
    CuPyMeasurand = NumPyMeasurand
    cp = np
    CUPY_AVAILABLE = False


def Measurand(val: Optional = None, std: Optional = None, use_cupy=True):
    """Factory function to return the appropriate Measurand class."""
    if use_cupy and CUPY_AVAILABLE:
        return CuPyMeasurand(val, std)
    return NumPyMeasurand(val, std)


def measurand_to_cupy(numpy_measurand: NumPyMeasurand):
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

    return CuPyMeasurand(val=ret_val, std=ret_std)


def measurand_to_numpy(cupy_measurand: CuPyMeasurand):
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

    return NumPyMeasurand(val=ret_val, std=ret_std)
