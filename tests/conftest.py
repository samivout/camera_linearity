import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np

# Global variable to track whether to use CuPy or NumPy
USE_CUPY = False
xp = np


def pytest_addoption(parser):
    """Add a command-line option to select NumPy or CuPy globally."""
    parser.addoption(
        "--use-cupy",
        action="store_true",
        default=False,
        help="Run tests with CuPy instead of NumPy"
    )


def pytest_configure(config):
    """Set USE_CUPY globally based on command-line arguments."""
    global USE_CUPY
    global xp
    USE_CUPY = config.getoption("--use-cupy")
    if USE_CUPY:
        xp = cp
    else:
        xp = np
