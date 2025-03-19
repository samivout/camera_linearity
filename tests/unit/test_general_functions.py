import pytest
from hypothesis import given
from hypothesis import strategies as st
from typing import Optional

from modules import general_functions as gf
from conftest import USE_CUPY, xp


@given(
    st.lists(st.integers(min_value=1), min_size=0, max_size=10), # Shape 1: List of positive integers (dims between 1 and 5)
    st.lists(st.integers(min_value=1), min_size=0, max_size=10)  # Shape 2: Another shape with dims between 1 and 5
)
def test_is_broadcastable(shape1, shape2):
    # Ensure the shapes are valid (non-empty lists, positive integers)
    if not shape1 or not shape2:
        with pytest.raises(ValueError, match='Shapes cannot be empty') as e:
            gf.is_broadcastable(shape1, shape2)
        assert e.type is ValueError
        return

    # Get the actual result from the is_broadcastable function
    result = gf.is_broadcastable(shape1, shape2)

    max_len = max(len(shape1), len(shape2))
    shape1 = [1] * (max_len - len(shape1)) + shape1
    shape2 = [1] * (max_len - len(shape2)) + shape2

    # Validate broadcasting compatibility dimension by dimension
    for dim1, dim2 in zip(shape1, shape2):
        if not (dim1 == 1 or dim2 == 1 or dim1 == dim2):
            # If any pair of dimensions is incompatible, result should be False
            assert not result
            return

    # If we didn't find any incompatible dimensions, result should be True
    assert result


@given(lower_limit=st.one_of(st.none(), st.integers()), upper_limit=st.one_of(st.none(), st.integers()),
       ICRF=st.from_type(Optional[xp.ndarray]))
def test_fuzz_map_linearity_limits(lower_limit: Optional[int], upper_limit: Optional[int], ICRF: Optional[xp.ndarray]):
    # TODO: this test needs to be looked into.
    gf.map_linearity_limits(lower_limit=lower_limit, upper_limit=upper_limit, ICRF=ICRF)
