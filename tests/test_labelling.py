import numpy as np
import pytest
import torch

import hm4.labelling as labelling
from hm4.labelling import N_CLASSES


def test_is_array_zero_indexed_zero_indexed_arr():
    # test 0-indexed numpy array
    arr = np.array([0, 1, 2, 3, N_CLASSES - 1])
    assert labelling.is_array_zero_indexed(arr)


def test_is_array_zero_indexed_one_indexed_arr():
    # test 1-indexed numpy array
    arr = np.array([1, 2, 3, 4, N_CLASSES])
    assert not labelling.is_array_zero_indexed(arr)


def test_is_array_zero_indexed_ambiguous_arr():
    # test ambiguous numpy array
    arr = np.array([0, 1, 2, 3, N_CLASSES])
    assert not labelling.is_array_zero_indexed(arr)


def test_make_array_zero_indexed_fwrds_arr():
    # test 1-indexed to 0-indexed with numpy array
    arr = np.array([1, 2, 3, 4, N_CLASSES])
    expected = np.array([0, 1, 2, 3, N_CLASSES - 1])
    assert np.array_equal(labelling.make_array_zero_indexed(arr), expected)


def test_make_array_zero_indexed_bckwrds_arr():
    # test 0-indexed to 1-indexed with numpy array
    arr = np.array([0, 1, 2, 3, N_CLASSES - 1])
    expected = np.array([1, 2, 3, 4, N_CLASSES])
    assert np.array_equal(
        labelling.make_array_zero_indexed(arr, forwards=False), expected
    )


def test_make_array_zero_indexed_fwrds_tensor():
    # test torch tensor forwards
    arr = torch.tensor([1, 2, 3, 4, N_CLASSES])
    expected = torch.tensor([0, 1, 2, 3, N_CLASSES - 1])
    assert torch.equal(labelling.make_array_zero_indexed(arr), expected)


def test_make_array_zero_indexed_fwrds_on_zero_indexed():
    # test 0-indexed array going to 0-indexed raises error
    arr = np.array([0, 1, 2, 3, N_CLASSES - 1])
    with pytest.raises(ValueError):
        labelling.make_array_zero_indexed(arr)


def test_make_array_zero_indexed_ambiguous_arr():
    # test ambiguous array raises error
    arr = np.array([0, 1, 2, 3, N_CLASSES])
    with pytest.raises(ValueError):
        labelling.make_array_zero_indexed(arr, forwards=False)


def test_make_array_zero_indexed_bckwrds_on_one_indexed():
    # test 1-indexed array going to 1-indexed raises error
    arr = np.array([1, 2, 3, 4, N_CLASSES])
    with pytest.raises(ValueError):
        labelling.make_array_zero_indexed(arr, forwards=False)


def test_make_array_zero_indexed_force():
    # test force conversion forwards
    arr = np.array([0, 1, 2, 3, 4])
    expected = np.array([-1, 0, 1, 2, 3])
    assert np.array_equal(
        labelling.make_array_zero_indexed(arr, force=True), expected
    )


def test_make_array_zero_indexed_force_bckwrds():
    # test force conversion backwards
    arr = np.array([1, 2, 3, 4, 5])
    expected = np.array([2, 3, 4, 5, 6])
    assert np.array_equal(
        labelling.make_array_zero_indexed(arr, forwards=False, force=True),
        expected,
    )


def test_change_pixel_label_default_args():
    # test default
    arr = np.array([0, 1, 2, 3, 38])
    expected = np.array([0, 1, 2, 3, 21])
    assert np.array_equal(labelling.change_pixel_label(arr), expected)


def test_change_pixel_label_custom_inout_vals():
    # test custom values
    arr = np.array([0, 1, 2, 3, 38])
    expected = np.array([0, 0, 2, 3, 38])
    assert np.array_equal(
        labelling.change_pixel_label(arr, in_val=1, out_val=0), expected
    )


def test_change_pixel_label_identity():
    # test identity (no change)
    arr = np.array([0, 1, 2, 3, 38])
    expected = np.array([0, 1, 2, 3, 38])
    assert np.array_equal(
        labelling.change_pixel_label(arr, in_val=10, out_val=10), expected
    )
