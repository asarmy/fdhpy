"""Unit tests for LoglinearScalingModel class."""

import numpy as np
import pytest

from fdhpy.loglinear_scaling_model import LoglinearScalingModel


def test_valid_params():
    """Should not raise errors or warnings."""
    p = {"intercept": 1.0, "slope": 2.0, "std_dev": 0.5}
    LoglinearScalingModel(regr_params=p)

    p = {"intercept": 1.0, "slope": 2.0, "std_dev": None}  # std_dev can be None
    LoglinearScalingModel(regr_params=p)


def test_invalid_params():
    """Should raise errors or warnings."""
    with pytest.raises(TypeError):
        # Regression parameters are required
        dummy = LoglinearScalingModel()

    with pytest.raises(TypeError):
        p = [1.0, 2.0, 0.5]
        # Dictionary is required
        LoglinearScalingModel(regr_params=p)

    with pytest.raises(AttributeError):
        p = {"intercept": 1.0, "slope": 2.0, "std_dev": 0.5}
        dummy = LoglinearScalingModel(regr_params=p)
        # Regression parameters are immutable
        dummy.regr_params = p

    with pytest.raises(KeyError):
        p = {"intercept": 1.0, "std_dev": None}
        # "slope" key is required
        LoglinearScalingModel(regr_params=p)

    with pytest.warns(UserWarning):
        p = {"intercept": 1.0, "slope": 2.0, "std_dev": None, "key": 1}
        # unexpected key
        LoglinearScalingModel(regr_params=p)


def test_mean_calc():
    p = {"intercept": -2.2192, "slope": 0.3244, "std_dev": 0.17}
    dummy = LoglinearScalingModel(regr_params=p, independent_var=7)
    assert np.isclose(dummy.mean_, 0.0516)

    with pytest.raises(ValueError):
        p = {"intercept": -2.2192, "slope": 0.3244, "std_dev": 0.17}
        # independent variable x is required
        dummy = LoglinearScalingModel(regr_params=p).mean_


def test_calc_prediction():
    p = {"intercept": -2.2192, "slope": 0.3244, "std_dev": 0.17}
    dummy = LoglinearScalingModel(regr_params=p, independent_var=7)
    result = dummy.calc_prediction(percentile=0.84)
    assert np.isclose(result, 1.662103)

    result = dummy.calc_prediction(percentile=-1)
    assert np.isclose(result, 1.21583)

    dummy = LoglinearScalingModel(regr_params=p, independent_var=[6, 7])
    result = dummy.calc_prediction(percentile=-1)
    assert np.allclose(result, np.array([0.576066, 1.21583]))

    with pytest.raises(ValueError):
        p = {"intercept": -2.2192, "slope": 0.3244, "std_dev": None}
        dummy = LoglinearScalingModel(regr_params=p, independent_var=7)
        # std_dev was not provided
        result = dummy.calc_prediction(percentile=0.84)
