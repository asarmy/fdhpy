"""Exceedance probability prediction tests for Chiou et al. (2025) model."""

import numpy as np
import pytest

from fdhpy import ChiouEtAl2025

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["chiou_2025_prob_exceed_model7.csv"])
def test_prob_exceed_model7(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 6.5
    xl = 0.25
    version = "model7"
    displ_arr = load_expected["displacement"]

    # Expected
    expected = load_expected["prob_exceed"]

    # Instantiate the model with the scenario attributes
    model = ChiouEtAl2025(magnitude=magnitude, xl=xl, version=version, displ_array=displ_arr)

    computed = model.prob_exceed

    # Testing
    np.testing.assert_allclose(
        expected,
        computed,  # type: ignore
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} test, Expected: {expected}, Computed: {computed}",
    )


@pytest.mark.parametrize("filename", ["chiou_2025_prob_exceed_model82.csv"])
def test_prob_exceed_model82(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 6.5
    xl = 0.25
    version = "model8.2"
    displ_arr = load_expected["displacement"]

    # Expected
    expected = load_expected["prob_exceed"]

    # Instantiate the model with the scenario attributes
    model = ChiouEtAl2025(magnitude=magnitude, xl=xl, version=version, displ_array=displ_arr)

    computed = model.prob_exceed

    # Testing
    np.testing.assert_allclose(
        expected,
        computed,  # type: ignore
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} test, Expected: {expected}, Computed: {computed}",
    )


def test_invalid_inputs(caplog):
    """Should raise errors or warnings."""

    with caplog.at_level("WARNING"):
        # Ignored attribute `percentile`
        ChiouEtAl2025(magnitude=7, xl=0.5, percentile=0.5).prob_exceed
