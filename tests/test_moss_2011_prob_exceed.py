"""Exceedance probability prediction tests for Moss and Ross (2011) model."""

import numpy as np
import pytest

from fdhpy import MossRoss2011

# Test setup
RTOL = 1e-2
ATOL = 1e-4


@pytest.mark.parametrize("filename", ["moss_2011_prob_exceed_d_ad.csv"])
def test_prob_exceed_d_ad(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 7
    xl = 0.5
    version = "d/ad"
    displ_arr = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Instantiate the model with the scenario attributes
    model = MossRoss2011(magnitude=magnitude, xl=xl, version=version, displ_array=displ_arr)

    computed = model.prob_exceed

    # Testing
    np.testing.assert_allclose(
        expected,
        computed,  # type: ignore
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} test, Expected: {expected}, Computed: {computed}",
    )


@pytest.mark.parametrize("filename", ["moss_2011_prob_exceed_d_md.csv"])
def test_prob_exceed_d_md(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 7
    xl = 0.5
    version = "d/md"
    displ_arr = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Instantiate the model with the scenario attributes
    model = MossRoss2011(magnitude=magnitude, xl=xl, version=version, displ_array=displ_arr)

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
        MossRoss2011(magnitude=7, xl=0.5, percentile=0.5, version="d/ad").prob_exceed
