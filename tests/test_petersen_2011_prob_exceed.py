"""Exceedance probability prediction tests for Petersen et al. (2011) model."""

import numpy as np
import pytest

from fdhpy import PetersenEtAl2011

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["petersen_2011_prob_exceed_elliptical.csv"])
def test_prob_exceed(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 7
    xl = 0.5
    version = "elliptical"
    displ_arr = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed"]

    # Instantiate the model with the scenario attributes
    model = PetersenEtAl2011(magnitude=magnitude, xl=xl, version=version, displ_array=displ_arr)

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
        PetersenEtAl2011(magnitude=7, xl=0.5, percentile=0.5).prob_exceed
