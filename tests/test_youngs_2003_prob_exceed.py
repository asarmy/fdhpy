"""Exceedance probability prediction tests for Youngs et al. (2003) model."""

import numpy as np
import pytest

from fdhpy import YoungsEtAl2003

# Test setup
RTOL = 1e-2
ATOL = 1e-4


@pytest.mark.parametrize("filename", ["youngs_2003_prob_exceed_d_ad.csv"])
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
    model = YoungsEtAl2003(magnitude=magnitude, xl=xl, version=version, displ_array=displ_arr)

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
        YoungsEtAl2003(magnitude=7, xl=0.5, percentile=0.5, version="d/ad").prob_exceed
