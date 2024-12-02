"""Exceedance probability prediction tests for Moss et al. (2024) model."""

import numpy as np
import pytest

from fdhpy import MossEtAl2024

# Test setup
RTOL = 1e-2
ATOL = 1e-4


@pytest.mark.parametrize("filename", ["moss_2024_prob_exceed_d_ad_girs.csv"])
def test_prob_exceed_d_ad_girs(load_expected):
    """
    Verify the predicted exceedance probabilities using the GIRS Report regressions for alpha and
    beta using the D/AD model.
    """

    # Inputs
    magnitude = 7
    xl = 0.5
    version = "d/ad"
    girs_flag = True
    displ_arr = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Instantiate the model with the scenario attributes
    model = MossEtAl2024(
        magnitude=magnitude, xl=xl, version=version, displ_array=displ_arr, use_girs=girs_flag
    )

    computed = model.prob_exceed

    # Testing
    np.testing.assert_allclose(
        expected,
        computed,  # type: ignore
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} test, Expected: {expected}, Computed: {computed}",
    )


@pytest.mark.parametrize("filename", ["moss_2024_prob_exceed_d_md_girs.csv"])
def test_prob_exceed_d_md_girs(load_expected):
    """
    Verify the predicted exceedance probabilities using the GIRS Report regressions for alpha and
    beta using the D/MD model.
    """

    # Inputs
    magnitude = 7
    xl = 0.5
    version = "d/md"
    girs_flag = True
    displ_arr = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Instantiate the model with the scenario attributes
    model = MossEtAl2024(
        magnitude=magnitude, xl=xl, version=version, displ_array=displ_arr, use_girs=girs_flag
    )

    computed = model.prob_exceed

    # Testing
    np.testing.assert_allclose(
        expected,
        computed,  # type: ignore
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} test, Expected: {expected}, Computed: {computed}",
    )


@pytest.mark.parametrize("filename", ["moss_2024_prob_exceed_d_ad_eqs.csv"])
def test_prob_exceed_d_ad_eqs(load_expected):
    """
    Verify the predicted exceedance probabilities using piecewise linear interpolation to obtain
    alpha and beta values from Table 2 for the D/AD model.
    """

    # Inputs
    magnitude = 7
    xl = 0.5
    version = "d/ad"
    girs_flag = False
    displ_arr = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Instantiate the model with the scenario attributes
    model = MossEtAl2024(
        magnitude=magnitude, xl=xl, version=version, displ_array=displ_arr, use_girs=girs_flag
    )

    computed = model.prob_exceed

    # Testing
    np.testing.assert_allclose(
        expected,
        computed,  # type: ignore
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} test, Expected: {expected}, Computed: {computed}",
    )


@pytest.mark.parametrize("filename", ["moss_2024_prob_exceed_d_md_eqs.csv"])
def test_prob_exceed_d_md_eqs(load_expected):
    """
    Verify the predicted exceedance probabilities using piecewise linear interpolation to obtain
    alpha and beta values from Table 2 for the D/MD model.
    """

    # Inputs
    magnitude = 7
    xl = 0.5
    version = "d/md"
    girs_flag = False
    displ_arr = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Instantiate the model with the scenario attributes
    model = MossEtAl2024(
        magnitude=magnitude, xl=xl, version=version, displ_array=displ_arr, use_girs=girs_flag
    )

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
        MossEtAl2024(magnitude=7, xl=0.5, percentile=0.5, version="d/ad").prob_exceed
