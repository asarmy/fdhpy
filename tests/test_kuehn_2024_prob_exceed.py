"""Exceedance probability prediction tests for Kuehn et al. (2024) model."""

import numpy as np
import pytest

from fdhpy import KuehnEtAl2024

# Test setup
RTOL = 1e-4
ATOL = 1e-7


@pytest.mark.parametrize(
    "filename", ["kuehn_2024_prob_exceed_median_coeffs_folded_mag_7p6_xl_0p7.csv"]
)
def test_prob_exceed_median_coeff(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 7.6
    xl = 0.7
    version = "median_coeffs"
    fold = True
    displ_arr = load_expected["displacement"]

    style_mapping = {
        "strike-slip": "strike_slip",  # Match the actual field name
        "reverse": "reverse",
        "normal": "normal",
    }

    for style, column_name in style_mapping.items():

        # Expected values
        expected = load_expected[column_name]

        # Instantiate the model with the scenario attributes
        model = KuehnEtAl2024(
            style=style,
            magnitude=magnitude,
            xl=xl,
            version=version,
            folded=fold,
            displ_array=displ_arr,
        )
        computed = model.prob_exceed

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"For {style} {version} test, Expected: {expected}, Computed: {computed}",
        )  # type: ignore


@pytest.mark.parametrize(
    "filename", ["kuehn_2024_prob_exceed_full_normal_folded_mag_7p2_xl_0p4.csv"]
)
def test_prob_exceed_full_coeff(load_expected):
    """Verify the predicted exceedance probabilities for the full coefficients version."""

    # Inputs
    style = "normal"
    magnitude = 7.2
    xl = 0.4
    version = "full_coeffs"
    fold = True
    displ_arr = load_expected["displacement"]

    # Expected
    data = np.array([list(row) for row in load_expected])
    expected = data[:, 1:]

    # Instantiate the model with the scenario attributes
    model = KuehnEtAl2024(
        style=style,
        magnitude=magnitude,
        xl=xl,
        version=version,
        folded=fold,
        displ_array=displ_arr,
    )
    computed = model.prob_exceed

    # Testing
    np.testing.assert_allclose(
        expected,
        computed,  # type: ignore
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {style} {version} test, Expected: {expected}, Computed: {computed}",
    )


def test_invalid_inputs(caplog):
    """Should raise errors or warnings."""

    with caplog.at_level("WARNING"):
        # Ignored attribute `percentile`
        KuehnEtAl2024(style="reverse", magnitude=7, xl=0.5, percentile=0.5).prob_exceed
