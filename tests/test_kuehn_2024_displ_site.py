"""Deterministic prediction tests for Kuehn et al. (2024) model."""

import numpy as np
import pytest

from fdhpy import KuehnEtAl2024

# Test setup
RTOL = 1e-4
ATOL = 1e-7


@pytest.mark.parametrize("filename", ["kuehn_2024_displ_site.csv"])
def test_displ_site_point_coeff(load_expected):
    """Verify the predicted displacements for the point coefficient versions."""

    for row in load_expected:
        style, version, fold, magnitude, xl, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = KuehnEtAl2024(
            style=style,
            magnitude=magnitude,
            xl=xl,
            percentile=percentile,
            version=version,
            folded=fold,
        )

        computed = model.displ_site

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"Mag {magnitude}, x/L {xl}, percentile {percentile}, for {style} {version} model "
                f"Expected: {expected}, Computed: {computed}"
            ),
        )  # type: ignore


@pytest.mark.parametrize(
    "filename", ["kuehn_2024_displ_site_full_normal_folded_mag_7p2_xl_0p4_ptile_0p16.csv"]
)
def test_displ_site_full_coeff(load_expected):
    """Verify the predicted displacements for the full coefficients version."""

    style, version, fold, magnitude, xl, percentile = "normal", "full_coeffs", True, 7.2, 0.4, 0.16

    expected = load_expected["displ"]

    # Instantiate the model with the scenario attributes
    model = KuehnEtAl2024(
        style=style,
        magnitude=magnitude,
        xl=xl,
        percentile=percentile,
        version=version,
        folded=fold,
    )

    computed = model.displ_site

    # Testing
    np.testing.assert_allclose(
        expected,
        computed,  # type: ignore
        rtol=RTOL,
        atol=ATOL,
        err_msg=(
            f"Mag {magnitude}, x/L {xl}, percentile {percentile}, for {style} {version} model "
            f"Expected: {expected}, Computed: {computed}"
        ),
    )
