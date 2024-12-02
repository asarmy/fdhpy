"""Deterministic prediction tests for Lavrentiadis & Abrahamson (2023) model."""

import numpy as np
import pytest

from fdhpy import LavrentiadisAbrahamson2023

# Test setup
RTOL = 1e-3
ATOL = 1e-6


@pytest.mark.parametrize("filename", ["lavrentiadis_2023_displ_site.csv"])
def test_displ_site(load_expected):
    """Verify the predicted displacements."""

    for row in load_expected:
        style, version, metric, psr_flag, magnitude, xl, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = LavrentiadisAbrahamson2023(
            style=style,
            magnitude=magnitude,
            xl=xl,
            percentile=percentile,
            version=version,
            metric=metric,
            include_prob_zero=psr_flag,
        )

        computed = model.displ_site

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,  # type: ignore
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"Mag {style}, {magnitude}, x/L {xl}, percentile {percentile}, for {version}, "
                f"{metric} model with include_prob_zero set to {psr_flag} "
                f"Expected: {expected}, Computed: {computed}"
            ),
        )


@pytest.mark.parametrize("filename", ["lavrentiadis_2023_displ_site_mean.csv"])
def test_displ_site_mean(load_expected):
    """Verify the predicted displacements."""

    for row in load_expected:
        style, version, metric, psr_flag, magnitude, xl, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = LavrentiadisAbrahamson2023(
            style=style,
            magnitude=magnitude,
            xl=xl,
            percentile=percentile,
            version=version,
            metric=metric,
            include_prob_zero=psr_flag,
        )

        computed = model.displ_site

        # Testing
        # NOTE: evaluate different tolerances for mean because of differences in results using
        # sampling and numerical integration
        np.testing.assert_allclose(
            np.round(expected, 2),
            np.round(computed, 2),  # type: ignore
            atol=0.01,
            err_msg=(
                f"Mag {style}, {magnitude}, x/L {xl}, percentile {percentile}, for {version}, "
                f"{metric} model with include_prob_zero set to {psr_flag} "
                f"Expected: {expected}, Computed: {computed}"
            ),
        )
