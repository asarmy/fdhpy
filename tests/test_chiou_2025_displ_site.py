"""Deterministic prediction tests for Chiou et al. (2025) model."""

import numpy as np
import pytest

from fdhpy import ChiouEtAl2025

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.filterwarnings("ignore::UserWarning")  # ignore warnings for magnitude range
@pytest.mark.parametrize("filename", ["chiou_2025_displ_site.csv"])
def test_displ_site(load_expected):
    """Verify the predicted displacements."""

    for row in load_expected:
        version, magnitude, xl, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = ChiouEtAl2025(magnitude=magnitude, xl=xl, percentile=percentile, version=version)

        computed = model.displ_site

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,  # type: ignore
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"Mag {magnitude}, x/L {xl}, percentile {percentile}, for {version} model "
                f"Expected: {expected}, Computed: {computed}"
            ),
        )
