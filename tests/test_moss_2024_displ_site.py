"""Deterministic prediction tests for Moss et al. (2024) model."""

import numpy as np
import pytest

from fdhpy import MossEtAl2024

# Test setup
RTOL = 1e-2
ATOL = 1e-4


@pytest.mark.parametrize("filename", ["moss_2024_displ_site.csv"])
def test_displ_site(load_expected):
    """Verify the predicted displacements."""

    for row in load_expected:
        version, girs_flag, magnitude, xl, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = MossEtAl2024(
            magnitude=magnitude,
            xl=xl,
            percentile=percentile,
            version=version,
            use_girs=girs_flag,
        )

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
