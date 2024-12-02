"""Deterministic prediction tests for Moss and Ross (2011) model."""

import numpy as np
import pytest

from fdhpy import MossRoss2011

# Test setup
RTOL = 1e-2
ATOL = 1e-4


@pytest.mark.parametrize("filename", ["moss_2011_displ_site.csv"])
def test_displ_site(load_expected):
    """Verify the predicted displacements."""

    for row in load_expected:
        version, magnitude, xl, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = MossRoss2011(magnitude=magnitude, xl=xl, percentile=percentile, version=version)

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
