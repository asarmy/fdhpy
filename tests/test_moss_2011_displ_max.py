"""Deterministic prediction tests for Moss and Ross (2011) model."""

import numpy as np
import pytest

from fdhpy import MossRoss2011

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["moss_2011_displ_max.csv"])
def test_displ_max(load_expected):
    """Verify the predicted displacements."""

    for row in load_expected:
        magnitude, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = MossRoss2011(magnitude=magnitude, percentile=percentile)

        computed = model.displ_max

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,  # type: ignore
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"Mag {magnitude}, percentile {percentile}"
                f"Expected MD: {expected}, Computed MD: {computed}"
            ),
        )


def test_invalid_inputs(caplog):
    """Should raise errors or warnings."""

    with caplog.at_level("WARNING"):
        # Ignored attribute `xl`
        MossRoss2011(magnitude=7, xl=0.5, percentile=0.5, version="d/ad").displ_max
