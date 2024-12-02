"""Deterministic prediction tests for Moss et al. (2024) model."""

import numpy as np
import pytest

from fdhpy import MossEtAl2024

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["moss_2024_displ_avg_complete_model.csv"])
def test_displ_avg(load_expected):
    """Verify the predicted displacements."""

    for row in load_expected:
        magnitude, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = MossEtAl2024(magnitude=magnitude, percentile=percentile)

        computed = model.displ_avg

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,  # type: ignore
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"Mag {magnitude}, percentile {percentile}"
                f"Expected AD: {expected}, Computed AD: {computed}"
            ),
        )


def test_invalid_inputs(caplog):
    """Should raise errors or warnings."""

    with caplog.at_level("WARNING"):
        # Ignored attribute `xl`
        MossEtAl2024(magnitude=7, xl=0.5, percentile=0.5, version="d/ad").displ_avg
