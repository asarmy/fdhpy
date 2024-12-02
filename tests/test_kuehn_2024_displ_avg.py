"""Average displacement prediction tests for Kuehn et al. (2024) model."""

import numpy as np
import pytest

from fdhpy import KuehnEtAl2024

# Test setup
RTOL = 1e-2
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["kuehn_2024_displ_avg.csv"])
def test_displ_avg(load_expected):
    """Verify the predicted displacements."""

    for row in load_expected:
        version, style, magnitude, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = KuehnEtAl2024(
            style=style, magnitude=magnitude, percentile=percentile, version=version
        )
        computed = model.displ_avg

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,  # type: ignore
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"Mag {magnitude} for {version}"
                f"Expected AD: {expected}, Computed AD: {computed}"
            ),
        )


def test_invalid_inputs(caplog):
    """Should raise errors or warnings."""

    with caplog.at_level("ERROR"):
        # only percentile=0.5 is allowed
        KuehnEtAl2024(style="normal", magnitude=7, percentile=0.84).displ_avg

    with caplog.at_level("ERROR"):
        # full coefficients not allowed
        KuehnEtAl2024(style="normal", magnitude=7, percentile=0.5, version="full").displ_avg

    with caplog.at_level("WARNING"):
        # Ignored attribute `xl`
        KuehnEtAl2024(style="normal", magnitude=7, xl=0.5, percentile=0.5).displ_avg
