"""Average displacement prediction tests for Lavrentiadis & Abrahamson (2023) model."""

import numpy as np
import pytest

from fdhpy import LavrentiadisAbrahamson2023

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["lavrentiadis_2023_displ_avg.csv"])
def test_displ_avg(load_expected):
    """Verify the predicted displacements."""

    for row in load_expected:
        style, version, metric, magnitude, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = LavrentiadisAbrahamson2023(
            style=style,
            magnitude=magnitude,
            percentile=percentile,
            version=version,
            metric=metric,
            include_prob_zero=False,
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
        LavrentiadisAbrahamson2023(
            style="reverse",
            magnitude=7,
            percentile=0.84,
            version="full rupture",
            metric="sum-of-principal",
        ).displ_avg

    with caplog.at_level("ERROR"):
        # only sum-of-principal is allowed
        LavrentiadisAbrahamson2023(
            style="reverse",
            magnitude=7,
            percentile=0.5,
            version="full rupture",
            metric="aggregate",
        ).displ_avg

    with caplog.at_level("ERROR"):
        # only full rupture is allowed
        LavrentiadisAbrahamson2023(
            style="reverse",
            magnitude=7,
            percentile=0.5,
            version="individual segment",
            metric="sum-of-principal",
        ).displ_avg

    with caplog.at_level("WARNING"):
        # psr is ignored for average displacement
        LavrentiadisAbrahamson2023(
            style="reverse",
            magnitude=7,
            percentile=0.5,
            version="full rupture",
            metric="sum-of-principal",
            include_prob_zero=True,
        ).displ_avg

    with caplog.at_level("WARNING"):
        # x/L is ignored for average displacement
        LavrentiadisAbrahamson2023(
            style="reverse",
            magnitude=7,
            xl=0.5,
            percentile=0.5,
            version="full rupture",
            metric="sum-of-principal",
            include_prob_zero=False,
        ).displ_avg
