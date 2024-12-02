"""Average displacement prediction tests for Petersen et al. (2011) model."""

import numpy as np
import pytest

from fdhpy import PetersenEtAl2011

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["petersen_2011_displ_avg.csv"])
def test_displ_avg(load_expected):
    """Verify the predicted displacements."""

    for row in load_expected:
        version, magnitude, percentile, expected = row

        # Instantiate the model with the scenario attributes
        model = PetersenEtAl2011(magnitude=magnitude, percentile=percentile, version=version)
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
        # only percentile=-1 is allowed
        PetersenEtAl2011(magnitude=7, percentile=0.5).displ_avg

    with caplog.at_level("WARNING"):
        # Ignored attribute `xl`
        PetersenEtAl2011(magnitude=7, xl=0.5, percentile=-1).displ_avg
