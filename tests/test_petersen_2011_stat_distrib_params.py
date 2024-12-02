"""Statistical distribution parameter tests for Petersen (2011) model."""

import numpy as np
import pytest

from fdhpy import PetersenEtAl2011

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["petersen_2011_params.csv"])
def test_params(load_expected):
    """Verify the predicted statistical distribution parameters (mean and standard deviation)."""

    for row in load_expected:
        version, magnitude, xl, mu_expected, sigma_expected = row

        # Instantiate the model with the scenario attributes
        model = PetersenEtAl2011(magnitude=magnitude, xl=xl, version=version)

        mu_calculated = model.stat_params_info["params"]["mu"]
        sigma_calculated = model.stat_params_info["params"]["sigma"]

        # Testing
        np.testing.assert_allclose(
            mu_calculated,  # type: ignore
            mu_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatched mu for magnitude {magnitude} at x/L {xl} for {version} model",
        )

        np.testing.assert_allclose(
            sigma_calculated,  # type: ignore
            sigma_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatched sigma for magnitude {magnitude} at x/L {xl} for {version} model",
        )
