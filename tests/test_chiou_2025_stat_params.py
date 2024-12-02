"""Statistical distribution parameter tests for Chiou et al. (2025) model."""

import numpy as np
import pytest

from fdhpy import ChiouEtAl2025

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.filterwarnings("ignore::UserWarning")  # ignore warnings for magnitude range
@pytest.mark.parametrize("filename", ["chiou_2025_params.csv"])
def test_params(load_expected):
    """Verify the predicted statistical distribution parameters."""

    for row in load_expected:
        version, magnitude, xl, mu_expected, sigma_expected, nu_expected = row

        # Instantiate the model with the scenario attributes
        model = ChiouEtAl2025(magnitude=magnitude, xl=xl, version=version)

        mu_calculated = model.stat_params_info["params"]["mu"]
        sigma_calculated = model.stat_params_info["params"]["sigma_prime"]
        nu_calculated = model.stat_params_info["params"]["nu"]

        # Testing
        np.testing.assert_allclose(
            mu_calculated,
            mu_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in mu for magnitude {magnitude}, xl {xl}, {version}",
        )

        np.testing.assert_allclose(
            sigma_calculated,
            sigma_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in sigma_prime for magnitude {magnitude}, xl {xl}, {version}",
        )

        np.testing.assert_allclose(
            nu_calculated,
            nu_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in nu for magnitude {magnitude}, xl {xl}, {version}",
        )
