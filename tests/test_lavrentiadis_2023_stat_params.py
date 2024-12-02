"""Statistical distribution parameter tests for Lavrentiadis & Abrahamson (2023) model."""

import numpy as np
import pytest

from fdhpy import LavrentiadisAbrahamson2023

# Test setup
RTOL = 1e-4
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["lavrentiadis_2023_params.csv"])
def test_params(load_expected):
    """Verify the predicted statistical distribution parameters."""

    for row in load_expected:
        style, version, metric, magnitude, xl, mu_expected, sigma_expected = row

        # Instantiate the model with the scenario attributes
        model = LavrentiadisAbrahamson2023(
            style=style, magnitude=magnitude, xl=xl, version=version, metric=metric
        )

        mu_calculated = model.stat_params_info["params"]["mu"]
        sigma_calculated = model.stat_params_info["params"]["sigma"]

        # Testing
        np.testing.assert_allclose(
            mu_calculated,  # type: ignore
            mu_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in mu for magnitude {magnitude}, xl {xl}, {version}, {metric}",
        )

        np.testing.assert_allclose(
            sigma_calculated,  # type: ignore
            sigma_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"Mismatch in sigma_prime for magnitude {magnitude}, xl {xl}, {version}, {metric}"
            ),
        )
