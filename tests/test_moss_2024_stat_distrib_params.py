"""Statistical distribution parameter tests for Moss et al. (2024) model."""

import numpy as np
import pytest

from fdhpy import MossEtAl2024

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["moss_2024_params_ad.csv"])
def test_params_ad(load_expected):
    """
    Verify the predicted statistical distribution parameters mean (mu) and standard deviation
    (sigma), in log10 units, for the "complete" magnitude-average displacement scaling relation.
    """

    for row in load_expected:
        magnitude, complete_flag, mu_expected, sigma_expected = row

        # Instantiate the model with the scenario attributes
        model = MossEtAl2024(magnitude=magnitude, version="d/ad", complete=complete_flag)

        mu_calculated = model.stat_params_info["params"]["mu"]
        sigma_calculated = model.stat_params_info["params"]["sigma"]

        # Testing
        np.testing.assert_allclose(
            mu_calculated,  # type: ignore
            mu_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in mu for magnitude {magnitude}",
        )

        np.testing.assert_allclose(
            sigma_calculated,  # type: ignore
            sigma_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in sigma for magnitude {magnitude}",
        )


@pytest.mark.parametrize("filename", ["moss_2024_params_md.csv"])
def test_params_md(load_expected):
    """
    Verify the predicted statistical distribution parameters mean (mu) and standard deviation
    (sigma), in log10 units, for the "complete" magnitude-average displacement scaling relation.
    """

    for row in load_expected:
        magnitude, complete_flag, mu_expected, sigma_expected = row

        # Instantiate the model with the scenario attributes
        model = MossEtAl2024(magnitude=magnitude, version="d/md", complete=complete_flag)

        mu_calculated = model.stat_params_info["params"]["mu"]
        sigma_calculated = model.stat_params_info["params"]["sigma"]

        # Testing
        np.testing.assert_allclose(
            mu_calculated,  # type: ignore
            mu_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in mu for magnitude {magnitude}",
        )

        np.testing.assert_allclose(
            sigma_calculated,  # type: ignore
            sigma_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in sigma for magnitude {magnitude}",
        )


@pytest.mark.parametrize("filename", ["moss_2024_params_normalized.csv"])
def test_params_normalized(load_expected):
    """Verify the predicted statistical distribution parameters alpha and beta."""

    for row in load_expected:
        version, girs_flag, xl, alpha_expected, beta_expected = row

        # Instantiate the model with the scenario attributes
        # Magnitude is required in fdhpy, use any value here
        model = MossEtAl2024(xl=xl, version=version, use_girs=girs_flag)

        alpha_calculated = model.stat_params_info["params"]["alpha"]
        beta_calculated = model.stat_params_info["params"]["beta"]

        # Testing
        np.testing.assert_allclose(
            alpha_calculated,  # type: ignore
            alpha_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in alpha for xl {xl} for {version} model",
        )

        np.testing.assert_allclose(
            beta_calculated,  # type: ignore
            beta_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in beta for xl {xl} for {version} model",
        )
