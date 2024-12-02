"""Statistical distribution parameter tests for Moss and Ross (2011) model."""

import numpy as np
import pytest

from fdhpy import MossRoss2011

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["moss_2011_params_ad.csv"])
def test_params_ad(load_expected):
    """
    Verify the predicted statistical distribution parameters mean (mu) and standard deviation
    (sigma), in log10 units, for the magnitude-average displacement scaling relation.
    """

    for row in load_expected:
        magnitude, mu_expected, sigma_expected = row

        # Instantiate the model with the scenario attributes
        model = MossRoss2011(magnitude=magnitude, version="d/ad")

        mu_calculated = model.stat_params_info["params"]["mu"]  # type: ignore
        sigma_calculated = model.stat_params_info["params"]["sigma"]  # type: ignore

        # Testing
        np.testing.assert_allclose(
            mu_calculated,  # type: ignore
            mu_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in mu for magnitude {magnitude}",
        )  # type: ignore

        np.testing.assert_allclose(
            sigma_calculated,  # type: ignore
            sigma_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in sigma for magnitude {magnitude}",
        )  # type: ignore


@pytest.mark.parametrize("filename", ["moss_2011_params_md.csv"])
def test_params_md(load_expected):
    """
    Verify the predicted statistical distribution parameters mean (mu) and standard deviation
    (sigma), in log10 units, for the magnitude-average displacement scaling relation.
    """

    for row in load_expected:
        magnitude, mu_expected, sigma_expected = row

        # Instantiate the model with the scenario attributes
        model = MossRoss2011(magnitude=magnitude, version="d/md")

        mu_calculated = model.stat_params_info["params"]["mu"]  # type: ignore
        sigma_calculated = model.stat_params_info["params"]["sigma"]  # type: ignore

        # Testing
        np.testing.assert_allclose(
            mu_calculated,  # type: ignore
            mu_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in mu for magnitude {magnitude}",
        )  # type: ignore

        np.testing.assert_allclose(
            sigma_calculated,  # type: ignore
            sigma_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in sigma for magnitude {magnitude}",
        )  # type: ignore


@pytest.mark.parametrize("filename", ["moss_2011_params_normalized.csv"])
def test_params_normalized(load_expected):
    """Verify the predicted statistical distribution parameters alpha and beta."""

    for row in load_expected:
        version, xl, alpha_expected, beta_expected = row

        # Instantiate the model with the scenario attributes
        # Magnitude is required in fdhpy, use any value here
        model = MossRoss2011(xl=xl, version=version, magnitude=7)

        alpha_calculated = model.stat_params_info["params"]["alpha"]  # type: ignore
        beta_calculated = model.stat_params_info["params"]["beta"]  # type: ignore

        # Testing
        np.testing.assert_allclose(
            alpha_calculated,  # type: ignore
            alpha_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in alpha for xl {xl} for {version} model",
        )  # type: ignore

        np.testing.assert_allclose(
            beta_calculated,  # type: ignore
            beta_expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in beta for xl {xl} for {version} model",
        )  # type: ignore
