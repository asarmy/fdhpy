"""Statistical distribution parameter tests for Youngs et al. (2003) model."""

import numpy as np
import pytest

from fdhpy import YoungsEtAl2003

# Test setup
RTOL = 1e-3
ATOL = 1e-8


@pytest.mark.parametrize("filename", ["youngs_2003_params_normalized.csv"])
def test_params_normalized(load_expected):
    """Verify the predicted statistical distribution parameters alpha and beta."""

    for row in load_expected:
        version, xl, alpha_expected, beta_expected = row

        # Instantiate the model with the scenario attributes
        # Magnitude is required in fdhpy, use any value here
        model = YoungsEtAl2003(xl=xl, version=version, magnitude=7)

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
