"""Exceedance probability prediction tests for Lavrentiadis & Abrahamson (2023) model."""

import numpy as np
import pytest

from fdhpy import LavrentiadisAbrahamson2023

# Test setup
RTOL = 1e-4
ATOL = 1e-7


@pytest.mark.parametrize(
    "filename",
    ["lavrentiadis_2023_prob_exceed_full_rupure_aggregate_exclude_pr_zero_mag_7p0_xl_0p6.csv"],
)
def test_prob_exceed_full_rupture_aggregate_no_pr_zero(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 7.0
    xl = 0.6
    metric = "aggregate"
    version = "full rupture"
    psr_flag = False
    displ_arr = load_expected["displacement"]

    style_mapping = {
        "strike-slip": "strike_slip",  # Match the actual field name
        "reverse": "reverse",
        "normal": "normal",
    }

    for style, column_name in style_mapping.items():

        # Expected values
        expected = load_expected[column_name]

        # Instantiate the model with the scenario attributes
        model = LavrentiadisAbrahamson2023(
            style=style,
            magnitude=magnitude,
            xl=xl,
            metric=metric,
            version=version,
            include_prob_zero=psr_flag,
            displ_array=displ_arr,
        )
        computed = model.prob_exceed

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,  # type: ignore
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"For mag {magnitude}, x/L {xl}, {metric} {version}, and include_prob_zero set to "
                f"{psr_flag}, Expected: {expected}, Computed: {computed}"
            ),
        )


@pytest.mark.parametrize(
    "filename",
    ["lavrentiadis_2023_prob_exceed_full_rupure_aggregate_with_pr_zero_mag_7p0_xl_0p6.csv"],
)
def test_prob_exceed_full_rupture_aggregate_with_pr_zero(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 7.0
    xl = 0.6
    metric = "aggregate"
    version = "full rupture"
    psr_flag = True
    displ_arr = load_expected["displacement"]

    style_mapping = {
        "strike-slip": "strike_slip",  # Match the actual field name
        "reverse": "reverse",
        "normal": "normal",
    }

    for style, column_name in style_mapping.items():

        # Expected values
        expected = load_expected[column_name]

        # Instantiate the model with the scenario attributes
        model = LavrentiadisAbrahamson2023(
            style=style,
            magnitude=magnitude,
            xl=xl,
            metric=metric,
            version=version,
            include_prob_zero=psr_flag,
            displ_array=displ_arr,
        )
        computed = model.prob_exceed

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,  # type: ignore
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"For mag {magnitude}, x/L {xl}, {metric} {version}, and include_prob_zero set to "
                f"{psr_flag}, Expected: {expected}, Computed: {computed}"
            ),
        )


@pytest.mark.parametrize(
    "filename",
    ["lavrentiadis_2023_prob_exceed_indiv_segment_aggregate_mag_7p0_xl_0p6.csv"],
)
def test_prob_exceed_indiv_segment_aggregate(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 7.0
    xl = 0.6
    metric = "aggregate"
    version = "individual segment"
    psr_flag = False
    displ_arr = load_expected["displacement"]

    style_mapping = {
        "strike-slip": "strike_slip",  # Match the actual field name
        "reverse": "reverse",
        "normal": "normal",
    }

    for style, column_name in style_mapping.items():

        # Expected values
        expected = load_expected[column_name]

        # Instantiate the model with the scenario attributes
        model = LavrentiadisAbrahamson2023(
            style=style,
            magnitude=magnitude,
            xl=xl,
            metric=metric,
            version=version,
            include_prob_zero=psr_flag,
            displ_array=displ_arr,
        )
        computed = model.prob_exceed

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,  # type: ignore
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"For mag {magnitude}, x/L {xl}, {metric} {version}, and include_prob_zero set to "
                f"{psr_flag}, Expected: {expected}, Computed: {computed}"
            ),
        )


@pytest.mark.parametrize(
    "filename",
    ["lavrentiadis_2023_prob_exceed_full_rupure_sum_prnc_exclude_pr_zero_mag_6p8_xl_0p6.csv"],
)
def test_prob_exceed_full_rupture_sum_prnc_no_pr_zero(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 6.8
    xl = 0.6
    metric = "sum-of-principal"
    version = "full rupture"
    psr_flag = False
    displ_arr = load_expected["displacement"]

    style_mapping = {
        "strike-slip": "strike_slip",  # Match the actual field name
        "reverse": "reverse",
        "normal": "normal",
    }

    for style, column_name in style_mapping.items():

        # Expected values
        expected = load_expected[column_name]

        # Instantiate the model with the scenario attributes
        model = LavrentiadisAbrahamson2023(
            style=style,
            magnitude=magnitude,
            xl=xl,
            metric=metric,
            version=version,
            include_prob_zero=psr_flag,
            displ_array=displ_arr,
        )
        computed = model.prob_exceed

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,  # type: ignore
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"For mag {magnitude}, x/L {xl}, {metric} {version}, and include_prob_zero set to "
                f"{psr_flag}, Expected: {expected}, Computed: {computed}"
            ),
        )


@pytest.mark.parametrize(
    "filename",
    ["lavrentiadis_2023_prob_exceed_full_rupure_sum_prnc_with_pr_zero_mag_6p8_xl_0p6.csv"],
)
def test_prob_exceed_full_rupture_sum_prnc_with_pr_zero(load_expected):
    """Verify the predicted exceedance probabilities."""

    # Inputs
    magnitude = 6.8
    xl = 0.6
    metric = "sum-of-principal"
    version = "full rupture"
    psr_flag = True
    displ_arr = load_expected["displacement"]

    style_mapping = {
        "strike-slip": "strike_slip",  # Match the actual field name
        "reverse": "reverse",
        "normal": "normal",
    }

    for style, column_name in style_mapping.items():

        # Expected values
        expected = load_expected[column_name]

        # Instantiate the model with the scenario attributes
        model = LavrentiadisAbrahamson2023(
            style=style,
            magnitude=magnitude,
            xl=xl,
            metric=metric,
            version=version,
            include_prob_zero=psr_flag,
            displ_array=displ_arr,
        )
        computed = model.prob_exceed

        # Testing
        np.testing.assert_allclose(
            expected,
            computed,  # type: ignore
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"For mag {magnitude}, x/L {xl}, {metric} {version}, and include_prob_zero set to "
                f"{psr_flag}, Expected: {expected}, Computed: {computed}"
            ),
        )


def test_invalid_inputs(caplog):
    """Should raise errors or warnings."""

    with caplog.at_level("WARNING"):
        # include_prob_zero is not applicable to individual segments
        LavrentiadisAbrahamson2023(
            style="normal",
            magnitude=6,
            xl=0.5,
            metric="aggregate",
            version="individual segment",
            include_prob_zero=True,
            displ_array=[1, 10],
        )
