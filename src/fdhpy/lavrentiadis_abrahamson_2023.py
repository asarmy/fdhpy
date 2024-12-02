"""
Lavrentiadis & Abrahamson (2023) fault displacement model
(https://doi.org/10.1177/87552930231201531).

Code is based on https://github.com/NHR3-UCLA/LA23_PFDHA_model. Supplemented with pers. comm.
Lavrentiadis to Sarmiento.
"""

import logging
from functools import cached_property
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.stats._distn_infrastructure import rv_continuous

from fdhpy.cli import cli_runner
from fdhpy.fault_displacement_model import FaultDisplacementModel
from fdhpy.utils import (
    AttributeIgnoredWarning,
    AttributeRequiredError,
    ValidListError,
    _required,
)


class LavrentiadisAbrahamson2023(FaultDisplacementModel):
    """
    Lavrentiadis & Abrahamson (2023) fault displacement model. Applicable to aggregate and
    sum-of-principal surface fault displacement.

    Parameters
    ----------
    style : str
        Style of faulting (case-insensitive).

    magnitude : float
        Earthquake moment magnitude. Recommended range is (5, 8.5).

    xl : float
        Normalized location x/L along the rupture length, range [0, 1.0].

    xl_step : float, optional
        Step size increment for slip profile calculations. Default is 0.05.

    percentile : float
        Aleatory quantile of interest. Use -1 for mean.

    metric : str
        Definition of displacement (case-insensitive).  Valid options are "sum-of-principal" and
        "aggregate".

    version : str, optional
        Name of the model formulation (case-insensitive). Valid options are "full rupture" or
        "individual segment". The "full rupture" corresponds to the "simplified FDM without
        segmentation" in Lavrentiadis & Abrahamson (2023). Default is "full rupture".

    displ_array : np.ndarray, optional
        Displacement test value(s) in meters. Default array is provided.

    include_prob_zero : bool, optional
        Include the probability of zero displacements. Default True. Not used in calculations for
        average displacement or maximum displacement.

    Notes
    -----
    - See Lavrentiadis & Abrahamson (2023) for discussion on the probability of zero displacements,
      P(Gap) and P(D_P=0|no gap), which are enabled with the `include_prob_zero` flag.

    - Model uncertainty is provided with the standard deviation of the predicted mean (in
      transformed units). This is referred to as "sigma_mu_agg" in Lavrentiadis & Abrahamson (2023)
      and is magnitude- and style-dependent. It can be accessed in an existing instance with
      `instance.sigma_mu_agg`. See Lavrentiadis & Abrahamson (2023) for recommendations on
      estimating within-model epistemic uncertainty using this uncertainty.

    See model help at the module level:

        .. code-block:: python

            from fdhpy import KuehnEtAl2024
            print(KuehnEtAl2024.__doc__)

    See model help in command line:

        .. code-block:: console

            $ fd-kea24 --help
    """

    _CONDITIONS = {
        "style": {
            "strike-slip": {"magnitude": (5, 8.5)},
            "reverse": {"magnitude": (5, 8.5)},
            "normal": {"magnitude": (5, 8.5)},
        },
        "metric": {
            "aggregate": {"version": ("full rupture", "individual segment")},
            "sum-of-principal": {"version": ("full rupture", "individual segment")},
        },
    }

    _MODEL_NAME = "LavrentiadisAbrahamson2023"

    # Override the init method to set model defaults
    def __init__(self, **kwargs):
        kwargs.setdefault("version", "full rupture")
        self.include_prob_zero = kwargs.pop("include_prob_zero", True)
        super().__init__(**kwargs)

    def __str__(self):
        parent_str = super().__str__()
        return parent_str[:-1] + f", include_prob_zero={self.include_prob_zero})"

    # NOTE: magnitude, xl, and version are validated in parent class initialization

    # Necessary optional methods in FaultDisplacementModel parent class
    @property
    def _folded_xl(self) -> Optional[float]:
        return self._calc_folded_xl()

    @cached_property
    def _coefficients(self) -> np.recarray:
        """Load the coefficients for the specified style of faulting."""

        try:
            df = self._load_data("lavrentiadis_2023_coefficients.csv")
            if df is not None:
                valid_styles = df["style"].values

                if self.style not in valid_styles:
                    raise ValidListError("style", self.style, valid_styles, self._MODEL_NAME)
                return df[df["style"] == self.style].to_records(index=True)

        except ValidListError as e:  # Other errors handled in `_load_data`
            logging.error(e)

    @property
    @_required("magnitude", "style")
    def sigma_mu_agg(
        self,
    ) -> Optional[float]:  # Use nomenclature in Lavrentiadis & Abrahamson (2023)
        """
        Standard deviation of the predicted median aggregate displacement  in transformed units.
        """
        if self.magnitude >= 7.1:
            return 0.035 + 0.025 * (self.magnitude - 7.1)
        else:
            c = np.array([0.064, 0.036, 0.036])
            s = np.array(["normal", "strike-slip", "reverse"])
            c_s = c[s == self.style]
            result = 0.035 + c_s * (7.1 - self.magnitude)
            return result.item()

        # return self._interpolate_uncertainties()[0]

    # Methods unique to model
    def _calc_wn_d_agg(self) -> Optional[float]:
        """
        Calculate aggregate displacement from wavenumber simulation (exponentiation of
        Equation 8 in Lavrentiadis & Abrahamson (2023)).
        """
        try:
            if self.xl is None:
                raise AttributeRequiredError("xl", self._MODEL_NAME)
        except AttributeRequiredError as e:
            logging.error(e)
            return None

        c = self._coefficients

        # Equation 8
        ln_dk_agg = (
            c["c_0"]
            + c["c_1"] * (self._folded_xl - 0.3)
            + c["c_2"] * (self._folded_xl - 0.3) ** 2
            + c["c_3"] * (self.magnitude - 7.0)
        )

        return np.exp(ln_dk_agg).item()

    def _calc_f_NDmu(self) -> Optional[float]:
        """
        Calculate shape normalization adjustment for simplified model (full rupture); Equation 20
        in Lavrentiadis & Abrahamson (2023).
        """
        try:
            if self.xl is None:
                raise AttributeRequiredError("xl", self._MODEL_NAME)
        except AttributeRequiredError as e:
            logging.error(e)
            return None

        c = self._coefficients

        xl_offset1 = np.minimum(self._folded_xl - 0.3, 0)
        xl_offset2 = np.maximum(self._folded_xl - 0.4, 0)
        f_NDmu = c["c_10"] + (
            c["c_11"] * xl_offset1
            + c["c_12"] * np.power(xl_offset1, 2)
            + c["c_13"] * np.power(xl_offset1, 3)
            + c["c_14"] * np.power(xl_offset1, 4)
            + c["c_14a"] * xl_offset2
        )
        return f_NDmu.item()

    def _calc_Dmu_max(self) -> Optional[float]:
        """
        Calculate amplitude  adjustment for simplified model (full rupture); Equation 21 in
        Lavrentiadis & Abrahamson (2023).
        """
        c = self._coefficients

        Dmu_max = (
            c["c_15"]
            + c["c_16"] * self.magnitude
            + c["c_17"] * np.power(self.magnitude - 6.7, 2)
            + c["c_17a"] * np.power(self.magnitude - 6.7, 3)
        )
        return Dmu_max.item()

    def _calc_phi_agg(self) -> Optional[float]:
        """
        Calculate the within-event variability on single segment in transformed units; Equation 15
        in Lavrentiadis & Abrahamson (2023).
        """
        phi_agg = np.clip(0.120 + 0.150 * (self.magnitude - 6.0), 0.120, 0.270)
        return phi_agg.item()

    def _calc_tau_agg(self) -> Optional[float]:
        """
        Calculate the between-event variability on single segment in transformed units; Equation 16
        in Lavrentiadis & Abrahamson (2023).
        """
        tau_agg = np.clip(0.115 + 0.060 * (self.magnitude - 6.0), 0.115, 0.205)
        return tau_agg.item()

    def _calc_phi_add(self) -> Optional[float]:
        """
        Calculate additional aleatory variability due to segmentation; Equation 23 in Lavrentiadis
        & Abrahamson (2023).
        """
        c = self._coefficients

        phi_add = (
            c["c_18"] + c["c_19"] * self.magnitude + c["c_20"] * np.power(self.magnitude - 6.7, 2)
        )
        return phi_add.item()

    def _calc_phi_prnc(self) -> Optional[float]:
        """
        Calculate additional aleatory variability due to segmentation; Equation 34 in Lavrentiadis
        & Abrahamson (2023).
        """
        c = self._coefficients
        phi_agg = self._calc_phi_agg()
        phi_prnc = np.sqrt(
            np.power(phi_agg, 2)
            + np.power(c["phi_b2"], 2)
            + 2 * c["rho_b2"] * phi_agg * c["phi_b2"]
        )
        return phi_prnc.item()

    def _calc_mu_agg_seg(self) -> Optional[float]:
        """
        Calculate mean aggregate displacement on single segment in transformed units; Equation 14
        in Lavrentiadis & Abrahamson (2023).
        """
        try:
            if self.xl is None:
                raise AttributeRequiredError("xl", self._MODEL_NAME)
        except AttributeRequiredError as e:
            logging.error(e)
            return None

        c = self._coefficients

        # x_seg/l_seg tapering (Equation 9)
        xl1 = 0.15 - 0.10 * np.clip(self.magnitude - 7.0, 0.0, 1.0)
        T_xl = np.minimum(self._folded_xl - xl1, 0) / xl1

        # magnitude tapering (Equation 13)
        T_m = np.clip((7.0 - self.magnitude) / (7.0 - c["M_1"]), 0.0, 1.0)

        dk_agg = self._calc_wn_d_agg()
        mu_agg_seg = (dk_agg * np.exp(c["c_5"] * T_xl)) ** 0.3 + c["c_6"] * T_m + c["c_7"]

        return mu_agg_seg.item()

    def _calc_mu_agg_prime(self) -> Optional[float]:
        """
        Calculate mean aggregate displacement for full rupture in transformed units; Equation 22
        in Lavrentiadis & Abrahamson (2023).
        """
        mu_agg_seg = self._calc_mu_agg_seg()
        f_NDmu = self._calc_f_NDmu()
        Dmu_max = self._calc_Dmu_max()
        return mu_agg_seg + Dmu_max * f_NDmu

    def _calc_mu_prnc_prime(self) -> Optional[float]:
        """
        Calculate mean principal displacement for full rupture in transformed units; Equation 33
        in Lavrentiadis & Abrahamson (2023).
        """
        c = self._coefficients
        mu_agg_prime = self._calc_mu_agg_prime()
        mu_prnc_prime = np.maximum(mu_agg_prime + c["b_2"], 0.0)
        return mu_prnc_prime.item()

    def _calc_mu_prnc_seg(self) -> Optional[float]:
        """Calculate mean principal displacement for full rupture in transformed units."""
        c = self._coefficients
        mu_agg_seg = self._calc_mu_agg_seg()
        mu_prnc_seg = np.maximum(mu_agg_seg + c["b_2"], 0.0)
        return mu_prnc_seg.item()

    def _calc_p_gap(self) -> Optional[float]:
        """
        Calculate the probability of being in a gap without any principal or distributed ruptures;
        Equation 25 in Lavrentiadis
        & Abrahamson (2023).
        """
        try:
            if self.xl is None:
                raise AttributeRequiredError("xl", self._MODEL_NAME)
        except AttributeRequiredError as e:
            logging.error(e)
            return None

        c = self._coefficients

        # Calculate max probability
        P_gap_max = (
            c["c_21"] + c["c_22"] * self.magnitude + c["c_23"] * np.power(self.magnitude - 6.5, 2)
        )

        # Calculate x/L adjustment
        c_25 = c["c_25"]

        c_24 = (
            c["c_24a"]
            + c["c_24b"] * (self.magnitude - 5)
            + c["c_24c"] * np.power(self.magnitude - 5, 3)
        )

        c_26 = (
            c["c_26a"]
            + c["c_26b"] * (self.magnitude - 5)
            + c["c_26c"] * np.power(self.magnitude - 5, 3)
        )

        f_NPGap = 20.0 * c_24 * np.clip(self._folded_xl - 0.10, 0.0, 0.05)  # first and second leg
        f_NPGap += (
            np.power(0.13, -1) * (1.0 - c_24) * np.clip(self._folded_xl - 0.15, 0.0, 0.13)
        )  # third leg and fourth
        f_NPGap -= 10.0 * (1.0 - c_25) * np.clip(self._folded_xl - 0.30, 0.0, 0.10)  # fifth leg
        f_NPGap -= 10.0 * (c_25 - c_26) * np.clip(self._folded_xl - 0.40, 0.0, 0.10)  # sixth leg

        return (P_gap_max * f_NPGap).item()

    def _calc_p_zero_slip(self) -> Optional[float]:
        """
        Calculate the probability of zero principal displacement given rupture has occurred at the
        site (i.e., P_gap = 0); Equation 32 in Lavrentiadis & Abrahamson (2023).
        """
        c = self._coefficients
        mu_agg_prime = self._calc_mu_agg_prime()
        P_zero_slip = 1 / (1 + np.exp(c["b_0"] + c["b_1"] * mu_agg_prime))
        return P_zero_slip.item()

    def _scale_by_zero_probability(
        self, value: Union[float, np.ndarray]
    ) -> Optional[Union[float, np.ndarray]]:
        """
        Scale the value by the probability of zero displacement based on the metric and
        `include_prob_zero` flag.
        """

        P_gap, P_zero_slip = 0, 0
        if self.include_prob_zero:
            P_gap = self._calc_p_gap()
            P_zero_slip = self._calc_p_zero_slip()

        if self.metric == "aggregate":
            return value * (1 - P_gap)
        elif self.metric == "sum-of-principal":
            return value * (1 - P_gap) * (1 - P_zero_slip)

    def _calc_power_normal_mean(
        self, *, stat_distrib: rv_continuous, stat_kwargs: Dict[str, float]
    ):
        """ "
        Calculate the mean of the power-normal distribution using SciPy's quadrature function for
        numerical integration. Return is in power-normal (X^0.3) units.
        """

        # FIXME: Numerical integration is much faster than sampling, but relative errors > 15% for
        # small magnitude at rupture ends; why?
        # np.random.seed(1)
        # sample = stat_distrib.rvs(**stat_kwargs, size=500_000)
        # sample = np.where(sample >= 0, sample, np.nan)
        # sample = np.nanmean(sample ** (10 / 3))
        # return np.power(sample, 0.3)

        def integrand(y):
            if y < 0:
                return 0
            else:
                return y ** (10 / 3) * stat_distrib.pdf(y, **stat_kwargs)

        integral, _ = quad(integrand, 0, np.inf)

        return np.power(integral, 0.3)

    # Mandated methods in FaultDisplacementModel parent class
    def _statistical_distribution_params(self) -> None:
        # Calculate standard deviations on single segments in transformed units
        phi_agg = self._calc_phi_agg()
        tau_agg = self._calc_tau_agg()

        # Helper functions to set instance distribution parameters
        def _stat_params_indiv(metric) -> Tuple[float, float]:
            if metric == "aggregate":
                mu = self._calc_mu_agg_seg()
                std_dev = np.sqrt(np.power(phi_agg, 2) + np.power(tau_agg, 2))

            elif metric == "sum-of-principal":
                mu = self._calc_mu_prnc_seg()
                phi_prnc = self._calc_phi_prnc()
                std_dev = np.sqrt(np.power(phi_prnc, 2) + np.power(tau_agg, 2))

            return mu, std_dev

        def _stat_params_full(metric) -> Tuple[float, float]:
            # Calculate additional aleatory variability due to segmentation
            phi_add = self._calc_phi_add()

            if metric == "aggregate":
                mu = self._calc_mu_agg_prime()
                std_dev = np.sqrt(
                    np.power(phi_agg, 2) + np.power(tau_agg, 2) + np.power(phi_add, 2)
                )

            elif metric == "sum-of-principal":
                mu = self._calc_mu_prnc_prime()
                phi_prnc = self._calc_phi_prnc()
                std_dev = np.sqrt(
                    np.power(phi_prnc, 2) + np.power(tau_agg, 2) + np.power(phi_add, 2)
                ).item()

            return mu, std_dev

        # Set the statical distribution parameters
        if self.version == "individual segment":
            self._mu, self._std_dev = _stat_params_indiv(self.metric)
        elif self.version == "full rupture":
            self._mu, self._std_dev = _stat_params_full(self.metric)

    def _calc_displ_site(self) -> Optional[float]:
        """Calculate deterministic scenario displacement."""

        stat_params = self.stat_params_info  # Access only once and store in a local variable

        if self.percentile == -1:
            pn_displ = self._calc_power_normal_mean(
                stat_distrib=stat_params["prob_distribution"],
                stat_kwargs=stat_params["prob_distribution_kwargs"],
            )
        else:
            pn_displ = stat_params["prob_distribution"].ppf(
                self.percentile, **stat_params["prob_distribution_kwargs"]
            )

        # Left-truncate at zero
        pn_displ = max(0, pn_displ)
        displ = np.power(pn_displ, 1 / 0.3)

        if self.version == "individual segment":
            if self.include_prob_zero:
                message = AttributeIgnoredWarning("include_prob_zero", "individual segments")
                logging.warning(message)
            else:
                return displ

        if self.version == "full rupture":
            # Check for `include_prob_zero` handled in helper function
            return self._scale_by_zero_probability(displ)

    def _calc_displ_avg(self) -> Optional[float]:
        """Calculate average displacement."""

        # if self.metric == "distributed":
        #     e = ValueError("Average displacement cannot be computed for distributed faults.")
        #     logging.error(e)
        #     return

        if self.percentile != 0.5:
            e = ValueError(
                f"\n\tThe `{self._MODEL_NAME}` model does not provide aleatory variability "
                "on the average displacement. Use `percentile=0.5` instead.\n\n"
            )
            logging.error(e)
            return

        if self.metric != "sum-of-principal":
            e = ValueError(
                f"\n\tThe {self.metric} 'metric' is not available for average displacement in the "
                f"`{self._MODEL_NAME}` model. Use `metric='sum-of-principal' instead.\n\n"
            )
            logging.error(e)
            return

        if self.version != "full rupture":
            e = ValueError(
                f"\n\tThe {self.version} 'version' is not available for average displacement in "
                f"the `{self._MODEL_NAME}` model. Use `version='full rupture' instead.\n\n"
            )
            logging.error(e)
            return

        if self.include_prob_zero:
            message = AttributeIgnoredWarning("include_prob_zero", "average displacement")
            logging.warning(message)

        # Store original values
        original_xl = self.xl
        original_zero_flag = self.include_prob_zero

        try:
            # Temporarily change the values
            self.xl = 0.25
            self.include_prob_zero = False

            c = self._coefficients

            ratio_ad = np.exp(c["b_3"] + c["b_4"] * np.exp(c["b_5"] * (self.magnitude - 5.0)))
            disp_prnc_prime = self._calc_displ_site()
            return ratio_ad.item() * disp_prnc_prime

        finally:
            # Restore original values
            self.xl = original_xl
            self.include_prob_zero = original_zero_flag

    def _calc_displ_max(self) -> Optional[float]:
        """Calculate maximum displacement."""

        # if self.metric == "distributed":
        #     e = ValueError("Maximum displacement cannot be computed for distributed faults.")
        #     logging.error(e)
        #     return

        if self.version != "full rupture":
            e = ValueError(
                f"\n\tThe {self.version} 'version' is not available for maximum displacement in "
                f"the `{self._MODEL_NAME}` model. Use `version='full rupture' instead.\n\n"
            )
            logging.error(e)
            return

        if self.include_prob_zero:
            message = AttributeIgnoredWarning("include_prob_zero", "maximum displacement")
            logging.warning(message)

        # Store original values
        original_xl = self.xl
        original_zero_flag = self.include_prob_zero
        original_percentile = self.percentile

        try:
            # Temporarily change the values
            self.xl = 0.25
            self.include_prob_zero = False
            self.percentile = 0.5  # needed to compute median disp_agg_prime
            # FIXME: Mixing of percentiles is error prone, need to rethink overall structure

            c = self._coefficients

            # Calculate disp_agg_prime adjustments for MD
            dm_pwr = c["e_1"]
            dm_pwr += c["e_2"] * np.clip(self.magnitude - 6.0, 0.0, 1.0)
            dm_pwr += c["e_3"] * np.maximum(self.magnitude - 7.0, 0.0) + c["e_4"] * np.power(
                np.maximum(self.magnitude - 7.0, 0.0), 2
            )

            # Calculate aleatory variability on MD
            sig_dm = 0.13 + 0.095 * np.clip(self.magnitude - 6.0, 0.0, 1.0)
            sig_dm += 0.050 * np.clip(self.magnitude - 7.0, 0.0, 0.5)
            if self.metric == "sum-of-principal":
                sig_dm = np.sqrt(np.power(sig_dm, 2) + np.power(c["phi_b2"], 2))

            # Calculate mean maximum displacement in transformed units
            # NOTE: `aggregate` or `sum-of-principal` metrics handled in `self._calc_displ_site()`
            disp_prime = self._calc_displ_site()
            mu = np.power(disp_prime, 0.3) + dm_pwr

            # Calculate maximum displacement
            stat_distrib = stats.norm
            stat_kwargs = {"loc": mu, "scale": sig_dm}
            if original_percentile == -1:
                pn_displ = self._calc_power_normal_mean(
                    stat_distrib=stat_distrib, stat_kwargs=stat_kwargs
                )
            else:
                pn_displ = stat_distrib.ppf(original_percentile, **stat_kwargs)

            # Left-truncate at zero
            pn_displ = max(0, pn_displ)
            displ = np.power(pn_displ, 1 / 0.3)
            return displ.item()

        finally:
            # Restore original values
            self.xl = original_xl
            self.include_prob_zero = original_zero_flag
            self.percentile = original_percentile

    def _calc_cdf(self) -> Optional[np.ndarray]:
        z = np.power(self.displ_array, 0.3)
        stat_params = self.stat_params_info
        cdf = stat_params["prob_distribution"].cdf(x=z, **stat_params["prob_distribution_kwargs"])
        if self.version == "individual segment":
            if self.include_prob_zero:
                message = AttributeIgnoredWarning("include_prob_zero", "individual segments")
                logging.warning(message)
            return cdf
        if self.version == "full rupture":
            # Check for `include_prob_zero` handled in helper function
            return self._scale_by_zero_probability(cdf)

    def _calc_prob_exceed(self) -> Optional[np.ndarray]:
        z = np.power(self.displ_array, 0.3)
        stat_params = self.stat_params_info
        ccdf = 1 - stat_params["prob_distribution"].cdf(
            x=z, **stat_params["prob_distribution_kwargs"]
        )
        if self.version == "individual segment":
            if self.include_prob_zero:
                message = AttributeIgnoredWarning("include_prob_zero", "individual segments")
                logging.warning(message)
            return ccdf
        if self.version == "full rupture":
            # Check for `include_prob_zero` handled in helper function
            return self._scale_by_zero_probability(ccdf)

    @property
    def stat_params_info(
        self,
    ) -> Dict[str, Union[Dict[str, float], rv_continuous, Dict[str, float]]]:
        """
        Dictionary of statistical parameters ("params"), probability distribution
        ("prob_distribution"), and its arguments ("prob_distribution_kwargs") for the instance.
        """
        self._statistical_distribution_params()

        statistical_parameters = {
            "mu": self._mu,
            "sigma": self._std_dev,
        }  # Use nomenclature in Lavrentiadis & Abrahamson (2023)

        probability_distribution = stats.norm

        probability_distribution_kwargs = {
            "loc": statistical_parameters["mu"],
            "scale": statistical_parameters["sigma"],
        }

        return {
            "params": statistical_parameters,
            "prob_distribution": probability_distribution,
            "prob_distribution_kwargs": probability_distribution_kwargs,
        }

    @staticmethod
    def _add_arguments(parser):
        # Add arguments specific to model
        parser.add_argument(
            "--include_prob_zero",
            default=True,
            type=bool,
            help=("Include the probability of zero displacements. Default True."),
        )

    @staticmethod
    def main():
        cli_runner(LavrentiadisAbrahamson2023, LavrentiadisAbrahamson2023._add_arguments)


if __name__ == "__main__":
    LavrentiadisAbrahamson2023.main()
