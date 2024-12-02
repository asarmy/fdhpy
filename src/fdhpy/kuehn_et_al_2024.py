"""Kuehn et al. (2024) fault displacement model (https://doi.org/10.1177/87552930241291077)."""

import logging
from functools import cached_property
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous

from fdhpy.cli import cli_runner
from fdhpy.fault_displacement_model import FaultDisplacementModel
from fdhpy.utils import (
    AttributeRequiredError,
    ValidListError,
    _required,
    _trapezoidal_integration,
)


class KuehnEtAl2024(FaultDisplacementModel):
    """
    Kuehn et al. (2024) fault displacement model. Applicable to aggregate surface fault
    displacement.

    Parameters
    ----------
    style : str
        Style of faulting (case-insensitive).

    magnitude : float
        Earthquake moment magnitude.

    xl : float
        Normalized location x/L along the rupture length, range [0, 1.0].

    xl_step : float, optional
        Step size increment for slip profile calculations. Default is 0.05.

    percentile : float
        Aleatory quantile of interest. Use -1 for mean.

    metric : str, optional
        Definition of displacement (case-insensitive). Valid options are "aggregate". Default is
        "aggregate".

    version : str, optional
        Name of the model formulation (case-insensitive). Valid options are "median_coeffs",
        "mean_coeffs", or "full_coeffs", which refer to using point estimates of model
        coefficients (i.e., "median" or "mean" coefficients, which will exclude within-model
        epistemic uncertainty) or the "full" set (n=1000) of model coefficients, which will
        include within-model epistemic uncertainty. Default is "median_coeffs".

    displ_array : np.ndarray, optional
        Displacement test value(s) in meters. Default array is provided.

    folded : bool, optional
        Calculate results for the folded location. Default True.

    Notes
    -----
    - This model predicts asymmetric displacement profiles that are right-skewed (i.e., the peak
      displacement is left of the midpoint). The default implementation uses the folded profile
      (`folded=True`) because the direction of skew is not known a priori.

    - This model provides epistemic uncertainty on the model predictions by providing a set of 1000
      correlated coefficients (per style of faulting) if `version="full_coeffs"` is used. This
      results in 1000 "answers" for each calculation. Alternatively, `version="median_coeffs"` or
      `version="mean_coeffs"` may be used for point estimates of the coefficients to get 1
      "answer" for each calculation.

    - Model uncertainty is also provided with standard deviations of the predicted mean and total
      standard deviation (in transformed units). These are referred to as "sd_med" and "sd_sigma_T"
      respectively in Kuehn et al. (2024) and depend on style, magnitude, and x/L. These can be
      accessed in an existing instance with `instance.sd_med` and `instance.sd_sigma_T`. See Kuehn
      et al. (2024) for recommendations on estimating within-model epistemic uncertainty using
      these uncertainties.

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
            "strike-slip": {"magnitude": (6, 8)},
            "reverse": {"magnitude": (5, 8)},
            "normal": {"magnitude": (6, 8)},
        },
        "metric": {
            "aggregate": {"version": ("median_coeffs", "mean_coeffs", "full_coeffs")},
        },
    }

    _MODEL_NAME = "KuehnEtAl2025"

    _MAG_BREAK, _DELTA = 7.0, 0.1

    _EXCLUDE_SIGMA_MAG = False  # Avg Displ calcs will use True

    # Override the init method to set model defaults
    def __init__(self, **kwargs):
        kwargs.setdefault("metric", "aggregate")
        kwargs.setdefault("version", "median_coeffs")
        self.folded = kwargs.pop("folded", True)
        super().__init__(**kwargs)

    def __str__(self):
        parent_str = super().__str__()
        return parent_str[:-1] + f", folded={self.folded})"

    # NOTE: magnitude, xl, and version are validated in parent class initialization

    # Necessary optional methods in FaultDisplacementModel parent class
    @cached_property
    def _coefficients(self) -> Optional[np.recarray]:
        posterior_files = {
            "strike-slip": "kuehn_2024_coefficients_posterior_SS_powtr.csv",
            "reverse": "kuehn_2024_coefficients_posterior_REV_powtr.csv",
            "normal": "kuehn_2024_coefficients_posterior_NM_powtr.csv",
        }

        valid_styles = tuple(posterior_files.keys())

        # Validate self.style before attempting to load the file
        if self.style not in valid_styles:
            e = ValidListError("style", self.style, valid_styles)
            logging.error(e)
            e = RuntimeError(
                f"Coefficients could not be loaded for `{self._MODEL_NAME}` because `style` is "
                "invalid."
            )
            logging.error(e)
            return None

        valid_versions = tuple(
            self._CONDITIONS.get("metric", {}).get(self.metric, {}).get("version", [])
        )

        # Validate self.version before attempting to load the file
        if self.version not in valid_versions:
            e = ValidListError("version", self.version, valid_versions)
            logging.error(e)
            e = RuntimeError(
                f"Coefficients could not be loaded for `{self._MODEL_NAME}` because `version` is "
                "invalid."
            )
            logging.error(e)
            return None

        try:
            df = self._load_data(posterior_files[self.style])
            if df is not None:
                if self.version == "median_coeffs":
                    coeffs = df.median(axis=0)
                    return pd.DataFrame([coeffs], columns=df.columns).to_records(index=True)
                elif self.version == "mean_coeffs":
                    coeffs = df.mean(axis=0)
                    return pd.DataFrame([coeffs], columns=df.columns).to_records(index=True)
                elif self.version == "full_coeffs":
                    return df.to_records(index=True)

        except ValidListError as e:  # Other errors handled in `_load_data`
            logging.error(e)

    @cached_property
    def _uncertainties(self) -> pd.DataFrame:
        unc_files = {
            "strike-slip": "kuehn_2024_uncertainty_SS.csv",
            "reverse": "kuehn_2024_uncertainty_REV.csv",
            "normal": "kuehn_2024_uncertainty_NM.csv",
        }

        valid_styles = tuple(unc_files.keys())

        try:
            if self.style not in valid_styles:
                raise ValidListError("style", self.style, valid_styles)
            return self._load_data(unc_files[self.style])
        except ValidListError as e:  # Other errors handled in `_load_data`
            logging.error(e)

    # Methods unique to model
    def _interpolate_uncertainties(self) -> Optional[Tuple[float, float]]:
        """
        Linearly interpolate standard deviations of the model uncertainties for the style,
        magnitude, and x/L of the instance.

        Returns
        -------
        tuple of float
            A tuple containing:

            - sd_mu : float
                Standard deviation of the predicted median in transformed units.

            - sd_sd : float
                Standard deviation of the predicted total standard deviation in transformed units.
        """

        # Check magnitude range
        if not (5 <= self.magnitude <= 8.2):
            e = ValueError(
                "\n\tMagnitude must be between 5 and 8.2 to calculate model uncertainty, but "
                f"{self.magnitude} was entered.\n\n"
            )
            logging.error(e)
            return

        # Access only once and store in local variables
        unc = self._uncertainties

        # Round location for efficiency, no need to interpolate on x/L
        xl = np.round(self.xl, 2)

        # Subset stdv data
        unc = unc[unc["Ustar"] == xl].sort_values(by="M")

        # Extract to np arrays
        mag_array = unc["M"].to_numpy()
        mu_array = unc["sd_med"].to_numpy()
        sig_array = unc["sd_sigma_T"].to_numpy()

        # Linear interpolation
        sd_mu = np.interp(self.magnitude, mag_array, mu_array)
        sd_sd = np.interp(self.magnitude, mag_array, sig_array)

        return sd_mu, sd_sd

    @property
    @_required("xl", "magnitude", "style")
    def sd_med(self) -> Optional[float]:  # Use nomenclature in Kuehn et al. (2024)
        """Standard deviation of the predicted median in transformed units."""
        return self._interpolate_uncertainties()[0]

    @property
    @_required("xl", "magnitude", "style")
    def sd_sigma_T(self) -> Optional[float]:  # Use nomenclature in Kuehn et al. (2024)
        """Standard deviation of the predicted standard deviation in transformed units."""
        return self._interpolate_uncertainties()[1]

    def _calc_fm(self) -> Union[float, np.ndarray]:
        """
        Calculate the magnitude scaling component. This is 'f_M(M)' in Kuehn et al. (2024).
        """

        c = self._coefficients
        fm = (
            c["c1"]
            + c["c2"] * (self.magnitude - self._MAG_BREAK)
            + (c["c3"] - c["c2"])
            * self._DELTA
            * np.log(1 + np.exp((self.magnitude - self._MAG_BREAK) / self._DELTA))
        )
        return fm.item() if fm.size == 1 else fm

    def _calc_sigma_mag(self) -> Union[float, np.ndarray]:
        """
        Calculate the magnitude aleatory variability. This is 'sigma_m' in Kuehn et al. (2024).
        """

        c = self._coefficients

        if self.style == "strike-slip":
            # Bilinear model
            std_dev = (
                c["s_m,s1"]
                + c["s_m,s2"] * (self.magnitude - c["s_m,s3"])
                - c["s_m,s2"]
                * self._DELTA
                * np.log(1 + np.exp((self.magnitude - c["s_m,s3"]) / self._DELTA))
            )

        elif self.style == "normal":
            # Sigmoidal model
            std_dev = c["s_m,n1"] - c["s_m,n2"] / (
                1 + np.exp(-1 * c["s_m,n3"] * (self.magnitude - self._MAG_BREAK))
            )

        elif self.style == "reverse":
            # Constant
            std_dev = c["s_m,r"]

        return std_dev.item() if std_dev.size == 1 else std_dev

    def _calc_sigma_xl(
        self,
    ) -> Optional[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]]:
        """
        Calculate the x/L aleatory variability. This is 'sigma_u' in Kuehn et al. (2024).
        Returns tuple of standard deviations for u1 and u2 (which correspond to "xl" and "1-xl",
        respectively).
        """
        try:
            if self.xl is None:
                raise AttributeRequiredError("xl", self._MODEL_NAME)
        except AttributeRequiredError as e:
            logging.error(e)
            return None

        c = self._coefficients

        if self.style == "strike-slip":
            std_dev_u1 = c["s_s1"] + c["s_s2"] * np.power(
                self.xl - c["alpha"] / (c["alpha"] + c["beta"]), 2
            )
            std_dev_u2 = c["s_s1"] + c["s_s2"] * np.power(
                (1 - self.xl) - c["alpha"] / (c["alpha"] + c["beta"]), 2
            )

        elif self.style == "normal":
            std_dev_u1 = c["sigma"]
            std_dev_u2 = c["sigma"]

        elif self.style == "reverse":
            std_dev_u1 = c["s_r1"] + c["s_r2"] * np.power(
                self.xl - c["alpha"] / (c["alpha"] + c["beta"]), 2
            )
            std_dev_u2 = c["s_r1"] + c["s_r2"] * np.power(
                (1 - self.xl) - c["alpha"] / (c["alpha"] + c["beta"]), 2
            )

        std_dev_u1 = std_dev_u1.item() if std_dev_u1.size == 1 else std_dev_u1
        std_dev_u2 = std_dev_u2.item() if std_dev_u2.size == 1 else std_dev_u2

        return std_dev_u1, std_dev_u2

    def _calc_mean_mu(self, u_star: float) -> Union[float, np.ndarray]:
        """Mean prediction in transformed units."""

        c = self._coefficients

        fm = self._calc_fm()

        a = fm - c["gamma"] * np.power(
            c["alpha"] / (c["alpha"] + c["beta"]), c["alpha"]
        ) * np.power(c["beta"] / (c["alpha"] + c["beta"]), c["beta"])

        mu = a + c["gamma"] * np.power(u_star, c["alpha"]) * np.power(1 - u_star, c["beta"])

        return mu.item() if mu.size == 1 else mu

    # Mandated methods in FaultDisplacementModel parent class
    def _statistical_distribution_params(self) -> None:
        # Calculate the mean `mu` for xl (u1) and the complementary location (u2)
        self._mean_u1 = self._calc_mean_mu(self.xl)
        self._mean_u2 = self._calc_mean_mu(1 - self.xl)

        # Calculate the total standard deviation for xl (u1) and the complementary location (u2)
        mag_std_dev = self._calc_sigma_mag()
        xl_std_dev_u1, xl_std_dev_u2 = self._calc_sigma_xl()

        self._total_std_dev_u1 = np.sqrt(np.power(mag_std_dev, 2) + np.power(xl_std_dev_u1, 2))
        self._total_std_dev_u2 = np.sqrt(np.power(mag_std_dev, 2) + np.power(xl_std_dev_u2, 2))

        # NOTE: Need to store sigma_xl_u1 attribute for Avg Displ calculation
        self._xl_std_dev_u1 = xl_std_dev_u1

    def _calc_displ_site(self) -> Optional[Union[float, np.ndarray]]:
        """Calculate deterministic scenario displacement."""

        # Access only once and store in local variables
        bc_parameter = self._coefficients["lambda"]
        stat_params = self.stat_params_info

        def _analytic_mean_meters(
            *,
            mu: Union[float, np.ndarray],
            sigma: Union[float, np.ndarray],
        ) -> Union[float, np.ndarray]:
            """
            Calculate the back-transformed predicted mean displacement in arithmetic units
            (meters).
            """
            return (np.power(bc_parameter * mu + 1, 1 / bc_parameter)) * (
                1
                + (np.power(sigma, 2) * (1 - bc_parameter))
                / (2 * np.power(bc_parameter * mu + 1, 2))
            )

        def _calc_transformed_displ(*, params_dict: Dict, u_key: str) -> Union[float, np.ndarray]:
            """
            Helper function to calculate transformed displacement ("Y" in Kuehn et al., 2024) for
            the aleatory quantile or mean.
            """
            if self.percentile == -1:
                loc = params_dict["prob_distribution_kwargs"][u_key]["loc"]
                scale = params_dict["prob_distribution_kwargs"][u_key]["scale"]
                displ_meters = _analytic_mean_meters(mu=loc, sigma=scale)
                # Convert from arithmetic units (meters) to transformed units
                return (np.power(displ_meters, bc_parameter) - 1) / bc_parameter
            else:
                return params_dict["prob_distribution"].ppf(
                    self.percentile, **params_dict["prob_distribution_kwargs"][u_key]
                )

        def _bc_to_meters(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            """
            Helper function to convert from transformed units (Y) to arithmetic units (meters).
            """
            with np.errstate(invalid="ignore"):
                # Handle values that are too small to calculate
                displ = np.power(value * bc_parameter + 1, 1 / bc_parameter)
                return np.where(np.isnan(displ), 0, displ)

        displ_u1_bc = _calc_transformed_displ(params_dict=stat_params, u_key="u1")

        if self.folded:
            displ_u2_bc = _calc_transformed_displ(params_dict=stat_params, u_key="u2")
            displ_fold_bc = np.mean([displ_u1_bc, displ_u2_bc], axis=0)
            result = _bc_to_meters(displ_fold_bc)
        else:
            result = _bc_to_meters(displ_u1_bc)

        return result.item() if result.size == 1 else result

    def _calc_displ_avg(self) -> Optional[float]:
        """
        Calculate average displacement in meters as the area under the mean displacement profile.

        Parameters
        ----------
        style : str
            Style of faulting (case-insensitive). Common options are 'strike-slip', 'reverse', or
            'normal'.

        magnitude : float
            Earthquake moment magnitude.

        percentile : float
            Aleatory quantile of interest. The Kuehn et al. (2024) model does not provide aleatory
            on the average displacement. This value must be 0.5.

        metric : str, optional
            Definition of displacement (case-insensitive). Valid options are "aggregate". Default
            is "aggregate".

        version : str, optional
            Name of the model formulation (case-insensitive). This value must be "median_coeffs"
            for this method ("full_coeffs" option is not implemented for the average displacement
            calculations).

        Returns
        -------
        float
            Displacement in meters.

        Raises
        ------
        ValueError
            If `percentile` is not 0.5 or if `version` is not "median_coeffs".
        """
        if self.percentile != 0.5:
            e = ValueError(
                f"\n\tThe `{self._MODEL_NAME}` model does not provide aleatory variability "
                "on the average displacement. Use `percentile=0.5` instead.\n\n"
            )
            logging.error(e)
            return

        if self.version == "full_coeffs":
            e = NotImplementedError(
                f"\n\tThe '_calc_displ_avg' method is not available for `{self._MODEL_NAME}` "
                "when `version='full_coeffs'`. Use version='median_coeffs'` "
                "instead.\n\n"
            )
            logging.error(e)
            return

        # Store original values
        original_xl_step = self.xl_step
        original_percentile = self.percentile
        original_sigma_flag = self._EXCLUDE_SIGMA_MAG
        original_folded = self.folded

        try:
            # Temporarily change the values
            self.xl_step = 0.01
            self.percentile = -1  # to compute mean profile
            # FIXME: Mixing how percentile is used is not good; should re-think overall structure
            self._EXCLUDE_SIGMA_MAG = True
            self.folded = False
            xl_array, displacements = self.displ_profile

            return _trapezoidal_integration(x_array=xl_array, y_array=displacements.squeeze())

        finally:
            # Restore original values
            self.xl_step = original_xl_step
            self.percentile = original_percentile
            self._EXCLUDE_SIGMA_MAG = original_sigma_flag
            self.folded = original_folded

    def _calc_displ_max(self) -> Optional[float]:
        """
        Not available: Kuehn et al. (2024) does not provide a maximum displacement model.
        """
        e = NotImplementedError(
            f"`{self._MODEL_NAME}` does not provide a model for maximum displacement."
        )
        logging.error(e)

    def _calc_cdf(self) -> Optional[np.ndarray]:
        # Access only once and store in local variables
        bc_parameter = self._coefficients["lambda"]
        stat_params = self.stat_params_info

        # Reshape arrays for broadcasting
        bc_parameter = bc_parameter.reshape(1, -1)  # Shape (1, N)
        self.displ_array = self.displ_array.reshape(-1, 1)  # Shape (M, 1)

        # Transform displacements
        z = (np.power(self.displ_array, bc_parameter) - 1) / bc_parameter  # Shape (M, N)

        cdf_u1 = stat_params["prob_distribution"].cdf(
            x=z, **stat_params["prob_distribution_kwargs"]["u1"]
        )

        if self.folded:
            cdf_u2 = stat_params["prob_distribution"].cdf(
                x=z, **stat_params["prob_distribution_kwargs"]["u2"]
            )
            return np.mean((cdf_u1, cdf_u2), axis=0).squeeze()

        else:
            return cdf_u1.squeeze()

    @property
    def displ_profile(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.version == "full_coeffs":
            e = NotImplementedError(
                f"The 'displ_profile' method is not available for `{self._MODEL_NAME}` "
                "when `version='full_coeffs'`. Use version='median_coeffs'` instead."
            )
            logging.error(e)
            return

        return super().displ_profile

    # Ensure statistical distribution parameters are updated for current instance
    @property
    def stat_params_info(
        self,
    ) -> Dict[str, Union[Dict[str, Dict[str, Union[float, np.ndarray]]], rv_continuous]]:
        """
        Dictionary of statistical parameters ("params"), probability distribution
        ("prob_distribution"), and its arguments ("prob_distribution_kwargs") for the instance.

        Notes
        -----
        - Keys "u1" and "u2" correspond to "xl" and "1-xl", respectively.
        """
        self._statistical_distribution_params()

        if self._EXCLUDE_SIGMA_MAG:  # Only used for Avg Displ calculations

            statistical_parameters = {
                "u1": {
                    "mu": self._mean_u1,
                    "sigma_xl": self._xl_std_dev_u1,
                }
            }

            probability_distribution_kwargs = {
                "u1": {
                    "loc": statistical_parameters["u1"]["mu"],
                    "scale": statistical_parameters["u1"]["sigma_xl"],
                }
            }

        else:
            statistical_parameters = {
                "u1": {
                    "mu": self._mean_u1,
                    "sigma": self._total_std_dev_u1,
                },
                "u2": {
                    "mu": self._mean_u2,
                    "sigma": self._total_std_dev_u2,
                },
            }

            probability_distribution_kwargs = {
                "u1": {
                    "loc": statistical_parameters["u1"]["mu"],
                    "scale": statistical_parameters["u1"]["sigma"],
                },
                "u2": {
                    "loc": statistical_parameters["u2"]["mu"],
                    "scale": statistical_parameters["u2"]["sigma"],
                },
            }

        probability_distribution = stats.norm

        return {
            "params": statistical_parameters,
            "prob_distribution": probability_distribution,
            "prob_distribution_kwargs": probability_distribution_kwargs,
        }

    @staticmethod
    def _add_arguments(parser):
        # Add arguments specific to model
        parser.add_argument(
            "-f",
            "--folded",
            default=True,
            type=bool,
            help=("Calculate results for the folded location. Default True."),
        )

    @staticmethod
    def main():
        cli_runner(KuehnEtAl2024, KuehnEtAl2024._add_arguments)


if __name__ == "__main__":
    KuehnEtAl2024.main()
