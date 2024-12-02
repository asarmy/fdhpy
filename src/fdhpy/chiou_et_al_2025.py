"""Chiou et al. (2025) fault displacement model (DOI TBD)."""

import logging
from functools import cached_property
from typing import Dict, Optional, Union

import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous

from fdhpy.cli import cli_runner
from fdhpy.fault_displacement_model import FaultDisplacementModel
from fdhpy.utils import AttributeRequiredError


class ChiouEtAl2025(FaultDisplacementModel):
    """
    Chiou et al. (2025) fault displacement model. Applicable to sum-of-principal surface fault
    displacement on strike-slip faults.

    Parameters
    ----------
    style : str, optional
        Style of faulting (case-insensitive). Default is "strike-slip".

    magnitude : float
        Earthquake moment magnitude. Recommended range is (6, 8.3).

    xl : float
        Normalized location x/L along the rupture length, range [0, 1.0].

    xl_step : float, optional
        Step size increment for slip profile calculations. Default is 0.05.

    percentile : float
        Aleatory quantile of interest. Use -1 for mean.

    metric : str, optional
        Definition of displacement (case-insensitive). Valid options are "sum-of-principal".
        Default is "sum-of-principal".

    version : str, optional
        Name of the model formulation (case-insensitive). Valid options are "model7", "model8.1",
        "model8.2", or "model8.3". Default is "model7".

    displ_array : np.ndarray
        Displacement test value(s) in meters. Default array is provided.

    Notes
    -----
    See model help at the module level:

        .. code-block:: python

            from fdhpy import ChiouEtAl2025
            print(ChiouEtAl2025.__doc__)

    See model help in command line:

        .. code-block:: console

            $ fd-cea25 --help
    """

    _CONDITIONS = {
        "style": {
            "strike-slip": {"magnitude": (6, 8.3)},
        },
        "metric": {
            "sum-of-principal": {"version": ("model7", "model8.1", "model8.2", "model8.3")},
        },
    }

    _MODEL_NAME = "ChiouEtAl2025"

    # Override the init method to set model defaults
    def __init__(self, **kwargs):
        kwargs.setdefault("metric", "sum-of-principal")
        kwargs.setdefault("version", "model7")
        kwargs.setdefault("style", "strike-slip")
        super().__init__(**kwargs)

    # NOTE: magnitude, xl, and version are validated in parent class initialization

    # Necessary optional methods in FaultDisplacementModel parent class
    @property
    def _xstar(self) -> Optional[float]:
        return self._calc_xstar()

    @property
    def _folded_xl(self) -> Optional[float]:
        return self._calc_folded_xl()

    @cached_property
    def _coefficients(self) -> np.recarray:
        """Load the coefficients for the specified model version."""

        try:
            if self.version is None:
                raise AttributeRequiredError("version", self._MODEL_NAME)

            df = self._load_data("chiou_2025_coefficients.csv")
            if df is not None:
                return df[df["version"] == self.version].to_records(index=True)

        # Other errors handled in `_load_data`
        except AttributeRequiredError as e:
            logging.error(e)

    # Methods unique to model
    def _calc_fm(self) -> Optional[float]:
        """
        Calculate the magnitude scaling component. This is 'f_M(M)' in Chiou et al. (2025).
        """
        c = self._coefficients
        fm = c["m2"] * (self.magnitude - c["m3"]) + (c["m2"] - c["m1"]) / c["cn"] * np.log(
            0.5 * (1 + np.exp(-c["cn"] * (self.magnitude - c["m3"])))
        )
        return fm.item()

    def _calc_sigma_mag(self) -> Optional[float]:
        """
        Calculate the magnitude aleatory variability. This is 'sigma_eq' in Chiou et al. (2025).
        """
        c = self._coefficients
        std_dev = np.maximum(
            0.4, c["cv1"] * np.exp(c["cv2"] * np.maximum(0, self.magnitude - 6.1))
        )
        return std_dev.item()

    def _calc_sigma_xl(self) -> Optional[float]:
        """
        Calculate the x/L aleatory variability. This is 'sigma' in Chiou et al. (2025).
        """
        try:
            if self.xl is None:
                raise AttributeRequiredError("xl", self._MODEL_NAME)
        except AttributeRequiredError as e:
            logging.error(e)
            return None

        c = self._coefficients
        std_dev = c["cv3"] * np.exp(c["cv4"] * np.maximum(0, self._folded_xl - c["ccap"]))
        return std_dev.item()

    # Mandated methods in FaultDisplacementModel parent class
    def _statistical_distribution_params(self) -> None:
        c = self._coefficients

        # Calculate the mean `mu` of the Gaussian component
        fm = self._calc_fm()
        self._mu = c["c0"] + fm + c["c1"] * (self._xstar - 1)

        # Calculate the total standard deviation `sigma_prime` of the Gaussian component
        std_dev_mag = self._calc_sigma_mag()
        std_dev_xl = self._calc_sigma_xl()
        self._sigma_prime = np.sqrt(np.power(std_dev_mag, 2) + np.power(std_dev_xl, 2))

        # Calculate the mean and standard deviation `nu` of the exponential component
        self._nu = c["cv5"].item()

        # NOTE: scipy parametrization of shape parameter is different than R gamless
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html
        self._k = 1 / (self._sigma_prime * (1 / self._nu))

    def _calc_displ_site(self) -> Optional[float]:
        """Calculate deterministic scenario displacement."""

        stat_params = self.stat_params_info  # Access only once and store in a local variable

        if self.percentile == -1:
            displ = np.exp(
                stat_params["params"]["mu"] + np.power(stat_params["params"]["sigma_prime"], 2) / 2
            ) / (stat_params["params"]["nu"] + 1)
        else:
            # NOTE: negative modifications required to "flip" EMG distribution to nEMG
            ln_displ = -1 * stat_params["prob_distribution"].ppf(
                1 - self.percentile,
                **stat_params["prob_distribution_kwargs"],
            )
            displ = np.exp(ln_displ)

        return displ.item()

    def _calc_displ_avg(self) -> Optional[float]:
        """Calculate average displacement."""

        if self.percentile != 0.5:
            e = ValueError(
                f"\n\tThe `{self._MODEL_NAME}` model does not provide aleatory variability "
                "on the average displacement. Use `percentile=0.5` instead.\n\n"
            )
            logging.error(e)
            return

        c = self._coefficients
        fm = self._calc_fm()
        return np.exp(c["c0"].item() + fm) * 0.3920

    def _calc_displ_max(self) -> Optional[float]:
        """
        Not available: Chiou et al. (2025) does not provide a maximum displacement model.
        """
        e = NotImplementedError(
            f"`{self._MODEL_NAME}` does not provide a model for maximum displacement."
        )
        logging.error(e)

    def _calc_cdf(self) -> Optional[np.ndarray]:
        z = np.log(self.displ_array)
        stat_params = self.stat_params_info
        # NOTE: negative modification and "1-" used to flip cdf & ccdf from EMG to nEMG
        return 1 - stat_params["prob_distribution"].cdf(
            x=z * -1, **stat_params["prob_distribution_kwargs"]
        )

    # Ensure statistical distribution parameters are updated for current instance
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
            "sigma_prime": self._sigma_prime,
            "nu": self._nu,
            "shape": self._k,
        }  # Use nomenclature in Chiou et al. (2025)

        probability_distribution = stats.exponnorm

        probability_distribution_kwargs = {
            "loc": statistical_parameters["mu"] * -1,  # NOTE: Use mu*-1 for negative EMG
            "scale": statistical_parameters["sigma_prime"],
            "K": statistical_parameters["shape"],
        }

        return {
            "params": statistical_parameters,
            "prob_distribution": probability_distribution,
            "prob_distribution_kwargs": probability_distribution_kwargs,
        }

    @staticmethod
    def main():
        cli_runner(ChiouEtAl2025)


if __name__ == "__main__":
    ChiouEtAl2025.main()
