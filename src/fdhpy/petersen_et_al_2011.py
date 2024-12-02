"""Petersen et al. (2011) fault displacement model (https://doi.org/10.1785/0120100035)."""

import logging
from typing import Dict, Optional, Union

import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous

from fdhpy.cli import cli_runner
from fdhpy.fault_displacement_model import FaultDisplacementModel
from fdhpy.utils import ValidListError, _trapezoidal_integration


class PetersenEtAl2011(FaultDisplacementModel):
    """
    Petersen et al. (2011) fault displacement model. Applicable to principal surface fault
    displacement on strike-slip faults.

    Parameters
    ----------
    style : str, optional
        Style of faulting (case-insensitive). Default is "strike-slip".

    magnitude : float
        Earthquake moment magnitude. Recommended range is (6, 8).

    xl : float
        Normalized location x/L along the rupture length, range [0, 1.0].

    xl_step : float, optional
        Step size increment for slip profile calculations. Default is 0.05.

    percentile : float
        Aleatory quantile of interest. Use -1 for mean.

    metric : str, optional
        Definition of displacement (case-insensitive). Valid options are "principal". Default is
        "principal".

    version : str, optional
        Name of the model formulation for the given metric (case-insensitive). Valid options are
        "elliptical" or "quadratic". Default is "elliptical".

    displ_array : np.ndarray, optional
        Displacement test value(s) in meters. Default array is provided.

    Notes
    -----
    - Distributed displacement models are not implemented yet.

    - The `ln(D)` versions of the principal displacement models are used; i.e., the `ln(D/AD)`
      models are not implemented.

    - Only the elliptical and quadratic shapes are implemented; the bilinear shape is not
      implemented.

    See model help at the module level:

        .. code-block:: python

            from fdhpy import PetersenEtAl2011
            print(PetersenEtAl2011.__doc__)

    See model help in command line:

        .. code-block:: console

            $ fd-pea11 --help
    """

    _CONDITIONS = {
        "style": {
            "strike-slip": {"magnitude": (6, 8)},
        },
        "metric": {
            "principal": {"version": ("elliptical", "quadratic")},
            "distributed": {"version": None},
        },
    }

    _MODEL_NAME = "PetersenEtAl2011"

    # Override the init method to set model defaults
    def __init__(self, **kwargs):
        kwargs.setdefault("metric", "principal")
        kwargs.setdefault("version", "elliptical")
        kwargs.setdefault("style", "strike-slip")
        super().__init__(**kwargs)

    # NOTE: magnitude, xl, and version are validated in parent class initialization

    # Necessary optional methods in FaultDisplacementModel parent class
    @property
    def _folded_xl(self) -> Optional[float]:
        return self._calc_folded_xl()

    @property
    def _xstar(self) -> Optional[float]:
        return self._calc_xstar()

    # Mandated methods in FaultDisplacementModel parent class
    def _statistical_distribution_params(self) -> None:
        """
        Calculate and set the predicted statistical distribution parameters (mean and standard
        deviation) for the instance.

        NOTE: Mean and standard deviation are in natural log units, and exp(mean) is in
        centimeters, not meters.
        """

        if self.metric == "distributed":
            e = NotImplementedError(
                f"Distributed model(s) are not implemented yet in `{self._MODEL_NAME}`."
            )
            logging.error(e)
            return

        if self.version == "elliptical":
            a, b, c = 1.7927, 3.3041, -11.2192
            self._std_dev = 1.1348
            self._mean = b * self._xstar + a * self.magnitude + c
        elif self.version == "quadratic":
            a, b, c, d = 1.7895, 14.4696, -20.1723, -10.54512
            self._std_dev = 1.1346
            self._mean = (
                a * self.magnitude + b * self._folded_xl + c * np.power(self._folded_xl, 2) + d
            )
        else:
            valid_version_options = (
                self._CONDITIONS.get("metric", {}).get(self.metric, {}).get("version", [])
            )
            e = ValidListError("version", self.version, valid_version_options, self._MODEL_NAME)
            logging.error(e)

    def _calc_displ_site(self) -> Optional[float]:
        """Calculate deterministic scenario displacement."""

        stat_params = self.stat_params_info  # Access only once and store in a local variable

        if self.percentile == -1:
            ln_displ_cm = (
                stat_params["prob_distribution_kwargs"]["loc"]
                + np.power(stat_params["prob_distribution_kwargs"]["scale"], 2) / 2
            )
        else:
            ln_displ_cm = stat_params["prob_distribution"].ppf(
                self.percentile, **stat_params["prob_distribution_kwargs"]  #
            )

        return float(np.exp(ln_displ_cm) / 100)

    def _calc_displ_avg(self) -> Optional[float]:
        """
        Calculate average displacement in meters as the area under the mean displacement profile.

        NOTE: Aleatory variability is not partitioned into between- and within-event terms in
        Petersen et al. (2011) model, so the mean displacement profile includes between-event
        aleatory variability.

        Parameters
        ----------
        style : str, optional
            Style of faulting (case-insensitive). Default is "strike-slip".
        magnitude : float
            Earthquake moment magnitude. Recommended range is (6, 8).
        percentile : float
            Aleatory quantile of interest. The Petersen et al. (2011) model only allows for the
            mean prediction for the average displacement. This value must be -1.
        metric : str, optional
            Definition of displacement (case-insensitive). This value must be "principal". Default
            is "principal".
        version : str, optional
            Name of the model formulation for the given metric (case-insensitive). Valid options
            are "elliptical" or "quadratic". Default is "elliptical".

        Returns
        -------
        float
            Displacement in meters.

            Raises
        ------
        ValueError
            If `percentile` is not -1 or `metric` is 'distributed'.
        """
        if self.metric == "distributed":
            e = ValueError(
                "Average displacement cannot be computed for distributed faults. Use "
                "`metric='principal'` instead."
            )
            logging.error(e)
            return

        if self.percentile != -1:
            e = ValueError(
                f"The `{self._MODEL_NAME}` model only allows for the mean prediction "
                "for the average displacement. Use `percentile=-1` instead."
            )
            logging.error(e)
            return

        # Store original values
        original_xl_step = self.xl_step
        original_xl = self.xl

        try:
            # Temporarily change the values
            self.xl_step = 0.01
            self.xl = None
            xl_array, displacements = self.displ_profile
            return _trapezoidal_integration(x_array=xl_array, y_array=displacements)

        finally:
            # Restore original values
            self.xl_step = original_xl_step
            self.xl = original_xl

    def _calc_displ_max(self) -> Optional[float]:
        """
        Not available: Petersen et al. (2011) does not provide a maximum displacement model.
        """
        e = NotImplementedError(
            f"`{self._MODEL_NAME}` does not provide a model for maximum displacement."
        )
        logging.error(e)

    def _calc_cdf(self) -> Optional[np.ndarray]:
        z = np.log(np.multiply(self.displ_array, 100))
        stat_params = self.stat_params_info
        return stat_params["prob_distribution"].cdf(x=z, **stat_params["prob_distribution_kwargs"])

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
            "mu": self._mean,
            "sigma": self._std_dev,
        }  # Use nomenclature in Petersen et al. (2011)

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
    def main():
        cli_runner(PetersenEtAl2011)


if __name__ == "__main__":
    PetersenEtAl2011.main()
