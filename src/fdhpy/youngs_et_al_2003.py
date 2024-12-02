"""Youngs et al. (2003) fault displacement model (https://doi.org/10.1193/1.1542891)."""

import logging
from typing import Dict, Optional, Union

import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous

from fdhpy.cli import cli_runner
from fdhpy.fault_displacement_model import FaultDisplacementModel
from fdhpy.normalized_fault_displacement_model import NormalizedFaultDisplacementModel


class YoungsEtAl2003(FaultDisplacementModel):
    """
    Youngs et al. (2003) fault displacement model. Applicable to principal surface fault
    displacement on normal faults.

    This is a normalized model (i.e., uses the D/AD or D/MD form). The reference displacement
    models (AD or MD) from Wells and Coppersmith (1994) for "All" styles are used. All calculations
    include total aleatory variability; in other words, the statistical distributions for D/AD and
    AD (or D/MD and MD) are convolved.

    Parameters
    ----------
    style : str, optional
        Style of faulting (case-insensitive). Default is "normal".

    magnitude : float
        Earthquake moment magnitude. Recommended range is (5.5, 8).

    xl : float, optional
        Normalized location x/L along the rupture length, range [0, 1.0].

    xl_step : float, optional
        Step size increment for slip profile calculations. Default is 0.05.

    percentile : float, optional
        Aleatory quantile of interest. Use -1 for mean.

    metric : str, optional
        Definition of displacement (case-insensitive). Valid options are "principal". Default is
        "principal".

    version : str
        Name of the model formulation for the given metric (case-insensitive). Valid options are
        "d/ad" or "d/md".

    displ_array : np.ndarray, optional
        Displacement test value(s) in meters. Default array is provided.

    Notes
    -----
    - Distributed displacement models are not implemented yet.

    See model help at the module level:

        .. code-block:: python

            from fdhpy import YoungsEtAl2003
            print(YoungsEtAl2003.__doc__)

    See model help in command line:

        .. code-block:: console

            $ fd-yea03 --help
    """

    _CONDITIONS = {
        "style": {
            "normal": {},  # Magnitude range not provided by model; default in parent class is used
        },
        "metric": {
            "principal": {"version": ("d/ad", "d/md")},
            "distributed": {"version": None},
        },
    }

    _MODEL_NAME = "YoungsEtAl2003"

    # Parameters based on Wells and Coppersmith (1994) All styles
    _AD_MAG_SCALE_PARAMS = {"intercept": -4.8, "slope": 0.69, "std_dev": 0.36}
    _MD_MAG_SCALE_PARAMS = {"intercept": -5.46, "slope": 0.82, "std_dev": 0.42}

    # Override the init method to set model defaults
    def __init__(self, **kwargs):
        kwargs.setdefault("metric", "principal")
        kwargs.setdefault("style", "normal")
        super().__init__(**kwargs)

    # NOTE: magnitude, xl, and version are validated in parent class initialization

    # Necessary optional methods in FaultDisplacementModel parent class
    @property
    def _folded_xl(self) -> Optional[float]:
        return self._calc_folded_xl()

    # Required methods for implementing normalized fault displacement models
    @property
    def _normalized_calcs(self) -> NormalizedFaultDisplacementModel:
        """Return an instance of the NormalizedFaultDisplacementModel for the current context."""
        return NormalizedFaultDisplacementModel(self)

    # Mandated methods in FaultDisplacementModel parent class
    def _statistical_distribution_params(self) -> None:
        """
        Compute and set the statistical distribution parameters mu, sigma, alpha, and
        beta for the model version (i.e., "d/ad" or "d/md").
        """
        if self.metric == "distributed":
            e = NotImplementedError(
                f"Distributed model(s) are not implemented yet in `{self._MODEL_NAME}`."
            )
            logging.error(e)
            return

        if self._folded_xl is not None:
            if self.version == "d/ad":
                a1, a2 = 1.628, -0.193
                b1, b2 = -0.476, 0.009
            elif self.version == "d/md":
                a1, a2 = 1.138, -0.705
                b1, b2 = -0.257, 0.421

            self._alpha = np.exp(a1 * self._folded_xl + a2)
            self._beta = np.exp(b1 * self._folded_xl + b2)

        # Compute mu and sigma parameters based on magnitude and model version
        regr_params_map = {
            "d/ad": self._AD_MAG_SCALE_PARAMS,
            "d/md": self._MD_MAG_SCALE_PARAMS,
        }
        # NOTE: magnitude is handled in self._normalized_calcs
        p = self._normalized_calcs._calc_mag_scale_stat_params(regr_params_map[self.version])

        self._mu, self._sigma = p[0], p[1]

        # Set to None if not computed
        self._alpha = getattr(self, "_alpha", None)
        self._beta = getattr(self, "_beta", None)
        self._mu = getattr(self, "_mu", None)
        self._sigma = getattr(self, "_sigma", None)

    def _calc_displ_site(self) -> Optional[float]:
        """Calculate deterministic scenario displacement."""
        return self._normalized_calcs._site_displ()

    def _calc_displ_avg(self) -> Optional[float]:
        """Calculate average displacement."""
        if self.metric == "distributed":
            e = ValueError(
                "Average displacement cannot be computed for distributed faults. Use "
                "`metric='principal'` instead."
            )
            logging.error(e)
            return
        return self._normalized_calcs._avg_displ(self._AD_MAG_SCALE_PARAMS)

    def _calc_displ_max(self) -> Optional[float]:
        """Calculate maximum displacement."""
        if self.metric == "distributed":
            e = ValueError(
                "Maximum displacement cannot be computed for distributed faults. Use "
                "`metric='principal'` instead."
            )
            logging.error(e)
            return
        return self._normalized_calcs._max_displ(self._MD_MAG_SCALE_PARAMS)

    def _calc_cdf(self) -> Optional[np.ndarray]:
        """Calculate cumulative probability."""
        return self._normalized_calcs._cdf()

    # Ensure statistical distribution parameters are updated for current instance
    @property
    def stat_params_info(
        self,
    ) -> Dict[str, Union[Dict[str, Optional[float]], rv_continuous, Dict[str, float]]]:
        """
        Dictionary of statistical parameters ("params"), probability distribution
        ("prob_distribution"), and its arguments ("prob_distribution_kwargs") for the instance.
        """
        self._statistical_distribution_params()

        statistical_parameters = {
            "mu": self._mu,
            "sigma": self._sigma,
            "alpha": self._alpha,
            "beta": self._beta,
        }  # Use nomenclature in Youngs et al. (2003)

        if self.version == "d/ad":
            probability_distribution = stats.gamma

            probability_distribution_kwargs = {
                "a": statistical_parameters["alpha"],
                "loc": 0,
                "scale": statistical_parameters["beta"],
            }
        elif self.version == "d/md":
            probability_distribution = stats.beta

            probability_distribution_kwargs = {
                "a": statistical_parameters["alpha"],
                "b": statistical_parameters["beta"],
            }

        return {
            "params": statistical_parameters,
            "prob_distribution": probability_distribution,
            "prob_distribution_kwargs": probability_distribution_kwargs,
        }

    @staticmethod
    def main():
        cli_runner(YoungsEtAl2003)


# Inherit the docstrings from parent class
YoungsEtAl2003._calc_displ_site.__doc__ = FaultDisplacementModel._calc_displ_site.__doc__
YoungsEtAl2003._calc_displ_avg.__doc__ = FaultDisplacementModel._calc_displ_avg.__doc__
YoungsEtAl2003.displ_max.__doc__ = FaultDisplacementModel.displ_max.__doc__
YoungsEtAl2003.displ_profile.__doc__ = FaultDisplacementModel.displ_profile.__doc__
YoungsEtAl2003.cdf.__doc__ = FaultDisplacementModel.cdf.__doc__
YoungsEtAl2003.prob_exceed.__doc__ = FaultDisplacementModel.prob_exceed.__doc__


if __name__ == "__main__":
    YoungsEtAl2003.main()
