"""Moss and Ross (2011) fault displacement model (https://doi.org/10.1785/0120100248)."""

from typing import Dict, Optional, Union

import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous

from fdhpy.cli import cli_runner
from fdhpy.fault_displacement_model import FaultDisplacementModel
from fdhpy.normalized_fault_displacement_model import NormalizedFaultDisplacementModel


class MossRoss2011(FaultDisplacementModel):
    """
    Moss and Ross (2011) fault displacement model. Applicable to principal surface fault
    displacement on reverse faults.

    This is a normalized model (i.e., uses the D/AD or D/MD form). The reference displacement
    models (AD or MD) from Moss and Ross (2011) are used. All calculations include total aleatory
    variability; in other words, the statistical distributions for D/AD and AD (or D/MD and MD) are
    convolved.

    Parameters
    ----------
    style : str, optional
        Style of faulting (case-insensitive). Default is "reverse".

    magnitude : float
        Earthquake moment magnitude. Recommended range is (5.5, 8).

    xl : float
        Normalized location x/L along the rupture length, range [0, 1.0].

    xl_step : float, optional
        Step size increment for slip profile calculations. Default is 0.05.

    percentile : float
        Aleatory quantile of interest. Use -1 for mean.

    metric : str, optional
        Definition of displacement (case-insensitive). Valid options are "principal". Default is
        "principal".

    version : str
        Name of the model formulation (case-insensitive). Valid options are "d/ad" or "d/md".

    displ_array : np.ndarray, optional
        Displacement test value(s) in meters. Default array is provided.

    Notes
    -----
    See model help at the module level:

        .. code-block:: python

            from fdhpy import MossRoss2011
            print(MossRoss2011.__doc__)

    See model help in command line:

        .. code-block:: console

            $ fd-mr11 --help
    """

    _CONDITIONS = {
        "style": {
            "reverse": {"magnitude": (5.5, 8)},
        },
        "metric": {
            "principal": {"version": ("d/ad", "d/md")},
        },
    }

    _MODEL_NAME = "MossRoss2011"

    _AD_MAG_SCALE_PARAMS = {"intercept": -2.2192, "slope": 0.3244, "std_dev": 0.17}
    _MD_MAG_SCALE_PARAMS = {"intercept": -3.1971, "slope": 0.5102, "std_dev": 0.31}

    # Override the init method to set model defaults
    def __init__(self, **kwargs):
        kwargs.setdefault("metric", "principal")
        kwargs.setdefault("style", "reverse")
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
        # Compute alpha and beta parameters based on folded x/L and model version
        if self._folded_xl is not None:
            if self.version == "d/ad":
                self._alpha = np.exp(
                    -30.4 * self._folded_xl**3
                    + 19.9 * self._folded_xl**2
                    - 2.29 * self._folded_xl
                    + 0.574
                )
                self._beta = np.exp(
                    50.3 * self._folded_xl**3
                    - 34.6 * self._folded_xl**2
                    + 6.6 * self._folded_xl
                    - 1.05
                )
            elif self.version == "d/md":
                a1, a2 = 0.901, 0.713
                b1, b2 = -1.86, 1.74
                self._alpha = a1 * self._folded_xl + a2
                self._beta = b1 * self._folded_xl + b2

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
        return self._normalized_calcs._avg_displ(self._AD_MAG_SCALE_PARAMS)

    def _calc_displ_max(self) -> Optional[float]:
        """Calculate maximum displacement."""
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
        }  # Use nomenclature in Moss & Ross (2011)

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
        cli_runner(MossRoss2011)


# Inherit the docstrings from parent class
MossRoss2011.displ_site.__doc__ = FaultDisplacementModel.displ_site.__doc__
MossRoss2011.displ_avg.__doc__ = FaultDisplacementModel.displ_avg.__doc__
MossRoss2011.displ_max.__doc__ = FaultDisplacementModel.displ_max.__doc__
MossRoss2011.displ_profile.__doc__ = FaultDisplacementModel.displ_profile.__doc__
MossRoss2011.cdf.__doc__ = FaultDisplacementModel.cdf.__doc__
MossRoss2011.prob_exceed.__doc__ = FaultDisplacementModel.prob_exceed.__doc__


if __name__ == "__main__":
    MossRoss2011.main()
