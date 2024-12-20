"""
Moss et al. (2024) fault displacement model (https://doi.org/10.1177/87552930241288560).
 """

import logging
from functools import cached_property
from typing import Dict, Optional, Union

import numpy as np
from scipy import stats
from scipy.interpolate import make_interp_spline
from scipy.stats._distn_infrastructure import rv_continuous

from fdhpy.cli import cli_runner
from fdhpy.fault_displacement_model import FaultDisplacementModel
from fdhpy.normalized_fault_displacement_model import NormalizedFaultDisplacementModel
from fdhpy.utils import AttributeRequiredError


class MossEtAl2024(FaultDisplacementModel):
    """
    Moss et al. (2024) fault displacement model. Applicable to principal surface fault
    displacement on reverse faults.

    This is a normalized model (i.e., uses the D/AD or D/MD form). The reference displacement
    models (AD or MD) from Moss et al. (2024) for the "complete" designation are used. All
    calculations include total aleatory variability; in other words, the statistical distributions
    for D/AD and AD (or D/MD and MD) are convolved.

    Parameters
    ----------
    style : str, optional
        Style of faulting (case-insensitive). Default is "reverse".

    magnitude : float
        Earthquake moment magnitude. Recommended range is (4.7, 8).

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

    use_girs : bool, optional
        If True, use statistical distribution parameters (alpha and beta) from the regression
        model on Figures 4.3 & 4.4 in the Moss et al. (2022) technical report at
        https://doi.org/10.34948/N3F595. If False, use piecewise linear interpolation to obtain
        statistical distribution parameters (alpha and beta) from Table 2 in the Earthquake Spectra
        paper at https://doi.org/10.1177/87552930241288560. Default is False.

    complete : bool, optional
        If True, use Moss et al. (2024) reference displacement models (AD or MD) with the
        "complete" designation. If False, use Moss et al. (2024) reference displacement models
        with the "all" designation. Default is True.

    Notes
    -----
    - The statistical distribution parameters (alpha and beta) can be computed from Table 2 in the
      Earthquake Spectra paper (default) or from the regression model in the GIRS technical report.
      See `use_girs` parameter above for more information.

    - The reference displacement (AD or MD) can be based on the Moss et al. (2024) "complete"
      subset or "all" data. See `complete` parameter above for more information.

    See model help at the module level:

        .. code-block:: python

            from fdhpy import MossEtAl2024
            print(MossEtAl2024.__doc__)

    See model help in command line:

        .. code-block:: console

            $ fd-mea24 --help
    """

    _CONDITIONS = {
        "style": {
            "reverse": {"magnitude": (4.7, 8)},
        },
        "metric": {
            "principal": {"version": ("d/ad", "d/md")},
        },
    }

    _MODEL_NAME = "MossEtAl2024"

    # Override the init method to set model defaults
    def __init__(self, **kwargs):
        kwargs.setdefault("metric", "principal")
        kwargs.setdefault("style", "reverse")
        self.use_girs = kwargs.pop("use_girs", False)
        self.complete = kwargs.pop("complete", True)
        super().__init__(**kwargs)

    def __str__(self):
        parent_str = super().__str__()
        return parent_str[:-1] + f", use_girs={self.use_girs}, complete={self.complete})"

    # NOTE: magnitude, xl, and version are validated in parent class initialization

    # Necessary optional methods in FaultDisplacementModel parent class
    @property
    def _folded_xl(self) -> Optional[float]:
        return self._calc_folded_xl()

    @cached_property
    def _model_data(self) -> np.recarray:
        data_files = {
            "d/ad": "moss_2024_gamma_distribution_parameters_d_ad.csv",
            "d/md": "moss_2024_gamma_distribution_parameters_d_md.csv",
        }

        try:
            if self.version is None:
                raise AttributeRequiredError("version", self._MODEL_NAME)

            df = self._load_data(data_files[self.version])
            if df is not None:
                return df.to_records(index=True)

        # Other errors handled in `_load_data`
        except AttributeRequiredError as e:
            logging.error(e)

    # Required methods for implementing normalized fault displacement models
    @property
    def _AD_MAG_SCALE_PARAMS(self):
        """Set parameters for loglinear magnitude scaling for average displacement."""
        if self.complete:
            return {"intercept": -2.87, "slope": 0.416, "std_dev": 0.2}
        else:
            return {"intercept": -2.98, "slope": 0.427, "std_dev": 0.25}

    @property
    def _MD_MAG_SCALE_PARAMS(self):
        """Set parameters for loglinear magnitude scaling for maximum displacement."""
        if self.complete:
            return {"intercept": -2.5, "slope": 0.415, "std_dev": 0.2}
        else:
            return {"intercept": -2.73, "slope": 0.422, "std_dev": 0.35}

    @property
    def _normalized_calcs(self) -> NormalizedFaultDisplacementModel:
        """Return an instance of the NormalizedFaultDisplacementModel for the current context."""
        return NormalizedFaultDisplacementModel(self)

    # Mandated methods in FaultDisplacementModel parent class
    def _statistical_distribution_params(self) -> None:
        if self.xl is not None:
            # Compute alpha and beta parameters based on folded x/L and model version using
            # regression model in GIRS report Figures 4.3 & 4.4
            if self.use_girs:
                if self.version == "d/ad":
                    a1, a2 = 4.2797, 1.6216
                    b1, b2 = -0.5003, 0.5133
                elif self.version == "d/md":
                    a1, a2 = 1.422, 1.856
                    b1, b2 = -0.0832, 0.1994

                self._alpha = a1 * self._folded_xl + a2
                self._beta = b1 * self._folded_xl + b2

            else:
                c = self._model_data

                params_to_interpolate = {
                    "_alpha": c["alpha"],
                    "_beta": c["beta"],
                }
                for _key, _array in params_to_interpolate.items():
                    # NOTE: np.interp does not extrapolate and scipy.interpolate.interp1d is being
                    # depreciated, so use scipy.interpolate.make_interp_spline with k=1 for
                    # piecewise linear interpolation with extrapolation.
                    # f = interp1d(
                    #     xl_array,
                    #     _array,
                    #     kind="linear",
                    #     fill_value="extrapolate",
                    # )
                    f = make_interp_spline(c["x_L"], _array, k=1)
                    setattr(self, _key, f(self._folded_xl))

        # Compute mu and sigma parameters based on magnitude and model version
        regr_params_map = {
            "d/ad": self._AD_MAG_SCALE_PARAMS,
            "d/md": self._MD_MAG_SCALE_PARAMS,
        }

        if self.magnitude is not None:
            # NOTE: mu based on magnitude is handled in self._normalized_calcs
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
        }  # Use nomenclature in Moss et al. (2024)

        probability_distribution = stats.gamma

        probability_distribution_kwargs = {
            "a": statistical_parameters["alpha"],
            "loc": 0,
            "scale": statistical_parameters["beta"],
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
            "--use_girs",
            default=False,
            type=bool,
            help=(
                "If True, use statistical distribution parameters (alpha and beta) from the "
                "regression model on Figures 4.3 & 4.4 in the Moss et al. (2022) technical report "
                "at https://doi.org/10.34948/N3F595. If False, use piecewise linear interpolation "
                "to obtain statistical distribution parameters (alpha and beta) from Table 2 in "
                "the Earthquake Spectra paper at https://doi.org/10.1177/87552930241288560. "
                "Default is False."
            ),
        )
        parser.add_argument(
            "--complete",
            default=True,
            type=bool,
            help=(
                "If True, use Moss et al. (2024) reference displacement models (AD or MD) with "
                "the 'complete' designation. If False, use Moss et al. (2024) reference "
                "displacement models with the 'all' designation. Default is True."
            ),
        )

    @staticmethod
    def main():
        cli_runner(MossEtAl2024, MossEtAl2024._add_arguments)


# Inherit the docstrings from parent class
MossEtAl2024.displ_site.__doc__ = FaultDisplacementModel.displ_site.__doc__
MossEtAl2024.displ_avg.__doc__ = FaultDisplacementModel.displ_avg.__doc__
MossEtAl2024.displ_max.__doc__ = FaultDisplacementModel.displ_max.__doc__
MossEtAl2024.displ_profile.__doc__ = FaultDisplacementModel.displ_profile.__doc__
MossEtAl2024.cdf.__doc__ = FaultDisplacementModel.cdf.__doc__
MossEtAl2024.prob_exceed.__doc__ = FaultDisplacementModel.prob_exceed.__doc__


if __name__ == "__main__":
    MossEtAl2024.main()
