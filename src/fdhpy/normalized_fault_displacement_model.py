import logging
from typing import Mapping, Optional, Tuple

import numpy as np
from scipy import stats

from fdhpy.loglinear_scaling_model import LoglinearScalingModel
from fdhpy.utils import AttributeIgnoredWarning, AttributeRequiredError, _required


class NormalizedFaultDisplacementModel:
    """
    Helper class responsible for implementing total aleatory variability calculations in normalized
    models.

    Intended to be composed in specific fault displacement model subclasses to implement those
    abstract methods (e.g., MossEtAl2011).
    """

    _N = 500_000  # used to generate samples in NormalizedFaultDisplacementModel class

    def __init__(self, context):
        self.context = context  # Store context for access to its attributes

    # NOTE: Checks for required scenario attributes (e.g., magnitude, xl, percentile) are handled
    # upstream in FaultDisplacementModel class.

    def _calc_mag_scale_stat_params(
        self, regr_params: Mapping[str, Optional[float]]
    ) -> Tuple[float, float]:
        """Calculate statistical distribution parameters in log10 units."""

        model = LoglinearScalingModel(
            regr_params=regr_params, independent_var=self.context.magnitude
        )

        return model.mean_, model.std_dev

    def _calc_displ_scaling(self, regr_params: Mapping[str, Optional[float]]) -> Optional[float]:
        """Calculate the displacement (in meters) assuming a loglinear scaling model."""

        try:
            if self.context.percentile is None:
                raise AttributeRequiredError("percentile", self.context._MODEL_NAME)
            if regr_params is None:
                raise AttributeRequiredError(
                    "Regression parameters dictionary", self.context._MODEL_NAME
                )
        except AttributeRequiredError as e:
            logging.error(e)
            return None

        scaling_model = LoglinearScalingModel(
            regr_params=regr_params, independent_var=self.context.magnitude
        )
        return scaling_model.calc_prediction(percentile=self.context.percentile)

    def _generate_sample(self):
        """
        Use sampling to convolve statistical distributions in normalized models to capture total
        aleatory variability.
        """
        # Access only once and store in a local variable
        stat_params = self.context.stat_params_info

        # Sample AD or MD values
        np.random.seed(1)
        samples_xd = np.power(
            10,
            stats.norm.rvs(
                loc=stat_params["params"]["mu"], scale=stat_params["params"]["sigma"], size=self._N
            ),
        )

        # Sample D/AD or D/MD values
        # NOTE: Seed must be explicitly set each time to force deterministic (repeatable)
        # behavior. See https://stackoverflow.com/a/21494630

        np.random.seed(1)
        samples_d_xd = stat_params["prob_distribution"].rvs(
            **stat_params["prob_distribution_kwargs"], size=self._N
        )

        # Truncation to correct for D/MD >1 (e.g., Moss et al. 2024)
        if self.context.version == "d/md":
            drop_idx = np.nonzero(samples_d_xd > 1)
            samples_d_xd[drop_idx] = np.nan

        return samples_d_xd * samples_xd

    def _calc_displ_from_sample(self) -> Optional[float]:
        """
        Calculate deterministic displacement (in meters) based on percentile from sample of
        displacements, with total aleatory variability, that is generated with Monte Carlo
        sampling.
        """
        samples = np.asarray(self._generate_sample())
        if self.context.percentile == -1:
            displ_meters = np.nanmean(samples)
        else:
            displ_meters = np.nanpercentile(samples, self.context.percentile * 100)

        displ_meters = float(displ_meters)

        return displ_meters

    def _calc_cdf_from_numerical_integration(self) -> np.ndarray:
        """
        Calculate the cumulative distribution with total aleatory variability in normalized models
        using numerical integration to convolve the statistical distributions.
        """
        # Access only once and store in a local variable
        stat_params = self.context.stat_params_info

        # Compute array of XD values
        n_eps, dz = 6, 0.1
        epsilons = np.arange(-n_eps, n_eps + dz, dz)
        z = np.power(10, stat_params["params"]["mu"] + epsilons * stat_params["params"]["sigma"])
        prob_z = stats.norm.pdf(epsilons)

        # Compute array of D/XD values
        y = self.context.displ_array / z[:, np.newaxis]
        y = y.T

        # Compute array of D/XD cumulative distributions
        cdf_matrix = stat_params["prob_distribution"].cdf(
            x=y, **stat_params["prob_distribution_kwargs"]
        )

        # Truncation to correct for D/MD >1 (e.g., Moss et al. 2024)
        if self.context.version == "d/md":
            cdf_value_at_1 = stat_params["prob_distribution"].cdf(
                x=1, **stat_params["prob_distribution_kwargs"]
            )
            cdf_matrix = np.where(y > 1, 1, cdf_matrix / cdf_value_at_1)

        # Compute weighted sum
        cdf = np.dot(cdf_matrix, prob_z) * dz

        return cdf

    # Private helper functions to handle validations once
    @_required("version", context_flag=True)
    def _site_displ(self) -> Optional[float]:
        return self._calc_displ_from_sample()

    def _avg_displ(self, params) -> Optional[float]:
        if self.context.version:
            message = AttributeIgnoredWarning(
                "version", "average displacement", self.context._MODEL_NAME
            )
            logging.warning(message)
        return self._calc_displ_scaling(params)

    def _max_displ(self, params) -> Optional[float]:
        if self.context.version:
            message = AttributeIgnoredWarning(
                "version", "maximum displacement", self.context._MODEL_NAME
            )
            logging.warning(message)
        return self._calc_displ_scaling(params)

    @_required("version", context_flag=True)
    def _cdf(self) -> Optional[np.ndarray]:
        return self._calc_cdf_from_numerical_integration()
