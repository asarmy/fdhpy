import warnings
from typing import Mapping, Optional, Union

import numpy as np
from scipy import stats


class LoglinearScalingModel:
    """Helper class responsible for implementing loglinear scaling models."""

    def __init__(
        self,
        *,
        regr_params: Mapping[str, Optional[float]],
        independent_var: Optional[Union[float, np.ndarray, list]] = None,
    ):
        """
        Initialize the model with a set of regression parameters and optionally set
        independent_var.

        Parameters
        ----------
        regr_params : dict
            A dictionary of regression parameters in the following format:
            {"intercept": float, "slope": float, "std_dev": Optional[float]}

        independent_var : float or np.ndarray, optional
            The independent variable in loglinear regression.
        """
        self._validate_regr_params(regr_params)
        self._regr_params = regr_params
        self._independent_var = independent_var

    @property
    def regr_params(self) -> Mapping[str, Optional[float]]:
        """Read-only property to access regression parameters."""
        return self._regr_params

    @regr_params.setter
    def regr_params(self, value):
        raise AttributeError("The egression parameters are immutable for an instance.")

    @property
    def independent_var(self) -> Optional[Union[float, np.ndarray]]:
        if isinstance(self._independent_var, list):
            self._independent_var = np.array(self._independent_var)
        return self._independent_var

    @independent_var.setter
    def independent_var(self, value: Optional[Union[float, np.ndarray]]) -> None:
        if value is not None and not isinstance(value, (float, int, np.ndarray)):
            raise ValueError(
                "Invalid type for `independent_var`. Expected float, int, list, or np.ndarray, "
                f"got {type(value).__name__}."
            )
        self._independent_var = value

    @property
    def std_dev(self) -> Optional[float]:
        """Read-only property to access the standard deviation in log10 units."""
        return self._regr_params.get("std_dev")

    @property
    def mean_(self) -> Union[float, np.ndarray]:
        """Read-only property to access the mean in log10 units."""

        if self.independent_var is None:
            raise ValueError(
                "Independent variable `independent_var` must be assigned in the instance."
            )

        intercept = self._regr_params["intercept"]
        slope = self._regr_params["slope"]

        return intercept + slope * self.independent_var  # type: ignore

    def _validate_regr_params(self, regr_params: Mapping[str, Optional[float]]) -> None:
        """
        Validate the given regression parameters dictionary.

        A dictionary of regr_params should be provided in the following format:
            {"intercept": float, "slope": float, "std_dev": Optional[float]}

        Raises
        ------
        TypeError
            If `regr_params` is not a dictionary.
        KeyError
            If required key is missing from `regr_params`.
        ValueError
            If required key has a value of an invalid type.

        Warns
        ------
        UserWarning
            If unexpected keys are present in `regr_params`.
        """
        REQUIRED_KEYS = {"intercept", "slope", "std_dev"}

        REQUIRED_KEYS = {"intercept", "slope", "std_dev"}

        if not isinstance(regr_params, dict):
            raise TypeError(
                f"Expected a dictionary for `regr_params`, got {type(regr_params).__name__}."
            )

        # Key validation
        missing_keys = REQUIRED_KEYS - regr_params.keys()
        if missing_keys:
            raise KeyError(
                f"Required key(s) '{', '.join(missing_keys)}' are missing from `params`."
            )

        unexpected_keys = regr_params.keys() - REQUIRED_KEYS
        if unexpected_keys:
            warnings.warn(
                f"Unexpected key(s) found in `params` will be ignored: "
                f"{', '.join(unexpected_keys)}.",
                UserWarning,
            )

        # Value types validation
        for key, value in regr_params.items():
            if key in REQUIRED_KEYS:
                if key == "std_dev":
                    if not isinstance(value, (float, type(None))):
                        raise ValueError(
                            f"Invalid type for key '{key}'. Expected float or None, but got "
                            f"{type(value).__name__}."
                        )
                else:
                    if not isinstance(value, float):
                        raise ValueError(
                            f"Invalid type for key '{key}'. Expected float, but got "
                            f"{type(value).__name__}."
                        )

    def calc_prediction(
        self,
        percentile: float = 0.5,
    ) -> Union[float, np.ndarray]:
        """
        Calculate value in arithmetic units assuming a lognormal distribution for a specific
        aleatory quantile.

        Parameters
        ----------
        percentile : float, optional
            Aleatory quantile of interest. Use -1 for mean. Default is median (0.5).

        Returns
        -------
        float or np.ndarray
            Predicted response in arithmetic units.

        Raises
        ------
        ValueError
            If `independent_var` is None.
            If `percentile` is not 0.5 when `regr_params["std_dev"]` is None.
            If `percentile` is not -1 or between 0 and 1.
        """

        if not isinstance(percentile, (float, int)):
            raise ValueError(
                "The `percentile` must be an int or float; only one value is allowed."
            )

        if not (percentile == -1 or 0 <= percentile <= 1):
            raise ValueError(
                "The `percentile` must be between 0 and 1 (inclusive), or use -1 for the mean."
            )

        std_dev = self.std_dev
        if self.std_dev is None and percentile != 0.5:
            raise ValueError(
                "Aleatory variability was not defined in `regr_params`, so `percentile` must be "
                "0.5."
            )

        if percentile == -1:
            log10_value = self.mean_ + (np.log(10) / 2 * np.power(self.std_dev, 2))  # type: ignore
        else:
            log10_value = stats.norm.ppf(q=percentile, loc=self.mean_, scale=std_dev)

        if isinstance(log10_value, np.ndarray):
            return np.power(10, log10_value)
        else:
            return (np.power(10, log10_value)).item()
