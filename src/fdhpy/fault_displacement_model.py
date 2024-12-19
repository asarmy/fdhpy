import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from attrs import define

from fdhpy.fault_displacement_model_variables import FaultDisplacementModelVariables
from fdhpy.utils import AttributeIgnoredWarning, _required


@define(slots=True, kw_only=True)
class FaultDisplacementModel(FaultDisplacementModelVariables, ABC):
    """
    Abstract base class for Fault Displacement Models.

    Intended to be subclassed for specific fault displacement models. Defines abstract methods to
    be implemented by subclassed models.
    """

    # Optional methods available to subclasses
    @_required("xl")
    def _calc_folded_xl(self) -> Optional[float]:
        """Calculate folded x/L."""
        return np.minimum(self.xl, 1 - self.xl)

    @_required("xl")
    def _calc_xstar(self) -> Optional[float]:
        """Calculate the elliptical shape scaling parameter in the Petersen and Chiou models."""
        return np.sqrt(1 - np.power(1 / 0.5, 2) * np.power(self.xl - 0.5, 2))

    # Optional model coefficient/parameter management available to subclasses
    @classmethod
    def _load_data(cls, filename: str) -> pd.DataFrame:
        """
        Load model data from a CSV file.

        Parameters
        ----------
        filename : str
            The name of the CSV file containing the model data.

        directory : Union[str, pathlib.Path]
            The directory containing the CSV file. Default is './src/fdhpy/data' directory.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the model data.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the provided filepath.
        ValueError
            If the DataFrame is empty or if there is an error parsing the file.
        pd.errors.EmptyDataError
            If the CSV file is empty (caught and re-raised as ValueError).
        pd.errors.ParserError
            If there is a parsing error with the CSV file (caught and re-raised as ValueError).
        Exception
            If an unexpected error occurs while reading the file.
        """
        # Ensure directory is a Path object
        directory = Path(os.path.join(os.path.dirname(__file__), "data"))
        filepath = directory / filename
        filepath = filepath.resolve()

        try:
            # Attempt to load the file
            df = pd.read_csv(filepath)
            if df.empty:
                raise ValueError(f"The DataFrame at {filepath} is empty.")
            return df

        except FileNotFoundError:
            message = FileNotFoundError(f"File {filepath} not found.")
            logging.error(message)

        except pd.errors.EmptyDataError:
            message = ValueError(f"The file at {filepath} is empty.")
            logging.error(message)

        except pd.errors.ParserError:
            message = ValueError(f"Error parsing the file {filepath}")
            logging.error(message)

        except Exception as e:
            message = ValueError(f"An unexpected error occurred: {e}")
            logging.error(message)

    # Required methods shared by all subclasses
    @abstractmethod
    def _statistical_distribution_params(self) -> None:
        """Compute and set the predicted statistical distribution parameters for the instance."""
        pass

    # NOTE: Property is used throughout for calculated attributes; can remove @property in the
    # future to allow the use of arguments for adjustments, such as mean shifts for epistemic
    # uncertainty or partially non-ergodic evaluations; or can use separate methods, such as
    # `displ_site_adjusted(self, *args)`
    @property
    @_required("magnitude", "xl", "style", "percentile")
    def displ_site(self) -> Optional[Union[float, np.ndarray]]:
        """
        Calculate deterministic scenario displacement in meters.

        Parameters
        ----------
        style : str
            Style of faulting (case-insensitive). Common options are 'strike-slip', 'reverse', or
            'normal'.

        magnitude : float
            Earthquake moment magnitude.

        xl : float
            Normalized location x/L along the rupture length, range [0, 1.0].

        percentile : float
            Aleatory quantile of interest. Use -1 for mean.

        metric : str
            Definition of displacement (case-insensitive), e.g. 'principal'. Varies by model. See
            model help for specifics.

        version : str
            Name of the model formulation for the given metric (case-insensitive), e.g. 'D/AD' or
            'median_coeffs'. Varies by model. See model help for specifics.

        Returns
        -------
        float
            Displacement in meters.

        Notes
        -----
        Parameters shown are typical; details vary with specific model.
        """
        return self._calc_displ_site()

    @abstractmethod
    def _calc_displ_site(self) -> Optional[Union[float, np.ndarray]]:
        """Abstract method to be implemented by specific fault displacement model subclass."""
        pass

    @property
    @_required("magnitude", "style", "percentile")
    def displ_avg(self) -> Optional[float]:
        """
        Calculate average displacement in meters.

        Parameters
        ----------
        style : str
            Style of faulting (case-insensitive). Common options are 'strike-slip', 'reverse', or
            'normal'.

        magnitude : float
            Earthquake moment magnitude.

        percentile : float
            Aleatory quantile of interest. Use -1 for mean. Not available in all models (i.e., not
            all models provide aleatory variability on the average displacement).

        metric : str
            Definition of displacement (case-insensitive), e.g. 'principal'. Varies by model. See
            model help for specifics.

        version : str
            Name of the model formulation for the given metric (case-insensitive), e.g. 'D/AD' or
            'median_coeffs'. Varies by model. See model help for specifics.

        Returns
        -------
        float
            Displacement in meters.

        Notes
        -----
        Parameters shown are typical; details vary with specific model.
        """
        if self.xl:
            message = AttributeIgnoredWarning("xl", "average displacement")
            logging.warning(message)
        return self._calc_displ_avg()

    @abstractmethod
    def _calc_displ_avg(self) -> Optional[float]:
        """Abstract method to be implemented by specific fault displacement model subclass."""
        pass

    @property
    @_required("magnitude", "style", "percentile")
    def displ_max(self) -> Optional[float]:
        """
        Calculate maximum displacement in meters. Not available in all models

        Parameters
        ----------
        style : str
            Style of faulting (case-insensitive). Common options are 'strike-slip', 'reverse', or
            'normal'.

        magnitude : float
            Earthquake moment magnitude.

        percentile : float
            Aleatory quantile of interest. Use -1 for mean. Not available in all models (i.e., not
            all models provide aleatory variability on the maximum displacement).

        metric : str
            Definition of displacement (case-insensitive), e.g. 'principal'. Varies by model. See
            model help for specifics.

        version : str
            Name of the model formulation for the given metric (case-insensitive), e.g. 'D/AD' or
            'median_coeffs'. Varies by model. See model help for specifics.

        Returns
        -------
        float
            Displacement in meters.

        Notes
        -----
        Parameters shown are typical; details vary with specific model.
        """
        if self.xl:
            message = AttributeIgnoredWarning("xl", "maximum displacement")
            logging.warning(message)
        return self._calc_displ_max()

    @abstractmethod
    def _calc_displ_max(self) -> Optional[float]:
        """Abstract method to be implemented by specific fault displacement model subclass."""
        pass

    @property
    @_required("magnitude", "style", "xl")
    def cdf(self) -> Optional[np.ndarray]:
        """
        Calculate the probability that the displacement is less than or equal to a specific value.

        Parameters
        ----------
        style : str
            Style of faulting (case-insensitive). Common options are 'strike-slip', 'reverse', or
            'normal'.

        magnitude : float
            Earthquake moment magnitude.

        xl : float
            Normalized location x/L along the rupture length, range [0, 1.0].

        displ_array : ArrayLike, optional
            Test values of displacement in meters. Default array is provided.

        metric : str
            Definition of displacement (case-insensitive), e.g. 'principal'. Varies by model. See
            model help for specifics.

        version : str
            Name of the model formulation for the given metric (case-insensitive), e.g. 'D/AD' or
            'median_coeffs'. Varies by model. See model help for specifics.

        Returns
        -------
        numpy.ndarray
            Cumulative probability.

        Notes
        -----
        Parameters shown are typical; details vary with specific model.
        """
        if self.percentile:
            message = AttributeIgnoredWarning("percentile", "cumulative probability")
            logging.warning(message)
        return self._calc_cdf()

    @abstractmethod
    def _calc_cdf(self) -> Optional[np.ndarray]:
        """Abstract method to be implemented by specific fault displacement model subclass."""
        pass

    @property
    @_required("magnitude", "style", "xl")
    def prob_exceed(self) -> Optional[np.ndarray]:
        """
        Calculate the probability that the displacement exceeds a specific value.

        Parameters
        ----------
        style : str
            Style of faulting (case-insensitive). Common options are 'strike-slip', 'reverse', or
            'normal'.

        magnitude : float
            Earthquake moment magnitude.

        xl : float
            Normalized location x/L along the rupture length, range [0, 1.0].

        displ_array : ArrayLike, optional
            Test values of displacement in meters. Default array is provided.

        metric : str
            Definition of displacement (case-insensitive), e.g. 'principal'. Varies by model. See
            model help for specifics.

        version : str
            Name of the model formulation for the given metric (case-insensitive), e.g. 'D/AD' or
            'median_coeffs'. Varies by model. See model help for specifics.

        Returns
        -------
        numpy.ndarray
            Exceedance probability.

        Notes
        -----
        Parameters shown are typical; details vary with specific model.
        """
        return self._calc_prob_exceed()

    def _calc_prob_exceed(self) -> Optional[np.ndarray]:
        """Helper method to calculate exceedance probability."""
        # NOTE: LA23 requires scaling for non-zero displacements; other models are `1-cdf`
        return 1 - self.cdf

    @property
    @_required("xl_step")
    def displ_profile(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate the displacement profile at 'xl_step' increments.

        Parameters
        ----------
        style : str
            Style of faulting (case-insensitive). Common options are 'strike-slip', 'reverse', or
            'normal'.

        magnitude : float
            Earthquake moment magnitude.

        percentile : float
            Aleatory quantile of interest. Use -1 for mean.

        xl_step : float, optional
            Step size increment for slip profile calculations. Default is 0.05.

        metric : str
            Definition of displacement (case-insensitive), e.g. 'principal'. Varies by model. See
            model help for specifics.

        version : str
            Name of the model formulation for the given metric (case-insensitive), e.g. 'D/AD' or
            'median_coeffs'. Varies by model. See model help for specifics.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing:

            - xl_array : numpy.ndarray
                Normalized location x/L along the rupture length.

            - displacement : numpy.ndarray
                Displacement in meters.

        Notes
        -----
        Parameters shown are typical; details vary with specific model.
        """

        if self.xl:
            message = (
                "`xl_step` is used in the profile calculation; `xl` attribute will be "
                f"ignored. The `xl_step` is currently set to {self.xl_step}."
            )
            logging.warning(message)
        return self._calc_displ_profile()

    def _calc_displ_profile(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Helper method to calculate displacement profile by looping over `displ_site`."""

        def _update_xl(instance: Any, _loc: float) -> float:
            """Update the xl of the instance and return the computed displacement."""
            instance.xl = _loc
            return instance.displ_site

        # Store original values
        original_xl = self.xl

        try:
            # Temporarily change the values
            xl_array = np.arange(0, 1 + self.xl_step, self.xl_step)
            displ_meters = np.array([_update_xl(self, loc) for loc in xl_array])

            # TODO: Check if fromiter is faster/better.
            # displ_meters = np.fromiter(
            #     (_update_xl(self, loc) for loc in xl_array), dtype=float, count=len(xl_array)
            # )

            return xl_array, displ_meters

        finally:
            # Restore original values
            self.xl = original_xl
