import logging
from typing import Optional

import numpy as np
from attrs import asdict, converters, define, field, setters, validators

from fdhpy.utils import (
    AttributeRequiredError,
    ColorFormatter,
    ValidListError,
    ValidListWarning,
    ValidNumericRangeWarning,
)

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = ColorFormatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@define(slots=True, kw_only=True)
class FaultDisplacementModelVariables:
    """
    Data class for defining input variables and performing validations for fault displacement
    models.

    Functions as a data container for predictor variables with built-in validation logic.
    """

    _CONDITIONS = {}  # provided in the specific model subclass
    _MODEL_NAME = None  # provided in the specific model subclass
    _DEFAULT_MAGNITUDE_RANGE = (5.5, 8.3)  # Default _CONDITIONS

    # NOTE: Some attributes are frozen to avoid introducing complex management of mutable states
    # (e.g., style-magnitude and metric-version combined validations).

    style: Optional[str] = field(
        default=None,
        converter=converters.optional(str.lower),
        on_setattr=setters.frozen,
    )

    magnitude: Optional[float] = field(
        default=None,
        converter=converters.optional(float),
    )

    xl: Optional[float] = field(
        default=None,
        converter=converters.optional(float),
        validator=validators.optional(
            [validators.ge(0), validators.le(1)]  # folding is handled separately if needed
        ),
    )

    percentile: Optional[float] = field(
        default=None,
        converter=converters.optional(float),
        validator=validators.optional(
            validators.and_(
                validators.or_(
                    validators.and_(validators.ge(0), validators.le(1)),
                    validators.in_([-1]),  # use -1 for mean
                )
            )
        ),
    )

    xl_step: float = field(  # profile step size
        default=0.05,
        converter=float,
        validator=validators.and_(validators.ge(0), validators.le(0.2)),
    )

    displ_array: np.ndarray = field(
        default=np.logspace(start=np.log(0.001), stop=np.log(80), num=25, base=np.e),
        converter=lambda x: np.atleast_1d(np.asarray(x, dtype=float)),
        validator=validators.deep_iterable(
            member_validator=validators.and_(validators.ge(0.001), validators.le(100)),
            iterable_validator=validators.instance_of(np.ndarray),
        ),
    )

    metric: Optional[str] = field(
        default=None,
        converter=converters.optional(str.lower),
        on_setattr=setters.frozen,
    )

    version: Optional[str] = field(
        default=None,
        converter=converters.optional(str.lower),
        on_setattr=setters.frozen,
    )

    # TODO: initialize attributes for distributed displacement models, e.g. distance to rupture

    # Customize instance string representation for key attributes
    def __str__(self):
        exclude_attrs = {"displ_array", "xl_step"}

        attributes = asdict(
            self, recurse=False, filter=lambda attr, value: attr.name not in exclude_attrs
        )

        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in attributes.items())})"

    # Additional attribute validations
    @style.validator  # type: ignore
    def _validate_style(self, attribute, value):
        """Validate style against valid options."""

        if value is None:
            return

        if value not in self._CONDITIONS.get("style", {}):
            valid_style_options = tuple(self._CONDITIONS.get("style", {}).keys())
            message = ValidListWarning(
                attribute.name, value, valid_style_options, self._MODEL_NAME
            )
            logging.warning(message)

    @magnitude.validator  # type: ignore
    def _validate_magnitude(self, attribute, value):
        """Validate magnitude against valid options."""

        if value is None:
            return

        min_, max_ = (
            self._CONDITIONS.get("style", {})
            .get(self.style, {})
            .get("magnitude", self._DEFAULT_MAGNITUDE_RANGE)
        )

        if not (min_ <= value <= max_):
            message = ValidNumericRangeWarning(attribute.name, value, min_, max_, self._MODEL_NAME)
            logging.warning(message)

    @metric.validator  # type: ignore
    def _validate_metric(self, attribute, value):
        """Validate metric against valid options."""

        try:
            if value is None:
                raise AttributeRequiredError(attribute.name)

            valid_version_options = self._CONDITIONS.get("metric", {})
            if value not in valid_version_options:
                raise ValidListError(
                    attribute.name, value, valid_version_options, self._MODEL_NAME
                )

        except (AttributeRequiredError, ValidListError) as e:
            logging.error(e)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

    @version.validator  # type: ignore
    def _validate_version(self, attribute, value):
        """Validate version against valid options."""

        try:
            # NOTE: Allow `None` because not all methods require a version (e.g., `displ_avg`).

            valid_version_options = (
                self._CONDITIONS.get("metric", {}).get(self.metric, {}).get("version", [])
            )

            if (
                value is not None
                and self.metric is not None
                and value not in valid_version_options
            ):
                raise ValidListError(
                    attribute.name, value, valid_version_options, self._MODEL_NAME
                )

        except (ValueError, ValidListError) as e:
            logging.error(e)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
