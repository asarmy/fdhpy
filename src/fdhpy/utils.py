"""Miscellaneous helper functions."""

import logging
from functools import wraps
from typing import Optional, Union

import numpy as np


def _required(*required_attrs, context_flag=False):
    """Decorator to ensure specified attributes are set before calling the method."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, "__req_attribute_check", False):
                # If validation has already failed, return None immediately
                return None

            # Determine the source for model_name and where to check attributes
            if context_flag:
                target = self.context
            else:
                target = self

            try:
                # Validate required attributes
                for attr in required_attrs:
                    if getattr(target, attr, None) is None:
                        e = AttributeRequiredError(attr, target._MODEL_NAME)
                        logging.error(e)
                        self.__req_attribute_check = True
                        return None

                # Call the original function
                return func(self, *args, **kwargs)

            except AttributeRequiredError as e:
                logging.error(e)
                self.__req_attribute_check = True
                return None

            except Exception:
                # logging.error(f"{e}`")
                self.__req_attribute_check = True
                return None

        return wrapper

    return decorator


def _trapezoidal_integration(*, x_array: np.ndarray, y_array: np.ndarray) -> float:
    """Numerical integration using trapezoid rule for the installed version of NumPy."""

    # NOTE: `trapz`` is being deprecated; `trapezoid` added in NumPy 2.0.0

    numpy_version = tuple(int(num) for num in np.__version__.split(".")[:3])
    if numpy_version[0] >= 2:
        return float(np.trapezoid(y_array, x_array))
    else:
        return float(np.trapz(y_array, x_array))  # type: ignore


class ColorFormatter(logging.Formatter):
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def format(self, record):
        if record.levelname == "ERROR":
            record.msg = f"{self.RED}{record.msg}{self.RESET}"
            record.levelname = f"{self.RED}{record.levelname}{self.RESET}"
        elif record.levelname == "WARNING":
            record.msg = f"{self.YELLOW}{record.msg}{self.RESET}"
            record.levelname = f"{self.YELLOW}{record.levelname}{self.RESET}"
        return super().format(record)


class AttributeRequiredError(Exception):
    def __init__(self, attribute_name: str, class_name: Optional[str] = None):
        if class_name:
            message = f"A '{attribute_name}' must be provided in `{class_name}`."
        else:
            message = f"A '{attribute_name}' must be provided."
        super().__init__(message)


class AttributeIgnoredWarning(UserWarning):
    def __init__(
        self, attribute_name: str, calculation_name: str, class_name: Optional[str] = None
    ):
        if class_name:
            message = (
                f"The '{attribute_name}' is not used in the {calculation_name} calculation in "
                f"`{class_name}`and will be ignored."
            )

        else:
            message = (
                f"The '{attribute_name}' is not used in the {calculation_name} calculation "
                "and will be ignored."
            )
        super().__init__(message)


class ValidNumericRangeWarning(UserWarning):
    def __init__(
        self,
        attribute_name: str,
        value: float,
        min_: float,
        max_: float,
        class_name: Optional[str] = None,
    ):
        if class_name:
            message = (
                f"The value '{value}' is not recommended for `{attribute_name}`. Expected a "
                f"value  between {min_} and {max_} for `{class_name}`. Calculations might not "
                "be valid."
            )
        else:
            message = (
                f"The value '{value}' is not recommended for `{attribute_name}`. Expected a "
                f"value between {min_} and {max_}. Calculations might not be valid."
            )
        super().__init__(message)


class ValidListWarning(UserWarning):
    def __init__(
        self,
        attribute_name: str,
        value: float,
        valid_options: Union[list, tuple],
        class_name: Optional[str] = None,
    ):
        if class_name:
            message = (
                f"The value '{value}' is not recommended for `{attribute_name}`. Expected one "
                f"of {valid_options} for `{class_name}`. Calculations might not be valid."
            )
        else:
            message = (
                f"The value '{value}' is not recommended for `{attribute_name}`. Expected one "
                f"of {valid_options}. Calculations might not be valid."
            )
        super().__init__(message)


class ValidListError(Exception):
    def __init__(
        self,
        attribute_name: str,
        value: float,
        valid_options: Union[list, tuple],
        class_name: Optional[str] = None,
    ):
        if class_name:
            message = (
                f"Invalid `{attribute_name}`: '{value}' was entered, but only {valid_options} "
                f"are allowed for `{class_name}`."
            )
        else:
            message = (
                f"Invalid `{attribute_name}`: '{value}' was entered, but only {valid_options} "
                "are allowed."
            )
        super().__init__(message)
