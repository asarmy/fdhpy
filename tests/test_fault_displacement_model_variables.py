"""
Unit tests for default initialization, value validation, and mutability constraints in the
FaultDisplacementModelVariables data class.
"""

import numpy as np
import pytest

from fdhpy.fault_displacement_model import FaultDisplacementModelVariables


class Dummy(FaultDisplacementModelVariables):
    _CONDITIONS = {
        "style": {
            "reverse": {"magnitude": (5.5, 8)},
            "normal": {"magnitude": (6, 8)},
        },
        "metric": {
            "principal": {"version": ("quadratic", "sinsqrt")},
            "distributed": {"version": ("elliptical",)},
        },
    }

    def __init__(self, **kwargs):
        kwargs.setdefault("metric", None)
        kwargs.setdefault("version", None)
        super().__init__(**kwargs)


def test_default_initialization():
    """Test default class attribute initialization."""
    # metric is the only one required; other attributes are instance defaults
    dummy = Dummy(metric="principal")
    assert dummy.metric == "principal"
    assert dummy.style is None
    assert dummy.magnitude is None
    assert dummy.xl is None
    assert dummy.xl_step == 0.05
    assert dummy.percentile is None
    assert dummy.version is None  #
    assert isinstance(dummy.displ_array, np.ndarray)


def test_valid_inputs():
    """Should not raise errors or warnings."""
    dummy = Dummy(magnitude=6.5, xl=0.7, percentile=0.5, style="reverse", metric="principal")
    assert dummy.magnitude == 6.5
    assert dummy.xl == 0.7
    assert dummy.percentile == 0.5
    assert dummy.style == "reverse"
    assert dummy.version is None

    # Magnitude, xl, percentile are mutable
    dummy.magnitude = 7
    assert dummy.magnitude == 7
    dummy.xl = 0
    assert dummy.xl == 0
    dummy.percentile = -1
    assert dummy.percentile == -1

    # Case-insensitive params
    dummy = Dummy(magnitude=6.5, style="ReVErSe", metric="DISTributed", version="ELLIPTICAL")
    assert dummy.style == "reverse"
    assert dummy.metric == "distributed"
    assert dummy.version == "elliptical"

    # Displacement array can be list or array
    dummy.displ_array = [0.01, 1, 10]
    assert np.allclose(dummy.displ_array, np.array([0.01, 1, 10]))


def test_invalid_inputs_logging(caplog):
    """Should raise errors or warnings managed with logging."""

    with caplog.at_level("ERROR"):
        # Missing `magnitude`
        Dummy(style="reverse", metric="principal")

    with caplog.at_level("ERROR"):
        # Missing `metric`
        Dummy(magnitude=6.5, style="reverse")

    with caplog.at_level("ERROR"):
        # Invalid `metric`
        Dummy(magnitude=6.5, style="reverse", metric="sum principal")

    with caplog.at_level("ERROR"):
        # Invalid `version`
        Dummy(magnitude=6.5, style="reverse", metric="principal", version="d/ad")

    with caplog.at_level("WARNING"):
        # Not recommended `style`
        Dummy(magnitude=6.5, style="strike-slip", metric="principal")

    with caplog.at_level("WARNING"):
        # Not recommended `magnitude`
        Dummy(magnitude=4, style="reverse", metric="principal")


def test_invalid_inputs_standard():
    """Should raise built-in errors or warnings derived from `attrs`."""

    with pytest.raises(ValueError):
        # Invalid `xl`
        Dummy(magnitude=6.5, xl=4, style="reverse", metric="principal")

    with pytest.raises(ValueError):
        # Invalid `percentile`
        Dummy(magnitude=6.5, percentile=50, style="reverse", metric="principal")

    with pytest.raises(AttributeError):
        dummy = Dummy(magnitude=6.5, style="reverse", metric="principal")
        # Immutable attribute `style`
        dummy.style = "normal"

    with pytest.raises(AttributeError):
        dummy = Dummy(magnitude=6.5, style="reverse", metric="principal")
        # Immutable attribute `metric`
        dummy.metric = "aggregate"

    with pytest.raises(TypeError):
        # Only one value allowed for `xl`
        Dummy(magnitude=6.5, xl=[0.2, 0.5], style="reverse", metric="principal")

    with pytest.raises(TypeError):
        # Only one value allowed for `magnitude`
        Dummy(magnitude=[6, 6.1], style="reverse", metric="principal")

    with pytest.raises(TypeError):
        # Only one value allowed for `percentile`
        Dummy(magnitude=6.5, percentile=[0.5, 0.84], style="reverse", metric="principal")

    with pytest.raises(TypeError):
        # Only one value allowed for `style`
        Dummy(magnitude=6.5, style=["normal", "reverse"], metric="principal")

    with pytest.raises(TypeError):
        # Only one value allowed for `metric`
        Dummy(magnitude=6.5, style="reverse", metric=["principal", "distributed"])

    with pytest.raises(TypeError):
        # lists not allowed
        Dummy(magnitude=6.5, xl=[0.2], style="reverse", metric="principal")

    with pytest.raises(TypeError):
        # lists not allowed
        Dummy(magnitude=[7], style="reverse", metric="principal")

    with pytest.raises(TypeError):
        # lists not allowed
        Dummy(magnitude=6.5, percentile=[0.5], style="reverse", metric="principal")

    with pytest.raises(TypeError):
        # lists not allowed
        Dummy(magnitude=6.5, style=["reverse"], metric="principal")
