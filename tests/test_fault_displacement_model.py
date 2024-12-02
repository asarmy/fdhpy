"""Unit tests for FaultDisplacementModel abstract base class."""

from typing import Optional

import numpy as np
import pandas as pd

from fdhpy.fault_displacement_model import FaultDisplacementModel


class Dummy(FaultDisplacementModel):
    _CONDITIONS = {
        "style": {
            "strike-slip": {"magnitude": (6, 8)},
        },
        "metric": {
            "principal": {"version": ("elliptical",)},
        },
    }

    @property
    def _folded_xl(self) -> Optional[float]:
        return self._calc_folded_xl()

    @property
    def _xstar(self) -> Optional[float]:
        return self._calc_xstar()

    def _statistical_distribution_params(self) -> None:
        pass

    def _calc_displ_site(self) -> Optional[float]:
        pass

    def _calc_displ_avg(self) -> Optional[float]:
        pass

    def _calc_displ_max(self) -> Optional[float]:
        pass

    def _calc_cdf(self) -> Optional[np.ndarray]:
        pass


def test_valid_inputs():
    """Should not raise errors."""
    dummy = Dummy(
        magnitude=7, xl=0.7, style="strike-slip", metric="principal", version="elliptical"
    )
    assert np.isclose(dummy._folded_xl, 0.3)
    assert np.isclose(dummy._xstar, 0.916515)

    dummy.xl_step = 0.2
    xl_array, _ = dummy.displ_profile
    assert np.allclose(xl_array, np.array([0, 0.2, 0.4, 0.6, 0.8, 1]))


def test_load_data():
    """Should not raise errors."""
    filename = "chiou_2025_coefficients.csv"
    df = Dummy._load_data(filename=filename)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.iloc[0]["c0"] == 1.82276
