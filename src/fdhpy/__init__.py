"""fdhpy: Fault displacement models implemented in Python."""

from .chiou_et_al_2025 import ChiouEtAl2025
from .kuehn_et_al_2024 import KuehnEtAl2024
from .lavrentiadis_abrahamson_2023 import LavrentiadisAbrahamson2023
from .moss_et_al_2024 import MossEtAl2024
from .moss_ross_2011 import MossRoss2011
from .petersen_et_al_2011 import PetersenEtAl2011
from .youngs_et_al_2003 import YoungsEtAl2003

__all__ = [
    "YoungsEtAl2003",
    "PetersenEtAl2011",
    "MossRoss2011",
    "LavrentiadisAbrahamson2023",
    "MossEtAl2024",
    "KuehnEtAl2024",
    "ChiouEtAl2025",
]


# Custom attribute to provide abridged help
#  python -c "import fdhpy; print(fdhpy.__quickstart__)"
__quickstart__ = """\nfdhpy Quick Usage Guide

Usage instructions for both CLI and module-level interactions with the `fdphy` package:

-----------------------------------
Available Models as Package Modules
-----------------------------------

The following fault displacement models are currently available and may be accessed as separate
modules in the `fdphy` package:

- YoungsEtAl2003 : The Youngs et al. (2003) model (https://doi.org/10.1193/1.1542891).
- PetersenEtAl2011 : The Petersen et al. (2011) model (https://doi.org/10.1785/0120100035).
- MossRoss2011 : The Moss and Ross (2011) model (https://doi.org/10.1785/0120100248).
- LavrentiadisAbrahamson2023: The Lavrentiadis & Abrahamson model
  (https://doi.org/10.1177/87552930231201531).
- MossEtAl2024 : The Moss et al. (2024) model (https://doi.org/10.1177/87552930241288560).
- KuehnEtAl2024 : The Kuehn et al. (2024) model (https://doi.org/10.1177/87552930241291077).
- ChiouEtAl2025 : The Chiou et al. (2025) model (DOI TBD).

--------------------
Available Properties
--------------------

The following properties are available for each model as calculated attributes of an instance:

- displ_site : Calculate deterministic scenario displacement in meters.
- displ_avg : Calculate average displacement (AD) in meters.
- displ_max : Calculate maximum displacement (MD) in meters.
- displ_profile : Calculate displacement profile in meters.
- cdf : Calculate probability that the displacement is less than or equal to a specific value.
- prob_exceed : Calculate probability that the displacement exceeds a specific value.

----------------------------
Command-Line Interface (CLI)
----------------------------

The CLI syntax follows this pattern:
    $ fd-ModelAbbreviation method [options]

For example:
    $ fd-mr11 displ_avg -m 7.5 -p 0.84

The options can be viewed with:
    $ fd-ModelAbbreviation --help

For example:
    $ fd-mr11 --help

The required options and valid inputs vary with the fault displacement model and method. The
documentation for a specific model can be viewed with:
    $ python -c "from fdhpy import fd-ModelAbbreviation; print(fd-ModelAbbreviation.__doc__)"

For example:
    $ python -c "from fdhpy import fd-mr11; print(fd-mr11.__doc__)"

Model Module Names & Abbreviations:

- YoungsEtAl2003 : yea03
- PetersenEtAl2011 : pea11
- MossRoss2011 : mr11
- LavrentiadisAbrahamson2023: la23
- MossEtAl2024 : mea24
- KuehnEtAl2024 : kea24
- ChiouEtAl2025 : cea25

----------------------
Module-Level Interface
----------------------

To use the package at the module level, you can import and call the methods directly:

    from fdhpy import PetersenEtAl2011
    print(PetersenEtAl2011.__doc__)
    print(PetersenEtAl2011.displ_site.__doc__)

    model = PetersenEtAl2011(
        magnitude=7,
        xl=0.1,
        percentile=0.84,
        metric="principal",
        version="elliptical",
    )
    result = model.displ_site

"""
