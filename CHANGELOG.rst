Version History
===============

Version 1.0.2 (2025-07-15)
--------------------------
Minor improvements:
- Fix typos in docstrings.
- Standardize return types: convert 1-element arrays to float across subclasses.
- KEA24: update `displ_site` to handle cases where back-transformed values are too small to calculate; now returns zero for displacements less than 1 mm.
- KEA24: update `stat_params_info` dictionary to include sigma_mag, sigma_xl, and Box-Cox transformation parameter lambda.
- LA23: expose gap and zero principal displacement probabilities as public properties.

Version 1.0.1 (2024-12-19)
--------------------------
- Fix boolean CLI arguments.
- Add `metric` attribute to CLI arguments.
- Allow smaller displacement values for cumulative distribution function and probability of exceedance calculations.
- Update MEA24 tests for alternative magnitude scaling models.
- Add alternative magnitude scaling model to MEA24 implementation.
- Update the order in which required attributes are checked for consistency.
- Improve handling of magnitude scaling parameters in normalized displacement models.
- Allow `None` for parameters in `stat_params` for normalized displacement models.

Version 1.0.0 (2024-12-01)
--------------------------
- Initial release as Python package.
