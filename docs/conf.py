# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# project = 'fdhpy'
# copyright = '2024, Alex Sarmiento'
# author = 'Alex Sarmiento'
# release = '0.1.0'

import toml
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

with open(os.path.join(os.path.dirname(__file__), "../pyproject.toml"), "r") as f:
    pyproject_data = toml.load(f)

project = pyproject_data["project"]["name"]
author = pyproject_data["project"]["authors"][0]["name"]
copyright = f"2024, {author}"
release = pyproject_data["project"]["version"]


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# -- Additional configurations -------------------------------------------------

autodoc_default_options = {
    "private-members": False,
    "members": True,
    "inherited-members": True,
}

# Needed to get docstrings to show on RTD
autodoc_mock_imports = ["numpy", "pandas", "scipy"]
