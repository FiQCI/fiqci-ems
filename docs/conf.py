import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FiQCI EMS"
copyright = "2026, FiQCI"
author = "FiQCI"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	"sphinx.ext.autodoc",
	"sphinx.ext.coverage",
	"sphinx.ext.napoleon",
	"sphinx.ext.autosummary",
	"myst_nb",
	"sphinx_design",
]

autodoc_default_options = {
	"members": True,
	"member-order": "bysource",
	"special-members": "__init__",
	"undoc-members": False,
	"show-inheritance": True,
	"private-members": False,
}

autosummary_generate = True
autosummary_ignore_patterns = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute", "build", "test", ".venv", "docs-venv"]

source_suffix = {".rst": "restructuredtext", ".ipynb": "myst-nb", ".md": "myst-nb"}

myst_enable_extensions = ["amsmath", "dollarmath"]

nb_execution_mode = "auto"

add_module_names = False

sys.path.insert(0, os.path.abspath("../src"))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {"collapse_navigation": True}
html_logo = "_static/images/FiQCI-logo-nobg.png"
html_favicon = "_static/images/FiQCI-logo-nobg.png"
html_title = "FiQCI EMS"
html_baseurl = "https://fiqci.fi/fiqci-ems/docs/"
html_static_path = ["_static"]
