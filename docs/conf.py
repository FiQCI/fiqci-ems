import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FiQCI EMS'
copyright = '2026, FiQCI'
author = 'FiQCI'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    "sphinx.ext.autosummary",
    'myst_nb',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'show-inheritance': True,
    'private-members': False,
}

autosummary_generate = True
autosummary_ignore_patterns = []

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "jupyter_execute",
    "build",
    "test",
    ".venv",
    "docs-venv",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.md': 'myst-nb',
}

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

nb_execution_mode = "off"

add_module_names = False

sys.path.insert(0, os.path.abspath('../src/fiqci/ems'))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "show_navbar_depth": 2,
    "collapse_navigation": True,
}
html_static_path = ['_static']
