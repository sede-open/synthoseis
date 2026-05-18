# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Make the repo root importable for autodoc
sys.path.insert(0, str(Path(__file__).parents[2]))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "Synthoseis"
copyright = "2024, sede-open contributors"
author = "sede-open contributors"
release = "0.1"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",        # Google / NumPy docstrings
    "sphinx.ext.viewcode",        # [source] links
    "sphinx_copybutton",          # copy button on code blocks
    "sphinx_design",              # cards, tabs, grids
    "myst_parser",                # Markdown support
]

# ---------------------------------------------------------------------------
# MyST (Markdown) options
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",   # ::: directive syntax
    "deflist",       # definition lists
    "fieldlist",
    "attrs_inline",
]
myst_heading_anchors = 3

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# HTML output — Furo theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "Synthoseis"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view"],
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/sede-open/synthoseis",
            "html": """<svg stroke="currentColor" fill="currentColor" stroke-width="0"
                viewBox="0 0 16 16" height="1em" width="1em"
                xmlns="http://www.w3.org/2000/svg">
                <path fill-rule="evenodd"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87
                2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18
                1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16
                1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73
                .54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8
                c0-4.42-3.58-8-8-8z"></path></svg>""",
            "class": "",
        },
    ],
}

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
nautodoc_preserve_defaults = True
nautodoc_mock_imports = []
# Silence malformed docstrings in the existing source code
nautodoc_show_class_docstring = True
autodoc_inherit_docstrings = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# Don't let upstream docstring formatting errors abort the build
nitpicky = False

# ---------------------------------------------------------------------------
# Suppress noisy warnings from malformed upstream docstrings
# ---------------------------------------------------------------------------
suppress_warnings = [
    "autodoc",          # catches duplicate object descriptions from napoleon
    "ref.python",
]
