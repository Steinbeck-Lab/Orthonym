"""Sphinx configuration for the Orthonym documentation site.

Build from the repository root:

    sphinx-build -W -b html guide _build/html

The site is Markdown (MyST) in this folder, themed in the Orthonym world with
the Shibuya theme. Everything the page needs is in _static: no font, script or
style is fetched from another site, so the built folder works offline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "_ext"))

# Importing orthonym for the API pages never names a molecule, so the OPSIN and
# centres jars are not needed to build the site.
os.environ.setdefault("ORTHONYM_SKIP_JAR_PREFETCH", "1")

import orthonym  # noqa: E402

project = "Orthonym"
author = "Kohulan Rajan, Achim Zielesny, Christoph Steinbeck"
copyright = "2026, Kohulan Rajan, Achim Zielesny and Christoph Steinbeck. MIT licence"
release = orthonym.__version__
version = release
language = "en"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "orthonym_site",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "DESIGN.md", "_data", "_ext", "**/.gitignore"]
templates_path = ["_templates"]

# ---------------------------------------------------------------- Markdown
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "attrs_inline", "attrs_block"]
myst_heading_anchors = 3
# "#target" links resolve against headings and Python objects only.
myst_ref_domains = ["std", "py"]

# ---------------------------------------------------------------- API pages
autodoc_member_order = "bysource"
toc_object_entries_show_parents = "hide"
autoclass_content = "class"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

# The internals pages render the docstrings of all engine modules as they are
# written for the developers. Many of those docstrings are notes in reStructured
# Text that is not quite well formed, or name objects that are not documented
# pages; those warnings are about the internal notes, not about this site, so
# they are silenced for the internals pages only (orthonym_site filters them
# by page). Every other page must build with no warning at all.
orthonym_internals_prefix = "reference/internals/"

# ---------------------------------------------------------------- copy button
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
copybutton_exclude = ".linenos, .gp, .go"

# ---------------------------------------------------------------- HTML
html_theme = "shibuya"
html_title = "Orthonym"
html_short_title = "Orthonym"
html_static_path = ["_static"]
html_css_files = ["orthonym.css"]
html_favicon = "_static/favicon.svg"
html_permalinks_icon = "#"
html_show_sourcelink = False
html_copy_source = False
html_last_updated_fmt = None
html_baseurl = os.environ.get("ORTHONYM_DOCS_BASEURL", "")

html_theme_options = {
    "color_mode": "auto",
    "accent_color": "gray",
    "globaltoc_expand_depth": 1,
    "toctree_collapse": False,
    "toctree_titles_only": True,
    "show_ai_links": False,
    "nav_links": [
        {"title": "Start", "url": "start/install"},
        {"title": "Use", "url": "use/command-line"},
        {"title": "Understand", "url": "checking"},
        {"title": "Reference", "url": "reference/python-api"},
        {"title": "Project", "url": "project/contributing"},
    ],
}
html_context = {
    "try_url": "https://orthonym.decimer.ai",
    "repo_url": "https://github.com/Beilstein-Institut/Orthonym",
}
html_sidebars = {"**": ["sidebars/localtoc.html"]}
