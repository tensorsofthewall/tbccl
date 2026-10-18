"""Sphinx configuration for TBCCL. Build with `make docs` from the repository root."""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

project = "TBCCL"
author = "TBCCL contributors"
copyright = "TBCCL contributors"

_m = re.search(r"VERSION\s+(\d+\.\d+\.\d+)", (ROOT / "CMakeLists.txt").read_text(), re.M)
release = _m.group(1) if _m else "unknown"
version = release

extensions = ["myst_parser", "sphinx_copybutton", "breathe"]
root_doc = "index"
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = f"TBCCL {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/brand/favicon.svg"

# Visual identity: the shared TBCCL mark (see branding/README.md) and the Furo colour and type settings. Keep these values in step with the
# other TBCCL documentation sites; docs/_static/custom.css holds the few rules Furo has no option for.
_FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Inter, "Helvetica Neue", Arial, sans-serif'
_MONO = 'ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'
html_theme_options = {
    "light_logo": "brand/tbccl-mark.svg",
    "dark_logo": "brand/tbccl-mark-white.svg",
    "light_css_variables": {
        "font-stack": _FONT,
        "font-stack--monospace": _MONO,
        "color-brand-primary": "#0b57c9",
        "color-brand-content": "#0b57c9",
        "color-sidebar-brand-text": "#0f172a",
        "color-sidebar-background": "#f4f8fd",
        "color-sidebar-item-background--current": "#e3eefb",
        "color-sidebar-link-text--top-level": "#0f172a",
        "color-highlighted-background": "#cfe6fd",
        "color-link-underline": "#9cbcec",
        "color-link-underline--hover": "#0b57c9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4db8ff",
        "color-brand-content": "#4db8ff",
        "color-sidebar-brand-text": "#e6edf6",
        "color-sidebar-background": "#0e1622",
        "color-sidebar-link-text--top-level": "#e6edf6",
        "color-sidebar-item-background--current": "#16253a",
        "color-highlighted-background": "#123b63",
        "color-link-underline": "#2f6fa8",
        "color-link-underline--hover": "#4db8ff",
    },
}

# C and C++ API reference: Doxygen extracts XML from the public headers, Breathe renders it.
_xml = HERE / "_build" / "doxygen" / "xml"
(HERE / "_build" / "doxygen").mkdir(parents=True, exist_ok=True)
subprocess.run(["doxygen", "Doxyfile"], cwd=HERE, check=True)
breathe_projects = {"tbccl": str(_xml)}
breathe_default_project = "tbccl"
breathe_domain_by_extension = {"h": "c", "hpp": "cpp"}


# Hosting, versions and cross-project links.
import os

# Read the Docs sets READTHEDOCS_VERSION_TYPE to "tag" only for a build of a release tag. Every other build (main, a branch, a pull request, a local
# build) is development documentation and says so; nothing here labels unreleased documentation as stable.
_released_build = os.environ.get("READTHEDOCS_VERSION_TYPE") == "tag"
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
if not _released_build:
    html_theme_options["announcement"] = "Development documentation (not a release)"

# The four documentation sites. Until a release exists the references use each site's development (latest) version.
# Cross-project references are opt-in: the build never needs the network unless DOCS_INTERSPHINX=1. A local inventory can be given with
# DOCS_INVENTORY_<PROJECT> (for example DOCS_INVENTORY_TORCH_TBCCL=/path/to/objects.inv).
_SITES = {
    "tbccl": "https://tbccl.tensorsofthewall.com/en/latest/",
    "torch-tbccl": "https://torch-tbccl.tensorsofthewall.com/en/latest/",
    "vllm-tbccl": "https://vllm-tbccl.tensorsofthewall.com/en/latest/",
    "exo-tbccl": "https://exo-tbccl.tensorsofthewall.com/en/latest/",
}
if os.environ.get("DOCS_INTERSPHINX") == "1":
    extensions.append("sphinx.ext.intersphinx")
    intersphinx_timeout = 10
    intersphinx_mapping = {
        _name: (_url, os.environ.get("DOCS_INVENTORY_" + _name.upper().replace("-", "_")))
        for _name, _url in _SITES.items()
        if _name != "tbccl"
    }
