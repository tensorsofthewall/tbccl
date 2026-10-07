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
html_static_path = []

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
    html_theme_options = {"announcement": "Development documentation (not a release)"}

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
