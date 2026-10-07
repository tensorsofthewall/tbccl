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
