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

# Visual identity: the shared TBCCL mark (branding/README.md in the tbccl repository), the neutral foundation and the project accent. Everything below _ACCENT is
# identical on the four TBCCL documentation sites; docs/_static/custom.css holds the few rules Furo has no option for.
# Project identity: the only values that differ between the TBCCL documentation sites. Light and dark are the accent used for links and
# controls on a light and on a dark page; the contrast of both is checked against the page, sidebar and tinted surfaces.
_ACCENT = {"light": "#1155BB", "dark": "#86B6FF"}

# Accents of all four projects (identical on every site): the ecosystem navigation and the architecture diagram colour each project with its own.
_ECO = {"tbccl": ("#1155BB", "#86B6FF"), "torch": ("#BF341A", "#FF9174"), "vllm": ("#523781", "#BCA4F4"), "exo": ("#026E2B", "#58D68D")}

_FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Inter, "Helvetica Neue", Arial, sans-serif'
_MONO = 'ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'


def _mix(color, base, share):
    """`share` of `color` over `base` (both #rrggbb), as #rrggbb."""
    c, b = (tuple(int(h[i : i + 2], 16) for i in (1, 3, 5)) for h in (color, base))
    return "#" + "".join(f"{round(x * share + y * (1 - share)):02x}" for x, y in zip(c, b))


# Shared neutral foundation (the same on every site) and the accent-derived tokens (hover, visited, tints, selection).
_LIGHT = {"page": "#ffffff", "sidebar": "#f6f7f9", "text": "#1a1d21", "muted": "#5b6168", "border": "#d9dde3", "stripe": "#f7f8fa"}
_DARK = {"page": "#131416", "sidebar": "#0e1013", "text": "#e6e8eb", "muted": "#a3a9b1", "border": "#2a2f36", "stripe": "#191c20"}


def _tokens(n, accent, dark):
    a = accent
    return {
        "color-brand-primary": a,
        "color-brand-content": a,
        "color-brand-visited": _mix(a, n["muted"], 0.6),
        "color-link": a,
        "color-link--hover": _mix(a, "#ffffff" if dark else "#000000", 0.75),
        "color-link--visited": _mix(a, n["muted"], 0.6),
        "color-link--visited--hover": _mix(a, "#ffffff" if dark else "#000000", 0.75),
        "color-link-underline": _mix(a, n["page"], 0.45),
        "color-link-underline--hover": a,
        "color-link-underline--visited": _mix(a, n["page"], 0.3),
        "color-link-underline--visited--hover": a,
        "color-sidebar-background": n["sidebar"],
        "color-sidebar-brand-text": n["text"],
        "color-sidebar-link-text--top-level": n["text"],
        "color-sidebar-item-background--hover": _mix(a, n["sidebar"], 0.07 if not dark else 0.12),
        "color-sidebar-item-background--current": _mix(a, n["sidebar"], 0.11 if not dark else 0.20),
        "color-highlighted-background": _mix(a, n["page"], 0.22 if not dark else 0.30),
        "color-announcement-background": _mix(_ACCENT["light"], "#000000", 0.7 if dark else 1.0),
        "color-announcement-text": "#ffffff",
        "tb-on-accent": "#0b0d10" if dark else "#ffffff",
        **{f"tb-accent-{k}": v[1 if dark else 0] for k, v in _ECO.items()},
        "tbccl-border": n["border"],
        "tbccl-stripe": n["stripe"],
        "tbccl-tint": _mix(a, n["page"], 0.06 if not dark else 0.10),
    }


html_theme_options = {
    "light_logo": "brand/tbccl-mark.svg",
    "dark_logo": "brand/tbccl-mark-white.svg",
    "light_css_variables": {"font-stack": _FONT, "font-stack--monospace": _MONO, **_tokens(_LIGHT, _ACCENT["light"], False)},
    "dark_css_variables": _tokens(_DARK, _ACCENT["dark"], True),
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
