#!/usr/bin/env python3
"""Build the TBCCL wordmark lockups (mark + project name + connector) as portable SVG.

The TBCCL mark is copied unchanged from branding/logo/tbccl-mark.svg. The project name is shaped with HarfBuzz (kerning included) and
converted to SVG paths, so the finished files need no fonts and no Fontconfig. The script never substitutes a typeface: a missing font,
a wrong weight or an unexpected family stops it with an error.

  python3 branding/tools/build_lockups.py                      # all projects, default typeface, into branding/lockups
  python3 branding/tools/build_lockups.py --project torch-tbccl --out /tmp/out
  python3 branding/tools/build_lockups.py --check              # regenerate and compare with the committed files
  python3 branding/tools/build_lockups.py --typeface plex-sans-600

Fonts are searched for by file name in --font-dir (repeatable), $TBCCL_FONT_DIRS (os.pathsep separated), ~/.local/share/fonts, ~/.fonts
and /usr/share/fonts. Requires: pip install uharfbuzz fonttools. Font binaries are not part of this repository.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRANDING = HERE.parent
SPEC = BRANDING / "lockups.json"
MARK_SVG = BRANDING / "logo" / "tbccl-mark.svg"
SCHEMA = 1

# typeface id -> how to find and verify the file. "variations" selects a real instance of a variable font; nothing is synthesised.
TYPEFACES = {
    "inter-600": dict(family="Inter", style="SemiBold", file="Inter-SemiBold.otf", weight=600, variations=None,
                      license="SIL Open Font License 1.1", copyright="Copyright 2020 The Inter Project Authors (https://github.com/rsms/inter)"),
    "space-grotesk-600": dict(family="Space Grotesk", style="SemiBold (wght=600 instance)", file="SpaceGrotesk[wght].ttf", weight=600,
                              variations={"wght": 600}, license="SIL Open Font License 1.1",
                              copyright="Copyright 2020 The Space Grotesk Project Authors (https://github.com/floriankarsten/space-grotesk)"),
    "plex-sans-600": dict(family="IBM Plex Sans", style="SemiBold", file="IBMPlexSans-SemiBold.ttf", weight=600, variations=None,
                          license="SIL Open Font License 1.1", copyright="Copyright 2017 IBM Corp. (https://github.com/IBM/plex)"),
    "plex-sans-400": dict(family="IBM Plex Sans", style="Regular", file="IBMPlexSans-Regular.ttf", weight=400, variations=None,
                          license="SIL Open Font License 1.1", copyright="Copyright 2017 IBM Corp. (https://github.com/IBM/plex)"),
    "plex-mono-400": dict(family="IBM Plex Mono", style="Regular", file="IBMPlexMono-Regular.ttf", weight=400, variations=None,
                          license="SIL Open Font License 1.1", copyright="Copyright 2017 IBM Corp. (https://github.com/IBM/plex)"),
}
DEFAULT_TYPEFACE = "inter-600"

# Geometry shared by every project. Lengths are in output pixels at the nominal size; the files scale freely.
LAYOUTS = {
    # mark height, type size, gap between mark and name, connector length / stroke / node radius (in type sizes), tracking (em)
    "horizontal": dict(mark=64, size=42, gap=20, c_len=0.90, c_stroke=0.075, c_node=0.105, c_gap=0.11, c_side=0.14, track=-0.010),
    "compact": dict(mark=44, size=31, gap=14, c_len=0.80, c_stroke=0.095, c_node=0.125, c_gap=0.13, c_side=0.13, track=0.0),
    "stacked": dict(mark=88, size=46, gap=22, c_len=0.90, c_stroke=0.075, c_node=0.105, c_gap=0.11, c_side=0.14, track=-0.010),
}
THEMES = {
    "light": dict(text="#1A1D21", mark="colour", accent="light"),
    "dark": dict(text="#E6E8EB", mark="white", accent="dark"),
    "mono-black": dict(text="#0F172A", mark="#0F172A", accent=None),
    "mono-white": dict(text="#FFFFFF", mark="#FFFFFF", accent=None),
}


class BuildError(Exception):
    pass


def n(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def font_dirs(extra):
    dirs = [Path(d) for d in extra]
    dirs += [Path(d) for d in os.environ.get("TBCCL_FONT_DIRS", "").split(os.pathsep) if d]
    dirs += [Path.home() / ".local/share/fonts", Path.home() / ".fonts", Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
    return [d for d in dirs if d.is_dir()]


def locate(filename, dirs):
    for d in dirs:
        for root, _, files in os.walk(d):
            if filename in files:
                return Path(root) / filename
    raise BuildError(f"font file {filename!r} not found in {[str(d) for d in dirs]}; install it or pass --font-dir (no substitute is used)")


class Face:
    """One verified font file, shaped and outlined through HarfBuzz."""

    def __init__(self, tid, dirs):
        import uharfbuzz as hb
        from fontTools.ttLib import TTFont

        self.hb, self.tid, self.info = hb, tid, TYPEFACES[tid]
        self.path = locate(self.info["file"], dirs)
        tt = TTFont(str(self.path), lazy=True)
        names = tt["name"]
        family = names.getDebugName(16) or names.getDebugName(1)
        if self.info["family"].lower() not in (family or "").lower():
            raise BuildError(f"{self.path}: family is {family!r}, expected {self.info['family']!r}")
        if self.info["variations"]:
            axes = {a.axisTag: (a.minValue, a.maxValue) for a in tt["fvar"].axes} if "fvar" in tt else {}
            for tag, val in self.info["variations"].items():
                if tag not in axes or not axes[tag][0] <= val <= axes[tag][1]:
                    raise BuildError(f"{self.path}: axis {tag}={val} is not available (axes: {axes})")
        elif tt["OS/2"].usWeightClass != self.info["weight"]:
            raise BuildError(f"{self.path}: weight class {tt['OS/2'].usWeightClass}, expected {self.info['weight']}")
        self.version = names.getDebugName(5) or ""
        self.sha256 = sha256(self.path)
        self.upem = tt["head"].unitsPerEm
        blob = hb.Blob.from_file_path(str(self.path))
        self.font = hb.Font(hb.Face(blob))
        if self.info["variations"]:
            self.font.set_variations(self.info["variations"])

    def run(self, text, size, x, base, track=0.0):
        """Outline `text` with its left origin at x and baseline at base. Returns (path data, ink bbox (x0, y0, x1, y1), advance)."""
        from fontTools.pens.boundsPen import BoundsPen
        from fontTools.pens.svgPathPen import SVGPathPen
        from fontTools.pens.teePen import TeePen
        from fontTools.pens.transformPen import TransformPen

        hb = self.hb
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.font, buf, {"kern": True, "liga": False})
        k = size / self.upem
        d, bounds, cx = [], BoundsPen(None), 0.0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            svg = SVGPathPen(None, ntos=n)
            pen = TransformPen(TeePen(svg, bounds), (k, 0, 0, -k, x + cx + pos.x_offset * k, base - pos.y_offset * k))
            self.font.draw_glyph_with_pen(info.codepoint, pen)
            d.append(svg.getCommands())
            cx += pos.x_advance * k + track * size
        cx -= track * size
        if bounds.bounds is None:
            raise BuildError(f"{self.path}: no outline for {text!r}")
        return " ".join(c for c in d if c), bounds.bounds, cx

    def hyphen_centre(self, size):
        """Height of the middle of the hyphen above the baseline, in output pixels (what the connector lines up with)."""
        _, (_, y0, _, y1), _ = self.run("-", size, 0, 0)
        return -(y0 + y1) / 2


def mark_parts():
    src = MARK_SVG.read_text()
    paths = re.findall(r'<path d="([^"]+)" fill="url\(#tbccl-g(\d)\)"/>', src)
    grads = re.search(r"<defs>(.*)</defs>", src, re.S).group(1)
    if len(paths) != 5:
        raise BuildError(f"{MARK_SVG}: expected five strand paths, found {len(paths)}")
    return paths, grads


MARK_X0, MARK_Y0, MARK_X1, MARK_Y1 = 46.0, 54.0, 840.0, 554.0  # extent of the artwork in its own coordinates


def mark_element(kind, x, y, h, pid, paths):
    k = h / (MARK_Y1 - MARK_Y0)
    t = f'transform="translate({n(x - MARK_X0 * k)} {n(y - MARK_Y0 * k)}) scale({k:.5f})"'
    if kind == "colour":
        body = "".join(f'<path d="{d}" fill="url(#{pid}-g{i})"/>' for d, i in paths)
        return f'<g aria-hidden="true" {t}>{body}</g>', (MARK_X1 - MARK_X0) * k
    fill = "#FFFFFF" if kind == "white" else kind
    body = "".join(f'<path d="{d}"/>' for d, _ in paths)
    return f'<g aria-hidden="true" fill="{fill}" {t}>{body}</g>', (MARK_X1 - MARK_X0) * k


def connector(x, cy, size, lay, color):
    """Two nodes joined by a cable with a small break. Returns (svg, length)."""
    r, t, gap, ln = lay["c_node"] * size, lay["c_stroke"] * size, lay["c_gap"] * size, lay["c_len"] * size
    seg = (ln - 2 * r - gap) / 2
    nodes = (f'<path fill="{color}" d="M{n(x)} {n(cy)}a{n(r)} {n(r)} 0 1 0 {n(2 * r)} 0a{n(r)} {n(r)} 0 1 0 {n(-2 * r)} 0z'
             f'M{n(x + ln - 2 * r)} {n(cy)}a{n(r)} {n(r)} 0 1 0 {n(2 * r)} 0a{n(r)} {n(r)} 0 1 0 {n(-2 * r)} 0z"/>')
    cable = (f'<path fill="none" stroke="{color}" stroke-width="{n(t)}" stroke-linecap="round" '
             f'd="M{n(x + 2 * r - t / 2)} {n(cy)}h{n(seg - r + t / 2)}M{n(x + r + seg + gap)} {n(cy)}h{n(seg - r + t / 2)}"/>')
    return nodes + cable, ln


def place(face, project, layout, theme, spec, paths, grads, dx=0.0, dy=0.0, live=False):
    """Lay out one lockup with its top-left at (dx, dy); returns (elements, bbox)."""
    lay, th = LAYOUTS[layout], THEMES[theme]
    pr = spec["projects"][project]
    accent = pr["accent"][th["accent"]] if th["accent"] else th["text"]
    size, track, H = lay["size"], lay["track"], lay["mark"]
    pid = project.split("-")[0]
    left, right = pr.get("left"), pr.get("right", project)
    parts = []

    def word(s, x, base, d):
        """Outlined path, or live text in the editable sources."""
        if not live:
            return d
        i = face.info
        return (f'<text x="{n(x)}" y="{n(base)}" font-family="{i["family"]}" font-size="{n(size)}" font-weight="{i["weight"]}" '
                f'letter-spacing="{n(track * size)}">{s}</text>')

    def text_block(x, base):
        """The name starting at x on baseline `base`; returns (elements, ink bbox, width)."""
        if left is None:
            d, bb, adv = face.run(right, size, x, base, track)
            return [word(right, x, base, f'<path fill="{th["text"]}" d="{d}"/>').replace("<text ", f'<text fill="{th["text"]}" ', 1)], bb, bb[2] - x
        side = lay["c_side"] * size
        d1, b1, _ = face.run(left, size, x, base, track)
        cx0 = b1[2] + side
        cn, ln = connector(cx0, base - face.hyphen_centre(size), size, lay, accent)
        probe = face.run(right, size, 0, base, track)[1]
        rx = cx0 + ln + side - probe[0]
        d2, b2, _ = face.run(right, size, rx, base, track)
        ink = (b1[0], min(b1[1], b2[1]), b2[2], max(b1[3], b2[3]))
        if live:
            t = lambda w, wx: word(w, wx, base, "").replace("<text ", f'<text fill="{th["text"]}" ', 1)
            return [t(left, x), cn, t(right, rx)], ink, ink[2] - x
        return [f'<path fill="{th["text"]}" d="{d1} {d2}"/>', cn], ink, ink[2] - x

    # first measure the text ink box at a zero baseline to centre it on the mark (horizontal, compact) or below it (stacked)
    els, ink0, _ = text_block(0.0, 0.0)
    ink_w, ink_h = ink0[2] - ink0[0], ink0[3] - ink0[1]
    mark_w = (MARK_X1 - MARK_X0) * H / (MARK_Y1 - MARK_Y0)
    if layout == "stacked":
        W = max(mark_w, ink_w)
        mx = dx + (W - mark_w) / 2
        m, _ = mark_element(th["mark"], mx, dy, H, pid, paths)
        top = dy + H + lay["gap"]
        base = top - ink0[1]
        tx = dx + (W - ink_w) / 2 - ink0[0]
        els, ink, _ = text_block(tx, base)
        box = (dx, dy, dx + W, top + ink_h)
    else:
        m, _ = mark_element(th["mark"], dx, dy, H, pid, paths)
        base = dy + H / 2 - (ink0[1] + ink0[3]) / 2
        tx = dx + mark_w + lay["gap"] - ink0[0]
        els, ink, _ = text_block(tx, base)
        box = (dx, dy, ink[2], dy + H)
    parts.append(m)
    parts.extend(els)
    return parts, box


def build_svg(face, project, layout, theme, spec, paths, grads, live=False):
    _, (x0, y0, x1, y1) = place(face, project, layout, theme, spec, paths, grads)
    pad = 0.05 * LAYOUTS[layout]["mark"]
    els, _ = place(face, project, layout, theme, spec, paths, grads, pad - x0, pad - y0, live)
    W, H = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    pid = project.split("-")[0]
    th = THEMES[theme]
    defs = f"<defs>{grads.replace('tbccl-g', pid + '-g')}</defs>" if th["mark"] == "colour" else ""
    pr = spec["projects"][project]
    title = project
    desc = pr["description"]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{round(W)}" height="{round(H)}" viewBox="0 0 {n(W)} {n(H)}" role="img" '
            f'aria-labelledby="t d"><title id="t">{title}</title><desc id="d">{desc}</desc>{defs}{"".join(els)}</svg>\n')


def build_editable(face, project, layout, spec):
    """Light-theme source with live text for editing in a vector editor (needs the font installed; not for publication)."""
    svg = build_svg(face, project, layout, "light", spec, *mark_parts(), live=True)
    i = face.info
    note = (f"<!-- Editable source for {project} ({layout}): live text in {i['family']} {i['style']}. "
            f"Production files are generated by branding/tools/build_lockups.py with the text outlined. -->")
    return svg.replace("<svg ", note + "\n<svg ", 1)


def generate(face, projects, spec, out, editable):
    paths, grads = mark_parts()
    files = {}
    for project in projects:
        for layout in LAYOUTS:
            for theme in THEMES:
                files[f"{project}-{layout}-{theme}.svg"] = build_svg(face, project, layout, theme, spec, paths, grads)
            if editable:
                files[f"source/{project}-{layout}-editable.svg"] = build_editable(face, project, layout, spec)
    return files


def manifest(face, projects, files, command, spec):
    import fontTools
    import uharfbuzz

    info = face.info
    return {
        "schema_version": SCHEMA,
        "generator": "branding/tools/build_lockups.py",
        "command": command,
        "tools": {"uharfbuzz": uharfbuzz.__version__, "fonttools": fontTools.version},
        "typeface": {"id": face.tid, "family": info["family"], "style": info["style"], "weight": info["weight"],
                     "variations": info["variations"], "file": info["file"], "version": face.version, "file_sha256": face.sha256,
                     "license": info["license"], "copyright": info["copyright"], "embedded": False,
                     "note": "Text is converted to paths; no font data is embedded or distributed."},
        "mark": {"source": "branding/logo/tbccl-mark.svg", "sha256": sha256(MARK_SVG), "modified": False},
        "projects": {p: {"accent": spec["projects"][p]["accent"]} for p in projects},
        "files": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sorted(files.items())},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--typeface", default=DEFAULT_TYPEFACE, choices=[t for t in TYPEFACES if t.endswith("600")])
    ap.add_argument("--font-dir", action="append", default=[], help="directory to search for the font file (repeatable)")
    ap.add_argument("--project", action="append", help="project to build (repeatable; default all in lockups.json)")
    ap.add_argument("--out", default=str(BRANDING / "lockups"))
    ap.add_argument("--editable", action="store_true", help="also write the live-text sources into <out>/source")
    ap.add_argument("--check", action="store_true", help="regenerate and compare with --out; write nothing")
    ap.add_argument("--manifest-name", default="MANIFEST.json")
    a = ap.parse_args(argv)
    try:
        spec = json.loads(SPEC.read_text())
        projects = a.project or list(spec["projects"])
        for p in projects:
            if p not in spec["projects"]:
                raise BuildError(f"unknown project {p!r}; known: {list(spec['projects'])}")
        face = Face(a.typeface, font_dirs(a.font_dir))
        files = generate(face, projects, spec, Path(a.out), a.editable)
        command = "python3 branding/tools/build_lockups.py" + "".join(f" --project {p}" for p in a.project or []) + (
            f" --typeface {a.typeface}" if a.typeface != DEFAULT_TYPEFACE else "") + (" --editable" if a.editable else "")
        man = manifest(face, projects, files, command, spec)
        out = Path(a.out)
        if a.check:
            bad = [name for name, text in files.items() if not (out / name).is_file() or (out / name).read_text() != text]
            if bad:
                raise BuildError(f"{len(bad)} file(s) differ from a fresh build: {bad[:4]}")
            print(f"ok: {len(files)} files match a fresh build with {face.info['family']} {face.version}")
            return 0
        for name, text in files.items():
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            (out / name).write_text(text)
        (out / a.manifest_name).write_text(json.dumps(man, indent=2) + "\n")
        print(f"wrote {len(files)} files to {out} using {face.info['family']} {face.version}")
        return 0
    except BuildError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"error: {e}; install uharfbuzz and fonttools", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
